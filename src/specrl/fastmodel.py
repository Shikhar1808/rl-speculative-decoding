"""CUDA-graph runner: removes per-step launch overhead for one-token-at-a-time decoding (DECISIONS.md #33).

On this laptop (Windows, RTX 3050) a batch-1 decode step is limited by how fast the CPU can launch GPU
kernels, not by model size: Qwen 0.5B and 1.5B both took ~40-45 ms per step. A CUDA graph records the whole
forward once and replays it with one launch (measured: 39.7 ms -> 9.7 ms for Qwen 0.5B).

GraphRunner keeps a StaticCache (fixed-size KV buffers) and one captured graph per input length 1..max_chunk
(draft steps feed 1-2 tokens, a verify feeds up to K+1). Longer inputs (the prompt) run eagerly into the same
cache. Rolling back = moving the write position back: stale entries past it are masked out and overwritten.

transformers 5 detail: StaticCache layers write at their own `cumulative_length` counter, not at the
cache_position passed in. So every forward first sets each layer's counter to the start position (inside
the graph, from one `start` tensor), and cache_position is computed from the same tensor.

Interface matches specdec._Runner: reset(), logits(ids) -> logits for the not-yet-seen part of ids, crop(n).
"""

import torch
from transformers import StaticCache

from specrl.config import K


class GraphRunner:
    def __init__(self, model, max_len=1024, max_chunk=K + 2):
        self.model, self.max_len = model, max_len
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        dev = model.device
        self.start = torch.zeros((), dtype=torch.long, device=dev)
        self.ids = {L: torch.zeros(1, L, dtype=torch.long, device=dev) for L in range(1, max_chunk + 1)}
        self.ar = {L: torch.arange(L, device=dev) for L in range(1, max_chunk + 1)}
        self.out, self.graphs = {}, {}
        with torch.no_grad():
            # warm up on a side stream (initialises the cache buffers; required before capture)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for L in self.ids:
                    for _ in range(2):
                        self._forward(self.ids[L], self.ar[L])
            torch.cuda.current_stream().wait_stream(s)
            pool = None
            for L in self.ids:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self.out[L] = self._forward(self.ids[L], self.ar[L])
                pool = g.pool()
                self.graphs[L] = g
        self.reset()

    def _forward(self, ids, ar):
        for layer in self.cache.layers:
            if layer.is_initialized:
                layer.cumulative_length.copy_(self.start)
        return self.model(ids, past_key_values=self.cache, cache_position=self.start + ar).logits[0]

    def reset(self):
        """Start a new sequence (old cache contents are masked out by position)."""
        self.n = 0
        return self

    @torch.no_grad()
    def logits(self, ids):
        start, end = self.n, ids.shape[1]
        if end > self.max_len:
            raise ValueError(f"sequence of {end} tokens exceeds GraphRunner max_len={self.max_len}")
        chunk = ids[:, start:end]
        L = chunk.shape[1]
        self.n = end
        self.start.fill_(start)
        if L in self.graphs:
            self.ids[L].copy_(chunk)
            self.graphs[L].replay()
            return self.out[L]  # graph output buffer: valid until the next replay of this length
        return self._forward(chunk, torch.arange(L, device=chunk.device))

    def crop(self, n):
        self.n = min(self.n, n)
