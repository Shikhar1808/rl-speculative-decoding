"""Step 5: real speculative decoding on the GPU, driven by any stopping policy (docs/05_live_benchmark.md).

    spec_generate(prompt_ids, draft, target, policy)   -> tokens + per-round trace
    target_generate(prompt_ids, target)                -> plain greedy decoding, the "target alone" baseline

`draft` / `target` can be plain models (DynamicCache, eager) or fastmodel.GraphRunner objects (StaticCache +
CUDA graphs, ~4x less per-step overhead on this laptop); the live benchmark and demo use GraphRunners for every
method, including the target-alone baseline. Use `load_fast_pair(pair)`.

Both use a KV cache (transformers DynamicCache). After each verify, both caches are cropped back so they
only hold accepted tokens. cache=False recomputes everything each step: slow, but the simplest correct
reference used by the tests.

Cache invariant: before a round, each cache holds a prefix of `ids` and at least the last token of `ids`
is NOT in it, so the next forward always has an input whose output predicts the next token.
"""

import torch
from transformers import DynamicCache

from specrl.config import K as K_DEFAULT
from specrl.config import MAX_NEW_TOKENS
from specrl.sim.env import CONTINUE, STOP
from specrl.live.fastmodel import GraphRunner


def crop_to(cache, n):
    """Keep only the first n tokens of a DynamicCache. (transformers 5: crop(-k) removes the last k tokens.)"""
    extra = cache.get_seq_length() - n
    if extra > 0:
        cache.crop(-extra)


def load_fast_pair(pair):
    """Tokenizer + GraphRunner for draft and target (captures CUDA graphs, ~10-20 s)."""
    from specrl.data.log_data import load_pair

    tok, draft, target = load_pair(pair)
    return tok, GraphRunner(draft), GraphRunner(target)


def _as_runner(m, cache):
    return m.reset() if isinstance(m, GraphRunner) else _Runner(m, cache)


def _eos_ids(m):
    gc = (m.model if isinstance(m, GraphRunner) else m).generation_config
    e = gc.eos_token_id
    return set(e if isinstance(e, (list, tuple)) else [e])


class _Runner:
    """One model + its cache; `next_logits(ids)` feeds whatever part of ids the cache hasn't seen."""

    def __init__(self, model, cache=True):
        self.model, self.use_cache = model, cache
        self.cache = DynamicCache() if cache else None
        self.n = 0  # tokens of ids already in the cache

    def logits(self, ids):
        """Logits for positions n..L-1 of ids (the part not seen before). ids: 1 x L, the full sequence so far."""
        start, self.n = self.n, ids.shape[1]
        if not self.use_cache:
            return self.model(ids).logits[0, start:]
        return self.model(ids[:, start:], past_key_values=self.cache, use_cache=True).logits[0]

    def crop(self, n):
        """Forget everything after the first n tokens."""
        self.n = min(self.n, n)
        if self.use_cache:
            crop_to(self.cache, self.n)


@torch.no_grad()
def target_generate(prompt_ids, target, max_new=MAX_NEW_TOKENS, cache=True, on_step=None):
    """Greedy decoding with the target alone. Returns list of new token ids. on_step(new_tokens) after each token."""
    eos = _eos_ids(target)
    run = _as_runner(target, cache)
    ids, new = prompt_ids, []
    while len(new) < max_new:
        tok = int(run.logits(ids)[-1].argmax())
        new.append(tok)
        if on_step:
            on_step([tok])
        ids = torch.cat([ids, ids.new_tensor([[tok]])], 1)
        if tok in eos:
            break
    return new


@torch.no_grad()
def spec_generate(prompt_ids, draft, target, policy, K=K_DEFAULT, max_new=MAX_NEW_TOKENS, cache=True, on_step=None):
    """Greedy speculative decoding. policy(obs) -> CONTINUE/STOP with obs = [s/K, p, H, q, a] (as in env.py).

    Returns (new_tokens, rounds), rounds = list of dicts: drafts, accepted n, s, p per draft, new tokens.
    on_step(round_dict) is called after each round (used by the demo to timestamp tokens).
    """
    eos = _eos_ids(target)
    d_run, t_run = _as_runner(draft, cache), _as_runner(target, cache)
    L0 = prompt_ids.shape[1]
    ids = prompt_ids
    acc_rate = 1.0
    rounds = []
    while ids.shape[1] - L0 < max_new:
        left = max_new - (ids.shape[1] - L0)
        # ---- draft
        drafts, ps, q, x = [], [], 1.0, ids
        while True:
            probs = d_run.logits(x)[-1].float().softmax(-1)
            p, tok = probs.max(-1)
            p, tok = float(p), int(tok)
            H = float(-(probs * probs.clamp_min(1e-9).log()).sum())
            drafts.append(tok)
            ps.append(p)
            q *= p
            s = len(drafts)
            x = torch.cat([x, x.new_tensor([[tok]])], 1)
            if s >= K or s >= left or tok in eos:
                break
            if policy([s / K, p, H, q, acc_rate]) == STOP:
                break
        # ---- verify: one target pass over the not-yet-cached tail + all drafts
        t_logits = t_run.logits(x)
        t_arg = t_logits[-(len(drafts) + 1) :].argmax(-1).tolist()  # target's choice at each draft + bonus
        n = 0
        while n < len(drafts) and t_arg[n] == drafts[n]:
            n += 1
        new = drafts[:n] + [t_arg[n]]
        new = new[:left]
        for i, tk in enumerate(new):  # stop at end-of-turn
            if tk in eos:
                new = new[: i + 1]
                break
        acc_rate = 0.8 * acc_rate + 0.2 * n / len(drafts)
        rounds.append({"s": len(drafts), "n": n, "drafts": drafts, "p": ps, "new": new})
        if on_step:
            on_step(rounds[-1])
        ids = torch.cat([ids, ids.new_tensor([new])], 1)
        keep = ids.shape[1] - 1  # invariant: last token not in either cache
        d_run.crop(keep)
        t_run.crop(keep)
        if new[-1] in eos:
            break
    return ids[0, L0:].tolist(), rounds


__all__ = ["spec_generate", "target_generate", "CONTINUE", "STOP"]
