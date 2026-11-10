# Live speculative decoding and the benchmark

The simulator trains the agents; this part proves the speedup is real on the GPU and that the output is
unchanged. Code: `src/specrl/live/specdec.py`, `src/specrl/live/fastmodel.py`, `src/specrl/live/bench.py`, test
`tests/test_specdec.py`.

## Why CUDA graphs (read this first)

On this laptop a one-token forward pass is limited by how fast the CPU can launch GPU kernels (Windows
driver), not by model size. Measured without graphs: Qwen 0.5B draft step 48.8 ms vs 1.5B verify 60.2 ms
(c = 0.81); SmolLM2 135M draft 47.3 ms vs 1.7B verify 32.8 ms (c = 1.44 — the small model is *slower*, it has
more layers). Speculation cannot win like that.

`fastmodel.GraphRunner` fixes it: each model gets a fixed-size KV cache (`StaticCache`, 1024 tokens) and one
recorded CUDA graph per input length 1–10. A step is then one graph replay instead of hundreds of kernel
launches: Qwen draft 10.7 ms, verify 26.5 ms (c = 0.40); SmolLM2 draft 6.6 ms, verify 26.7 ms (c = 0.25).
Inputs longer than 10 tokens (the prompt) run normally into the same cache. Every method in the benchmark,
including the target-alone baseline, uses the same runners.

transformers 5 detail: `StaticCache` layers write at their own internal counter, not at the position passed
in, so `GraphRunner` sets every layer's counter to the start position at the beginning of each forward
(inside the graph). Without that, writes landed in the wrong slots and outputs were wrong.

`load_fast_pair(pair)` returns the tokenizer and both runners (~15 s, includes capturing the graphs).

## How one round works (`spec_generate`)

1. **Draft.** The draft model produces one token at a time. After each token we compute the same 5
   features as the simulator (`[s/K, p, H, q, a]`) and ask the policy: CONTINUE or STOP. Drafting also stops
   at `s = K = 8`, at the token budget, or if the draft produces the end-of-turn token.
2. **Verify.** One target forward pass over the drafts (plus the last accepted token, which the cache has
   not seen yet). It returns the target's greedy choice at every draft position plus one more.
3. **Accept.** Drafts are accepted left to right while they equal the target's choice (`n` of them); then
   the target's own token at position `n+1` is added. So a round always gains `n + 1` tokens.
4. **Roll back the caches.** Both caches are cut back to the accepted length, so rejected drafts are
   forgotten. With `GraphRunner` that just moves the write position back (stale entries are masked out
   and overwritten); with a plain model it calls `crop_to` on the `DynamicCache`.

Cache invariant: at the start of a round each cache holds a prefix of the text that excludes at least the
last token, so the next forward always has an input whose output predicts the next token. When all drafts
are accepted, the draft cache is missing the last draft + bonus token and feeds them (2 tokens) next round.

`cache=False` recomputes the full sequence every step: slow but obviously correct; the tests compare the two.

## Target alone (`target_generate`)

Plain greedy decoding with the same KV-cache code path. Using our own loop (not `model.generate`) for the
baseline means both sides pay the same Python overhead, so the speedup measures speculation, not
implementation differences. A test checks it gives exactly `model.generate`'s output.

## Correctness test

```bash
uv run pytest tests/test_specdec.py -s
```

- `target_generate` == `model.generate(do_sample=False, repetition_penalty=1.0)` on 5 prompts.
- CUDA-graph target decoding == plain decoding, token for token.
- Cache and no-cache speculative decoding agree.
- For fixed k = 1, 4, 8 and threshold 0.4 on 20 test prompts (graph and plain paths): wherever the
  speculative output differs from the target's own output, the target's top-2 logit gap at that point must
  be < 0.1 (a near-tie). Result: 16–18 of 20 identical; every difference at a gap of 0.000–0.016.

Why not 100%: the target verifies several tokens in one fp16 pass but generates one at a time alone; the
two paths round differently, and when the top two tokens are almost tied the argmax can flip. A real bug
(wrong acceptance or cache indexing) would diverge where the target has a clear preference — none did.

## Benchmark (`bench.py`)

```bash
uv run python -m specrl.live.bench --pair qwen      # 30 test prompts (10 per task), 3 warm-ups, 3 runs
```

| Rule | Why |
|---|---|
| Same PC, plugged in, nothing else on the GPU | timings must be comparable |
| 3 warm-up generations first | first CUDA calls are slow (kernel loading, allocator) |
| `torch.cuda.synchronize()` before and after each timed generation | GPU work is asynchronous |
| Method order rotated per prompt | spreads thermal / clock drift over all methods |
| 3 runs over all prompts, totals summed | tokens/s = total tokens ÷ total seconds |

Methods: target alone, best fixed k (from the simulator), fixed k = 4, threshold 0.4, VI, Q-learning, DQN,
REINFORCE. Speedup = method tokens/s ÷ target-alone tokens/s, overall and per task.
`exact_match_pct` = share of generations identical to the target alone.

Outputs: `results/live_<pair>.csv`, `results/live_<pair>_by_task.csv`, `results/live_<pair>_raw.csv`.
