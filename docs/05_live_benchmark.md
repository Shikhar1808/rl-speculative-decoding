# Live speculative decoding and the benchmark

The simulator trains the agents; this part proves the speedup is real on the GPU and that the output is
unchanged. Code: `src/specrl/specdec.py`, `src/specrl/bench.py`, test `tests/test_specdec.py`.

## How one round works (`spec_generate`)

1. **Draft.** The draft model produces one token at a time. After each token we compute the same 5
   features as the simulator (`[s/K, p, H, q, a]`) and ask the policy: CONTINUE or STOP. Drafting also stops
   at `s = K = 8`, at the token budget, or if the draft produces the end-of-turn token.
2. **Verify.** One target forward pass over the drafts (plus the last accepted token, which the cache has
   not seen yet). It returns the target's greedy choice at every draft position plus one more.
3. **Accept.** Drafts are accepted left to right while they equal the target's choice (`n` of them); then
   the target's own token at position `n+1` is added. So a round always gains `n + 1` tokens.
4. **Roll back the caches.** Both KV caches are cropped (`DynamicCache.crop`) to the accepted length, so
   rejected drafts are forgotten.

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

- `target_generate` == `model.generate(do_sample=False)` on 5 prompts.
- cache and no-cache speculative decoding agree.
- For fixed k = 1, 4, 8 and threshold 0.4 on 20 test prompts, speculative output == target output.
  Allowed: up to 2 prompts that diverge late, ≥ 95% of tokens equal before the first divergence.

Why not 100%: the target verifies 5 tokens in one fp16 pass but generates one at a time alone; the two
paths round differently, and when the top two tokens are almost tied the argmax can flip. The guide treats
more than 1% bad tokens as a bug.

## Benchmark (`bench.py`)

```bash
uv run python -m specrl.bench --pair qwen      # 30 test prompts (10 per task), 3 warm-ups, 3 runs
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
