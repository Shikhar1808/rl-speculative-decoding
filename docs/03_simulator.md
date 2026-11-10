# Logging and the simulator

Training never runs the language models. We log each prompt once on the GPU, then replay any policy on the
CPU from the logs.

## Why this works

With greedy decoding the final text is always the target's own greedy answer, whatever the stopping policy.
So we only need, at every answer position, whether the draft model would have guessed that token. A round
starting at position `t` with `s` drafts accepts `n` = consecutive matches from `t`, capped at `s`.

## Step 1: prompts

```bash
uv run python -m specrl.data.prompts        # -> data/prompts.jsonl (committed; already built)
```

| Task | Source | Prompts | Test |
|---|---|---|---|
| chat | `tatsu-lab/alpaca` (instructions with no input field) | 167 | 33 |
| code | `openai/openai_humaneval` (all) | 164 | 33 |
| math | `openai/gsm8k` test split, + "Solve step by step." | 167 | 33 |

Shuffled with seed 0; the first 20% of each task is the test split. Prompts are wrapped in each model's chat
template at logging time.

## Step 2: logging

```bash
uv run python -m specrl.data.log_data --pair qwen                  # all prompts, one PC
uv run python -m specrl.data.log_data --pair smol --start 0 --end 250   # split across PCs
```

Per prompt: `target.generate` (greedy, 128 new tokens), then one draft forward pass over prompt + answer.
Saved to `logs/<pair>_<start>-<end>.npz`:

| Array | Per | Meaning |
|---|---|---|
| `p`, `H`, `match`, `tokens` | answer token (concatenated) | draft top prob, draft entropy, draft == target, the target's token |
| `offsets` | prompt + 1 | where each prompt's tokens start in the arrays above |
| `id`, `task`, `split` | prompt | from `data/prompts.jsonl` |

`load_logs(pair, split)` merges every file of a pair (both PCs) and returns one dict per prompt. Copy the
other PC's `.npz` files into `logs/` before training.

## Step 3: the simulator

`SpecEnv(episodes, lam, c)` in `src/specrl/sim/env.py`, gym-like `reset()` / `step(a)`; MDP in
[02_mdp.md](02_mdp.md). `evaluate(policy, episodes, c)` runs a policy over all episodes and returns tokens,
target calls, mean accepted length and the sim speedup.

Tests: `uv run pytest tests/test_env.py` (includes the guide's check: match `[1,1,0,1]`, s=3 → n=2, gained=3).

## Baselines in the simulator

```bash
uv run python -m specrl.sim.baselines --pair qwen --c <c from measure_c>
```

Fixed k = 1..8 and threshold 0.4 → `results/sim_baselines_<pair>_<split>.csv`. The best fixed-k throughput
is the starting λ.

## The one approximation

After the first wrong draft token, a real draft model continues from its own wrong token; the log continues
from the true text. Those later drafts are rejected anyway, so `n` is exact. What can differ is `p` and `H`
for drafts after the first mistake, which the agent sees before deciding to stop. The live benchmark (P6)
measures the size of this gap.
