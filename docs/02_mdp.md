# The MDP

The agent decides, after each drafted token, whether to draft one more or send the round to the target.
Code: `src/specrl/sim/env.py`.

| Part | Definition |
|---|---|
| Episode | one prompt's full answer (≤ 128 tokens), many draft→verify rounds |
| Observation | `[s/K, p, H, q, a]` — see below |
| Actions | `0 = CONTINUE`, `1 = STOP`. STOP is forced at `s = K = 8` and at the end of the answer |
| Reward | `0` on CONTINUE; on STOP `r = gained − λ(1 + s·c)` |
| Discount | γ = 1 for tabular methods, 0.99 for DQN/REINFORCE |

Observation features:

| Feature | Meaning |
|---|---|
| `s/K` | tokens drafted so far this round, scaled to 0–1 |
| `p` | draft model's top probability for the latest draft token |
| `H` | entropy of the draft distribution at that token |
| `q` | product of `p` over this round ≈ chance all drafts survive |
| `a` | moving average (0.8 old / 0.2 new) of `n/s` over recent rounds |

## Reward details

- `n` = number of drafts accepted (consecutive matches from the round start).
- `gained = min(n + 1, tokens left)`: accepted drafts plus the target's own token (correction or bonus),
  never past the answer's end.
- `c = t_draft / t_target`: cost of one draft step relative to one target verify pass. Measured per model
  pair with `uv run python -m specrl.live.measure_c --pair <pair>` → `results/c_<pair>.json`.
- `λ`: price of one target call, in tokens.

## Choosing λ (Dinkelbach)

Maximising throughput `tokens / time` is the same as maximising `tokens − λ·time` when λ equals the best
achievable throughput. Recipe:

1. λ₀ = throughput of the best fixed-k baseline (`specrl.sim.baselines` prints it).
2. Train with λ₀, measure the new throughput λ₁.
3. Retrain once with λ₁.
4. For the report, also sweep λ around that value.

## Time and the sim speedup

Time is counted in units of one target call: a round costs `1 + s·c`. The target alone makes 1 token per
target call, so the simulator's speedup estimate is

```
sim_speedup = tokens / (target_calls + c · draft_steps)
```

`specrl.sim.env.evaluate(policy, episodes, c)` returns it along with the mean accepted length. The live benchmark
(P6) checks whether this estimate ranks methods the same way as real timings.
