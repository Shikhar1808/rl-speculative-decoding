# The four algorithms

All four learn the same stopping decision in the simulator ([02_mdp.md](02_mdp.md)) on the **train** split,
and are scored on the **test** split. Code: `src/specrl/algos/`, entry point `src/specrl/algos/train.py`.

```bash
uv run python -m specrl.algos.train --pair qwen --algo all        # all four, Dinkelbach lambda, ~minutes on CPU
uv run python -m specrl.algos.train --pair qwen --algo dqn --lam 2.5 --rounds 1   # one algo, fixed lambda
```

| Algorithm | Course topic | State | Key settings |
|---|---|---|---|
| Value Iteration (`vi.py`) | DP, Bellman equations | (s, q-bin): 8 × 10 = 80 states | Model counted from logs; iterate until Bellman residual < 1e-9 |
| Q-learning (`qlearn.py`) | Temporal difference | same 80 states | α = 0.05, ε 1.0 → 0.05 linear over 80% of episodes, 10 000 episodes |
| DQN (`dqn.py`) | Function approx, deep RL | `[s/K, p, H/5, q, a]` | MLP 5→64→64→2, replay 50k, batch 128, update every 4 steps, target sync 500 steps, Huber, Adam 1e-3, 1000 warm-up steps |
| REINFORCE (`reinforce.py`) | Policy gradient | same 5 features | softmax MLP 5→64→2, Adam 1e-3, baseline = mean return per prompt, advantages ÷ std |

"Episode" in the settings = one prompt's answer replayed in the simulator (many rounds).

## Design choice: one round = one decision episode

Every algorithm treats each draft→verify round as a short episode of at most 8 decisions, with γ = 1:
CONTINUE bootstraps from the next state in the same round; STOP ends the round, and its reward is the
whole return.

Why, instead of the whole answer as one episode:

- An answer has a fixed number of tokens whatever the policy, so over a whole answer
  `Σ reward = tokens − λ·time` only rewards spending less time, and λ would just rescale it.
- Per round, maximising `E[gained − λ·cost]` with λ = best throughput is exactly Dinkelbach's method for
  maximising `tokens / time` (renewal-reward argument). This is the formulation in the proposal.
- The guide already recommends it for the tabular methods; using it for all four keeps the comparison fair.
- REINFORCE gets short, low-variance episodes.

The acceptance-rate feature `a` still carries over between rounds, so DQN and REINFORCE see the recent history.

## Forced STOPs

At `s = K` and at the end of the answer the environment stops whatever the agent picks
(`common.forced(env)`). Those steps are stored as STOP for Q-learning/DQN, and get no gradient in REINFORCE.

CONTINUE at `s = K` is never trained, so it must not appear in a bootstrap target: Q-learning keeps it at −∞,
DQN masks it when `s' = K`. Without this, its untouched value leaked into `max_a' Q(s',a')` and Q-learning
learned "always continue" (DECISIONS.md #35).

## Lambda (Dinkelbach), automatic in `train.py`

1. λ₀ = throughput of the best fixed-k on the train split.
2. Train → measure the policy's train throughput λ₁.
3. Train again with λ₁ (`--rounds 2`, the default). The saved policy is from the last round.

## Outputs

| File | Content |
|---|---|
| `policies/<pair>_<algo>.pt` | table or network weights + λ, c, training time; load with `train.load_policy(pair, algo)` |
| `results/curve_<pair>_<algo>.csv` | training curve: mean round reward per 50 episodes (VI: Bellman residual per iteration) |
| `results/sim_rl_<pair>.csv` | test-split sim speedup, mean accepted length, λ, training seconds per algorithm |

## Tests

`uv run pytest tests/test_algos.py`: every algorithm trains on synthetic logs, produces every token, and beats
the target alone; Value Iteration matches or beats the best fixed k.
