# Results, and how each algorithm works

Status: **being filled in as experiments finish.** Sections marked _(pending)_ have no numbers yet.
All numbers come from `results/*.csv`; figures are in `figures/`. Decisions behind every setting:
[DECISIONS.md](../DECISIONS.md).

---

## 1. The task in one paragraph

A small draft model guesses tokens one at a time; after each guess the agent says CONTINUE (guess one
more) or STOP (let the big target model check all guesses in one pass). The target keeps the guesses that
match its own choice, plus one token of its own. Output is always exactly the target's greedy output; the
only thing the agent changes is **speed**. Reward per round = tokens gained − λ·(1 + s·c), where
s = tokens drafted, c = cost of a draft step relative to a target pass, λ = price of a target call in tokens.

## 2. Measured constants _(pending)_

| Pair | Draft step | Target verify (5 tokens) | c = ratio | Draft match rate |
|---|---|---|---|---|
| Qwen 0.5B → 1.5B | | | | |
| SmolLM2 135M → 1.7B | | | | |

## 3. How each algorithm works

All four solve the same MDP, learn on the 399 train prompts, are scored on the 99 test prompts. Each round
is one short episode (≤ 8 decisions), see [04_algorithms.md](04_algorithms.md).

### Value Iteration — dynamic programming (model-based)

1. **Discretise.** State = (s, q-bin): s = 1..8 drafted tokens, q = chance all drafts survive, cut into
   10 bins → 80 states.
2. **Build a model from the logs.** For every logged answer position treated as a round start, and every
   s = 1..8, it counts: which state the round is in, what the STOP reward would be there
   (`min(n,s)+1 − λ(1+s c)`), and which q-bin the next draft lands in on CONTINUE. Averages of these counts
   give R̄(s,b) and the transition probabilities P(b' | s, b).
3. **Iterate the Bellman optimality equation** until values stop changing:
   `V(s,b) = max( R̄(s,b), Σ_b' P(b'|s,b) · V(s+1,b') )`, with STOP forced at s = 8.
4. **Policy** = in each state, the action with the larger value.

Never interacts with the simulator while learning: it plans with the counted model. Because s only grows,
it converges in about 8 sweeps. It is the "near-optimal reference" for the tabular state.

### Q-learning — temporal difference (model-free)

1. Same 80 states, a table Q(state, action) initialised to 0.
2. Plays rounds in the simulator. ε-greedy: random action with probability ε (1.0 → 0.05 over the first
   80% of 3000 episodes), otherwise the best action in the table.
3. After each step, nudges the table toward the observed target:
   `Q(s,a) ← Q(s,a) + 0.05 · [ r + max_a' Q(s',a') − Q(s,a) ]` (no bootstrap term after STOP — the round ended).
4. Policy = argmax of the table.

Learns the same thing as Value Iteration, but from sampled experience, without a model.

### DQN — function approximation + deep RL

1. No discretisation: input is 5 continuous features `[s/8, p, H/5, q, a]`, so it also sees the latest
   draft confidence p, the entropy H and the recent acceptance rate a.
2. Network 5→64→64→2 outputs Q(state, CONTINUE) and Q(state, STOP).
3. Experience replay: every step goes into a 50 000-step buffer; every 4 steps it trains on a random batch
   of 128 (breaks the correlation between consecutive steps).
4. Target network: a frozen copy, synced every 500 steps, provides the bootstrap target
   `r + max_a' Q_target(s', a')` (stops the target from chasing itself).
5. Huber loss, Adam (lr 1e-3), ε-greedy like Q-learning.

### REINFORCE — policy gradient

1. Learns the policy directly: network 5→64→2 outputs action probabilities (softmax).
2. Samples actions from those probabilities while playing one prompt's answer.
3. Every decision in a round gets that round's reward as its return G (γ = 1, reward only at STOP).
4. Baseline: subtract the mean return of the batch, divide by its std → advantage A.
5. Gradient step on `−Σ log π(a|s) · A`: makes actions that led to above-average rounds more likely.
6. Forced STOPs (s = 8, end of answer) get no gradient — the agent didn't choose them.

### Baselines (no learning)

| Baseline | Rule |
|---|---|
| Target alone | no speculation, speedup 1.00× by definition |
| Fixed k | always draft exactly k tokens (k = 1..8; we report the best k and k = 4) |
| Threshold 0.4 | stop after a draft token with confidence p < 0.4 (HuggingFace's dynamic speculation) |

### What the comparison is designed to show

| Pair | Contrast |
|---|---|
| Value Iteration vs Q-learning | model-based planning vs model-free learning, same states |
| Q-learning vs DQN | table vs neural network (and 2 vs 5 state features) |
| DQN vs REINFORCE | value-based vs policy-based |

## 4. Simulator results _(pending)_

## 5. Live results on the RTX 3050 _(pending)_

## 6. What the agents learned _(pending)_

## 7. Per-task behaviour _(pending)_

## 8. Effect of λ _(pending)_

## 9. Simulator vs live _(pending)_

## 10. Hypotheses from the proposal, checked _(pending)_

## 11. Limitations _(pending)_
