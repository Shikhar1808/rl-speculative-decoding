# Results, and how each algorithm works

Status: final (10 Nov 2026). Simulator: 99 test prompts. Live: 30 test prompts × 3 runs on the RTX 3050.
All numbers come from `results/*.csv`; figures are in `figures/`. Decisions behind every setting:
[DECISIONS.md](../DECISIONS.md).

---

## 1. The task in one paragraph

A small draft model guesses tokens one at a time; after each guess the agent says CONTINUE (guess one
more) or STOP (let the big target model check all guesses in one pass). The target keeps the guesses that
match its own choice, plus one token of its own. Output is always exactly the target's greedy output; the
only thing the agent changes is **speed**. Reward per round = tokens gained − λ·(1 + s·c), where
s = tokens drafted, c = cost of a draft step relative to a target pass, λ = price of a target call in tokens.

## 2. Measured constants

Measured on the RTX 3050 6 GB laptop (`results/c_<pair>.json`), the way the live loop runs (CUDA graphs).

| Pair | Draft step | Target verify (5 tokens) | c | c without CUDA graphs | Draft match rate (chat / code / math) |
|---|---|---|---|---|---|
| Qwen2.5 0.5B → 1.5B | 10.7 ms | 26.5 ms | **0.40** | 0.81 | 0.78 / 0.92 / 0.92 |
| SmolLM2 135M → 1.7B | 6.6 ms | 26.7 ms | **0.25** | 1.44 | 0.73 / 0.95 / 0.83 |

**Without CUDA graphs, speculation could not win on this laptop.** A one-token step was limited by how fast
the CPU launches GPU kernels (Windows driver), not by model size, so the small model was nearly as slow as
the big one. SmolLM2's 135M draft (30 layers) was even *slower* than its 1.7B target (24 layers): c = 1.44.
Recording each forward pass as a CUDA graph removed that overhead (DECISIONS #33).

Code is the easiest text to predict (draft matches the target 92–95% of the time), chat the hardest (73–78%).

## 3. How each algorithm works

All four solve the same MDP, learn on the 399 train prompts, and are scored on the 99 test prompts. Each
round is one short episode (≤ 8 decisions), see [04_algorithms.md](04_algorithms.md).

### Value Iteration — dynamic programming (model-based)

1. **Discretise.** State = (s, q-bin): s = 1..8 drafted tokens, q = chance all drafts survive, cut into
   10 bins → 80 states.
2. **Build a model from the logs.** For every logged answer position treated as a round start, and every
   s = 1..8, it counts which state the round is in, what the STOP reward would be there
   (`min(n,s)+1 − λ(1+s c)`), and which q-bin the next draft lands in on CONTINUE. Averages of these counts
   give R̄(s,b) and the transition probabilities P(b' | s, b).
3. **Iterate the Bellman optimality equation** until values stop changing:
   `V(s,b) = max( R̄(s,b), Σ_b' P(b'|s,b) · V(s+1,b') )`, with STOP forced at s = 8.
4. **Policy** = in each state, the action with the larger value.

It never interacts with the simulator while learning: it plans with the counted model. Because s only
grows, it converges in about 8 sweeps.

### Q-learning — temporal difference (model-free)

1. Same 80 states, a table Q(state, action) initialised to 0 (CONTINUE at s = 8 fixed at −∞: impossible).
2. Plays rounds in the simulator. ε-greedy: random action with probability ε (1.0 → 0.05 over the first
   80% of 10 000 episodes), otherwise the best action in the table.
3. After each step, nudges the table toward the observed target:
   `Q(s,a) ← Q(s,a) + 0.05 · [ r + max_a' Q(s',a') − Q(s,a) ]` (no bootstrap term after STOP — the round ended).
4. Policy = argmax of the table.

It learns the same values as Value Iteration, but from sampled experience, without a model.

### DQN — function approximation + deep RL

1. No discretisation: input is 5 continuous features `[s/8, p, H/5, q, a]`, so it also sees the latest
   draft confidence p, the entropy H and the recent acceptance rate a.
2. Network 5→64→64→2 outputs Q(state, CONTINUE) and Q(state, STOP).
3. Experience replay: every step goes into a 50 000-step buffer; every 4 steps it trains on a random batch
   of 128 (breaks the correlation between consecutive steps).
4. Target network: a frozen copy, synced every 500 steps, gives the bootstrap target
   `r + max_a' Q_target(s', a')` (CONTINUE masked out when s' = 8).
5. Huber loss, Adam (lr 1e-3), ε-greedy like Q-learning, 3000 episodes.

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
| Fixed k | always draft exactly k tokens (k = 1..8) |
| Threshold 0.4 | stop after a draft token with confidence p < 0.4 (HuggingFace's dynamic speculation) |

### What the comparison is designed to show

| Pair | Contrast |
|---|---|
| Value Iteration vs Q-learning | model-based planning vs model-free learning, same states |
| Q-learning vs DQN | table vs neural network (and 2 vs 5 state features) |
| DQN vs REINFORCE | value-based vs policy-based |

## 4. Simulator results (99 unseen test prompts)

Sim speedup = tokens ÷ time, with time counted in target calls (`1 + s·c` per round); target alone = 1.00×.
"Tokens/call" = mean tokens gained per target call.

| Method | Qwen speedup | Qwen tokens/call | SmolLM2 speedup | SmolLM2 tokens/call |
|---|---|---|---|---|
| Fixed k = 1 | 1.34× | 1.88 | 1.48× | 1.84 |
| Fixed k = 3 (best fixed, Qwen) | 1.50× | 3.31 | 1.79× | 3.10 |
| Fixed k = 4 (best fixed, SmolLM2) | 1.50× | 3.89 | 1.81× | 3.58 |
| Fixed k = 8 | 1.33× | 5.52 | 1.63× | 4.78 |
| Threshold p < 0.4 (HuggingFace) | 1.53× | 5.17 | 1.91× | 4.43 |
| **Value Iteration** | **1.66×** | 3.88 | **2.02×** | 3.84 |
| Q-learning | 1.59× | 4.77 | 1.94× | 4.42 |
| DQN | 1.63× | 4.78 | 2.01× | 4.10 |
| REINFORCE | 1.64× | 4.36 | 2.01× | 3.95 |

Against the best baseline (threshold 0.4): Value Iteration +8.5% (Qwen) / +5.9% (SmolLM2), REINFORCE
+7.6% / +5.1%, DQN +6.5% / +5.4%, Q-learning +4.4% / +1.6%. **All four RL agents beat every baseline on
both pairs.**

The RL agents get *fewer* tokens per target call than the threshold rule, yet are faster: they stop earlier
when a rejection is likely, so they waste fewer draft steps. The goal is tokens per unit of time, not tokens
per call.

### Training cost

| Algorithm | Qwen: time to train | SmolLM2: time to train | Experience used |
|---|---|---|---|
| Value Iteration | 1.5 s | 1.0 s | counts over all logged positions, ~8 sweeps |
| Q-learning | 8.0 s | 8.2 s | 10 000 simulated answers |
| DQN | 182 s | 162 s | 3 000 simulated answers, replay buffer |
| REINFORCE | 149 s | 156 s | 3 000 simulated answers |

Time for the second of two Dinkelbach rounds, laptop CPU, no GPU.

### Comparing the algorithms

- **Model-based vs model-free (Value Iteration vs Q-learning, same 80 states).** Value Iteration wins
  (1.66× vs 1.59× on Qwen) and trains 5× faster. It plans with an exact count-based model of the logs, while
  Q-learning estimates the same values from noisy sampled rounds; its table is visibly noisier
  (`figures/policy_heatmap_*.png`) and it needed 10 000 episodes to get close. When a cheap, accurate model
  exists (our simulator), planning beats learning from samples.
- **Tabular vs function approximation (Q-learning vs DQN).** DQN wins (1.63× vs 1.59× Qwen; 2.01× vs
  1.94× SmolLM2). It sees 5 continuous features instead of 2 binned ones, including the latest token's
  confidence p and entropy H, and generalises across similar states instead of learning 80 cells
  separately. Cost: ~20× more training time.
- **Value-based vs policy-based (DQN vs REINFORCE).** About a tie (1.63× vs 1.64× Qwen; 2.01× vs 2.01×
  SmolLM2). REINFORCE's curve rises fastest early (`figures/training_curves_*.png`); per-round episodes keep
  its variance low, so the usual "policy gradient is noisy" disadvantage barely shows here.

### A bug the comparison exposed

Q-learning first scored 1.28× (worse than fixed k = 1). Comparing its table with Value Iteration's on the
same states exposed it: CONTINUE at s = 8 is impossible, so its table entry was never updated and stayed 0;
that 0 won the `max` in the bootstrap target and taught "always continue". DQN had the same flaw. Fixed by
excluding that action from the target (DECISIONS #35); a regression test now guards it.

## 5. What the agents learned

`figures/policy_heatmap_qwen.png`: share of STOP decisions in each (s, q) cell on the test prompts.

- **Value Iteration learned a clean threshold on q**: STOP when q < 0.7 (lower at small s), else CONTINUE.
  This is the shape SpecDec++ proves optimal: stop when the chance that a draft has already been rejected
  (1 − q) gets too high. Nobody told it this rule; it came out of the Bellman equation.
- **DQN and REINFORCE learned soft thresholds** around q ≈ 0.3–0.5: within one cell they sometimes stop
  and sometimes continue, because they also use p (latest token's confidence) and H (entropy), which q does
  not capture.
- **Q-learning** shows the same idea (stop at low q) but noisily, matching its lower score.

## 6. Per-task behaviour (Qwen, simulator)

`figures/per_task_draft_len_qwen.png`, `results/sim_per_task_qwen.csv`

| Method | Chat: drafted/round · speedup | Code | Math |
|---|---|---|---|
| Value Iteration | 2.0 · 1.41× | 4.0 · 1.78× | 4.2 · 1.74× |
| Q-learning | 3.3 · 1.36× | 5.8 · 1.73× | 6.1 · 1.66× |
| DQN | 3.1 · 1.39× | 5.6 · 1.76× | 6.0 · 1.69× |
| REINFORCE | 2.4 · 1.42× | 5.0 · 1.78× | 5.3 · 1.69× |
| Threshold 0.4 | 4.1 · 1.30× | 6.8 · 1.66× | 7.1 · 1.58× |

**The agents draft about twice as long on code and maths as on chat**, without being told the task: they
read it from the draft's confidence. Code and maths answers are formulaic (the draft matches 92% of the
time); chat is open-ended (78%). Every RL agent beats the threshold rule on every task; the biggest gain is
on chat (+9% for REINFORCE), where knowing when to stop matters most.

## 7. Effect of λ (Value Iteration, Qwen)

`figures/lambda_sweep_qwen.png`, `results/lambda_sweep_qwen.csv`

| λ (× tuned value) | 0.5 | 0.7 | 0.9 | **1.0** | 1.1 | 1.2 | 1.4 |
|---|---|---|---|---|---|---|---|
| Sim speedup | 1.47× | 1.60× | 1.66× | **1.66×** | 1.64× | 1.61× | 1.34× |
| Tokens drafted per round | 6.4 | 5.1 | 3.9 | 3.3 | 2.9 | 2.5 | 1.0 |

λ is the price of a target call. A cheap target (low λ) makes the agent draft long rounds and waste
rejected drafts; an expensive one (high λ) makes it stop early, and at 1.4× it collapses to "always stop
after 1 token" (the failure mode in the proposal's risk table). **The best speedup sits exactly at the λ
the Dinkelbach update chose**, which confirms the method.

## 8. Live results on the RTX 3050 (wall clock)

`results/live_<pair>.csv`, `figures/speedup_live_<pair>.png`. 30 test prompts (10 per task) × 3 runs, every
method on the same CUDA-graph runners, method order rotated per prompt. Speedup = tokens/s ÷ target-alone
tokens/s (target alone: 39.3 tok/s Qwen, 39.2 tok/s SmolLM2).

| Method | Qwen tok/s | Qwen speedup | SmolLM2 tok/s | SmolLM2 speedup |
|---|---|---|---|---|
| Target alone | 39.3 | 1.00× | 39.2 | 1.00× |
| Best fixed k (k=3 Qwen, k=4 SmolLM2) | 54.3 | 1.38× | 61.4 | 1.57× |
| Threshold 0.4 (HuggingFace) | 55.7 | 1.42× | 64.6 | 1.65× |
| **Value Iteration** | **59.3** | **1.51×** | **68.4** | **1.75×** |
| REINFORCE | 58.8 | 1.50× | 66.7 | 1.70× |
| DQN | 57.9 | 1.47× | 66.4 | 1.70× |
| Q-learning | 57.7 | 1.47× | 65.4 | 1.67× |

**All four RL agents beat every baseline in real time, on both pairs.** Value Iteration is 6% faster than the
best baseline on Qwen and 6% on SmolLM2, and 9% / 11% faster than the best fixed draft length.

### Live speedup by task (Value Iteration vs best baseline)

| Task | Qwen: VI | Qwen: threshold 0.4 | SmolLM2: VI | SmolLM2: threshold 0.4 |
|---|---|---|---|---|
| Chat | 1.31× | 1.25× | 1.48× | 1.39× |
| Code | 1.61× | 1.55× | **2.03×** | 1.96× |
| Math | 1.60× | 1.45× | 1.81× | 1.69× |

Full per-task tables for every method: `results/live_<pair>_by_task.csv`.

### Same output?

`exact_match_pct` in the live CSVs: 83–97% of generations identical to the target alone, token for token.
The rest differ only from an fp16 near-tie onward (section 11); chat diverges most often (more near-ties in
open-ended text).

## 9. Simulator vs live

`figures/sim_vs_live_<pair>.png`

| | Qwen | SmolLM2 |
|---|---|---|
| Rank correlation, simulator vs live (Spearman) | 1.00 | 0.94 |
| Live speedup below simulator, average | 8% | 14% |

The simulator orders the methods exactly as the live run does (SmolLM2: DQN and REINFORCE, 0.005× apart, swap
places). It is optimistic by 8–14% because it counts only model time; live decoding also pays for computing
the features (entropy over the vocabulary), calling the policy, and Python control flow every step. That
fixed per-step cost weighs more when the draft step is cheaper (SmolLM2: 6.6 ms), hence the larger gap.
**Conclusion: the simulator is reliable for comparing and choosing policies; its absolute numbers are
optimistic.**

## 10. Hypotheses from the proposal, checked

| # | Hypothesis | Result |
|---|---|---|
| 1 | All four RL agents match or beat the best fixed-k baseline | **Yes**, live and simulated, both pairs. Live: +6% to +9% over best fixed k (Qwen 1.47–1.51× vs 1.38×), +6% to +11% (SmolLM2 1.67–1.75× vs 1.57×) |
| 2 | Value Iteration is the strongest tabular method; Q-learning approaches it | **Yes**: VI 1.66× vs QL 1.59× (Qwen), 2.02× vs 1.94× (SmolLM2) |
| 3 | DQN and REINFORCE gain from the richer state; REINFORCE trains noisier | **Partly**: both beat Q-learning but not Value Iteration; REINFORCE was not noisier with per-round episodes |
| 4 | Learned policies look like a threshold on q | **Yes** (section 5) |
| 5 | Speedups larger on SmolLM2 (12× size gap) than on Qwen (3×) | **Yes**: live 1.67–1.75× vs 1.47–1.51×; simulated ~2.0× vs ~1.6× |

## 11. Correctness

Speculative decoding must give exactly the target's output. On 20 test prompts × 4 policies, 16–18 of 20
outputs were identical; every difference started at an exact near-tie of the target's top two tokens
(logit gap 0.000–0.016), where fp16 rounding of a multi-token verify pass differs from one-token steps.
No difference at a clear gap, so no acceptance or cache bug (`tests/test_specdec.py`, 9/9 pass).

## 12. Limitations

- Greedy decoding only (sampling-based speculative decoding not covered).
- Small models on a 6 GB laptop GPU; absolute speedups depend on the hardware's c (section 2).
- Simulator approximation: drafts after the first rejected token see the true text, not the draft's own
  mistake. `n` is exact; only the p and H the agent sees after a rejection can differ. The simulator is also
  8–14% optimistic because it ignores per-step overhead (section 9).
- One seed per algorithm; gaps of 1–2% between DQN, REINFORCE and Value Iteration are within what a
  different seed could change.
