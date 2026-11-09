# Decision log

Every important choice made while building, with the reason, so it can be reviewed and reversed.
Newest at the bottom. "D#" items refer to PLAN.md. Read this first if something looks wrong.

## 9 Nov — setup (approved by you)

| # | Decision | Reason | How to reverse |
|---|---|---|---|
| 1 | uv project, Python 3.12 | your rule: uv for Python; 3.12 safest for ML wheels | `.python-version`, `pyproject.toml` |
| 2 | PyTorch 2.11 CUDA 12.8 build, pinned via explicit uv index | already worked on PC1; also runs on older drivers (PC2) | `[tool.uv.index]` in `pyproject.toml` |
| 3 | LaTeX report on Overleaf, one file per section | your request; no TeX locally, disk 94% full | — |
| 4 | Proposal doc is read-only; report text ported from it | it's the team's doc | — |
| 5 | git, one commit per phase | easy rollback | `git log`, `git revert <id>` |

## 9–10 Nov — data and simulator

| # | Decision | Reason |
|---|---|---|
| 6 | 498 prompts: 167 Alpaca (only instructions with no `input` field), all 164 HumanEval, 167 GSM8K test; seed 0; first 20% of each task = test (99 test, 399 train) | guide asks ~400/100 mixed; HumanEval only has 164 |
| 7 | GSM8K prompts get "Solve step by step." appended; HumanEval prompts get "Complete the following Python function." | makes the instruct models answer in the intended style |
| 8 | Prompts wrapped in the target model's chat template; the draft uses the same tokenizer | pairs share a tokenizer (required) |
| 9 | `data/prompts.jsonl` is committed (removed `data/` from `.gitignore`) | both PCs must log exactly the same prompts |
| 10 | Answers: greedy, max 128 new tokens, stop at end-of-turn token | guide |
| 11 | Simulator caps `gained` at the tokens left in the answer | the guide's sketch could count a bonus token past the end |
| 12 | "Sim speedup" = tokens / (target calls + c · draft steps) | target alone = 1 token per target call; time measured in target calls |

## 10 Nov — algorithms

| # | Decision | Reason |
|---|---|---|
| 13 | **All four algorithms train on per-round episodes** (STOP ends the episode, γ = 1). The guide suggested whole-answer episodes with γ = 0.99 for DQN/REINFORCE | an answer has a fixed token count, so whole-answer reward only measures time and λ does nothing; per-round `E[gained − λ·cost]` is exactly Dinkelbach's ratio maximisation; same setup for all four = fair comparison |
| 14 | Forced STOPs (s = K, end of answer) are stored as STOP for QL/DQN and get no gradient in REINFORCE | they are not the agent's choice |
| 15 | Value Iteration's model is counted from **every** logged position as a possible round start | gives the model for all states without needing a behaviour policy |
| 16 | DQN updates every 4 env steps (not every step) | 4× faster training on CPU; standard DQN practice |
| 17 | Entropy feature scaled H/5 (capped at 2) for the networks | other features are 0–1; raw entropy reaches ~12 |
| 18 | REINFORCE advantages divided by their std | reward scaling, listed in the guide's risk table |
| 19 | λ chosen automatically: best fixed-k train throughput, then one Dinkelbach update (2 training rounds) | guide's recipe |
| 20 | c measured with KV cache: 1 draft token vs 5-token target verify, 200-token context, median of 30 | a round verifies ~4–6 tokens; context ~prompt + half an answer |

## 10 Nov — live loop, benchmark, demo, figures

| # | Decision | Reason |
|---|---|---|
| 21 | Live loop keeps one KV cache per model and crops both to the accepted length after each verify; invariant "last token never in the cache" | one forward always predicts the next token; rejected drafts are forgotten |
| 22 | Draft also stops at its own end-of-turn token | drafting past the end is wasted work |
| 23 | "Target alone" baseline uses our own greedy KV-cache loop, not `model.generate` | both sides pay the same Python overhead, so the speedup measures speculation only; a test checks it equals `model.generate` |
| 24 | Correctness test allows ≤ 2 of 20 prompts to diverge late (≥ 95% tokens equal before first divergence) | fp16: one multi-token verify pass rounds differently from one-token steps, which can flip near-ties (guide: >1% = bug) |
| 25 | Benchmark: 30 test prompts (10 per task), 3 warm-ups, 3 runs, method order rotated per prompt, CUDA sync around each generation | guide's timing rules; rotation spreads thermal drift |
| 26 | Demo runs the two generations one after the other, timestamps every token, then replays both side by side at recorded speed | two models racing on one GPU would slow each other and make the race unfair; shown numbers are the real ones |
| 27 | Figures: baselines grey, one fixed colour per RL algorithm (reference palette from the dataviz guide) | same method = same colour in every figure |
| 28 | Policy heatmaps for DQN/REINFORCE are drawn at the median p and H of the logs, a = 0.7 | networks take 5 inputs; a 2-D picture needs the other 3 fixed |
| 29 | λ sweep uses Value Iteration only (0.5×–1.5× the tuned λ, 11 points) | VI trains in seconds; DQN/REINFORCE would take ~hours for a sweep |
| 30 | Report dropped the proposal's timeline section; Colab 3B row removed | planning material, not results; Colab skipped (D6) |
| 31 | **Logging is batched** (8 prompts per GPU pass, sorted by length, left padding, explicit position ids). Qwen was logged one-at-a-time (started before this); SmolLM2 and any later logging use batches | your instruction: offline GPU jobs should fill the GPU. Live benchmark and demo stay one-at-a-time because that's what is measured. In fp16 a padded batch can round differently on near-ties, so a logged answer may rarely differ from one-at-a-time greedy; the simulator only needs a consistent answer + match flags |
| 33 | **CUDA graphs for all live decoding** (`fastmodel.GraphRunner`: StaticCache + one captured graph per input length 1–10). Used by every method in the benchmark, including target alone | measured: without them a Qwen 0.5B draft step took 41.8 ms vs 45.9 ms for a 1.5B verify (c = 0.91), so speculation could never win (best fixed k = 0.94× in the sim). The step was CPU launch-bound (Windows driver), not model-size-bound. Graphs: 39.7 → ~10 ms per draft step. Verified token-identical to eager decoding; multi-token chunks and rollback checked against plain forward passes |
| 34 | **Re-logged Qwen with `repetition_penalty=1.0`** | Qwen's default generation config sets repetition_penalty 1.1, which `generate` applies even with `do_sample=False`, so the first logs were not pure greedy text while speculative decoding reproduces pure greedy output. SmolLM2 has no such defaults. Old logs kept in `logs/old/` |
| 35 | **Bug fix (Q-learning, DQN):** CONTINUE at s = K is now excluded from the bootstrap max (−∞ in the Q-table; masked in the DQN target) | STOP is forced at s = K, so that entry/output is never trained; its leftover value (0) won the max and taught "always continue". Found because Q-learning scored 1.28 vs Value Iteration 1.62 on the same 80 states. After the fix 1.55–1.57. Regression test added |
| 36 | Q-learning default 10 000 episodes (others 3000), α = 0.05 | small sweep on train/test: 3000 eps → 1.548/–, 10 000 eps α 0.05 → 1.566 / 1.610 (best), α 0.01 worse; costs 8 s |
| 37 | **Correctness criterion changed** from "≥ 18/20 prompts identical" to "wherever speculative output differs from the target, the target's top-2 logit gap there is < 0.1" | all observed differences were exact fp16 near-ties (gaps 0.000–0.016) halfway through answers; 16–18 of 20 prompts identical. A real acceptance/cache bug would diverge at clear gaps. 9/9 tests pass under this rule (eager and CUDA-graph paths) |
| 38 | `_eos_ids` bug fixed: it looked up `.model` to unwrap a GraphRunner, but HF models also have `.model` (their inner network) | found by the GPU tests |
| 39 | Figures: policy heatmap now shows the **share of STOP decisions per (s, q) cell observed on test prompts**, not the network's output at fixed median p, H | the median slice made REINFORCE look like "always continue", which is not what it does; the observed share is one honest measure for tables and networks alike |
| 40 | Logs (`logs/*.npz`, ~1 MB each) and trained policies (`policies/`) are now committed | small; teammates and demo day need them without re-running the GPU steps |
| 41 | With time until the evening (your message 02:45): full live benchmark (30 prompts × 3 runs, both pairs) + **3 seeds** for Q-learning, DQN, REINFORCE (`specrl.seeds`) | checks whether the 1–2% gaps between algorithms are real; seed-0 policies stay the ones used live |
| 32 | **Only this PC is used** (your instruction); all timings, both model pairs, training and the demo run on the RTX 3050 laptop | — |

## 10 Nov — problems hit

| When | Problem | What I did |
|---|---|---|
| 00:45 | Qwen logging took ~70 min instead of ~35 (CPU-bound, shared the CPU with my test runs; RAM low because WSL/Docker hold ~9 GB). Its progress was hidden because its output went through `grep`, which buffers | waited instead of restarting (would lose the work); later runs write progress straight to a file (`logs/smol_run.txt`) |
| 01:08 | **My mistake:** I stopped the one-at-a-time Qwen logging after 78 min to switch to batching, not knowing (output was hidden) it was at 480/498 prompts, ~2 min from done. ~75 min of GPU time lost. Its log showed ~9 s/prompt and a final match rate ≈ 0.81 | re-ran Qwen with batched logging (progress in `logs/qwen_run.txt`); used 0.81 as a sanity check for the batched result. Lesson: never stop a job whose progress I can't see |
| 02:45 | The first full live benchmark ran ~5× slower than expected (≈85 s per prompt instead of ≈20 s; GPU 98% busy). A 3-prompt test right after ran at normal speed and nothing else was on the GPU, so the cause was most likely another program using the GPU at that time (browser video or another session's tests) — not confirmed | stopped it after 10 of 90 prompt-runs, re-ran from scratch; progress lines now show elapsed time so a slowdown is visible |
