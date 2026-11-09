# Plan — RL Adaptive Speculative Decoding

Status: **APPROVED 9 Nov** (defaults D1–D15, Overleaf for LaTeX). Progress: P0 done · P1 code done, Qwen logging running · P2 code + tests done · P3 script ready.
Sources: `RL_SpecDecoding_Project_Guide.pdf` + the team proposal doc
(https://claude.ai/artifact/Ng8gGdLpuKhmK7CWns2pqK — Shikhar Saxena, Vaibhav Sundriyal, Angad Shukla, Thapar).
Deadline: **17 Nov 2026**. Today: 9 Nov. Dates below follow the proposal's timeline (section 8).

Goal: an RL agent decides, token by token, when the small draft model should stop drafting and let the
big model verify. Output stays identical to the big model; generation gets faster.

---

## 1. Decisions (change any of these)

| # | Decision | My default | Why |
|---|---|---|---|
| D1 | Python env | uv project, Python 3.12 | Your preference; 3.12 safest for ML packages |
| D2 | PyTorch | CUDA 12.8 build via uv index `pytorch-cu128` | Already tested working here; also runs on older drivers (PC2). Driver supports 13.x too if you prefer |
| D3 | Project folder | `C:\Users\saxen\Desktop\work\rl\setup` | Current working dir |
| D4 | Model pair 1 | Qwen2.5-0.5B-Instruct → Qwen2.5-1.5B-Instruct | From guide |
| D5 | Model pair 2 (demo) | SmolLM2-135M-Instruct → SmolLM2-1.7B-Instruct | Bigger size gap → bigger speedup |
| D6 | Colab 3B pair | Skip unless time is left | Optional in guide |
| D7 | Prompts | 400 train / 100 test, equal mix of Alpaca (chat), HumanEval (code), GSM8K (math) | From guide |
| D8 | Answer length | 128 new tokens, greedy | From guide |
| D9 | Max draft length K | 8 | From guide |
| D10 | Git | `git init` + commit after each phase | Easy rollback; say no if you don't want it |
| D11 | Tests | pytest, run with `uv run pytest` | Unit test for env + correctness test for spec decoding |
| D12 | Report format | **LaTeX** in `report/` (`main.tex` + `references.bib`), text ported from the proposal doc, results + figures filled in from `results/` and `figures/` | Your choice |
| D13 | Compiling LaTeX | **Overleaf** (decided). Upload `report/`, all 3 teammates edit. See `docs/07_report.md` | Disk is 94% full, no local TeX |
| D14 | Slides | Beamer (LaTeX) in `report/slides.tex`, same figures | Matches the report; say if you want .pptx |
| D15 | Proposal doc | Read-only source; I don't edit it unless you ask | It's your team's doc |

---

## 2. Project layout

```
setup/
  PLAN.md                 this file
  README.md               how to install, run every step, reproduce results
  pyproject.toml uv.lock  uv project (commit both so PC2 gets the same env)
  docs/
    01_setup.md           install on a new PC (uv, CUDA check)
    02_mdp.md             state, actions, reward, λ, Bellman equations
    03_simulator.md       smart-logging trick + its one approximation
    04_algorithms.md      VI, Q-learning, DQN, REINFORCE: design + hyperparameters
    05_live_benchmark.md  real spec-decoding loop, KV cache, timing rules
    06_results.md         tables + figures, filled in as results arrive
  src/specrl/
    config.py             model pairs, paths, K, λ — one place for settings
    prompts.py            load + split datasets, apply chat template
    log_data.py           step 2: logging pass  (--pair, --start, --end)
    env.py                step 3: SpecEnv simulator
    baselines.py          fixed k=1..8, threshold p<0.4
    algos/vi.py qlearn.py dqn.py reinforce.py
    train.py              trains any algo, saves policy + curve
    specdec.py            step 5: real draft/verify loop (no-cache first, then KV cache)
    bench.py              timings + correctness check → results/*.csv
    plots.py              all figures
    app.py                Gradio race demo
  report/
    main.tex              report: abstract → conclusion, same sections as the proposal doc
    sections/*.tex        one file per section (easy to split among 3 people)
    references.bib        the 11 references from the proposal
    slides.tex            Beamer deck (~10 min talk order from the guide)
    figures/              copied/symlinked from ../figures
  tests/
    test_env.py           guide's check: match=[1,1,0,1], s=3 → n=2, gained=3
    test_specdec.py       spec output == target greedy output on 20 prompts
  data/  logs/  policies/  results/  figures/   (generated; logs/ etc. git-ignored)
```

Every script has a docstring + `--help`. Every phase updates its `docs/` page and the README.

---

## 3. Phases

Each phase ends with a check. I stop after each one so you can run the GPU parts and paste results.

| Phase | Work | Check (done when) | Who runs it | Est. |
|---|---|---|---|---|
| **P0 Setup** | `uv init`, add torch (cu128) + transformers, datasets, accelerate, gradio, numpy, pandas, matplotlib, pytest; README + docs/01 | `uv run python -c "torch.cuda.is_available()"` → True | Me | 15 min |
| **P1 Data + logging** | prompts.py, log_data.py, docs/03 | `logs/qwen_*.npz` + `logs/smol_*.npz` exist; acceptance rate printed and sensible (≈0.5–0.8) | You (GPU, ~10–30 min per pair) | 2 h code |
| **P2 Simulator + baselines** | env.py, baselines.py, test_env.py, docs/02 | unit test passes; baseline table (fixed k, threshold) in sim | Me (CPU) | 1.5 h |
| **P3 Measure c, set λ** | time draft vs target forward pass; λ = best fixed-k throughput | `c` and `λ` saved in config | You (GPU, 2 min) | 30 min |
| **P4 Tabular RL** | vi.py, qlearn.py, train.py, docs/04 | VI + QL beat best fixed-k in sim (or we know why not) | Me (CPU) | 2 h |
| **P5 Deep RL** | dqn.py, reinforce.py; Dinkelbach λ re-tune; λ sweep | all 4 trained, curves saved | Me (CPU, minutes) | 3 h |
| **P6 Live loop** | specdec.py (no-cache → KV-cache with `DynamicCache.crop`), bench.py, test_specdec.py, docs/05 | correctness test ≤1% mismatch; live speedup table | You (GPU) | 4 h — highest risk |
| **P7 Analysis + figures** | plots.py: training curves, policy heatmap, speedup bars, per-task, λ sweep; docs/06 | all figures in `figures/` | Me | 2 h |
| **P8 Demo** | app.py: race (target alone vs target+agent), coloured tokens, confidence meter | demo runs on LAN | You test | 4 h |
| **P9 Report + slides** | LaTeX report: port the proposal text, Bellman equations as `align`, tables from `results/*.csv`, figures, related work + bib; Beamer slides | compiles on Overleaf with no errors | Me write, you edit | 3 h |
| **P10 Buffer** | backup demo video, rehearse, submit | submitted | You | 16–17 Nov |

Calendar (matches proposal section 8):

| Date | Phases |
|---|---|
| Nov 9 (tonight) | P0 setup; LaTeX skeleton of the report from the proposal (sections 1–5 are already written) |
| Nov 10 | P1 logging (both laptops), P2 simulator + baselines |
| Nov 11 | P3 c/λ, P4 VI + Q-learning, P5 DQN + REINFORCE |
| Nov 12 | P6 live loop + correctness + benchmark |
| Nov 13 | P7 figures, P8 Gradio demo |
| Nov 14 | P9 report results + slides |
| Nov 15–17 | P10 buffer, backup video, rehearsal, submit |

---

## 4. Two PCs

- Both PCs: install uv, `git clone`/copy the folder, `uv sync`.
- P1: PC1 logs Qwen, PC2 logs SmolLM2 → copy `.npz` files into one `logs/`.
- P6: all compared timings come from **one** PC, plugged in.
- Demo day: demo ready on both PCs.

---

## 5. Risks

| Risk | Plan |
|---|---|
| KV-cache rollback bugs | Ship the no-cache loop first (correct, slow); add cache after. Training never needs it. |
| Small speedup on 6 GB GPU | Use SmolLM2 pair for the demo; also report speedup vs best fixed-k |
| Agent learns "always stop at 1" | Re-measure c, re-tune λ, check reward signs |
| DQN/REINFORCE unstable | Feature normalisation, reward scaling, baseline; tabular results still stand |
| Disk 94% full (67 GB free) | Models need ~10 GB — fine, but don't download extra models |
| Second PC unchecked | Run `docs/01_setup.md` check there before P1 |

---

## 6. Not doing (unless you ask)

- Colab 3B benchmark (D6), sampling (non-greedy) speculative decoding, FastAPI, gymnasium dependency,
  hyperparameter search beyond the λ sweep.

---

## 7. Your review

Tell me: changes to any **D1–D15**, phases to add/drop, or a different order. Then I start with P0.
