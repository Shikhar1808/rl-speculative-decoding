# Learning When to Stop Drafting

RL for adaptive speculative decoding in LLMs: a small **draft** model guesses tokens, a big **target**
model checks them in one pass, and an RL agent decides after each guess whether to keep drafting or stop
and verify. Output is identical to the target alone; generation is faster.

Course project, Thapar Institute — Shikhar Saxena, Vaibhav Sundriyal, Angad Shukla. Deadline 17 Nov 2026.

| Start here | |
|---|---|
| [docs/06_results.md](docs/06_results.md) | **All results**, how each algorithm works, the comparison |
| [DECISIONS.md](DECISIONS.md) | Every decision and problem hit, with reasons — read this if something looks wrong |
| Report | LaTeX report + Beamer slides, on Overleaf; not in git ([docs/07_report.md](docs/07_report.md)) |
| [PLAN.md](PLAN.md) | Original plan and phases |
| Proposal doc | https://claude.ai/artifact/Ng8gGdLpuKhmK7CWns2pqK |

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a recent driver. Full steps: [docs/01_setup.md](docs/01_setup.md).

```bash
uv sync                      # creates .venv with Python 3.12 + CUDA 12.8 PyTorch
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
uv run python -m specrl.app  # the race demo (needs the trained policies in policies/)
```

## Pipeline (reproduces every number in the report)

Run for each pair (`qwen`, then `smol`). Times are on the RTX 3050 laptop.

| Step | Command | Runs on | Time | Output |
|---|---|---|---|---|
| 0. Build prompts (done, committed) | `uv run python -m specrl.prompts` | CPU | 1 min | `data/prompts.jsonl` |
| 1. Log model outputs | `uv run python -m specrl.log_data --pair qwen` | GPU | 7 min | `logs/qwen_0-498.npz` |
| 2. Measure c | `uv run python -m specrl.measure_c --pair qwen` | GPU | 1 min | `results/c_qwen.json` |
| 3. Sim baselines | `uv run python -m specrl.baselines --pair qwen --c <c> --split test` | CPU | 10 s | `results/sim_baselines_*.csv` |
| 4. Train agents | `uv run python -m specrl.train --pair qwen --algo all` | CPU | 12 min | `policies/`, `results/sim_rl_*.csv`, `results/curve_*.csv` |
| 5. Seed check | `uv run python -m specrl.seeds --pair qwen --seeds 3` | CPU | 30 min | `results/seeds_*.csv` |
| 6. Live benchmark | `uv run python -m specrl.bench --pair qwen` | GPU | 30 min | `results/live_*.csv` |
| 7. Figures | `uv run python -m specrl.plots --pair qwen` | CPU | 2 min | `figures/*.png`, analysis CSVs |
| 8. Report tables | `uv run python -m specrl.report_tables` | CPU | 5 s | `report/generated/`, `report/figures/` |
| 9. Demo | `uv run python -m specrl.app` | GPU | — | Gradio app at localhost:7860 |

Tests: `uv run pytest tests/test_env.py tests/test_algos.py` (CPU, ~3 min) and
`uv run pytest tests/test_specdec.py -s` (GPU correctness, ~20 min).

## Layout

```
src/specrl/
  config.py        model pairs, paths, K, answer length
  prompts.py       build the prompt set            log_data.py   logging pass (batched)
  env.py           simulator + evaluate()          baselines.py  fixed k, threshold 0.4
  algos/           vi.py qlearn.py dqn.py reinforce.py common.py
  train.py         training + Dinkelbach λ         seeds.py      multi-seed check
  fastmodel.py     CUDA-graph runner               specdec.py    live speculative decoding
  measure_c.py     c on this GPU                   bench.py      live benchmark
  plots.py         figures                         report_tables.py  LaTeX tables
  app.py           Gradio race demo
tests/        test_env.py, test_algos.py (CPU), test_specdec.py (GPU)
docs/         one page per part of the system
report/       LaTeX report + slides (local only, git-ignored; lives on Overleaf)
data/ results/ figures/   committed     logs/ policies/   generated, not committed
```

## Docs

| Page | Covers |
|---|---|
| [01_setup.md](docs/01_setup.md) | Install on a new PC, CUDA check |
| [02_mdp.md](docs/02_mdp.md) | State, actions, reward, c, λ, sim speedup |
| [03_simulator.md](docs/03_simulator.md) | Prompts, logging, simulator, baselines |
| [04_algorithms.md](docs/04_algorithms.md) | VI, Q-learning, DQN, REINFORCE: settings, per-round episodes, λ tuning |
| [05_live_benchmark.md](docs/05_live_benchmark.md) | Live speculative decoding, CUDA graphs, correctness test, timing rules |
| [06_results.md](docs/06_results.md) | All results and the algorithm comparison |
| [07_report.md](docs/07_report.md) | How the LaTeX report is organised and synced with Overleaf |
| [08_demo.md](docs/08_demo.md) | Running and presenting the race demo |
