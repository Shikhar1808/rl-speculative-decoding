# Learning When to Stop Drafting

RL for adaptive speculative decoding in LLMs: a small **draft** model guesses tokens, a big **target**
model checks them in one pass, and an RL agent decides after each guess whether to keep drafting or stop
and verify. Output is identical to the target alone; generation is faster.

Course project, Thapar Institute — Shikhar Saxena, Vaibhav Sundriyal, Angad Shukla. Deadline 17 Nov 2026.

- Plan and progress: [PLAN.md](PLAN.md)
- Proposal doc: https://claude.ai/artifact/Ng8gGdLpuKhmK7CWns2pqK
- Report (LaTeX, Overleaf): [report/](report/) — see [docs/07_report.md](docs/07_report.md)

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a recent driver. Full steps: [docs/01_setup.md](docs/01_setup.md).

```bash
uv sync                      # creates .venv with Python 3.12 + CUDA 12.8 PyTorch
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Pipeline

| Step | Command | Runs on | Output |
|---|---|---|---|
| 0. Build prompts (done, committed) | `uv run python -m specrl.prompts` | CPU | `data/prompts.jsonl` |
| 1. Log model outputs | `uv run python -m specrl.log_data --pair qwen` | GPU | `logs/*.npz` |
| 2. Measure c | `uv run python -m specrl.measure_c --pair qwen` | GPU | `results/c_qwen.json` |
| 3. Sim baselines | `uv run python -m specrl.baselines --pair qwen --c <c>` | CPU | `results/sim_baselines_*.csv` |
| 4. Train agents | *(P4–P5)* | CPU | `policies/` |
| 5. Live benchmark | *(P6)* | GPU | `results/*.csv` |
| 6. Figures | *(P7)* | CPU | `figures/` |
| 7. Demo | *(P8)* | GPU | Gradio app |

This table is filled in as each phase lands.

## Layout

```
src/specrl/   Python package (all code)
tests/        pytest tests: uv run pytest
docs/         one page per part of the system
report/       LaTeX report for Overleaf
logs/ data/ policies/   generated, not committed
results/ figures/       generated, committed (the report uses them)
```

## Docs

| Page | Covers |
|---|---|
| [01_setup.md](docs/01_setup.md) | Install on a new PC, CUDA check |
| [02_mdp.md](docs/02_mdp.md) | State, actions, reward, c, λ, sim speedup |
| [03_simulator.md](docs/03_simulator.md) | Prompts, logging, simulator, baselines |
| [07_report.md](docs/07_report.md) | How the LaTeX report is organised and synced with Overleaf |

More pages (MDP, simulator, algorithms, live benchmark, results) are added with each phase.
