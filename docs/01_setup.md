# Setup on a new PC

Every teammate's PC needs the same environment. uv reads `pyproject.toml` + `uv.lock`, so one command
recreates it exactly.

## Requirements

| Item | Needed | Check |
|---|---|---|
| NVIDIA GPU | 6 GB+ VRAM (both laptops: RTX 3050 6 GB) | `nvidia-smi` |
| NVIDIA driver | supports CUDA 12.8 or newer (driver 570+) | top-right of `nvidia-smi` shows "CUDA Version" ≥ 12.8 |
| uv | any recent version | `uv --version` |
| Disk | ~15 GB free (env ~5 GB, 4 models ~10 GB) | |
| Internet | huggingface.co reachable (models + datasets download on first run) | |

You do **not** need to install Python, CUDA toolkit or PyTorch yourself: uv fetches Python 3.12 and the
CUDA 12.8 build of PyTorch.

## Steps

1. Install uv (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Get the project folder (git clone or copy, without `.venv`).
3. In the project folder: `uv sync`
4. Check the GPU works:
   ```bash
   uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expected: `2.x.x+cu128 True NVIDIA GeForce RTX 3050 ...`

If it prints `False` or `+cpu`, the CPU-only PyTorch got installed — check that `pyproject.toml` still has
the `pytorch-cu128` index and run `uv sync` again.

## Why CUDA 12.8

The PyTorch build is pinned to CUDA 12.8 (`[tool.uv.index]` in `pyproject.toml`). It was tested working on
PC1, and it also runs on drivers older than the newest CUDA 13 builds need, which keeps both laptops safe.
The RTX 3050 gets no benefit from CUDA 13.

## Running things

Always through uv, never bare `python`/`pip`:

```bash
uv run python -m specrl.<module> --help   # any script
uv run pytest                              # tests
uv add <package>                           # new dependency (updates pyproject + lock; commit both)
```

## Timing rule

All timings that are compared in the report must come from **one** PC, plugged in, with nothing else using
the GPU.
