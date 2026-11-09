"""Measure c = t_draft / t_target on this GPU (docs/02_mdp.md).

t_draft  = one draft decode step (1 new token, KV cache)
t_target = one target verify pass (5 tokens at once, KV cache) — about what a round sends
Both measured after a ~200-token context, 3 warm-ups, CUDA-synchronised, median of N runs.

Usage:
    uv run python -m specrl.measure_c --pair qwen
Writes results/c_<pair>.json. Use the same PC as the live benchmark.
"""

import argparse
import json
import statistics
import time

import torch
from transformers import DynamicCache

from specrl.config import PAIRS, RESULTS_DIR
from specrl.log_data import encode, load_pair
from specrl.prompts import load_prompts


@torch.no_grad()
def time_step(model, ctx, n_new, runs=30):
    cache = DynamicCache()
    model(ctx, past_key_values=cache, use_cache=True)
    L = ctx.shape[1]
    x = ctx[:, -n_new:]  # any n_new token ids work for timing
    times = []
    for i in range(runs + 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t0)
        cache.crop(L)  # roll back so every run sees the same context length
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    ap.add_argument("--verify-len", type=int, default=5)
    a = ap.parse_args()
    tok, draft, target = load_pair(a.pair)
    ctx = encode(tok, load_prompts("train")[0]["text"])
    ctx = torch.cat([ctx] * (1 + 200 // ctx.shape[1]), 1)[:, :200]
    t_d = time_step(draft, ctx, 1)
    t_t = time_step(target, ctx, a.verify_len)
    t_t1 = time_step(target, ctx, 1)
    res = {
        "pair": a.pair,
        "t_draft_ms": t_d * 1e3,
        "t_target_ms": t_t * 1e3,
        "t_target_1tok_ms": t_t1 * 1e3,
        "c": t_d / t_t,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(res, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"c_{a.pair}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
