"""Non-RL baselines as policies `obs -> action` (0 = CONTINUE, 1 = STOP), plus a sim baseline table.

Usage:
    uv run python -m specrl.baselines --pair qwen --c 0.25

Prints the fixed-k (k=1..8) and threshold-0.4 results in the simulator and saves results/sim_baselines_<pair>.csv.
The best fixed-k throughput is the starting lambda for training (docs/02_mdp.md).
"""

import argparse

import pandas as pd

from specrl.config import K, PAIRS, RESULTS_DIR
from specrl.env import CONTINUE, STOP, evaluate


def fixed_k(k):
    """Always draft exactly k tokens (obs[0] = s/K)."""
    return lambda obs: STOP if round(obs[0] * K) >= k else CONTINUE


def threshold(tau=0.4):
    """HuggingFace dynamic speculation: stop after a draft token whose top prob is below tau."""
    return lambda obs: STOP if obs[1] < tau else CONTINUE


def baseline_policies():
    pols = {f"fixed_k={k}": fixed_k(k) for k in range(1, K + 1)}
    pols["threshold_0.4"] = threshold(0.4)
    return pols


def table(episodes, c):
    rows = [{"method": name, **evaluate(pol, episodes, c)} for name, pol in baseline_policies().items()]
    return pd.DataFrame(rows)


def main():
    from specrl.log_data import load_logs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    ap.add_argument("--c", type=float, required=True, help="t_draft / t_target, from specrl.measure_c")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    a = ap.parse_args()
    df = table(load_logs(a.pair, a.split), a.c)
    print(df.round(3).to_string(index=False))
    fixed = df[df.method.str.startswith("fixed")]
    best = fixed.loc[fixed.throughput.idxmax()]
    print(f"\nbest fixed k: {best.method}, throughput {best.throughput:.3f} -> starting lambda")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"sim_baselines_{a.pair}_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()
