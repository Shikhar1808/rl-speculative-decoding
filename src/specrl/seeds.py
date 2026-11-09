"""Seed robustness: retrain the sampling-based algorithms with several seeds (simulator only).

Value Iteration is deterministic, so it needs one run. Q-learning, DQN and REINFORCE are retrained with
seeds 0..N-1 using the same Dinkelbach recipe as train.py; saved policies (seed 0, used by the live benchmark
and demo) are NOT overwritten.

Usage:
    uv run python -m specrl.seeds --pair qwen --seeds 3
Writes results/seeds_<pair>.csv (one row per algorithm x seed) and prints mean ± std of the test speedup.
"""

import argparse

import pandas as pd

from specrl.config import K, PAIRS, RESULTS_DIR
from specrl.env import evaluate
from specrl.log_data import load_logs
from specrl.train import ALGOS, best_fixed_k, read_c, to_policy


def run(pair, n_seeds, rounds=2):
    c = read_c(pair)
    train_eps, test_eps = load_logs(pair, "train"), load_logs(pair, "test")
    _, lam0 = best_fixed_k(train_eps, c)
    rows = []
    for algo in ALGOS:
        for seed in range(1 if algo == "vi" else n_seeds):
            lam = lam0
            for _ in range(rounds):
                pol = to_policy(ALGOS[algo].train(train_eps, lam=lam, c=c, K=K, seed=seed))
                lam = evaluate(pol, train_eps, c)["throughput"]
            te = evaluate(pol, test_eps, c)
            rows.append({"pair": pair, "method": algo, "seed": seed, "test_sim_speedup": te["sim_speedup"],
                         "test_mean_accepted_len": te["mean_accepted_len"]})
            print(f"  {algo} seed {seed}: test sim speedup {te['sim_speedup']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"seeds_{pair}.csv", index=False)
    summary = df.groupby("method").test_sim_speedup.agg(["mean", "std", "min", "max", "count"])
    print(summary.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    run(a.pair, a.seeds)


if __name__ == "__main__":
    main()
