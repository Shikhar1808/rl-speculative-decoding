"""Train an RL stopping policy in the simulator, with Dinkelbach lambda tuning (docs/04_algorithms.md).

    lambda_0 = throughput of the best fixed-k baseline on the train split
    train with lambda_0 -> measure throughput lambda_1 -> retrain with lambda_1   (--rounds 2, default)

Usage:
    uv run python -m specrl.algos.train --pair qwen --algo vi
    uv run python -m specrl.algos.train --pair qwen --algo all
    uv run python -m specrl.algos.train --pair qwen --algo dqn --lam 2.5 --rounds 1     # fixed lambda (sweeps)

c is read from results/c_<pair>.json (specrl.live.measure_c) unless --c is given.
Outputs: policies/<pair>_<algo>.pt, results/curve_<pair>_<algo>.csv, rows in results/sim_rl_<pair>.csv
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from specrl.algos import dqn, qlearn, reinforce, vi
from specrl.algos.common import NetPolicy, TablePolicy, mlp
from specrl.sim.baselines import baseline_policies
from specrl.config import K, PAIRS, POLICY_DIR, RESULTS_DIR
from specrl.sim.env import evaluate
from specrl.data.log_data import load_logs

ALGOS = {"vi": vi, "qlearn": qlearn, "dqn": dqn, "reinforce": reinforce}


def to_policy(saved):
    if saved["kind"] == "table":
        return TablePolicy(saved["table"])
    net = mlp(saved["arch"])
    net.load_state_dict(saved["state_dict"])
    return NetPolicy(net)


def load_policy(pair, algo):
    """Greedy policy `obs -> action` from policies/<pair>_<algo>.pt."""
    return to_policy(torch.load(POLICY_DIR / f"{pair}_{algo}.pt", weights_only=False))


def read_c(pair):
    f = RESULTS_DIR / f"c_{pair}.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: run `uv run python -m specrl.live.measure_c --pair {pair}` or pass --c")
    return json.loads(f.read_text())["c"]


def best_fixed_k(episodes, c):
    res = {name: evaluate(pol, episodes, c) for name, pol in baseline_policies().items() if name.startswith("fixed")}
    name = max(res, key=lambda n: res[n]["throughput"])
    return name, res[name]["throughput"]


def train_one(pair, algo, train_eps, test_eps, c, lam, rounds, n_episodes, seed):
    for rd in range(rounds):
        t0 = time.time()
        kw = {} if n_episodes is None else {"n_episodes": n_episodes}  # None: each algorithm's own default
        saved = ALGOS[algo].train(train_eps, lam=lam, c=c, K=K, seed=seed, **kw)
        secs = time.time() - t0
        pol = to_policy(saved)
        tr = evaluate(pol, train_eps, c)
        print(f"  [{algo}] round {rd + 1}: lambda={lam:.3f}  train throughput {tr['throughput']:.3f}  ({secs:.1f}s)")
        used_lam, lam = lam, tr["throughput"]  # Dinkelbach update
    te = evaluate(pol, test_eps, c)
    saved.update(pair=pair, algo=algo, lam=used_lam, c=c, K=K, train_seconds=secs)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(saved, POLICY_DIR / f"{pair}_{algo}.pt")
    pd.DataFrame(saved["curve"]).to_csv(RESULTS_DIR / f"curve_{pair}_{algo}.csv", index=False)
    return {"method": algo, "lam": used_lam, "train_seconds": secs, **{f"test_{k}": v for k, v in te.items()},
            "train_throughput": tr["throughput"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    ap.add_argument("--algo", choices=[*ALGOS, "all"], required=True)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None, help="default: best fixed-k throughput (train split)")
    ap.add_argument("--rounds", type=int, default=2, help="Dinkelbach rounds")
    ap.add_argument("--episodes", type=int, default=None,
                    help="training episodes (default per algorithm: QL 10000, DQN/REINFORCE 3000)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = a.c if a.c is not None else read_c(a.pair)
    train_eps, test_eps = load_logs(a.pair, "train"), load_logs(a.pair, "test")
    name, thr = best_fixed_k(train_eps, c)
    lam = a.lam if a.lam is not None else thr
    print(f"{a.pair}: {len(train_eps)} train / {len(test_eps)} test episodes, c={c:.3f}, best fixed: {name} ({thr:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [train_one(a.pair, al, train_eps, test_eps, c, lam, a.rounds, a.episodes, a.seed)
            for al in (ALGOS if a.algo == "all" else [a.algo])]

    out = RESULTS_DIR / f"sim_rl_{a.pair}.csv"
    df = pd.DataFrame(rows)
    if out.exists():  # keep other algorithms' rows, replace the ones just trained
        old = pd.read_csv(out)
        df = pd.concat([old[~old.method.isin(df.method)], df], ignore_index=True)
    df.to_csv(out, index=False)

    base = {n: evaluate(p, test_eps, c)["throughput"] for n, p in baseline_policies().items()}
    best_base = max(base, key=base.get)
    print(f"\ntest split, sim speedup vs target alone (best baseline: {best_base} {base[best_base]:.3f}):")
    for _, r in df.iterrows():
        gain = (r.test_throughput / base[best_base] - 1) * 100
        print(f"  {r.method:10s} {r.test_throughput:.3f}   ({gain:+.1f}% vs best baseline)  "
              f"mean accepted len {r.test_mean_accepted_len:.2f}")


if __name__ == "__main__":
    main()
