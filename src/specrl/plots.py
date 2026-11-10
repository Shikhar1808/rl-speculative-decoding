"""All figures for the report and slides (docs/06_results.md). CPU only; reads logs/, policies/, results/.

Usage:
    uv run python -m specrl.plots --pair qwen
Writes figures/<name>_<pair>.png and the analysis tables results/sim_per_task_<pair>.csv,
results/lambda_sweep_<pair>.csv.

Colour rule: baselines are grey, each RL algorithm has one fixed hue in every figure.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from specrl.algos.common import Q_BINS, forced, tab_state  # noqa: E402
from specrl.algos import vi  # noqa: E402
from specrl.baselines import baseline_policies  # noqa: E402
from specrl.config import ALGO_NAMES as NAMES  # noqa: E402
from specrl.config import FIG_DIR, K, PAIRS, POLICY_DIR, RESULTS_DIR  # noqa: E402
from specrl.env import SpecEnv, evaluate  # noqa: E402
from specrl.log_data import load_logs  # noqa: E402
from specrl.train import ALGOS, load_policy, read_c, to_policy  # noqa: E402

# reference categorical palette (dataviz skill), fixed slot per algorithm
COL = {"vi": "#2a78d6", "qlearn": "#eb6834", "dqn": "#1baf7a", "reinforce": "#6250d6"}
GREY, GREY_D, INK, INK2 = "#b9b8b2", "#7d7c76", "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def label(m):
    return NAMES.get(m, m.replace("_", " "))


def color(m):
    return COL.get(m, GREY)


def save(fig, name, pair):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}_{pair}.png")
    plt.close(fig)
    print(f"  figures/{name}_{pair}.png")


def mean_round_reward(policy, episodes, lam, c):
    env, total, n = SpecEnv(episodes, lam, c), 0.0, 0
    for ep in episodes:
        obs, done = env.reset(ep), False
        while not done:
            obs, r, done, info = env.step(policy(obs))
            if info:
                total, n = total + r, n + 1
    return total / n


def training_curves(pair, train_eps, c):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for algo in ("qlearn", "dqn", "reinforce"):
        f = RESULTS_DIR / f"curve_{pair}_{algo}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        y = d.mean_round_reward.rolling(5, min_periods=1).mean()
        ax.plot(d.episode, y, color=COL[algo], lw=2, label=NAMES[algo])
    saved = torch.load(POLICY_DIR / f"{pair}_vi.pt", weights_only=False)
    vi_r = mean_round_reward(to_policy(saved), train_eps, saved["lam"], c)
    ax.axhline(vi_r, color=COL["vi"], lw=1.5, ls="--", label="Value Iteration (model-based, no episodes)")
    ax.set_xlabel("training episode (one prompt's answer)")
    ax.set_ylabel("mean reward per round")
    ax.set_title(f"Training curves ({pair}), 5-point moving average", loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "training_curves", pair)


def policy_grid(pair, algo, episodes, min_visits=20):
    """Share of decisions that were STOP in each (s, q-bin) cell, measured by running the policy on the test
    prompts (forced STOPs excluded). Same measure for tables and networks; NaN where a cell has < min_visits."""
    pol = load_policy(pair, algo)
    env = SpecEnv(episodes, 0.0, read_c(pair))
    stop, n = np.zeros((K, Q_BINS)), np.zeros((K, Q_BINS))
    for ep in episodes:
        obs, done = env.reset(ep), False
        while not done:
            a = pol(obs)
            if not forced(env):
                s_, b = tab_state(obs)
                n[s_ - 1, b] += 1
                stop[s_ - 1, b] += a
            obs, _, done, _ = env.step(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_visits, stop / n, np.nan)


def heatmaps(pair, test_eps):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#efefec")
    for ax, algo in zip(axes, ALGOS):
        g = policy_grid(pair, algo, test_eps)
        im = ax.imshow(g[:-1], origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto",
                       extent=[0, 1, 0.5, K - 0.5])
        ax.set_title(NAMES[algo], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("q (chance all drafts survive)")
        ax.grid(False)
    axes[0].set_ylabel("s (tokens drafted this round)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("share of decisions = STOP", color=INK2)
    fig.suptitle(f"What each agent does ({pair}, test prompts; grey = state rarely visited; s = 8 always stops)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "policy_heatmap", pair)


def speedup_bars(pair, df, col, title, name):
    df = df.sort_values(col)
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(df) + 1))
    colors = [color(m) for m in df.method]
    ax.barh([label(m) for m in df.method], df[col], color=colors, height=0.6)
    ax.axvline(1.0, color=GREY_D, lw=1, ls=":")
    for i, v in enumerate(df[col]):
        ax.text(v + 0.01, i, f"{v:.2f}×", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("speedup vs target alone")
    ax.set_xlim(0, df[col].max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    save(fig, name, pair)


def per_task(pair, test_eps, c):
    pols = {m: load_policy(pair, m) for m in ALGOS}
    base = baseline_policies()
    pols["threshold_0.4"] = base["threshold_0.4"]
    rows = []
    for task in ("chat", "code", "math"):
        eps = [e for e in test_eps if e["task"] == task]
        for m, pol in pols.items():
            r = evaluate(pol, eps, c)
            rows.append({"task": task, "method": m, "mean_draft_len": r["draft_steps"] / r["target_calls"],
                         "mean_accepted_len": r["mean_accepted_len"], "sim_speedup": r["sim_speedup"],
                         "draft_match_rate": float(np.concatenate([e["match"] for e in eps]).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"sim_per_task_{pair}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    methods = list(pols)
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        d = df[df.method == m]
        ax.bar(np.arange(3) + (i - (len(methods) - 1) / 2) * w, d.mean_draft_len, w * 0.9, color=color(m),
               label=label(m))
    ax.set_xticks(range(3), ["chat", "code", "math"])
    ax.set_ylabel("mean tokens drafted per round")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, df.mean_draft_len.max() * 1.35)
    ax.set_title(f"How long each policy drafts, by task ({pair}, test prompts, simulator)", loc="left",
                 color=INK, fontsize=11)
    save(fig, "per_task_draft_len", pair)


def lambda_sweep(pair, train_eps, test_eps, c):
    lam0 = torch.load(POLICY_DIR / f"{pair}_vi.pt", weights_only=False)["lam"]
    rows = []
    for f in np.linspace(0.5, 1.5, 11):
        lam = lam0 * f
        pol = to_policy(vi.train(train_eps, lam=lam, c=c, K=K))
        r = evaluate(pol, test_eps, c)
        rows.append({"lambda": lam, "lambda_over_tuned": f, "sim_speedup": r["sim_speedup"],
                     "mean_draft_len": r["draft_steps"] / r["target_calls"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"lambda_sweep_{pair}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, yl in zip(axes, ("sim_speedup", "mean_draft_len"), ("sim speedup vs target alone",
                                                                     "mean tokens drafted per round")):
        ax.plot(df["lambda"], df[col], color=COL["vi"], lw=2, marker="o", ms=5)
        ax.axvline(lam0, color=GREY_D, ls=":", lw=1)
        ax.text(lam0, ax.get_ylim()[1], " tuned λ", color=INK2, va="top", fontsize=8)
        ax.set_xlabel("λ (price of one target call, tokens)")
        ax.set_ylabel(yl)
    fig.suptitle(f"Effect of λ on Value Iteration ({pair}, test prompts, simulator)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    save(fig, "lambda_sweep", pair)


def sim_vs_live(pair, sim, live):
    """Dumbbell: one row per method, open dot = simulator, filled dot = live wall clock."""
    m = sim.merge(live, on="method").sort_values("speedup")
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(m) + 1.2))
    for i, r in enumerate(m.itertuples()):
        ax.plot([r.speedup, r.sim_speedup], [i, i], color=GREY, lw=2, zorder=1)
        ax.scatter(r.sim_speedup, i, s=60, facecolors="white", edgecolors=color(r.method), linewidths=2, zorder=3)
        ax.scatter(r.speedup, i, s=60, color=color(r.method), edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_yticks(range(len(m)), [label(x) for x in m.method])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("speedup vs target alone")
    ax.scatter([], [], s=60, facecolors="white", edgecolors=INK2, linewidths=2, label="simulator")
    ax.scatter([], [], s=60, color=INK2, label="live (wall clock)")
    ax.legend(loc="lower right", fontsize=8)
    gap = float((1 - m.speedup / m.sim_speedup).mean())
    rho = m.speedup.rank().corr(m.sim_speedup.rank())  # Spearman rank correlation
    ax.set_title(f"Simulator vs live ({pair}): live {gap:.0%} lower on average, rank corr. {rho:.2f}", loc="left",
                 color=INK, fontsize=11)
    save(fig, "sim_vs_live", pair)


def sim_table(pair, test_eps, c):
    pols = {**baseline_policies(), **{m: load_policy(pair, m) for m in ALGOS}}
    return pd.DataFrame([{"method": m, **evaluate(p, test_eps, c)} for m, p in pols.items()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    a = ap.parse_args()
    pair, c = a.pair, read_c(a.pair)
    train_eps, test_eps = load_logs(pair, "train"), load_logs(pair, "test")

    training_curves(pair, train_eps, c)
    heatmaps(pair, test_eps)
    per_task(pair, test_eps, c)
    lambda_sweep(pair, train_eps, test_eps, c)
    sim = sim_table(pair, test_eps, c)
    sim.to_csv(RESULTS_DIR / f"sim_all_{pair}.csv", index=False)
    keep = sim[~sim.method.str.startswith("fixed") | sim.method.isin(
        [sim[sim.method.str.startswith("fixed")].sort_values("sim_speedup").method.iloc[-1], "fixed_k=4"])]
    speedup_bars(pair, keep, "sim_speedup", f"Simulated speedup ({pair}, test prompts)", "speedup_sim")
    live_f = RESULTS_DIR / f"live_{pair}.csv"
    if live_f.exists():
        live = pd.read_csv(live_f)
        speedup_bars(pair, live, "speedup", f"Live speedup on RTX 3050 ({pair}, wall clock)", "speedup_live")
        sim_vs_live(pair, sim, live[live.method != "target_alone"])


if __name__ == "__main__":
    main()
