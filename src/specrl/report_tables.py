"""Turn results/*.csv into LaTeX tables and copy figures for the report (docs/07_report.md).

Usage:
    uv run python -m specrl.report_tables
Writes report/generated/*.tex (\\input by the report) and copies figures/*.png to report/figures/.
Re-run after any new result, then upload report/generated and report/figures to Overleaf.
"""

import argparse
import json
import shutil

import pandas as pd

from specrl.config import FIG_DIR, PAIRS, RESULTS_DIR, ROOT

OUT = ROOT / "report" / "generated"
NAMES = {
    "target_alone": "Target alone",
    "threshold_0.4": "Threshold 0.4",
    "vi": "Value Iteration",
    "qlearn": "Q-learning",
    "dqn": "DQN",
    "reinforce": "REINFORCE",
}
RL = ["vi", "qlearn", "dqn", "reinforce"]


def name(m):
    return NAMES.get(m, m.replace("fixed_k=", "Fixed $k=") + "$" if m.startswith("fixed") else m)


def read(f):
    p = RESULTS_DIR / f
    return pd.read_csv(p) if p.exists() else None


def main_table():
    """Live speedup + sim speedup + accepted length per method, both pairs side by side."""
    rows = {}
    for pair in PAIRS:
        live, sim, rl = read(f"live_{pair}.csv"), read(f"sim_all_{pair}.csv"), read(f"sim_rl_{pair}.csv")
        if sim is None:
            continue
        for _, r in sim.iterrows():
            rows.setdefault(r.method, {})[f"{pair}_sim"] = r.sim_speedup
            rows[r.method][f"{pair}_acc"] = r.mean_accepted_len
        if live is not None:
            for _, r in live.iterrows():
                rows.setdefault(r.method, {})[f"{pair}_live"] = r.speedup
                rows[r.method][f"{pair}_exact"] = r.exact_match_pct
        if rl is not None:
            for _, r in rl.iterrows():
                rows.setdefault(r.method, {})[f"{pair}_train_s"] = r.train_seconds

    keep = ["target_alone"] + [m for m in rows if m.startswith("fixed") and any(
        k.endswith("_live") for k in rows[m])] + ["threshold_0.4"] + RL
    if not any(k.endswith("_live") for r in rows.values() for k in r):  # no live yet: show sim fixed k=4
        keep = ["fixed_k=4", "threshold_0.4"] + RL

    def f(v, fmt="{:.2f}"):
        return "--" if v is None or pd.isna(v) else fmt.format(v)

    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Qwen2.5 0.5B$\to$1.5B} & \multicolumn{3}{c}{SmolLM2 135M$\to$1.7B} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Live speedup & Sim speedup & Tokens/call & Live speedup & Sim speedup & Tokens/call \\",
        r"\midrule",
    ]
    for m in keep:
        if m not in rows:
            continue
        r = rows[m]
        cells = []
        for pair in ("qwen", "smol"):
            live = r.get(f"{pair}_live")
            cells += [f(live, "{:.2f}$\\times$") if live is not None else "--",
                      f(r.get(f"{pair}_sim"), "{:.2f}$\\times$") if m != "target_alone" else "1.00$\\times$",
                      f(r.get(f"{pair}_acc")) if m != "target_alone" else "1.00"]
        if m == "threshold_0.4":
            lines.append(r"\midrule")
        lines.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
        if m == "threshold_0.4":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_task_table(pair):
    live = read(f"live_{pair}_by_task.csv")
    sim = read(f"sim_per_task_{pair}.csv")
    if sim is None:
        return None
    methods = ["threshold_0.4"] + RL
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Method & Chat & Code & Math \\", r"\midrule"]
    for m in methods:
        cells = []
        for task in ("chat", "code", "math"):
            d = sim[(sim.task == task) & (sim.method == m)].mean_draft_len
            sp = None
            if live is not None:
                x = live[(live.task == task) & (live.method == m)].speedup
                sp = x.iloc[0] if len(x) else None
            cell = f"{d.iloc[0]:.2f}" if len(d) else "--"
            if sp is not None:
                cell += f" / {sp:.2f}$\\times$"
            cells.append(cell)
        lines.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
    match = sim.groupby("task").draft_match_rate.first()
    lines.append(r"\midrule")
    lines.append(r"Draft match rate & " + " & ".join(f"{match.get(t, float('nan')):.2f}" for t in
                                                    ("chat", "code", "math")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def training_table():
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{SmolLM2} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Algorithm & $\lambda$ & Train time (s) & $\lambda$ & Train time (s) \\", r"\midrule"]
    data = {p: read(f"sim_rl_{p}.csv") for p in PAIRS}
    for m in RL:
        cells = []
        for p in ("qwen", "smol"):
            d = data.get(p)
            r = d[d.method == m] if d is not None else []
            cells += [f"{r.lam.iloc[0]:.2f}", f"{r.train_seconds.iloc[0]:.0f}"] if len(r) else ["--", "--"]
        lines.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def c_values():
    out = {}
    for p in PAIRS:
        f = RESULTS_DIR / f"c_{p}.json"
        out[p] = json.loads(f.read_text()) if f.exists() else None
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main_results.tex").write_text(main_table())
    (OUT / "training.tex").write_text(training_table())
    for p in PAIRS:
        t = per_task_table(p)
        if t:
            (OUT / f"per_task_{p}.tex").write_text(t)
    cv = c_values()
    macros = []
    for p, key in (("qwen", "Qwen"), ("smol", "Smol")):
        v = cv[p]
        macros.append(rf"\newcommand{{\numC{key.lower()}}}{{{v['c']:.2f}}}" if v else
                      rf"\newcommand{{\numC{key.lower()}}}{{?}}")
        if v:
            macros.append(rf"\newcommand{{\tDraft{key}}}{{{v['t_draft_ms']:.1f}}}")
            macros.append(rf"\newcommand{{\tTarget{key}}}{{{v['t_target_ms']:.1f}}}")
    (OUT / "numbers.tex").write_text("\n".join(macros) + "\n")
    dst = ROOT / "report" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for png in FIG_DIR.glob("*.png"):
        shutil.copy2(png, dst / png.name)
    print(f"wrote {OUT} and copied {len(list(FIG_DIR.glob('*.png')))} figures to report/figures")


if __name__ == "__main__":
    main()
