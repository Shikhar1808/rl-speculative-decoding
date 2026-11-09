"""Gradio race demo: target alone vs target + RL agent on the same prompt (docs/08_demo.md).

Both generations run one after the other on the GPU (running them at the same time would make them share
the GPU and slow each other). Every token is timestamped as it is produced, then both panes are replayed
side by side at the recorded speed, so the race shows the real timings.

Right pane colours: green = accepted draft token, red struck-through = rejected draft, bold = the target's
own token (correction or bonus), thin bar = where the agent pressed STOP. Hover a draft token to see its
draft confidence p.

Usage:
    uv run python -m specrl.app                  # SmolLM2 pair, http://localhost:7860
    uv run python -m specrl.app --pair qwen --share-lan    # also reachable from the other laptop
"""

import argparse
import html
import time

import gradio as gr
import torch

from specrl.baselines import fixed_k, threshold
from specrl.config import PAIRS
from specrl.log_data import encode
from specrl.specdec import load_fast_pair, spec_generate, target_generate
from specrl.train import ALGOS, load_policy

FRAME = 0.05  # seconds between replay frames
NAMES = {"vi": "Value Iteration", "qlearn": "Q-learning", "dqn": "DQN", "reinforce": "REINFORCE"}
BOX = ("font-family:ui-monospace,Consolas,monospace;font-size:14px;line-height:1.7;white-space:pre-wrap;"
       "padding:12px;border:1px solid #d8d7d2;border-radius:8px;min-height:260px;background:#fcfcfb;color:#0b0b0b")


def build(pair):
    tok, draft, target = load_fast_pair(pair)
    policies = {NAMES[a]: load_policy(pair, a) for a in ALGOS}
    policies["Fixed k=4 (baseline)"] = fixed_k(4)
    policies["Threshold 0.4 (HuggingFace baseline)"] = threshold(0.4)
    warm = encode(tok, "Hello")
    target_generate(warm, target, max_new=8)
    spec_generate(warm, draft, target, fixed_k(4), max_new=8)

    def esc(t):
        return html.escape(tok.decode([t]))

    def record(prompt, policy):
        ids = encode(tok, prompt)
        t_times, t_toks = [], []
        t0 = time.perf_counter()
        target_generate(ids, target, on_step=lambda toks: (t_toks.extend(toks),
                                                          t_times.append(time.perf_counter() - t0)))
        r_times, rounds = [], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec_generate(ids, draft, target, policy, on_step=lambda r: (rounds.append(r),
                                                                     r_times.append(time.perf_counter() - t0)))
        return t_toks, t_times, rounds, r_times

    def render_target(toks):
        return f"<div style='{BOX}'>" + "".join(esc(t) for t in toks) + "</div>"

    def render_spec(rounds):
        out = []
        for r in rounds:
            new, n, drafts = r["new"], r["n"], r["drafts"]
            acc = min(n, len(new))
            for t, p in zip(new[:acc], r["p"]):
                out.append(f"<span title='draft p={p:.2f}' style='background:#d9f2e4'>{esc(t)}</span>")
            for t, p in zip(drafts[n:], r["p"][n:]):
                out.append(f"<span title='rejected, draft p={p:.2f}' style='color:#c23b3b;"
                           f"text-decoration:line-through;opacity:.75'>{esc(t)}</span>")
            if len(new) > acc:
                out.append(f"<b>{esc(new[acc])}</b>")
            out.append("<span style='border-left:2px solid #2a78d6;margin:0 1px'></span>")
        return f"<div style='{BOX}'>" + "".join(out) + "</div>"

    def meter(rounds):
        if not rounds:
            return ""
        r = rounds[-1]
        bars = "".join(
            f"<div style='display:flex;align-items:center;gap:6px;font-size:12px'>"
            f"<span style='width:56px;font-family:monospace'>draft {i + 1}</span>"
            f"<div style='height:10px;width:{int(160 * p)}px;background:{'#1baf7a' if i < r['n'] else '#e34948'};"
            f"border-radius:0 4px 4px 0'></div><span>{p:.2f}</span></div>"
            for i, p in enumerate(r["p"])
        )
        return (f"<div style='font-size:13px;margin-bottom:4px'>Latest round: drafted {r['s']}, "
                f"accepted {r['n']}, agent pressed STOP after draft {r['s']}</div>{bars}")

    def stats(n_tok, secs, label):
        tps = n_tok / secs if secs > 0 else 0.0
        return f"**{label}**: {n_tok} tokens in {secs:.2f}s = **{tps:.1f} tok/s**", tps

    def race(prompt, method):
        if not prompt.strip():
            raise gr.Error("Type a prompt first.")
        t_toks, t_times, rounds, r_times = record(prompt, policies[method])
        end = max(t_times[-1], r_times[-1])
        start = time.perf_counter()
        while True:
            el = time.perf_counter() - start
            nt = sum(1 for x in t_times if x <= el)
            nr = sum(1 for x in r_times if x <= el)
            shown_r = rounds[:nr]
            n_spec = sum(len(r["new"]) for r in shown_r)
            s_t, tps_t = stats(nt, min(el, t_times[-1]), "Target alone")
            s_r, tps_r = stats(n_spec, min(el, r_times[-1]), f"Target + {method}")
            summary = f"{s_t}  \n{s_r}"
            if el >= end:
                tps_t, tps_r = len(t_toks) / t_times[-1], n_spec / r_times[-1]
                acc = n_spec / len(rounds)
                same = "identical" if sum((r["new"] for r in rounds), []) == t_toks else "DIFFERENT (fp16 near-tie)"
                summary += (f"  \n\n**Speedup {tps_r / tps_t:.2f}×** · {len(rounds)} target calls instead of "
                            f"{len(t_toks)} · {acc:.2f} tokens per call · output {same}")
            yield render_target(t_toks[:nt]), render_spec(shown_r), meter(shown_r), summary
            if el >= end:
                break
            time.sleep(FRAME)

    return race, list(policies)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, default="smol")
    ap.add_argument("--share-lan", action="store_true", help="listen on 0.0.0.0 so other PCs on the LAN can open it")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    race, methods = build(a.pair)

    with gr.Blocks(title="Learning When to Stop Drafting") as demo:
        gr.Markdown(
            f"## Learning When to Stop Drafting — live race ({a.pair})\n"
            "A small **draft** model guesses tokens; the big **target** model checks them in one pass. "
            "The RL agent decides after every guess whether to keep drafting or stop and verify. "
            "Output is identical to the target alone, just faster."
        )
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="Write a Python function that checks if a number is prime.",
                                lines=2, scale=4)
            method = gr.Dropdown(methods, value=methods[0], label="Stopping policy", scale=2)
        go = gr.Button("Race", variant="primary")
        summary = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Target alone** (one token per big-model call)")
                left = gr.HTML()
            with gr.Column():
                gr.Markdown("**Target + RL agent** — green accepted · red rejected · **bold** target's token · "
                            "blue bar = STOP")
                right = gr.HTML()
        conf = gr.HTML(label="Draft confidence")
        go.click(race, [prompt, method], [left, right, conf, summary])
        gr.Examples(
            [["Write a Python function that checks if a number is prime."],
             ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
              "How many clips did Natalia sell altogether in April and May? Solve step by step."],
             ["Give three tips for staying healthy."]],
            inputs=[prompt],
        )
    demo.queue().launch(server_name="0.0.0.0" if a.share_lan else "127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
