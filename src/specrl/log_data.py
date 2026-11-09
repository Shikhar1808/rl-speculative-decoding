"""Step 2: the logging pass ("smart logging", docs/03_simulator.md).

For each prompt: generate the target's greedy answer once, run the draft model once over the full text, and
store per answer token t:
    p[t]     draft top-token probability
    H[t]     entropy of the draft distribution
    match[t] draft's top token == target's actual token
The simulator replays any stopping policy from these arrays without touching the GPU again.

Usage:
    uv run python -m specrl.log_data --pair qwen
    uv run python -m specrl.log_data --pair smol --start 0 --end 250    # split prompts across 2 PCs

Output: logs/<pair>_<start>-<end>.npz  (load all parts of a pair with load_logs(pair))
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specrl.config import LOG_DIR, MAX_NEW_TOKENS, PAIRS
from specrl.prompts import load_prompts


def load_pair(pair, device="cuda"):
    draft_name, target_name = PAIRS[pair]
    tok = AutoTokenizer.from_pretrained(target_name)
    kw = dict(dtype=torch.float16, device_map=device)
    draft = AutoModelForCausalLM.from_pretrained(draft_name, **kw).eval()
    target = AutoModelForCausalLM.from_pretrained(target_name, **kw).eval()
    return tok, draft, target


def encode(tok, text, device="cuda"):
    """Prompt -> input ids using the model's chat template."""
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def log_prompt(prompt_ids, tok, draft, target, max_new=MAX_NEW_TOKENS):
    full = target.generate(
        prompt_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    )
    L = prompt_ids.shape[1]
    logits = draft(full).logits[0, L - 1 : -1].float()  # position i predicts answer token L+i
    probs = logits.softmax(-1)
    p_top, d_arg = probs.max(-1)
    ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
    match = d_arg == full[0, L:]
    return p_top.cpu().numpy(), ent.cpu().numpy(), match.cpu().numpy(), full[0, L:].cpu().numpy()


def run(pair, start, end):
    prompts = load_prompts()[start:end]
    tok, draft, target = load_pair(pair)
    parts = {k: [] for k in ("p", "H", "match", "tokens")}
    lengths, ids, tasks, splits = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(prompts):
        p, H, m, toks = log_prompt(encode(tok, r["text"]), tok, draft, target)
        for k, v in zip(parts, (p, H, m, toks)):
            parts[k].append(v)
        lengths.append(len(p))
        ids.append(r["id"]), tasks.append(r["task"]), splits.append(r["split"])
        if (i + 1) % 20 == 0 or i + 1 == len(prompts):
            acc = np.concatenate(parts["match"]).mean()
            print(f"[{pair}] {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s  draft match rate so far {acc:.3f}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{pair}_{start}-{start + len(prompts)}.npz"
    np.savez_compressed(
        out,
        p=np.concatenate(parts["p"]).astype(np.float32),
        H=np.concatenate(parts["H"]).astype(np.float32),
        match=np.concatenate(parts["match"]).astype(bool),
        tokens=np.concatenate(parts["tokens"]).astype(np.int32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        id=np.array(ids),
        task=np.array(tasks),
        split=np.array(splits),
    )
    print(f"wrote {out}")
    for task in sorted(set(tasks)):
        sel = [j for j, t in enumerate(tasks) if t == task]
        m = np.concatenate([parts["match"][j] for j in sel])
        print(f"  {task}: match rate {m.mean():.3f}, mean answer length {np.mean([lengths[j] for j in sel]):.0f} tokens")


def load_logs(pair, split=None):
    """All logged episodes of a pair (merges the parts from both PCs).

    Returns a list of dicts: {"id", "task", "split", "p", "H", "match", "tokens"} — one per prompt.
    """
    eps = []
    files = sorted(LOG_DIR.glob(f"{pair}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no logs for pair '{pair}' in {LOG_DIR}; run specrl.log_data first")
    for f in files:
        z = np.load(f)
        o = z["offsets"]
        for j in range(len(z["id"])):
            sl = slice(o[j], o[j + 1])
            ep = {"id": str(z["id"][j]), "task": str(z["task"][j]), "split": str(z["split"][j])}
            ep.update({k: z[k][sl] for k in ("p", "H", "match", "tokens")})
            eps.append(ep)
    seen = {}
    for ep in eps:  # same prompt logged on both PCs -> keep one
        seen.setdefault(ep["id"], ep)
    return [ep for ep in seen.values() if split is None or ep["split"] == split]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=PAIRS, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    run(a.pair, a.start, a.end)


if __name__ == "__main__":
    main()
