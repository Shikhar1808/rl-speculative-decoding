"""Step 2: the logging pass ("smart logging", docs/03_simulator.md).

For each prompt: generate the target's greedy answer once, run the draft model once over the full text, and
store per answer token t:
    p[t]     draft top-token probability
    H[t]     entropy of the draft distribution
    match[t] draft's top token == target's actual token
The simulator replays any stopping policy from these arrays without touching the GPU again.

Usage:
    uv run python -m specrl.log_data --pair qwen
    uv run python -m specrl.log_data --pair smol --batch 8             # 8 prompts per GPU pass (default)
    uv run python -m specrl.log_data --pair smol --start 0 --end 16   # a slice, e.g. for a quick test

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
    """Prompt -> input ids (1 x L) using the model's chat template."""
    return tok(_chat(tok, text), return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def _chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def log_batch(texts, tok, draft, target, max_new=MAX_NEW_TOKENS):
    """Log several prompts at once (left-padded batch). Returns one (p, H, match, tokens) tuple per prompt.

    Batching only changes how fast the logs are made: each row is masked so it only sees its own tokens.
    In fp16 a padded batch can round differently from a single prompt, so on rare near-ties the logged answer
    may differ slightly from batch-1 greedy decoding (DECISIONS.md #31); the simulator only needs a
    consistent answer + match flags, so this does not bias training.
    """
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = "left"
    enc = tok([_chat(tok, t) for t in texts], return_tensors="pt", padding=True, add_special_tokens=False).to(
        target.device
    )
    # repetition_penalty=1.0: Qwen's default config sets 1.1, which applies even with do_sample=False (DECISIONS #34)
    full = target.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, pad_token_id=pad)
    L = enc.input_ids.shape[1]
    gen = full[:, L:]
    eos = target.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, (list, tuple)) else [eos], device=gen.device)
    is_eos = torch.isin(gen, eos)
    # answer length per row: up to and including the first end-of-turn token
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1) + 1, torch.full_like(is_eos[:, 0], gen.shape[1],
                                                                                         dtype=torch.long))
    gen_mask = torch.arange(gen.shape[1], device=gen.device)[None, :] < first[:, None]
    mask = torch.cat([enc.attention_mask, gen_mask.long()], 1)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)  # real positions despite left padding
    logits = draft(full, attention_mask=mask, position_ids=pos).logits[:, L - 1 : -1]
    out = []
    for i in range(len(texts)):
        n = int(first[i])
        probs = logits[i, :n].float().softmax(-1)  # one row at a time keeps memory small
        p_top, d_arg = probs.max(-1)
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
        out.append((p_top.cpu().numpy(), ent.cpu().numpy(), (d_arg == gen[i, :n]).cpu().numpy(),
                    gen[i, :n].cpu().numpy()))
    return out


def run(pair, start, end, batch=8):
    prompts = load_prompts()[start:end]
    # sort by prompt length so each batch pads little; results are stored per prompt id, so order doesn't matter
    prompts = sorted(prompts, key=lambda r: len(r["text"]))
    tok, draft, target = load_pair(pair)
    parts = {k: [] for k in ("p", "H", "match", "tokens")}
    lengths, ids, tasks, splits = [], [], [], []
    t0 = time.time()
    for b in range(0, len(prompts), batch):
        rows = prompts[b : b + batch]
        for r, (p, H, m, toks) in zip(rows, log_batch([r["text"] for r in rows], tok, draft, target)):
            for k, v in zip(parts, (p, H, m, toks)):
                parts[k].append(v)
            lengths.append(len(p))
            ids.append(r["id"]), tasks.append(r["task"]), splits.append(r["split"])
        acc = np.concatenate(parts["match"]).mean()
        print(f"[{pair}] {len(ids)}/{len(prompts)}  {time.time() - t0:.0f}s  draft match rate so far {acc:.3f}",
              flush=True)

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
    ap.add_argument("--batch", type=int, default=8, help="prompts per GPU pass (1 = one at a time)")
    a = ap.parse_args()
    run(a.pair, a.start, a.end, a.batch)


if __name__ == "__main__":
    main()
