"""Build the prompt set: Alpaca (chat), HumanEval (code), GSM8K (math), split into train/test.

Usage:
    uv run python -m specrl.prompts            # writes data/prompts.jsonl (once; commit it)

Every row: {"id", "task", "split", "text"}. The file is committed so both PCs log the exact same prompts.
"""

import argparse
import json
import random

from datasets import load_dataset

from specrl.config import PROMPTS_FILE, SEED

PER_TASK = 167  # HumanEval only has 164, so it gives all of them -> ~500 total
TEST_FRACTION = 0.2


def _alpaca():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    for r in ds:
        if not r["input"]:  # keep instructions that need no extra input field
            yield r["instruction"]


def _humaneval():
    for r in load_dataset("openai/openai_humaneval", split="test"):
        yield "Complete the following Python function.\n\n" + r["prompt"]


def _gsm8k():
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        yield r["question"] + "\nSolve step by step."


SOURCES = {"chat": _alpaca, "code": _humaneval, "math": _gsm8k}


def build():
    rng = random.Random(SEED)
    rows = []
    for task, gen in SOURCES.items():
        texts = list(gen())
        rng.shuffle(texts)
        texts = texts[:PER_TASK]
        n_test = round(len(texts) * TEST_FRACTION)
        for i, t in enumerate(texts):
            rows.append({"id": f"{task}-{i:03d}", "task": task, "split": "test" if i < n_test else "train", "text": t})
    return rows


def load_prompts(split=None):
    """Read data/prompts.jsonl; optionally only one split ("train" or "test")."""
    rows = [json.loads(line) for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if split is None or r["split"] == split]


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = build()
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for task in SOURCES:
        n = [r for r in rows if r["task"] == task]
        print(f"{task}: {len(n)} prompts ({sum(r['split'] == 'test' for r in n)} test)")
    print(f"wrote {len(rows)} prompts -> {PROMPTS_FILE}")


if __name__ == "__main__":
    main()
