"""GPU correctness test (guide 6.3): speculative decoding must reproduce the target's greedy output.

    uv run pytest tests/test_specdec.py -s        (needs CUDA + the Qwen models; ~2 min)

Rare mismatches come from fp16 numerics: one multi-token verify pass rounds differently from one-token
steps, which can flip a near-tie between the target's top two tokens. So the rule checked here: wherever the
speculative output first differs from the target's own greedy output, the target's top-2 logit gap at that
position must be tiny (< TIE_GAP). A real bug (wrong acceptance or cache indexing) diverges at clear gaps.
"""

import pytest
import torch

from specrl.baselines import fixed_k, threshold
from specrl.log_data import encode, load_pair
from specrl.prompts import load_prompts
from specrl.fastmodel import GraphRunner
from specrl.specdec import spec_generate, target_generate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N_PROMPTS = 20
TIE_GAP = 0.1  # logits; observed fp16 divergences: gaps 0.000-0.016


@pytest.fixture(scope="module")
def models():
    tok, draft, target = load_pair("qwen")
    rows = load_prompts("test")
    rows = [r for t in ("chat", "code", "math") for r in rows if r["task"] == t][:: max(1, 99 // N_PROMPTS)][:N_PROMPTS]
    return tok, draft, target, [encode(tok, r["text"]) for r in rows]


@torch.no_grad()
def _check_divergence(ids, out, ref, target):
    """Return None if out == ref, else the target's top-2 logit gap where they first differ."""
    if out == ref:
        return None
    i = next((j for j, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
    prefix = torch.cat([ids, ids.new_tensor([ref[:i]])], 1) if i else ids
    top = target(prefix).logits[0, -1].float().topk(2).values
    return float(top[0] - top[1])




def test_target_loop_matches_hf_generate(models):
    tok, _, target, prompts = models
    for ids in prompts[:5]:
        ref = target.generate(ids, max_new_tokens=128, do_sample=False, repetition_penalty=1.0,
                              pad_token_id=tok.eos_token_id)[0, ids.shape[1] :]
        assert target_generate(ids, target) == ref.tolist()


def test_cache_equals_no_cache(models):
    _, draft, target, prompts = models
    for ids in prompts[:3]:
        a, _ = spec_generate(ids, draft, target, fixed_k(4), cache=True)
        b, _ = spec_generate(ids, draft, target, fixed_k(4), cache=False)
        gap = _check_divergence(ids, a, b, target)
        assert gap is None or gap < TIE_GAP, gap


@pytest.mark.parametrize("policy", [fixed_k(4), threshold(0.4)], ids=["k4", "thr"])
def test_spec_matches_target(models, policy):
    _, draft, target, prompts = models
    exact, gaps = 0, []
    for ids in prompts:
        ref = target_generate(ids, target)
        out, rounds = spec_generate(ids, draft, target, policy)
        assert sum(len(r["new"]) for r in rounds) == len(out)
        gap = _check_divergence(ids, out, ref, target)
        exact += gap is None
        gaps += [] if gap is None else [round(gap, 3)]
    print(f"\n  eager: exact {exact}/{len(prompts)}; top-2 gaps where they diverge: {gaps}")
    assert all(g < TIE_GAP for g in gaps)


@pytest.fixture(scope="module")
def fast(models):
    _, draft, target, prompts = models
    return GraphRunner(draft), GraphRunner(target), prompts


def test_graph_target_equals_eager(models, fast):
    _, _, target, _ = models
    gd, gt, prompts = fast
    for ids in prompts[:5]:
        assert target_generate(ids, gt) == target_generate(ids, target)


@pytest.mark.parametrize("policy", [fixed_k(1), fixed_k(4), fixed_k(8), threshold(0.4)], ids=["k1", "k4", "k8", "thr"])
def test_graph_spec_matches_target(fast, policy):
    gd, gt, prompts = fast
    exact, gaps = 0, []
    for ids in prompts:
        ref = target_generate(ids, gt)
        out, rounds = spec_generate(ids, gd, gt, policy)
        assert sum(len(r["new"]) for r in rounds) == len(out)
        gap = _check_divergence(ids, out, ref, gt.model)
        exact += gap is None
        gaps += [] if gap is None else [round(gap, 3)]
    print(f"\n  graph: exact {exact}/{len(prompts)}; top-2 gaps where they diverge: {gaps}")
    assert all(g < TIE_GAP for g in gaps)
