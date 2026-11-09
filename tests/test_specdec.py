"""GPU correctness test (guide 6.3): speculative decoding must reproduce the target's greedy output.

    uv run pytest tests/test_specdec.py -s        (needs CUDA + the Qwen models; ~2 min)

Rare mismatches can come from fp16 numerics (one 5-token verify pass vs one-token steps give slightly
different logits, which can flip a near-tie). The guide's rule: more than 1% bad tokens means a bug.
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


@pytest.fixture(scope="module")
def models():
    tok, draft, target = load_pair("qwen")
    rows = load_prompts("test")
    rows = [r for t in ("chat", "code", "math") for r in rows if r["task"] == t][:: max(1, 99 // N_PROMPTS)][:N_PROMPTS]
    return tok, draft, target, [encode(tok, r["text"]) for r in rows]


def _token_agreement(a, b):
    """Tokens equal up to the first difference, over the reference length."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n, len(b)


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
        n, total = _token_agreement(a, b)
        assert n / total > 0.95, (n, total)


@pytest.mark.parametrize("policy", [fixed_k(1), fixed_k(4), fixed_k(8), threshold(0.4)], ids=["k1", "k4", "k8", "thr"])
def test_spec_matches_target(models, policy):
    _, draft, target, prompts = models
    good = total = exact = 0
    for ids in prompts:
        ref = target_generate(ids, target)
        out, rounds = spec_generate(ids, draft, target, policy)
        assert sum(len(r["new"]) for r in rounds) == len(out)
        n, t = _token_agreement(out, ref)
        good, total, exact = good + n, total + t, exact + (out == ref)
    print(f"\n  exact {exact}/{len(prompts)} prompts, {good}/{total} tokens agree before first divergence")
    assert exact >= len(prompts) - 2
    assert good / total > 0.95


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
    good = total = exact = 0
    for ids in prompts:
        ref = target_generate(ids, gt)
        out, _ = spec_generate(ids, gd, gt, policy)
        n, t = _token_agreement(out, ref)
        good, total, exact = good + n, total + t, exact + (out == ref)
    print(f"
  graph: exact {exact}/{len(prompts)} prompts, {good}/{total} tokens agree before first divergence")
    assert exact >= len(prompts) - 2
    assert good / total > 0.95
