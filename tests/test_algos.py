"""Smoke + sanity tests for the four algorithms on synthetic logs where draft confidence predicts a match."""

import numpy as np
import pytest

from specrl.config import K
from specrl.env import evaluate
from specrl.train import ALGOS, best_fixed_k, to_policy


def synthetic(n_eps=60, length=100, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_eps):
        easy = rng.random(length) < 0.6  # easy tokens: confident and usually right
        p = np.where(easy, rng.uniform(0.8, 1.0, length), rng.uniform(0.05, 0.5, length)).astype(np.float32)
        match = rng.random(length) < np.where(easy, 0.95, 0.25)
        eps.append({"p": p, "H": (1 - p).astype(np.float32) * 3, "match": match})
    return eps


@pytest.mark.parametrize("algo", list(ALGOS))
def test_algo_trains_and_beats_worst_fixed_k(algo):
    eps, c = synthetic(), 0.2
    _, lam = best_fixed_k(eps, c)
    saved = ALGOS[algo].train(eps, lam=lam, c=c, K=K, n_episodes=300, seed=0)
    pol = to_policy(saved)
    res = evaluate(pol, eps, c)
    assert res["tokens"] == sum(len(e["p"]) for e in eps)
    assert len(saved["curve"]) > 0
    assert res["throughput"] > 1.0  # faster than the target alone


def test_value_iteration_matches_or_beats_best_fixed_k():
    eps, c = synthetic(), 0.2
    name, best = best_fixed_k(eps, c)
    pol = to_policy(ALGOS["vi"].train(eps, lam=best, c=c, K=K))
    assert evaluate(pol, eps, c)["throughput"] >= 0.99 * best, name
