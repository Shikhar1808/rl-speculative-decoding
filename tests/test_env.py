import numpy as np

from specrl.sim.baselines import fixed_k, threshold
from specrl.sim.env import CONTINUE, STOP, SpecEnv, evaluate


def ep(match, p=None):
    n = len(match)
    return {
        "match": np.array(match, dtype=bool),
        "p": np.array(p if p is not None else [0.9] * n, dtype=np.float32),
        "H": np.zeros(n, dtype=np.float32),
    }


def test_guide_example():
    # Guide 6.2: match [1,1,0,1], s=3 -> n=2, gained=3
    env = SpecEnv([], lam=1.0, c=0.1)
    env.reset(ep([1, 1, 0, 1]))
    env.step(CONTINUE)
    env.step(CONTINUE)
    obs, r, done, info = env.step(STOP)
    assert info == {"n": 2, "s": 3, "gained": 3}
    assert np.isclose(r, 3 - 1.0 * (1 + 3 * 0.1))
    assert env.t == 3 and not done


def test_bonus_token_when_all_accepted_and_capped_at_end():
    env = SpecEnv([], lam=0.0, c=0.0)
    env.reset(ep([1, 1, 1, 1, 1]))
    env.step(CONTINUE)
    _, _, done, info = env.step(STOP)
    assert info["n"] == 2 and info["gained"] == 3  # 2 accepted + bonus
    env.step(CONTINUE)  # s=2, t+s=5 == end -> next CONTINUE is forced to STOP
    _, _, done, info = env.step(CONTINUE)
    assert info["n"] == 2 and info["gained"] == 2 and done  # bonus would be past the end


def test_forced_stop_at_K():
    env = SpecEnv([], lam=0.0, c=0.0, K=3)
    env.reset(ep([1] * 10))
    env.step(CONTINUE)
    env.step(CONTINUE)
    _, _, _, info = env.step(CONTINUE)  # s == K -> STOP
    assert info["s"] == 3 and info["gained"] == 4


def test_q_is_product_of_p():
    env = SpecEnv([], lam=0.0, c=0.0)
    obs = env.reset(ep([1, 1, 1], p=[0.5, 0.4, 0.9]))
    assert np.isclose(obs[3], 0.5)
    obs, *_ = env.step(CONTINUE)
    assert np.isclose(obs[3], 0.2)


def test_evaluate_counts_every_token():
    eps = [ep([1, 0, 1, 1, 0, 0, 1]), ep([0, 0, 1])]
    for pol in (fixed_k(1), fixed_k(4), fixed_k(8), threshold(0.4)):
        res = evaluate(pol, eps, c=0.2)
        assert res["tokens"] == 10  # every answer token is produced exactly once


def test_target_alone_equivalent():
    # all drafts wrong, k=1, c=0: one token per target call -> speedup 1
    res = evaluate(fixed_k(1), [ep([0] * 6)], c=0.0)
    assert res["target_calls"] == 6 and np.isclose(res["sim_speedup"], 1.0)
