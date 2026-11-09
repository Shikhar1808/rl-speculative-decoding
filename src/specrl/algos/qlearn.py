"""Q-learning (temporal difference), tabular state (s, q-bin), epsilon-greedy with linear decay.

Each round is a short episode (gamma = 1): CONTINUE bootstraps from the next state of the same round,
STOP ends the round, so its target is just the reward.
    Q(s,a) <- Q(s,a) + alpha [ r + max_a' Q(s',a') * (not round_end) - Q(s,a) ]
"""

import random

import numpy as np

from specrl.algos.common import CONTINUE, Q_BINS, STOP, forced, tab_state
from specrl.env import SpecEnv


def train(episodes, lam, c, K, n_episodes=3000, alpha=0.05, eps_start=1.0, eps_end=0.05, seed=0, log_every=50, **_):
    rng = random.Random(seed)
    env = SpecEnv(episodes, lam, c, K, seed=seed)
    Q = np.zeros((K + 1, Q_BINS, 2))
    curve, window = [], []
    for e in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * e / (0.8 * n_episodes))
        obs, done = env.reset(), False
        while not done:
            st = tab_state(obs)
            if forced(env):
                a = STOP
            elif rng.random() < eps:
                a = rng.randrange(2)
            else:
                a = int(Q[st].argmax())
            obs2, r, done, info = env.step(a)
            round_end = bool(info)
            target = r if round_end else r + Q[tab_state(obs2)].max()
            Q[st][a] += alpha * (target - Q[st][a])
            if round_end:
                window.append(r)
            obs = obs2
        if (e + 1) % log_every == 0:
            curve.append({"episode": e + 1, "mean_round_reward": float(np.mean(window)), "epsilon": eps})
            window = []
    Q[K, :, CONTINUE] = -np.inf  # never CONTINUE at s = K
    table = np.where(Q[..., CONTINUE] > Q[..., STOP], CONTINUE, STOP)
    return {"kind": "table", "table": table, "Q": np.where(np.isfinite(Q), Q, np.nan), "curve": curve}
