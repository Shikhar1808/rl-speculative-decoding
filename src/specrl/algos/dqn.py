"""Deep Q-Network: MLP 5->64->64->2 on [s/K, p, H, q, a], replay buffer, target network, Huber loss, Adam.

Same per-round episodes as Q-learning (STOP ends the round, gamma = 1 inside a round). In bootstrap targets,
CONTINUE at s' = K is masked out: it is never trained (STOP is forced there), so its output is meaningless.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from specrl.algos.common import CONTINUE, STOP, feats, forced, mlp
from specrl.sim.env import SpecEnv


def train(
    episodes,
    lam,
    c,
    K,
    n_episodes=3000,
    lr=1e-3,
    batch=128,
    buffer=50_000,
    warmup=1000,
    sync_every=500,
    train_every=4,
    eps_start=1.0,
    eps_end=0.05,
    seed=0,
    log_every=50,
    **_,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    env = SpecEnv(episodes, lam, c, K, seed=seed)
    net, tgt = mlp([5, 64, 64, 2]), mlp([5, 64, 64, 2])
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = deque(maxlen=buffer)
    steps, curve, window, losses = 0, [], [], []
    for e in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * e / (0.8 * n_episodes))
        obs, done = env.reset(), False
        while not done:
            f = feats(obs)
            if forced(env):
                a = STOP
            elif rng.random() < eps:
                a = rng.randrange(2)
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(f)).argmax())
            obs2, r, done, info = env.step(a)
            round_end = bool(info)
            f2 = feats(obs2) if obs2 is not None else np.zeros(5, np.float32)
            buf.append((f, a, r, f2, float(round_end)))
            if round_end:
                window.append(r)
            obs = obs2
            steps += 1

            if len(buf) >= warmup and steps % train_every == 0:
                o, act, rew, o2, d = map(np.array, zip(*rng.sample(buf, batch)))
                o, o2 = torch.from_numpy(o), torch.from_numpy(o2)
                act = torch.from_numpy(act).long()
                rew, d = torch.tensor(rew, dtype=torch.float32), torch.tensor(d, dtype=torch.float32)
                q_sa = net(o).gather(1, act[:, None]).squeeze(1)
                with torch.no_grad():
                    q2 = tgt(o2)
                    q2[o2[:, 0] > 0.999, CONTINUE] = -float("inf")  # s' = K: CONTINUE impossible (never trained)
                    target = rew + (1 - d) * q2.max(1).values
                loss = F.smooth_l1_loss(q_sa, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if steps % sync_every == 0:
                tgt.load_state_dict(net.state_dict())
        if (e + 1) % log_every == 0:
            curve.append(
                {
                    "episode": e + 1,
                    "mean_round_reward": float(np.mean(window)),
                    "epsilon": eps,
                    "loss": float(np.mean(losses)) if losses else float("nan"),
                }
            )
            window, losses = [], []
    return {"kind": "net", "arch": [5, 64, 64, 2], "state_dict": net.state_dict(), "curve": curve}
