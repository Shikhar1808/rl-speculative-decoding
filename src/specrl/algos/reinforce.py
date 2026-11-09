"""REINFORCE (policy gradient): softmax policy MLP 5->64->2 on [s/K, p, H, q, a].

Per-round episodes, gamma = 1, so every decision in a round gets that round's STOP reward as its return.
Baseline = mean return over the batch of rounds from one prompt; advantages are divided by their std
(reward scaling). Forced STOPs (s = K, answer end) are not the agent's choice, so they get no gradient.
"""

import numpy as np
import torch

from specrl.algos.common import STOP, feats, forced, mlp
from specrl.env import SpecEnv


def train(episodes, lam, c, K, n_episodes=3000, lr=1e-3, seed=0, log_every=50, **_):
    torch.manual_seed(seed)
    env = SpecEnv(episodes, lam, c, K, seed=seed)
    net = mlp([5, 64, 2])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    curve, window = [], []
    for e in range(n_episodes):
        obs, done = env.reset(), False
        logps, returns, round_logps = [], [], []
        while not done:
            if forced(env):
                a = STOP
            else:
                dist = torch.distributions.Categorical(logits=net(torch.from_numpy(feats(obs))))
                a_t = dist.sample()  # uses the global torch RNG, seeded above
                round_logps.append(dist.log_prob(a_t))
                a = int(a_t)
            obs, r, done, info = env.step(a)
            if info:  # round ended: its reward is the return of every decision in it
                logps += round_logps
                returns += [r] * len(round_logps)
                round_logps = []
                window.append(r)
        if logps:
            G = torch.tensor(returns, dtype=torch.float32)
            adv = G - G.mean()
            if len(adv) > 1 and adv.std() > 0:
                adv = adv / adv.std()
            loss = -(torch.stack(logps) * adv).sum() / len(logps)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (e + 1) % log_every == 0:
            curve.append({"episode": e + 1, "mean_round_reward": float(np.mean(window))})
            window = []
    return {"kind": "net", "arch": [5, 64, 2], "state_dict": net.state_dict(), "curve": curve}
