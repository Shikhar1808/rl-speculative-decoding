"""Shared pieces for the four algorithms: state encodings, forced-STOP check, saved-policy format."""

import numpy as np
import torch
from torch import nn

from specrl.config import K
from specrl.env import CONTINUE, STOP

Q_BINS = 10


def q_bin(q):
    return min(int(q * Q_BINS), Q_BINS - 1)


def tab_state(obs):
    """Tabular state (s, q-bin): s in 1..K, q in 10 equal bins. 80 states."""
    return int(round(obs[0] * K)), q_bin(obs[3])


def feats(obs):
    """Continuous features for the networks. Entropy is rescaled to ~0-2 (others are already 0-1)."""
    s, p, H, q, a = obs
    return np.array([s, p, min(H / 5.0, 2.0), q, a], dtype=np.float32)


def forced(env):
    """True when the env will STOP whatever the agent picks (s == K or the answer ends)."""
    return env.s >= env.K or env.t + env.s >= env.end


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class TablePolicy:
    """Greedy policy from an action table indexed [s, q-bin]."""

    def __init__(self, table):
        self.table = np.asarray(table)

    def __call__(self, obs):
        s, b = tab_state(obs)
        return int(self.table[s, b])


class NetPolicy:
    """Greedy policy from a network that outputs 2 scores (Q-values or logits) per state."""

    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def __call__(self, obs):
        return int(self.net(torch.from_numpy(feats(obs))).argmax())


__all__ = ["CONTINUE", "STOP", "Q_BINS", "q_bin", "tab_state", "feats", "forced", "mlp", "TablePolicy", "NetPolicy"]
