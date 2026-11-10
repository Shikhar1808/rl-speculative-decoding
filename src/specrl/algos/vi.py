"""Value Iteration (dynamic programming). Model-based: the MDP is estimated by counting in the logs.

Tabular state (s, q-bin). For every logged position t treated as a round start and every s = 1..K we count
    R[s,b]      sum of STOP rewards  (min(n,s)+1 - lam(1+s c))
    T[s,b,b']   transitions to (s+1, b') on CONTINUE
then iterate the Bellman optimality update until it stops changing:
    V(s,b) = max( R̄(s,b),  Σ_b' P(b'|s,b) V(s+1,b') )
"""

import numpy as np

from specrl.algos.common import CONTINUE, Q_BINS, STOP
from specrl.algos.common import q_bin as _bin


def estimate_model(episodes, lam, c, K):
    R = np.zeros((K + 2, Q_BINS))
    N = np.zeros((K + 2, Q_BINS))
    T = np.zeros((K + 2, Q_BINS, Q_BINS))
    for ep in episodes:
        p, m = ep["p"], ep["match"]
        L = len(p)
        run = np.zeros(L + 1, dtype=int)  # run[t] = consecutive matches starting at t
        for t in range(L - 1, -1, -1):
            run[t] = run[t + 1] + 1 if m[t] else 0
        for t in range(L):
            q = 1.0
            for s in range(1, K + 1):
                if t + s - 1 >= L:
                    break
                q *= p[t + s - 1]
                b = _bin(q)
                gained = min(min(run[t], s) + 1, L - t)
                R[s, b] += gained - lam * (1 + s * c)
                N[s, b] += 1
                if s < K and t + s < L:
                    T[s, b, _bin(q * p[t + s])] += 1
    return R, N, T


def value_iteration(R, N, T, K, tol=1e-9, max_iter=1000):
    Rbar = np.where(N > 0, R / np.maximum(N, 1), 0.0)
    V = np.zeros((K + 2, Q_BINS))
    Q = np.zeros((K + 2, Q_BINS, 2))
    history = []
    for it in range(max_iter):
        Q[..., STOP] = Rbar
        Tn = T.sum(-1, keepdims=True)
        P = np.where(Tn > 0, T / np.maximum(Tn, 1), 0.0)
        cont = np.einsum("sbc,sc->sb", P[:-1], V[1:])  # Σ_b' P(b'|s,b) V(s+1,b')
        Q[:-1, :, CONTINUE] = np.where(Tn[:-1, :, 0] > 0, cont, -np.inf)
        Q[K:, :, CONTINUE] = -np.inf  # STOP forced at s = K
        V_new = np.where(N > 0, Q.max(-1), 0.0)
        delta = np.abs(V_new - V).max()
        V = V_new
        history.append(delta)
        if delta < tol:
            break
    return Q, V, history


def train(episodes, lam, c, K, **_):
    R, N, T = estimate_model(episodes, lam, c, K)
    Q, V, history = value_iteration(R, N, T, K)
    table = np.where(np.isfinite(Q[..., CONTINUE]) & (Q[..., CONTINUE] > Q[..., STOP]), CONTINUE, STOP)
    return {
        "kind": "table",
        "table": table,
        "Q": np.where(np.isfinite(Q), Q, np.nan),
        "visits": N,
        "curve": [{"iteration": i + 1, "bellman_residual": d} for i, d in enumerate(history)],
    }
