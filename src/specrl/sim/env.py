"""Step 3: the simulator. Replays logged episodes so any stopping policy can be trained/evaluated on the CPU.

One episode = one prompt's answer, made of draft->verify rounds. After each drafted token the agent picks
    0 = CONTINUE (draft one more)   or   1 = STOP (send to the target)
Observation (5 floats): [s/K, p, H, q, a]
    s  tokens drafted this round (1..K)      p  draft top prob of the latest token
    H  draft entropy at that token           q  product of p over this round
    a  moving average of the acceptance rate over recent rounds
Reward: 0 on CONTINUE; on STOP r = gained - lam * (1 + s*c), gained = min(s, n) + 1 (docs/02_mdp.md).
"""

import random

import numpy as np

from specrl.config import K as K_DEFAULT

CONTINUE, STOP = 0, 1


class SpecEnv:
    def __init__(self, episodes, lam, c, K=K_DEFAULT, seed=None):
        self.episodes, self.lam, self.c, self.K = episodes, lam, c, K
        self.rng = random.Random(seed)

    def reset(self, ep=None):
        """Start an episode: a random logged answer, or `ep` if given (for deterministic evaluation)."""
        self.ep = ep if ep is not None else self.rng.choice(self.episodes)
        self.end = len(self.ep["p"])
        self.t, self.s, self.q, self.acc_rate = 0, 1, 1.0, 1.0
        return self._obs()

    def _obs(self):
        i = self.t + self.s - 1  # index of the s-th draft token of this round
        p, H = float(self.ep["p"][i]), float(self.ep["H"][i])
        self.q_now = self.q * p
        return np.array([self.s / self.K, p, H, self.q_now, self.acc_rate], dtype=np.float32)

    def step(self, a):
        """Returns (obs, reward, done, info). obs is None when done. info has the round stats on STOP."""
        if a == CONTINUE and self.s < self.K and self.t + self.s < self.end:
            self.q = self.q_now
            self.s += 1
            return self._obs(), 0.0, False, {}
        m = self.ep["match"][self.t : self.t + self.s]
        n = len(m) if m.all() else int(np.argmin(m))  # consecutive accepted drafts
        gained = min(n + 1, self.end - self.t)  # +1 = target's correction/bonus token; never past the answer end
        s = self.s
        self.acc_rate = 0.8 * self.acc_rate + 0.2 * n / s
        r = gained - self.lam * (1 + s * self.c)
        self.t += gained
        self.s, self.q = 1, 1.0
        done = self.t >= self.end
        return (None if done else self._obs()), r, done, {"n": n, "s": s, "gained": gained}


def evaluate(policy, episodes, c, K=K_DEFAULT):
    """Run `policy(obs) -> action` once over every episode. Time is in units of one target call.

    Returns tokens, target calls, draft steps, mean accepted length and sim speedup vs the target alone
    (target alone = 1 token per target call, so speedup = tokens / (calls + c * draft_steps)).
    """
    env = SpecEnv(episodes, lam=0.0, c=c, K=K)
    tokens = calls = drafts = 0
    for ep in episodes:
        obs, done = env.reset(ep), False
        while not done:
            obs, _, done, info = env.step(policy(obs))
            if info:
                tokens += info["gained"]
                calls += 1
                drafts += info["s"]
    time = calls + c * drafts
    return {
        "tokens": tokens,
        "target_calls": calls,
        "draft_steps": drafts,
        "mean_accepted_len": tokens / calls,
        "throughput": tokens / time,  # tokens per target-call time; = sim speedup
        "sim_speedup": tokens / time,
    }
