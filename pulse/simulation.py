"""Poisson-cluster event generation and trial protocols (App. 'Continuous-time
event generation and observation')."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import protocol


def simulate_events(network, horizon: float, rng, burn_in: float = protocol.BURN_IN,
                    extra_immigrants=()):
    """Cluster construction on [0, horizon] with immigrants on [-burn_in, horizon].

    Each event of source type j generates Poisson(sum_i K_ij) offspring; child
    types are drawn proportionally to K_ij and delays from the edge's Gamma
    density. Returns (times, types) of events with 0 <= t <= horizon.
    """
    stack = []
    n_types = network.n
    for i in range(n_types):
        n_i = rng.poisson(network.mu[i] * (horizon + burn_in))
        stack.extend((float(t), i) for t in rng.uniform(-burn_in, horizon, size=n_i))
    stack.extend((float(t), int(i)) for t, i in extra_immigrants)

    K, shapes, means = network.K, network.kernels.shapes, network.kernels.means
    times_out, types_out = [], []
    while stack:
        t, src = stack.pop()
        if 0.0 <= t <= horizon:
            times_out.append(t)
            types_out.append(src)
        if t > horizon:
            continue
        mean_children = K[:, src].sum()
        if mean_children <= 0.0:
            continue
        count = rng.poisson(mean_children)
        if count == 0:
            continue
        children = rng.choice(n_types, size=count, p=K[:, src] / mean_children)
        for child in children:
            k = shapes[child, src]
            delay = rng.gamma(k, means[child, src] / k)
            t_child = t + delay
            if t_child <= horizon:
                stack.append((t_child, int(child)))
    order = np.argsort(times_out) if times_out else []
    return (np.asarray(times_out, dtype=float)[order],
            np.asarray(types_out, dtype=int)[order])


@dataclass
class TrialPool:
    """Reusable per-target pool of paired pulse/sham blocks (App. protocols).

    The estimator receives only observed count/event data; ancestry, hidden
    events, and the excitation matrix stay held out.
    """

    target: int
    horizon: float
    pulse: list  # per-block (times, types) for the probe arm
    sham: list   # per-block (times, types) for the sham arm

    def __len__(self) -> int:
        return len(self.pulse)

    def contrast(self, k: int, p: int) -> np.ndarray:
        """Pulse-minus-sham observed count vector of block k (seed included)."""
        diff = np.zeros(p)
        for events, sign in ((self.pulse[k], 1.0), (self.sham[k], -1.0)):
            times, types = events
            obs = types < p
            diff += sign * np.bincount(types[obs], minlength=p)
        return diff


def generate_trial(network, target: int, horizon: float, rng,
                   contaminate_epsilon: float = 0.0):
    """One paired block: independent backgrounds, matched protocol (Assumption 1).

    The pulse arm adds the known unit immigrant at (0, target). With
    probability contaminate_epsilon it additionally recruits one uniformly
    selected hidden population (App. 'Direct recruitment of hidden
    populations'); this is an input-model violation used only for stress tests.
    """
    extra = [(0.0, target)]
    if contaminate_epsilon > 0.0 and network.q > 0 and rng.random() < contaminate_epsilon:
        hidden = network.p + int(rng.integers(network.q))
        extra.append((0.0, hidden))
    pulse = simulate_events(network, horizon, rng, extra_immigrants=extra)
    sham = simulate_events(network, horizon, rng)
    return pulse, sham


def generate_pool(network, target: int, n_blocks: int, horizon: float, seed,
                  contaminate_epsilon: float = 0.0) -> TrialPool:
    rng = np.random.default_rng(seed)
    pulse, sham = [], []
    for _ in range(n_blocks):
        p_ev, s_ev = generate_trial(network, target, horizon, rng, contaminate_epsilon)
        pulse.append(p_ev)
        sham.append(s_ev)
    return TrialPool(target=target, horizon=horizon, pulse=pulse, sham=sham)


def generate_all_pools(network, n_blocks: int, horizon: float, seed_base: int,
                       targets=None, contaminate_epsilon: float = 0.0):
    """One pool per eligible target; seeds derive deterministically from the
    network seed so every method inspects identical pools."""
    if targets is None:
        targets = range(network.p)
    return [
        generate_pool(network, j, n_blocks, horizon, seed_base * 1000 + j,
                      contaminate_epsilon)
        for j in targets
    ]


def simulate_passive(network, duration: float, seed) -> np.ndarray:
    """Passive observed counts on [0, duration] with the same burn-in."""
    rng = np.random.default_rng(seed)
    times, types = simulate_events(network, duration, rng)
    obs = types < network.p
    return np.bincount(types[obs], minlength=network.p)


def baseline_rate_estimate(network, duration: float, seed) -> np.ndarray:
    """r_hat from the separate passive recording (fixed across allocation)."""
    return simulate_passive(network, duration, seed) / duration
