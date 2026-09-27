"""Decision-focused probe acquisition (Algorithm 1) and matched comparators."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import protocol
from .estimation import TargetStats, clipped_moments
from .intervention import single_target_value, value_variance

RULE_PULSE = "pulse"                    # full decision-focused rule
RULE_UNIFORM = "uniform"                # batch to a least-sampled target
RULE_RESPONSE = "response_focused"      # whole-response variance reduction
RULE_NOCOV = "pulse_nocov"              # PULSE without the covariance term
RULES = (RULE_PULSE, RULE_UNIFORM, RULE_RESPONSE, RULE_NOCOV)


@dataclass
class AcquisitionResult:
    chosen: int
    values: np.ndarray        # final plug-in values
    allocation: np.ndarray    # blocks drawn per target
    history: list = field(default_factory=list)  # per-step (target, n_used)


def _estimates(stats, r_hat, c, eta, include_covariance):
    """Plug-in (b, d, V, v) per target with physical clipping."""
    n_t = len(stats)
    V = np.zeros(n_t)
    v = np.zeros(n_t)
    for j, st in enumerate(stats):
        b_hat, d_hat = clipped_moments(st.xbar, st.ybar, c[st.target])
        V[j] = single_target_value(r_hat[st.target], b_hat, d_hat, eta)
        cov = st.covariance_xy()
        v[j] = value_variance(r_hat[st.target], b_hat, d_hat, eta, cov,
                              include_covariance)
    return V, v


def _select_target(rule, stats, V, v, batch):
    """Allocation index of the active rule, over targets with pool remaining."""
    n = np.array([st.n for st in stats], dtype=float)
    available = np.array([st.n < st.pool_size for st in stats])
    if rule == RULE_UNIFORM:
        order = np.argsort(np.where(available, n, np.inf), kind="stable")
        return int(order[0]) if available.any() else None
    if rule == RULE_RESPONSE:
        index = np.array([
            np.trace(st.covariance_vector()) / (st.n * (st.n + batch))
            for st in stats
        ])
    else:
        # leader and strongest challenger by upper uncertainty index
        leader = int(np.argmax(V))
        upper = V + 2.0 * np.sqrt(v / n)
        upper[leader] = -np.inf
        challenger = int(np.argmax(upper))
        index = v / (n * (n + batch))
        mask = np.zeros(len(stats), dtype=bool)
        mask[[leader, challenger]] = True
        # finite-pool fallback: any available target by the same index
        if not (available & mask).any():
            mask = available
        else:
            mask &= available
        index = np.where(mask, index, -np.inf)
        return int(np.argmax(index)) if available.any() else None
    index = np.where(available, index, -np.inf)
    return int(np.argmax(index)) if available.any() else None


def run_acquisition(rule, pools, r_hat, c, eta, budget,
                    init_blocks=protocol.INIT_BLOCKS,
                    batch=protocol.BATCH_BLOCKS) -> AcquisitionResult:
    """Algorithm 1 with the finite-pool convention (App. protocols).

    A method may inspect only the first n_j blocks it has selected at target
    j, with sum_j n_j = budget; pools are shared identically across rules.
    """
    assert rule in RULES
    p = len(r_hat)
    stats = [TargetStats(p=p, c=c, target=pool.target) for pool in pools]
    for st, pool in zip(stats, pools):
        st.pool_size = len(pool)
    include_covariance = rule != RULE_NOCOV

    used = 0
    for j, (st, pool) in enumerate(zip(stats, pools)):
        for k in range(min(init_blocks, len(pool))):
            st.update(pool.contrast(k, p))
        used += st.n

    history = []
    while used < budget:
        V, v = _estimates(stats, r_hat, c, eta, include_covariance)
        j = _select_target(rule, stats, V, v, batch)
        if j is None:
            break
        take = int(min(batch, budget - used, stats[j].pool_size - stats[j].n))
        if take <= 0:
            break
        pool = pools[j]
        for k in range(stats[j].n, stats[j].n + take):
            stats[j].update(pool.contrast(k, p))
        used += take
        history.append((j, used))

    V, _ = _estimates(stats, r_hat, c, eta, include_covariance)
    allocation = np.array([st.n for st in stats])
    return AcquisitionResult(chosen=int(np.argmax(V)), values=V,
                             allocation=allocation, history=history)
