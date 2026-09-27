"""Target-scoring comparison under identical uniform allocation (Table 1):
evoked reach, uncorrected transport, PULSE value, known-direct-edges reference,
rate-only, and passive observed-only Hawkes fits (one and three bases)."""
from __future__ import annotations

import numpy as np

from pulse import (baselines, estimation, intervention, metrics, networks,
                   protocol, simulation)

try:  # package import (python -m experiments.scoring_comparison)
    from .common import ground_truth_values
except ImportError:  # direct script execution (python scoring_comparison.py)
    from common import ground_truth_values

ETA = protocol.ETA
BUDGET = 5120


def run_network(net, net_seed):
    p = net.p
    c = np.ones(p)
    r_true, true_vals = ground_truth_values(net, ETA, c)
    pools = simulation.generate_all_pools(
        net, protocol.POOL_PER_TARGET, protocol.TRIAL_WINDOW, seed_base=net_seed)
    r_hat = simulation.baseline_rate_estimate(
        net, protocol.PASSIVE_DURATION, seed=net_seed + 1)

    n_per = BUDGET // p  # uniform allocation
    R_hat = estimation.full_response_estimate(pools, [n_per] * p, p)
    b_hat = np.maximum(c, c @ R_hat)
    d_hat = np.maximum(1.0, np.diag(R_hat))

    selections = {
        "evoked_reach": baselines.select_evoked_reach(R_hat, c),
        "uncorrected": baselines.select_uncorrected(r_hat, R_hat, c),
        "pulse_value": int(np.argmax(
            intervention.single_target_value(r_hat, b_hat, d_hat, ETA))),
        "known_direct_edges": int(np.argmax(
            baselines.known_direct_edges_values(net, r_hat, ETA, c))),
        "rate_only": baselines.select_rate_only(r_hat),
    }

    # passive observed-only Hawkes fits on 50-ms bins of the passive record
    times, types = simulation.simulate_events(
        net, protocol.PASSIVE_DURATION, np.random.default_rng(net_seed + 2))
    counts = baselines.bin_passive_counts(times, types, p,
                                          protocol.PASSIVE_DURATION)
    for name, decays in (("passive_1basis", (0.15,)),
                         ("passive_3basis", (0.05, 0.15, 0.5))):
        K_fit = baselines.fit_passive_hawkes(counts, decays)
        vals = baselines.passive_fit_values(K_fit, r_hat, ETA, c)
        selections[name] = int(np.argmax(vals))

    return {k: metrics.selection_regret(true_vals, j)
            for k, j in selections.items()}


def main():
    per_method = {}
    for q in protocol.HIDDEN_COUNTS:
        for u in range(protocol.NETWORKS_PER_SETTING):
            seed = protocol.main_seed(q, u)
            net = networks.generate_network(seed, p=8, q=q,
                                            radius=protocol.MAIN_RADIUS)
            for name, reg in run_network(net, seed).items():
                per_method.setdefault(name, []).append(reg)
    for name, vals in per_method.items():
        mean, lo, hi = metrics.bootstrap_ci(vals)
        print(f"{name:20s} regret {100*mean:.2f} [{100*lo:.2f}, {100*hi:.2f}] %")


if __name__ == "__main__":
    main()
