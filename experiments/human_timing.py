"""Computational benchmark with human-derived timing priors (Sec. 'Decisions
on empirical time scales', App. 'timing'): 74 participant-specific networks,
250-ms response window, budgets 1,280 and 2,560 blocks.

Requires per-participant N1 latencies (ms); see pulse/human_timing.py.
Usage: python human_timing.py latencies.json
"""
from __future__ import annotations

import sys

import numpy as np

from pulse import (acquisition, baselines, estimation, human_timing, metrics,
                   networks, simulation)

from common import ground_truth_values

ETA = 0.8
HORIZON = 0.25
POOL_PER_TARGET = 1280
BUDGETS = (1280, 2560)
PARTICIPANTS = range(1, 75)


def main():
    latencies = human_timing.load_n1_latencies(sys.argv[1])
    regrets = {name: {m: [] for m in BUDGETS}
               for name in ("pulse", "uniform")}
    ablation_regret = {"evoked_reach": [], "uncorrected": [], "rate_only": []}
    exact = {name: {m: [] for m in BUDGETS} for name in ("pulse", "uniform")}

    for u in PARTICIPANTS:
        net = networks.timing_network(u, latencies[u])
        p = net.p
        c = np.ones(p)
        _, true_vals = ground_truth_values(net, ETA, c)
        pools = simulation.generate_all_pools(
            net, POOL_PER_TARGET, HORIZON, seed_base=31000 + u)
        r_hat = simulation.baseline_rate_estimate(
            net, 2400.0, seed=31000 + u + 1)

        for name, rule in (("pulse", acquisition.RULE_PULSE),
                           ("uniform", acquisition.RULE_UNIFORM)):
            for m in BUDGETS:
                res = acquisition.run_acquisition(rule, pools, r_hat, c, ETA, m)
                regrets[name][m].append(
                    metrics.selection_regret(true_vals, res.chosen))
                exact[name][m].append(int(res.chosen == int(np.argmax(true_vals))))

        # scoring ablations share the uniform allocation at 2,560 blocks
        n_per = max(BUDGETS) // p
        R_hat = estimation.full_response_estimate(pools, [n_per] * p, p)
        b_hat = np.maximum(c, c @ R_hat)
        d_hat = np.maximum(1.0, np.diag(R_hat))
        picks = {
            "evoked_reach": baselines.select_evoked_reach(R_hat, c),
            "uncorrected": baselines.select_uncorrected(r_hat, R_hat, c),
            "rate_only": baselines.select_rate_only(r_hat),
        }
        for name, j in picks.items():
            ablation_regret[name].append(
                metrics.selection_regret(true_vals, j))

    for name in ("pulse", "uniform"):
        for m in BUDGETS:
            mean, lo, hi = metrics.bootstrap_ci(regrets[name][m])
            acc = 100 * np.mean(exact[name][m])
            print(f"{name:8s} @{m}: regret {100*mean:.2f} "
                  f"[{100*lo:.2f}, {100*hi:.2f}] %, exact {acc:.1f}%")
    for name, vals in ablation_regret.items():
        mean, lo, hi = metrics.bootstrap_ci(vals)
        print(f"{name:13s} @2560: regret {100*mean:.2f} "
              f"[{100*lo:.2f}, {100*hi:.2f}] %")


if __name__ == "__main__":
    main()
