"""Main acquisition benchmark: 96 networks, four matched acquisition rules,
budgets 320..5120 (Sec. 'Better target decisions with fewer measurements',
App. 'Matched acquisition objectives')."""
from __future__ import annotations

import numpy as np

from pulse import (acquisition, estimation, metrics, networks, protocol,
                   response, simulation)

try:  # package import (python -m experiments.acquisition_benchmark)
    from .common import ground_truth_values
except ImportError:  # direct script execution (python acquisition_benchmark.py)
    from common import ground_truth_values

ETA = protocol.ETA


def run_network(net, net_seed):
    """Per-network regret curves for all four rules plus response errors."""
    p = net.p
    c = np.ones(p)
    _, true_vals = ground_truth_values(net, ETA, c)
    pools = simulation.generate_all_pools(
        net, protocol.POOL_PER_TARGET, protocol.TRIAL_WINDOW, seed_base=net_seed)
    r_hat = simulation.baseline_rate_estimate(
        net, protocol.PASSIVE_DURATION, seed=net_seed + 1)

    regrets = {rule: [] for rule in acquisition.RULES}
    rel_err = {}
    for rule in acquisition.RULES:
        for budget in protocol.BUDGETS:
            res = acquisition.run_acquisition(rule, pools, r_hat, c, ETA, budget)
            regrets[rule].append(metrics.selection_regret(true_vals, res.chosen))
        final = acquisition.run_acquisition(
            rule, pools, r_hat, c, ETA, max(protocol.BUDGETS))
        n_per = [int(n) for n in final.allocation]
        R_hat = estimation.full_response_estimate(pools, n_per, p)
        rel_err[rule] = metrics.relative_frobenius(
            R_hat, response.integrated_response(net))
    return regrets, rel_err


def main():
    all_regrets = {rule: [] for rule in acquisition.RULES}
    all_err = {rule: [] for rule in acquisition.RULES}
    for q in protocol.HIDDEN_COUNTS:
        for u in range(protocol.NETWORKS_PER_SETTING):
            seed = protocol.main_seed(q, u)
            net = networks.generate_network(seed, p=8, q=q,
                                            radius=protocol.MAIN_RADIUS)
            regrets, rel_err = run_network(net, seed)
            for rule in acquisition.RULES:
                all_regrets[rule].append(regrets[rule])
                all_err[rule].append(rel_err[rule])

    for bi, budget in enumerate(protocol.BUDGETS):
        row = []
        for rule in acquisition.RULES:
            vals = [r[bi] for r in all_regrets[rule]]
            mean, lo, hi = metrics.bootstrap_ci(vals)
            row.append(f"{rule}: {100*mean:.2f} [{100*lo:.2f}, {100*hi:.2f}]")
        print(f"budget {budget:5d} | " + " | ".join(row))

    # paired PULSE-minus-uniform regret difference at 2,560 blocks
    bi = protocol.BUDGETS.index(2560)
    diffs = [all_regrets[acquisition.RULE_PULSE][i][bi]
             - all_regrets[acquisition.RULE_UNIFORM][i][bi]
             for i in range(len(all_regrets[acquisition.RULE_PULSE]))]
    mean, lo, hi = metrics.bootstrap_ci(diffs)
    print(f"paired regret difference at 2560: {100*mean:.3f} "
          f"[{100*lo:.3f}, {100*hi:.3f}] pp")
    for rule in acquisition.RULES:
        print(f"relative response error @5120, {rule}: "
              f"{np.mean(all_err[rule]):.3f}")


if __name__ == "__main__":
    main()
