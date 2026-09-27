"""Hidden coverage and probe-calibration experiments (Sec. 'Hidden coverage
and probe calibration', App. 'Spatial coverage and probe calibration')."""
from __future__ import annotations

import numpy as np

from pulse import (acquisition, estimation, intervention, metrics, networks,
                   protocol, response, simulation)

from common import ground_truth_values

ETA = protocol.ETA
EPSILONS = (0.0, 0.1, 0.3, 0.6)
BLOCKS_PER_TARGET = 800


def coverage_by_hidden_count():
    """PULSE regret at 5,120 blocks, stratified by hidden-population count
    (24 networks per setting)."""
    c = np.ones(8)
    per_q = {q: [] for q in protocol.HIDDEN_COUNTS}
    for q in protocol.HIDDEN_COUNTS:
        for u in range(protocol.NETWORKS_PER_SETTING):
            seed = protocol.main_seed(q, u)
            net = networks.generate_network(seed, p=8, q=q,
                                            radius=protocol.MAIN_RADIUS)
            _, true_vals = ground_truth_values(net, ETA, c)
            pools = simulation.generate_all_pools(
                net, protocol.POOL_PER_TARGET, protocol.TRIAL_WINDOW,
                seed_base=seed)
            r_hat = simulation.baseline_rate_estimate(
                net, protocol.PASSIVE_DURATION, seed=seed + 1)
            res = acquisition.run_acquisition(
                acquisition.RULE_PULSE, pools, r_hat, c, ETA,
                max(protocol.BUDGETS))
            per_q[q].append(metrics.selection_regret(true_vals, res.chosen))
    for q, vals in per_q.items():
        mean, lo, hi = metrics.bootstrap_ci(vals)
        print(f"q={q:2d}: regret {100*mean:.2f} [{100*lo:.2f}, {100*hi:.2f}] %")


def hidden_recruitment_stress():
    """Direct probe recruitment of hidden populations with probability eps;
    16 networks, 800 blocks per target, known baseline rates. Reports value
    NMAE from estimated responses and from contaminated population responses."""
    nmae_est = {e: [] for e in EPSILONS}
    nmae_pop = {e: [] for e in EPSILONS}
    for seed in protocol.RECRUITMENT_SEEDS:
        net = networks.generate_network(seed, p=8, q=8,
                                        radius=protocol.MAIN_RADIUS)
        c = np.ones(8)
        _, true_vals = ground_truth_values(net, ETA, c)
        r_pop = response.observed_stationary_rates(net)
        for eps in EPSILONS:
            pools = simulation.generate_all_pools(
                net, BLOCKS_PER_TARGET, protocol.TRIAL_WINDOW, seed_base=seed,
                contaminate_epsilon=eps)
            est_vals = np.zeros(8)
            for j, pool in enumerate(pools):
                col = estimation.response_column_estimate(
                    pool, BLOCKS_PER_TARGET, 8)
                b_hat, d_hat = estimation.clipped_moments(
                    float(c @ col), col[j], c[j])
                est_vals[j] = intervention.single_target_value(
                    r_pop[j], b_hat, d_hat, ETA)
            nmae_est[eps].append(metrics.value_nmae(true_vals, est_vals))

            Rc = networks.contaminated_response(net, eps)
            pop_vals = intervention.single_target_value(
                r_pop, c @ Rc, np.diag(Rc), ETA)
            nmae_pop[eps].append(metrics.value_nmae(true_vals, pop_vals))
    for eps in EPSILONS:
        m_e, lo_e, hi_e = metrics.bootstrap_ci(nmae_est[eps])
        m_p, lo_p, hi_p = metrics.bootstrap_ci(nmae_pop[eps])
        print(f"eps={eps:.1f}: estimated {100*m_e:.1f} [{100*lo_e:.1f}, "
              f"{100*hi_e:.1f}] %, population {100*m_p:.1f} [{100*lo_p:.1f}, "
              f"{100*hi_p:.1f}] %")


def main():
    coverage_by_hidden_count()
    hidden_recruitment_stress()


if __name__ == "__main__":
    main()
