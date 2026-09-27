"""Response-window experiments (Sec. 'Response windows and empirical timing',
App. 'Feedback strength and response-window sensitivity' and 'Response horizon
near criticality')."""
from __future__ import annotations

import numpy as np

from pulse import estimation, intervention, metrics, networks, protocol, response, simulation

try:  # package import (python -m experiments.response_windows)
    from .common import ground_truth_values
except ImportError:  # direct script execution (python response_windows.py)
    from common import ground_truth_values

ETA = protocol.ETA
RADII = (0.45, 0.60, 0.72, 0.82, 0.90)
HORIZON_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
REGIME_BLOCKS = 320
CRITICAL_HORIZONS = (0.5, 2.0, 6.0, 12.0)
CRITICAL_BLOCKS = 1600


def _nmae_at_horizon(net, horizon, n_blocks, seed_base):
    """Value NMAE with known baseline rates at one response horizon."""
    p = net.p
    c = np.ones(p)
    _, true_vals = ground_truth_values(net, ETA, c)
    r_pop = response.observed_stationary_rates(net)
    pools = simulation.generate_all_pools(net, n_blocks, horizon,
                                          seed_base=seed_base)
    est_vals = np.zeros(p)
    for j, pool in enumerate(pools):
        col = estimation.response_column_estimate(pool, n_blocks, p)
        b_hat, d_hat = estimation.clipped_moments(float(c @ col), col[j], c[j])
        est_vals[j] = intervention.single_target_value(r_pop[j], b_hat, d_hat, ETA)
    return metrics.value_nmae(true_vals, est_vals)


def feedback_strength_grid():
    """8 replicates x 5 spectral radii x 6 horizons; connectivity proportions,
    immigrant rates, and Gamma timing retained within each replicate (same
    seed, rescaled). Conditions within a replicate are paired."""
    nmae = {(rho, H): [] for rho in RADII for H in HORIZON_GRID}
    for seed in protocol.REGIME_SEEDS:
        for rho in RADII:
            net = networks.generate_network(seed, p=8, q=8, radius=rho)
            for H in HORIZON_GRID:
                nmae[(rho, H)].append(_nmae_at_horizon(net, H, REGIME_BLOCKS, seed))
    for rho in RADII:
        row = " ".join(f"H={H}: {100*np.mean(nmae[(rho, H)]):.1f}%"
                       for H in HORIZON_GRID)
        print(f"rho={rho:.2f} | {row}")


def near_critical_horizons():
    """12 networks at rho = 0.90, 1,600 blocks per target, horizons
    0.5/2/6/12 s; reports relative response error and value NMAE."""
    for H in CRITICAL_HORIZONS:
        errs, nmaes = [], []
        for seed in protocol.WINDOW_SEEDS:
            net = networks.generate_network(seed, p=8, q=8,
                                            radius=protocol.HIGH_RADIUS)
            nmaes.append(_nmae_at_horizon(net, H, CRITICAL_BLOCKS, seed))
            pools = simulation.generate_all_pools(net, CRITICAL_BLOCKS, H,
                                                  seed_base=seed)
            R_hat = estimation.full_response_estimate(
                pools, [CRITICAL_BLOCKS] * 8, 8)
            errs.append(metrics.relative_frobenius(
                R_hat, response.integrated_response(net)))
        print(f"H={H:4.1f}s: response error {100*np.mean(errs):.1f}%, "
              f"value NMAE {100*np.mean(nmaes):.1f}%")


def main():
    feedback_strength_grid()
    near_critical_horizons()


if __name__ == "__main__":
    main()
