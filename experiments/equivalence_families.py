"""Response-equivalent families: identical benefits across hidden realizations,
and finite-measurement convergence (Sec. 'From local probes to suppression
benefits', App. 'Constructing and evaluating response-equivalent networks')."""
from __future__ import annotations

import numpy as np

from pulse import estimation, intervention, metrics, networks, protocol, response, simulation

try:  # package import (python -m experiments.equivalence_families)
    from .common import ground_truth_values
except ImportError:  # direct script execution (python equivalence_families.py)
    from common import ground_truth_values

ETA = 0.8
C = np.ones(4)
BLOCK_GRID = (40, 160, 640)
DOSE_GRID = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))


def check_population_invariance():
    """All variants in a family share (R, r) and hence every V_j(eta)."""
    max_R = max_r = max_V = 0.0
    for f in range(24):
        variants = networks.equivalence_family(f)
        ref = variants[1]
        R0 = response.integrated_response(ref)
        r0 = response.observed_stationary_rates(ref)
        for k in (2, 4):
            R1 = response.integrated_response(variants[k])
            r1 = response.observed_stationary_rates(variants[k])
            max_R = max(max_R, float(np.abs(R1 - R0).max()))
            max_r = max(max_r, float(np.abs(r1 - r0).max()))
            for j in range(4):
                for eta in DOSE_GRID:
                    V0 = intervention.single_target_value(
                        r0[j], C @ R0[:, j], R0[j, j], eta)
                    V1 = intervention.single_target_value(
                        r1[j], C @ R1[:, j], R1[j, j], eta)
                    max_V = max(max_V, abs(float(V0 - V1)))
    print(f"max |dR| = {max_R:.3e}, max |dr| = {max_r:.3e}, max |dV| = {max_V:.3e}")


def finite_measurement_convergence():
    """Value NMAE at 40/160/640 blocks per target with the common baseline
    fixed at its population value; variants averaged within family before
    bootstrap. Trial seeds 61000 + 10 f + v, v in {0, 1, 2}."""
    nmae = {n: [] for n in BLOCK_GRID}
    for f in range(24):
        per_variant = {n: [] for n in BLOCK_GRID}
        for v, k in enumerate((1, 2, 4)):
            net = networks.equivalence_family(f)[k]
            _, true_vals = ground_truth_values(net, ETA, C)
            r_pop = response.observed_stationary_rates(net)
            pools = simulation.generate_all_pools(
                net, max(BLOCK_GRID), protocol.TRIAL_WINDOW,
                seed_base=protocol.EQUIVALENCE_TRIAL_BASE + 10 * f + v)
            for n in BLOCK_GRID:
                est_vals = np.zeros(4)
                for j, pool in enumerate(pools):
                    col = estimation.response_column_estimate(pool, n, 4)
                    b_hat, d_hat = estimation.clipped_moments(
                        float(C @ col), col[j], C[j])
                    est_vals[j] = intervention.single_target_value(
                        r_pop[j], b_hat, d_hat, ETA)
                per_variant[n].append(metrics.value_nmae(true_vals, est_vals))
        for n in BLOCK_GRID:
            nmae[n].append(float(np.mean(per_variant[n])))
    for n in BLOCK_GRID:
        mean, lo, hi = metrics.bootstrap_ci(nmae[n])
        print(f"{n:4d} blocks/target: value NMAE {mean:.4f} [{lo:.4f}, {hi:.4f}]")


def main():
    check_population_invariance()
    finite_measurement_convergence()


if __name__ == "__main__":
    main()
