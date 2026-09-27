"""Effective-operator recovery (App. 'Nonnegative operator estimation', Fig.
kernel_recovery): 12 networks (seeds 22000-22011) with 8
observed and 8 hidden populations, 2,560 blocks per target, relative Frobenius
error averaged over four Laplace frequencies per network, then network
bootstrap."""
from __future__ import annotations

import numpy as np

from pulse import estimation, metrics, networks, operator_fit, protocol, response, simulation

BLOCKS_PER_TARGET = 2560
FREQUENCIES = (0.5, 1.0, 2.0, 5.0)  # s^-1; the four averaged Laplace frequencies


def main():
    per_network = []
    for seed in protocol.OPERATOR_SEEDS:
        net = networks.generate_network(seed, p=8, q=8,
                                        radius=protocol.MAIN_RADIUS)
        pools = simulation.generate_all_pools(
            net, BLOCKS_PER_TARGET, protocol.TRIAL_WINDOW, seed_base=seed)
        errs = []
        for s in FREQUENCIES:
            R_hat = estimation.full_response_estimate(
                pools, [BLOCKS_PER_TARGET] * 8, 8, s=s)
            Phi_hat = operator_fit.fit_effective_operator(
                R_hat, BLOCKS_PER_TARGET * 8)
            Phi_true = response.effective_kernel_laplace(net, s)
            errs.append(metrics.relative_frobenius(Phi_hat, Phi_true))
        per_network.append(float(np.mean(errs)))
    mean, lo, hi = metrics.bootstrap_ci(per_network)
    print(f"operator relative Frobenius error: {mean:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
