"""Machine-precision checks of Theorem 2 against direct complete-model solves
(App. 'Complete experimental specification', identity checks): 50 random
networks with q = 0..9 and rho(K) in [0.55, 0.942]; single-target identity at
four suppression levels for all eight targets, plus two- and three-target
identities. Networks are indexed by seeds 70000+."""
from __future__ import annotations

import numpy as np

from pulse import intervention, networks, response

N_NETWORKS = 50
ETAS = (0.25, 0.5, 0.8, 1.0)


def main():
    rng = np.random.default_rng(0)
    max_single = max_multi = 0.0
    for rep in range(N_NETWORKS):
        q = rep % 10
        radius = 0.55 + (0.942 - 0.55) * rng.random()
        net = networks.generate_network(seed=70000 + rep, p=8, q=q, radius=radius)
        R = response.integrated_response(net)
        r = response.observed_stationary_rates(net)

        for j in range(net.p):
            for eta in ETAS:
                pred = intervention.single_target_reduction(R, r, j, eta)
                true = r - net.intervened_observed_rates([j], [eta])
                max_single = max(max_single, float(np.abs(pred - true).max()))

        for targets in ((0, 3), (1, 4, 6)):
            etas = [0.4] * len(targets)
            pred = intervention.multi_target_reduction(R, r, targets, etas)
            true = r - net.intervened_observed_rates(targets, etas)
            max_multi = max(max_multi, float(np.abs(pred - true).max()))

    print(f"single-target max abs error: {max_single:.3e} events/s")
    print(f"multi-target  max abs error: {max_multi:.3e} events/s")


if __name__ == "__main__":
    main()
