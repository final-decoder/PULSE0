"""Independent post-action validation of intervention means (Sec. 'Checking
sustained benefits against independent simulations', App. 'Independent
validation of intervention means').

Four networks (seeds 26000-26003), targets 0/2/4/6, suppression levels
0.25/0.5/0.8/1.0: predictions use the population (R, r) before intervention;
measured reductions use four independent 12,000-s streams of the intervened
complete model."""
from __future__ import annotations

import numpy as np

from pulse import intervention, networks, protocol, response, simulation

TARGETS = (0, 2, 4, 6)
ETAS = (0.25, 0.5, 0.8, 1.0)
N_STREAMS = 4
STREAM_SECONDS = 12000.0


def main():
    discrepancies = []
    for seed in protocol.VALIDATION_SEEDS:
        net = networks.generate_network(seed=seed, p=8, q=8,
                                        radius=protocol.MAIN_RADIUS)
        R = response.integrated_response(net)
        r = response.observed_stationary_rates(net)
        for j in TARGETS:
            for eta in ETAS:
                pred = float(np.sum(intervention.single_target_reduction(R, r, j, eta)))
                post = net.intervened([j], [eta])
                rates = []
                for s in range(N_STREAMS):
                    times, types = simulation.simulate_events(
                        post, STREAM_SECONDS,
                        np.random.default_rng(seed * 100 + s))
                    obs = types < net.p
                    rates.append(obs.sum() / STREAM_SECONDS)
                measured = float(r.sum() - np.mean(rates))
                discrepancies.append(abs(pred - measured))
    print(f"{len(discrepancies)} conditions, mean abs discrepancy "
          f"{np.mean(discrepancies):.4f} events/s")


if __name__ == "__main__":
    main()
