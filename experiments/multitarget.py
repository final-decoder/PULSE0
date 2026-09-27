"""Joint suppression and overlapping benefits (App. 'Joint suppression and
overlapping benefits'): all 28 unordered target pairs in each of the 96 main
networks at eta = 0.4; pair values from Eq. (9) checked against complete-model
solves; regret of the best-isolated-target pair against the best joint pair."""
from __future__ import annotations

import itertools

import numpy as np

from pulse import intervention, metrics, networks, protocol, response

ETA_PAIR = 0.4


def main():
    max_disc = 0.0
    pair_regrets = []
    disagree = 0
    for q in protocol.HIDDEN_COUNTS:
        for u in range(protocol.NETWORKS_PER_SETTING):
            seed = protocol.main_seed(q, u)
            net = networks.generate_network(seed, p=8, q=q,
                                            radius=protocol.MAIN_RADIUS)
            p = net.p
            c = np.ones(p)
            R = response.integrated_response(net)
            r = response.observed_stationary_rates(net)
            burden = float(c @ r)

            singles = intervention.single_target_value(
                r, c @ R, np.diag(R), ETA_PAIR)
            pair_vals = {}
            for a, b in itertools.combinations(range(p), 2):
                pred = intervention.multi_target_reduction(
                    R, r, [a, b], [ETA_PAIR, ETA_PAIR])
                true = r - net.intervened_observed_rates([a, b],
                                                         [ETA_PAIR, ETA_PAIR])
                max_disc = max(max_disc, float(np.abs(pred - true).max()))
                pair_vals[(a, b)] = float(c @ pred) / burden

            best_pair = max(pair_vals, key=pair_vals.get)
            order = np.argsort(-singles)
            isolated = (int(order[0]), int(order[1]))
            isolated = (min(isolated), max(isolated))
            if isolated != best_pair:
                disagree += 1
            pair_regrets.append(
                (pair_vals[best_pair] - pair_vals[isolated]) / pair_vals[best_pair])

    mean, lo, hi = metrics.bootstrap_ci(pair_regrets)
    print(f"max |rate discrepancy|: {max_disc:.3e} events/s")
    print(f"best-isolated pair disagrees with best joint pair in {disagree}/96")
    print(f"relative pair-value regret: {100*mean:.2f} [{100*lo:.2f}, "
          f"{100*hi:.2f}] %")


if __name__ == "__main__":
    main()
