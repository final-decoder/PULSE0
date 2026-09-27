"""Shared helpers for experiment scripts."""
from __future__ import annotations

import numpy as np


def ground_truth_values(network, eta, c):
    """Observed baseline rates and per-target values from direct complete-model
    stationary solves (never from estimated responses)."""
    p = network.p
    c = np.asarray(c, dtype=float)
    r = network.stationary_rates()[:p]
    values = np.array([
        c @ (r - network.intervened_observed_rates([j], [eta]))
        for j in range(p)
    ])
    return r, values
