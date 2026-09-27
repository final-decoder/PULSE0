"""Evaluation metrics and resampling intervals (App. 'Ground truth, metrics,
and replication units')."""
from __future__ import annotations

import numpy as np

from . import protocol


def selection_regret(true_values, chosen: int) -> float:
    """L = (max_j V_j - V_chosen) / max_j V_j."""
    true_values = np.asarray(true_values, dtype=float)
    best = true_values.max()
    return float((best - true_values[chosen]) / best)


def value_nmae(true_values, est_values) -> float:
    """Mean absolute target-value error divided by the best target's benefit."""
    true_values = np.asarray(true_values, dtype=float)
    est_values = np.asarray(est_values, dtype=float)
    return float(np.mean(np.abs(true_values - est_values)) / true_values.max())


def relative_frobenius(est, ref) -> float:
    """||est - ref||_F / ||ref||_F."""
    return float(np.linalg.norm(est - ref) / np.linalg.norm(ref))


def bootstrap_ci(samples, stat=np.mean, resamples=protocol.BOOTSTRAP_RESAMPLES,
                 seed=0, alpha=0.05):
    """Percentile interval over resampled units (whole networks / families /
    participants, never individual intervention conditions)."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)
    stats = np.empty(resamples)
    for b in range(resamples):
        stats[b] = stat(rng.choice(samples, size=samples.size, replace=True))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(samples)), float(lo), float(hi)
