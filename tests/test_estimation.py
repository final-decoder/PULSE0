"""Tests for the pulse/sham estimator and decision moments."""
import numpy as np

from pulse import estimation
from pulse.simulation import TrialPool


def _pool_with_seed(target: int, p: int) -> TrialPool:
    """One block whose pulse arm contains only the unit seed at t = 0."""
    empty = (np.array([]), np.array([], dtype=int))
    seed_only = (np.array([0.0]), np.array([target]))
    return TrialPool(target=target, horizon=1.0,
                     pulse=[seed_only], sham=[empty])


def test_unit_seed_counted_exactly_once():
    """R_jj must include the direct seed contribution once (App. calibration)."""
    p = 4
    for j in range(p):
        est = estimation.response_column_estimate(_pool_with_seed(j, p), 1, p)
        assert np.allclose(est, np.eye(p)[j])


def test_sham_subtracted():
    """Identical pulse and sham arms give a zero contrast at s = 0."""
    events = (np.array([0.0, 0.4]), np.array([1, 2]))
    pool = TrialPool(target=1, horizon=1.0, pulse=[events], sham=[events])
    assert np.allclose(estimation.response_column_estimate(pool, 1, 4), 0.0)


def test_laplace_weighting_applies_exp_decay():
    pool = TrialPool(target=0, horizon=1.0,
                     pulse=[(np.array([0.0, np.log(2.0)]), np.array([0, 1]))],
                     sham=[(np.array([]), np.array([], dtype=int))])
    est = estimation.response_column_estimate(pool, 1, 2, s=1.0)
    assert np.allclose(est, [1.0, 0.5])  # e^{-s t} at t = log 2


def test_clipped_moments_enforce_physical_bounds():
    b, d = estimation.clipped_moments(0.5, 0.9, c_j=2.0)
    assert (b, d) == (2.0, 1.0)
    b, d = estimation.clipped_moments(3.0, 1.7, c_j=2.0)
    assert (b, d) == (3.0, 1.7)


def test_target_stats_covariance_unbiased():
    rng = np.random.default_rng(0)
    stats = estimation.TargetStats(p=3, c=np.ones(3), target=0)
    draws = rng.normal(size=(200, 3))
    for row in draws:
        stats.update(row)
    cov = stats.covariance_xy()
    x, y = draws @ np.ones(3), draws[:, 0]
    assert np.allclose(cov, np.cov(np.vstack([x, y])), atol=1e-10)


def test_covariance_zero_before_two_blocks():
    stats = estimation.TargetStats(p=2, c=np.ones(2), target=0)
    assert np.allclose(stats.covariance_xy(), 0.0)
    assert np.allclose(stats.covariance_vector(), 0.0)
