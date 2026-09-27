"""Tests for the intervention-transport formulas (Thm 2)."""
import numpy as np
import pytest

from pulse import intervention, networks, response

ETAS = (0.25, 0.5, 0.8, 1.0)


@pytest.mark.parametrize("seed", [70000, 70001, 70007, 70013])
@pytest.mark.parametrize("q", [0, 3, 9])
def test_single_target_identity(seed, q):
    """Eq. (8) reduction equals a direct complete-model solve."""
    net = networks.generate_network(seed, p=8, q=q, radius=0.8)
    R = response.integrated_response(net)
    r = response.observed_stationary_rates(net)
    for j in range(net.p):
        for eta in ETAS:
            pred = intervention.single_target_reduction(R, r, j, eta)
            true = r - net.intervened_observed_rates([j], [eta])
            assert np.allclose(pred, true, atol=1e-10)


@pytest.mark.parametrize("targets", [(0, 3), (1, 4, 6)])
def test_multi_target_identity(targets):
    """Eq. (9) reduction equals a direct complete-model solve."""
    net = networks.generate_network(70002, p=8, q=5, radius=0.8)
    R = response.integrated_response(net)
    r = response.observed_stationary_rates(net)
    etas = [0.4] * len(targets)
    pred = intervention.multi_target_reduction(R, r, targets, etas)
    true = r - net.intervened_observed_rates(targets, etas)
    assert np.allclose(pred, true, atol=1e-10)


def test_transport_agrees_with_multi_target():
    """General retain vector D vs the multi-target form on the same gains."""
    net = networks.generate_network(70004, p=6, q=4, radius=0.75)
    R = response.integrated_response(net)
    r = response.observed_stationary_rates(net)
    targets, etas = [1, 4], [0.3, 0.6]
    retain = np.ones(net.p)
    retain[targets] = 1.0 - np.asarray(etas)
    via_transport = intervention.transport_rate_reduction(R, r, retain)
    via_multi = intervention.multi_target_reduction(R, r, targets, etas)
    assert np.allclose(via_transport, via_multi, atol=1e-12)


def test_value_intervals_contain_value():
    """L_j <= V_j <= U_j whenever the moments lie inside the endpoints."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.uniform(0.01, 0.05)
        b = rng.uniform(1.0, 3.0)
        d = rng.uniform(1.0, 2.5)
        eta = 0.8
        V = intervention.single_target_value(r, b, d, eta)
        lo, hi = intervention.value_intervals(
            r * 0.9, r * 1.1, max(1.0, b - 0.2), b + 0.2,
            1.0, d + 0.2, eta)
        assert lo - 1e-12 <= V <= hi + 1e-12


def test_value_variance_without_covariance_uses_diagonal():
    r_j, b_j, d_j, eta = 0.02, 2.0, 1.5, 0.8
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    full = intervention.value_variance(r_j, b_j, d_j, eta, cov)
    diag = intervention.value_variance(r_j, b_j, d_j, eta, cov,
                                       include_covariance=False)
    g = intervention.value_gradient(r_j, b_j, d_j, eta)
    assert diag == pytest.approx(g[0] ** 2 * 4.0 + g[1] ** 2 * 9.0)
    assert diag != pytest.approx(full)
