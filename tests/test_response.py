"""Tests for the response-identification formulas (Thm 1)."""
import numpy as np
import pytest

from pulse import networks, response


def test_integrated_response_matches_complete_resolvent():
    """R(0) must equal the observed block of (I - K)^{-1}."""
    for seed in (13000, 13100, 13300, 22000):
        net = networks.generate_network(seed, p=6, q=4, radius=0.7)
        Q = np.linalg.inv(np.eye(net.n) - net.K)
        assert np.allclose(response.integrated_response(net), Q[: net.p, : net.p])


def test_response_includes_identity_and_is_nonnegative():
    net = networks.generate_network(13000, p=6, q=4, radius=0.7)
    R = response.integrated_response(net)
    assert (R >= 0).all()
    assert (np.diag(R) >= 1.0).all()  # identity atom included


def test_effective_kernel_is_subcritical():
    """K_eff = I - R^{-1} is nonnegative with rho < 1 (App. stability)."""
    net = networks.generate_network(22000, p=6, q=8, radius=0.9)
    K_eff = response.effective_integrated(response.integrated_response(net))
    assert (K_eff >= -1e-12).all()
    assert np.max(np.abs(np.linalg.eigvals(K_eff))) < 1.0


@pytest.mark.parametrize("f", [0, 5, 11, 23])
def test_equivalence_family_shares_response_and_rates(f):
    """All clones in a family share (R, r) despite different hidden counts."""
    variants = networks.equivalence_family(f)
    R0 = response.integrated_response(variants[1])
    r0 = response.observed_stationary_rates(variants[1])
    for k in (2, 4):
        assert np.allclose(response.integrated_response(variants[k]), R0)
        assert np.allclose(response.observed_stationary_rates(variants[k]), r0)


def test_laplace_response_reduces_to_integrated_at_zero():
    net = networks.generate_network(13000, p=6, q=4, radius=0.7)
    assert np.allclose(response.response_laplace(net, 0.0),
                       response.integrated_response(net))


def test_effective_baseline_reduces_to_immigrant_rate_without_hidden():
    net = networks.generate_network(13000, p=6, q=0, radius=0.7)
    assert np.allclose(response.effective_baseline(net), net.mu[: net.p])
