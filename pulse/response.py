"""Observed probe-response operator and effective kernel (Thm 1, Eqs. (4)-(7))."""
from __future__ import annotations

import numpy as np


def effective_kernel_laplace(network, s: float) -> np.ndarray:
    """hat Phi_eff(s) = hat Phi_OO + hat Phi_OH (I - hat Phi_HH)^{-1} hat Phi_HO.

    Schur complement over hidden populations (Eq. (5)); valid for any hidden
    dimension, unequal hidden-path lengths, and edge-specific kernels.
    """
    Phi = network.kernels.laplace(s)
    p, q = network.p, network.q
    if q == 0:
        return Phi[:p, :p]
    Phi_OO, Phi_OH = Phi[:p, :p], Phi[:p, p:]
    Phi_HO, Phi_HH = Phi[p:, :p], Phi[p:, p:]
    hidden = np.linalg.solve(np.eye(q) - Phi_HH, Phi_HO)
    return Phi_OO + Phi_OH @ hidden


def response_laplace(network, s: float) -> np.ndarray:
    """R(s) = [I - hat Phi_eff(s)]^{-1}, including the unit seed (Eq. (4))."""
    F_eff = effective_kernel_laplace(network, s)
    return np.linalg.inv(np.eye(network.p) - F_eff)


def integrated_response(network) -> np.ndarray:
    """R = R(0), the integrated observed response."""
    return response_laplace(network, 0.0)


def effective_integrated(R: np.ndarray) -> np.ndarray:
    """K_eff = I - R^{-1} (Sec. 'Response-equivalent networks')."""
    return np.eye(R.shape[0]) - np.linalg.inv(R)


def effective_baseline(network) -> np.ndarray:
    """beta = mu_O + K_OH (I - K_HH)^{-1} mu_H (Eq. (7))."""
    p, q = network.p, network.q
    if q == 0:
        return network.mu[:p].copy()
    K = network.K
    return network.mu[:p] + K[:p, p:] @ np.linalg.solve(
        np.eye(q) - K[p:, p:], network.mu[p:]
    )


def observed_stationary_rates(network) -> np.ndarray:
    """Observed rates r = (I - K_eff)^{-1} beta = R beta."""
    return integrated_response(network) @ effective_baseline(network)
