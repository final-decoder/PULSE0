"""Complete Hawkes network with observed and hidden populations (paper Eq. (2)).

Intensity: lambda(t) = mu + int_{u < t} Phi(t - u) dN(u), with nonnegative
edge-wise Gamma kernels and rho(K) < 1 for K = int_0^inf Phi(t) dt.
Matrix entry (i, j) always means excitation from source j into receiver i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KernelBank:
    """Phi_ij(t) = K_ij * GammaPDF(t; shape=k_ij, scale=tau_ij / k_ij)."""

    weights: np.ndarray  # integrated excitation K_ij (receiver i, source j)
    shapes: np.ndarray   # Gamma shapes k_ij
    means: np.ndarray    # mean delays tau_ij in seconds

    def laplace(self, s: float) -> np.ndarray:
        """hat Phi(s); Gamma(k, tau/k) has transform (1 + s tau / k)^-k."""
        active = self.weights > 0
        shapes = np.where(active, self.shapes, 1.0)
        means = np.where(active, self.means, 1.0)
        transform = (1.0 + s * means / shapes) ** (-shapes)
        return np.where(active, self.weights * transform, 0.0)

    @property
    def integrated(self) -> np.ndarray:
        """K = hat Phi(0)."""
        return self.weights


@dataclass
class HawkesNetwork:
    """Stationary complete network; observed populations are indices [0, p)."""

    mu: np.ndarray       # immigrant rates, length n = p + q
    kernels: KernelBank
    p: int               # number of observed populations

    @property
    def n(self) -> int:
        return self.mu.shape[0]

    @property
    def q(self) -> int:
        return self.n - self.p

    @property
    def K(self) -> np.ndarray:
        return self.kernels.integrated

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.K))))

    def stationary_rates(self) -> np.ndarray:
        """Complete stationary rates (I - K)^{-1} mu."""
        return np.linalg.solve(np.eye(self.n) - self.K, self.mu)

    def intervened(self, targets, etas) -> HawkesNetwork:
        """Ground-truth local gain suppression: for each targeted observed
        population, scale its receiving row of K and its immigrant rate by
        1 - eta (App. 'Ground truth, metrics, and replication units')."""
        K_mod = self.K.copy()
        mu_mod = self.mu.copy()
        for j, eta in zip(targets, etas):
            K_mod[j] *= 1.0 - eta
            mu_mod[j] *= 1.0 - eta
        kernels = KernelBank(K_mod, self.kernels.shapes, self.kernels.means)
        return HawkesNetwork(mu=mu_mod, kernels=kernels, p=self.p)

    def intervened_observed_rates(self, targets, etas) -> np.ndarray:
        """Observed stationary rates of the intervened complete model."""
        return self.intervened(targets, etas).stationary_rates()[: self.p]
