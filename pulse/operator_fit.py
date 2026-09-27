"""Nonnegative effective-operator recovery (App. 'Nonnegative operator
estimation'). The intervention-value estimator does not use this inversion or
the spectral projection."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def project_spectral_radius(M: np.ndarray, cap: float) -> np.ndarray:
    """Uniform rescaling onto the spectral-norm ball when rho(M) > cap."""
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    return M * (cap / rho) if rho > cap else M


def fit_effective_operator(R_hat: np.ndarray, n_samples: int,
                           ridge: float | None = None) -> np.ndarray:
    """Row-wise NNLS of (I - Phi_eff) R = I:
    min_{a_i >= 0} ||a_i R_hat - (R_hat - I)_{i:}||^2 + n^{-1/2} ||a_i||^2,
    solved as NNLS with an augmented design matrix; result projected to
    spectral radius <= 0.98."""
    p = R_hat.shape[0]
    lam = n_samples ** -0.5 if ridge is None else ridge
    design = np.vstack([R_hat.T, np.sqrt(lam) * np.eye(p)])
    rhs_full = (R_hat - np.eye(p)).T  # column i is the transposed row target
    Phi = np.zeros((p, p))
    for i in range(p):
        target = np.concatenate([rhs_full[:, i], np.zeros(p)])
        Phi[i], _ = nnls(design, target)
    return project_spectral_radius(Phi, 0.98)
