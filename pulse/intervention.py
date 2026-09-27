"""Suppression-benefit formulas and decision statistics (Thm 2, Eqs. (1), (8)-(10))."""
from __future__ import annotations

import numpy as np


def single_target_value(r_j, b_j, d_j, eta: float):
    """V_j(eta) = eta r_j b_j / (1 + eta (d_j - 1)) (Eq. (1)).

    b_j = c^T R e_j is the total weighted probe response, d_j = R_jj the
    own-site response; d_j - 1 measures recurrent returns interrupted by
    suppression. Vectorized over targets.
    """
    r_j = np.asarray(r_j, dtype=float)
    b_j = np.asarray(b_j, dtype=float)
    d_j = np.asarray(d_j, dtype=float)
    return eta * r_j * b_j / (1.0 + eta * (d_j - 1.0))


def transport_rate_reduction(R: np.ndarray, r: np.ndarray, retain) -> np.ndarray:
    """r - r^D = (I - D K_eff)^{-1} Gamma r, Gamma = I - D (Eq. (8)).

    retain: per-population retained gains alpha_i in [0, 1].
    """
    p = R.shape[0]
    D = np.diag(np.asarray(retain, dtype=float))
    K_eff = np.eye(p) - np.linalg.inv(R)
    Gamma = np.eye(p) - D
    return np.linalg.solve(np.eye(p) - D @ K_eff, Gamma @ r)


def multi_target_reduction(R, r, targets, etas) -> np.ndarray:
    """r - r^D = R E Lambda [I + E^T (R - I) E Lambda]^{-1} E^T r (Eq. (9)).

    Valid up to complete suppression; no inverse of Lambda or D is needed.
    """
    p = R.shape[0]
    k = len(targets)
    E = np.zeros((p, k))
    E[np.asarray(targets), np.arange(k)] = 1.0
    Lam = np.diag(np.asarray(etas, dtype=float))
    middle = np.eye(k) + E.T @ (R - np.eye(p)) @ E @ Lam
    return R @ E @ Lam @ np.linalg.solve(middle, E.T @ r)


def single_target_reduction(R: np.ndarray, r: np.ndarray, j: int, eta: float):
    """eta r_j / (1 + eta (R_jj - 1)) * R e_j (App. single-target form)."""
    scale = eta * r[j] / (1.0 + eta * (R[j, j] - 1.0))
    return scale * R[:, j]


def value_gradient(r_j: float, b_j: float, d_j: float, eta: float) -> np.ndarray:
    """g_j = dV_j/d(b_j, d_j) = (eta r_j / z_j, -eta^2 r_j b_j / z_j^2)^T
    with z_j = 1 + eta (d_j - 1) (Eq. (10))."""
    z = 1.0 + eta * (d_j - 1.0)
    return np.array([eta * r_j / z, -(eta ** 2) * r_j * b_j / z ** 2])


def value_variance(r_j, b_j, d_j, eta, cov_xy, include_covariance=True) -> float:
    """v_j = g_j^T Cov(X_j, Y_j) g_j (Eq. (10)).

    The cross term 2 g1 g2 Cov(X, Y) is included unless include_covariance is
    False (the no-covariance ablation of App. 'Matched acquisition objectives').
    """
    g = value_gradient(r_j, b_j, d_j, eta)
    cov = np.asarray(cov_xy, dtype=float)
    if not include_covariance:
        cov = np.diag(np.diag(cov))
    return float(g @ cov @ g)


def value_intervals(r_lo, r_hi, b_lo, b_hi, d_lo, d_hi, eta):
    """Simultaneous-interval value bounds (App. 'interval decision result'):
    L_j = eta r_lo b_lo / (1 + eta (d_hi - 1)),
    U_j = eta r_hi b_hi / (1 + eta (d_lo - 1)).
    Selecting argmax L_j certifies regret at most max U - L_argmax on the
    coverage event. Endpoints must satisfy r >= 0, b >= c_j, d >= 1."""
    lower = eta * r_lo * b_lo / (1.0 + eta * (d_hi - 1.0))
    upper = eta * r_hi * b_hi / (1.0 + eta * (d_lo - 1.0))
    return lower, upper
