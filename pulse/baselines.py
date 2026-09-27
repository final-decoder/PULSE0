"""Reference scoring rules and passive Hawkes fits (App. 'Passive baselines
and ablations')."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .intervention import single_target_value


def select_rate_only(r_hat) -> int:
    """argmax_j r_hat_j."""
    return int(np.argmax(r_hat))


def select_evoked_reach(R_hat, c) -> int:
    """argmax_j c^T R_hat e_j (no return correction, no baseline weighting)."""
    return int(np.argmax(np.asarray(c) @ R_hat))


def select_uncorrected(r_hat, R_hat, c) -> int:
    """argmax_j r_hat_j * c^T R_hat e_j (no return-correction denominator)."""
    return int(np.argmax(np.asarray(r_hat) * (np.asarray(c) @ R_hat)))


def known_direct_edges_values(network, r, eta, c) -> np.ndarray:
    """Additional-information reference: insert the physical K_OO into the
    observed-only calculation, omitting hidden pathways."""
    p = network.p
    R_direct = np.linalg.inv(np.eye(p) - network.K[:p, :p])
    b = np.asarray(c) @ R_direct
    d = np.diag(R_direct)
    return single_target_value(np.asarray(r), b, d, eta)


def bin_passive_counts(times, types, p, duration, binwidth=0.05) -> np.ndarray:
    """(T, p) binned observed counts; 50-ms bins in the paper protocol."""
    n_bins = int(duration / binwidth)
    counts = np.zeros((n_bins, p))
    idx = (times / binwidth).astype(int)
    keep = (idx >= 0) & (idx < n_bins) & (types < p)
    np.add.at(counts, (idx[keep], types[keep]), 1.0)
    return counts


def _exponential_history(counts: np.ndarray, decay: float, binwidth: float) -> np.ndarray:
    """h[t] = sum_{tau <= t-1} counts[tau] exp(-(t - tau) binwidth / decay);
    previous bins only, one column per source population."""
    T, p = counts.shape
    h = np.zeros((T, p))
    decay_factor = np.exp(-binwidth / decay)
    for t in range(1, T):
        h[t] = decay_factor * (h[t - 1] + counts[t - 1])
    return h


def fit_passive_hawkes(counts: np.ndarray, decays, binwidth: float = 0.05,
                       penalty: float = 0.3, max_obs: int = 25000):
    """Observed-only nonnegative Poisson GLM on filtered histories.

    For each target: log E[count] = intercept + sum_{j,k} w_jk h_jk with
    intercept >= 1e-7, w_jk in [0, 2]; objective adds penalty * sum(w).
    L-BFGS-B, <= 600 iterations, ftol 1e-10, up to max_obs evenly spaced
    bins. Returns the integrated coefficient matrix, rescaled to spectral
    radius <= 0.97 (identical stabilization for every fit).
    """
    T, p = counts.shape
    features = [_exponential_history(counts, d, binwidth) for d in decays]
    H = np.concatenate(features, axis=1)  # (T, p * n_bases)
    if T > max_obs:
        rows = np.linspace(0, T - 1, max_obs).astype(int)
    else:
        rows = np.arange(T)
    Hs = H[rows]
    n_coef = H.shape[1]

    K_fit = np.zeros((p, p))
    for i in range(p):
        y = counts[rows, i]

        def nll(params):
            a, w = params[0], params[1:]
            eta_lin = a + Hs @ w
            lam = np.exp(eta_lin)
            diff = lam - y
            f = float(diff.sum() + penalty * w.sum())
            g = np.concatenate([[diff.sum()], Hs.T @ diff + penalty])
            return f, g

        x0 = np.concatenate([[np.log(max(y.mean(), 1e-7))], np.zeros(n_coef)])
        bounds = [(1e-7, None)] + [(0.0, 2.0)] * n_coef
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 600, "ftol": 1e-10})
        Wm = res.x[1:].reshape(len(decays), p)  # rows: bases, cols: sources
        # discrete filter mass of a unit event: sum_{m >= 1} e^{-m dt / theta}
        mass = np.array([1.0 / (np.exp(binwidth / d) - 1.0) for d in decays])
        K_fit[i] = Wm.T @ mass

    rho = np.max(np.abs(np.linalg.eigvals(K_fit)))
    if rho > 0.97:
        K_fit *= 0.97 / rho
    return K_fit


def passive_fit_values(K_fit, r_hat, eta, c) -> np.ndarray:
    """Evaluate the gain-intervention formula with the fitted observed-only
    resolvent and the common empirical baseline rates."""
    p = K_fit.shape[0]
    R_fit = np.linalg.inv(np.eye(p) - K_fit)
    b = np.asarray(c) @ R_fit
    d = np.diag(R_fit)
    return single_target_value(np.asarray(r_hat), b, d, eta)
