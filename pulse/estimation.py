"""Pulse/sham response estimation (Eq. (6)) and online decision moments."""
from __future__ import annotations

import numpy as np


def response_column_estimate(pool, n_blocks: int, p: int, s: float = 0.0) -> np.ndarray:
    """hat R_:j,H(s) = (1/n_j) sum_l int_[0,H] e^{-st} [dN^{+,j,l} - dN^{-,j,l}].

    The pulse arm includes the known unit seed at time zero (App. 'Known input
    calibration': it must be counted exactly once, also inside R_jj).
    """
    est = np.zeros(p)
    for k in range(n_blocks):
        for events, sign in ((pool.pulse[k], 1.0), (pool.sham[k], -1.0)):
            times, types = events
            obs = types < p
            est += sign * np.bincount(
                types[obs], weights=np.exp(-s * times[obs]), minlength=p
            )
    return est / n_blocks


def full_response_estimate(pools, n_blocks, p: int, s: float = 0.0) -> np.ndarray:
    """Stack per-target column estimates into an estimate of R(s)."""
    cols = [response_column_estimate(pool, n, p, s) for pool, n in zip(pools, n_blocks)]
    return np.column_stack(cols)


def clipped_moments(xbar: float, ybar: float, c_j: float):
    """Physical constraints b_j >= c_j and d_j >= 1 on plug-in estimates."""
    return max(c_j, xbar), max(1.0, ybar)


class TargetStats:
    """Online sufficient summaries for one probe target (App. 'Acquisition'):
    trial count, sums of X = c^T dN_O and Y = dN_j, their second moments, and
    full contrast-vector moments (only the latter serve response-focused
    acquisition; single-target selection needs the scalar moments)."""

    def __init__(self, p: int, c: np.ndarray, target: int):
        self.p = p
        self.c = np.asarray(c, dtype=float)
        self.target = target
        self.n = 0
        self.pool_size = None  # set by the acquisition loop (finite pools)
        self.sum_vec = np.zeros(p)
        self.sum_outer = np.zeros((p, p))
        self.sum_x = self.sum_y = 0.0
        self.sum_xx = self.sum_yy = self.sum_xy = 0.0

    def update(self, contrast: np.ndarray) -> None:
        """contrast: pulse-minus-sham observed count vector of one block."""
        x = float(self.c @ contrast)
        y = float(contrast[self.target])
        self.n += 1
        self.sum_vec += contrast
        self.sum_outer += np.outer(contrast, contrast)
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    @property
    def xbar(self) -> float:
        return self.sum_x / self.n

    @property
    def ybar(self) -> float:
        return self.sum_y / self.n

    def covariance_xy(self) -> np.ndarray:
        """Sample covariance of (X, Y)."""
        if self.n < 2:
            return np.zeros((2, 2))
        mean = np.array([self.xbar, self.ybar])
        second = np.array(
            [[self.sum_xx, self.sum_xy], [self.sum_xy, self.sum_yy]]
        ) / self.n
        return (second - np.outer(mean, mean)) * (self.n / (self.n - 1))

    def covariance_vector(self) -> np.ndarray:
        """Sample covariance of the full observed contrast vector."""
        if self.n < 2:
            return np.zeros((self.p, self.p))
        mean = self.sum_vec / self.n
        second = self.sum_outer / self.n
        return (second - np.outer(mean, mean)) * (self.n / (self.n - 1))
