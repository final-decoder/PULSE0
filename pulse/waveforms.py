"""Participant-held-out waveform completion on released CCEP derivatives
(App. 'Public human responses and anatomical sampling')."""
from __future__ import annotations

import numpy as np

N_TIMES = 118        # 2-ms grid from 15 to 249 ms
RIDGE = 0.01
N_FOLDS = 5
SAMPLES_PER_BLOCK = 10  # contiguous 20-ms blocks


def unit_normalize(waveforms):
    """u = y / ||y||_2 per waveform, for basis learning."""
    out = []
    for y in waveforms:
        n = np.linalg.norm(y)
        if n > 0:
            out.append(np.asarray(y, dtype=float) / n)
    return out


def held_out_basis(participants, exclude: int, k: int) -> np.ndarray:
    """k leading eigenvectors of the uncentered training covariance
    C_{-s} = 1/(S-1) sum_{a != s} (1/L_a) sum_l u_al u_al^T.
    Each training participant receives equal weight; the held-out participant
    contributes nothing."""
    C = None
    count = 0
    for a, waves in enumerate(participants):
        if a == exclude:
            continue
        U = unit_normalize(waves)
        if not U:
            continue
        U = np.asarray(U)
        cov = U.T @ U / len(U)
        C = cov if C is None else C + cov
        count += 1
    C /= count
    _, vecs = np.linalg.eigh(C)
    return vecs[:, ::-1][:, :k]


def ridge_completion(y, basis, visible_mask, ridge=RIDGE):
    """a_hat = argmin ||y_O - B_O a||^2 + ridge ||a||^2; predict all samples."""
    y = np.asarray(y, dtype=float)
    B_O = basis[visible_mask]
    a = np.linalg.solve(B_O.T @ B_O + ridge * np.eye(basis.shape[1]),
                        B_O.T @ y[visible_mask])
    return basis @ a


def five_fold_masks(n_times: int = N_TIMES):
    """Each contiguous 20-ms block is assigned to a fold, cycling labels."""
    labels = (np.arange(n_times) // SAMPLES_PER_BLOCK) % N_FOLDS
    return [labels == f for f in range(N_FOLDS)]


def participant_error(waveforms, basis) -> float:
    """E_{s,k}: out-of-fold normalized squared error, averaged over the
    participant's waveforms; every sample is predicted exactly once."""
    errs = []
    for y in waveforms:
        y = np.asarray(y, dtype=float)
        y_hat = np.zeros_like(y)
        for withheld in five_fold_masks(len(y)):
            visible = ~withheld
            y_hat[withheld] = ridge_completion(y, basis, visible)[withheld]
        denom = float(y @ y)
        if denom > 0:
            errs.append(float(((y - y_hat) ** 2).sum() / denom))
    return float(np.mean(errs))


def linear_interpolate(y, visible_mask) -> np.ndarray:
    """Training-free comparator: linear interpolation between visible samples,
    nearest visible endpoint held constant outside their range."""
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y))
    vis = np.asarray(visible_mask, dtype=bool)
    return np.interp(t, t[vis], y[vis])


def gap_error(y, y_hat, mask) -> float:
    """E_M = |M|^{-1} sum_{t in M} (y_hat - y)^2 / (N^{-1} sum_t y_t^2)."""
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    num = ((y_hat - y)[mask] ** 2).mean()
    den = (y ** 2).mean()
    return float(num / den)


def gap_cases(n_times: int = N_TIMES, binwidth_ms: float = 2.0):
    """(duration_ms, start_index) grid: durations 10/20/40/80 ms, start times
    15..215 ms in 20-ms steps, intervals ending no later than 250 ms."""
    cases = []
    for dur_ms in (10, 20, 40, 80):
        length = int(dur_ms / binwidth_ms)
        for start in range(0, 101, 10):  # 15 ms + start * 2 ms <= 215 ms
            if start + length <= n_times:
                cases.append((dur_ms, start))
    return cases
