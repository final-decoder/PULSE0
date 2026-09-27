"""Participant-held-out waveform completion on released CCEP derivatives
(App. 'Public human responses and anatomical sampling').

Input: NPZ with object array `waveforms`, one entry per participant, each a
list of baseline-corrected length-118 responses (2-ms grid, 15-249 ms).
Usage: python waveform_completion.py waveforms.npz
"""
from __future__ import annotations

import sys

import numpy as np

from pulse import metrics, waveforms

BOOTSTRAP_SEED = 7357
RANKS = (1, 4)


def main():
    data = np.load(sys.argv[1], allow_pickle=True)
    participants = [list(w) for w in data["waveforms"]]

    errors = {k: [] for k in RANKS}
    interp_errors = []
    for s, waves in enumerate(participants):
        for k in RANKS:
            basis = waveforms.held_out_basis(participants, exclude=s, k=k)
            errors[k].append(waveforms.participant_error(waves, basis))
        # interpolation with the same five-fold visible/withheld split
        errs = []
        for y in waves:
            y = np.asarray(y, dtype=float)
            y_hat = np.zeros_like(y)
            for withheld in waveforms.five_fold_masks(len(y)):
                y_hat[withheld] = waveforms.linear_interpolate(y, ~withheld)[withheld]
            if y @ y > 0:
                errs.append(float(((y - y_hat) ** 2).sum() / (y @ y)))
        interp_errors.append(float(np.mean(errs)))

    for k in RANKS:
        mean, lo, hi = metrics.bootstrap_ci(errors[k], seed=BOOTSTRAP_SEED)
        print(f"{k} basis: error {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    mean, lo, hi = metrics.bootstrap_ci(interp_errors, seed=BOOTSTRAP_SEED)
    print(f"linear interpolation: {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    diffs = [a - b for a, b in zip(errors[4], errors[1])]
    mean, lo, hi = metrics.bootstrap_ci(diffs, seed=BOOTSTRAP_SEED)
    print(f"paired 4-minus-1 basis difference: {mean:.3f} [{lo:.3f}, {hi:.3f}]")

    # duration-by-position gap sweep (one contiguous interval per waveform)
    for dur_ms, start in waveforms.gap_cases():
        per_participant = {k: [] for k in (*RANKS, "interp")}
        for s, waves in enumerate(participants):
            mask = np.zeros(waveforms.N_TIMES, dtype=bool)
            mask[start:start + dur_ms // 2] = True
            if not mask.any():
                continue
            for k in RANKS:
                basis = waveforms.held_out_basis(participants, exclude=s, k=k)
                errs = [waveforms.gap_error(y, waveforms.ridge_completion(
                    y, basis, ~mask), mask) for y in waves]
                per_participant[k].append(float(np.mean(errs)))
            errs = [waveforms.gap_error(y, waveforms.linear_interpolate(
                y, ~mask), mask) for y in waves]
            per_participant["interp"].append(float(np.mean(errs)))
        row = {k: float(np.mean(v)) for k, v in per_participant.items() if v}
        print(f"gap {dur_ms:3d} ms @ idx {start:3d}: " +
              ", ".join(f"{k}={v:.3f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
