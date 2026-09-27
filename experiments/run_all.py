"""Run every self-contained experiment and archive the printed summaries.

Each experiment script is executed as a subprocess so a failure in one does
not stop the rest; stdout/stderr are captured under ``results/<name>.txt``.
The two data-dependent scripts (``human_timing``, ``waveform_completion``)
are skipped unless their input files are supplied.

Usage (from the repository root):

    python -m experiments.run_all                # all self-contained scripts
    python -m experiments.run_all --list         # show what would run
    python -m experiments.run_all --only multitarget,operator_recovery
    python -m experiments.run_all --latencies latencies.json \
        --waveforms waveforms.npz                # include data-dependent ones
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# (script, needs) — "data" scripts require an external input file
SCRIPTS = {
    "check_identities": None,
    "equivalence_families": None,
    "intervention_validation": None,
    "multitarget": None,
    "scoring_comparison": None,
    "acquisition_benchmark": None,
    "coverage_calibration": None,
    "response_windows": None,
    "operator_recovery": None,
    "human_timing": "latencies",
    "waveform_completion": "waveforms",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="list scripts and exit")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset of scripts")
    parser.add_argument("--latencies", type=str, default=None,
                        help="JSON of per-participant N1 latencies")
    parser.add_argument("--waveforms", type=str, default=None,
                        help="NPZ of per-participant CCEP waveforms")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="output directory (default: ./results)")
    args = parser.parse_args()

    selected = list(SCRIPTS)
    if args.only:
        selected = [s.strip() for s in args.only.split(",")]
        unknown = [s for s in selected if s not in SCRIPTS]
        if unknown:
            parser.error(f"unknown scripts: {unknown}")

    inputs = {"latencies": args.latencies, "waveforms": args.waveforms}
    runnable = []
    for name in selected:
        need = SCRIPTS[name]
        if need is None or inputs.get(need):
            runnable.append(name)
        else:
            print(f"[skip] {name}: requires --{need}")

    if args.list:
        for name in runnable:
            print(f"would run: {name}")
        return 0

    out_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for name in runnable:
        cmd = [sys.executable, "-m", f"experiments.{name}"]
        need = SCRIPTS[name]
        if need:
            cmd.append(inputs[need])
        print(f"[run] {name} ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        (out_dir / f"{name}.txt").write_text(
            f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"[{status}] {name} in {elapsed:.1f}s -> {out_dir / (name + '.txt')}")
        if proc.returncode != 0:
            failures.append(name)

    if failures:
        print(f"failed: {', '.join(failures)}")
        return 1
    print(f"all {len(runnable)} scripts finished; summaries in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
