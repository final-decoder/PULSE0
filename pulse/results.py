"""Saving and loading of experiment outputs (JSON / CSV).

Experiment scripts print human-readable summaries; anything persisted for
later analysis goes through these helpers so on-disk results carry the
protocol metadata (package version, seeds, and the script that produced
them) alongside the numbers.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _version() -> str:
    """Package version, resolved lazily to avoid an import cycle."""
    from . import __version__
    return __version__


def _to_builtin(obj):
    """Recursively convert numpy scalars/arrays to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(payload: dict, path, script: str | None = None) -> Path:
    """Write a JSON document with provenance metadata attached."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "pulse_version": _version(),
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": _to_builtin(payload),
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return path


def load_json(path) -> dict:
    """Read a document written by :func:`save_json`; returns the results."""
    with open(path) as fh:
        return json.load(fh)["results"]


def save_rows(rows, path, fieldnames=None, script: str | None = None) -> Path:
    """Write a list of flat dicts as CSV with a leading provenance comment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_to_builtin(r) for r in rows]
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row.keys()))
    with open(path, "w", newline="") as fh:
        fh.write(f"# pulse_version={_version()} script={script}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
