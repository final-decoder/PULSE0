"""Empirical N1 latency input for the 250-ms timing benchmark (App. 'timing').

Per-participant deduplicated N1 latencies (milliseconds) are provided as JSON:
{"1": [12.3, ...], ..., "74": [...]} for participants u = 1..74.
"""
from __future__ import annotations

import json

import numpy as np


def load_n1_latencies(path) -> dict[int, np.ndarray]:
    with open(path) as fh:
        raw = json.load(fh)
    return {int(k): np.asarray(v, dtype=float) for k, v in raw.items()}
