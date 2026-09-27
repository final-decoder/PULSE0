"""Tests for metrics, resampling intervals, and result persistence."""
import numpy as np
import pytest

from pulse import metrics, results


def test_selection_regret_zero_for_best():
    vals = np.array([0.1, 0.5, 0.3])
    assert metrics.selection_regret(vals, 1) == 0.0
    assert metrics.selection_regret(vals, 0) == pytest.approx(0.8)


def test_value_nmae_zero_for_perfect_estimate():
    vals = np.array([1.0, 2.0, 3.0])
    assert metrics.value_nmae(vals, vals.copy()) == 0.0


def test_bootstrap_ci_is_deterministic():
    samples = np.linspace(0.0, 1.0, 50)
    a = metrics.bootstrap_ci(samples)
    b = metrics.bootstrap_ci(samples)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi


def test_json_roundtrip_with_numpy_types(tmp_path):
    payload = {"regret": np.float64(0.0158), "chosen": np.int64(3),
               "curve": np.arange(4)}
    path = results.save_json(payload, tmp_path / "out.json", script="pytest")
    loaded = results.load_json(path)
    assert loaded["regret"] == pytest.approx(0.0158)
    assert loaded["chosen"] == 3
    assert loaded["curve"] == [0, 1, 2, 3]


def test_csv_rows_written_with_provenance(tmp_path):
    rows = [{"q": 0, "regret": 1.5}, {"q": 4, "regret": 1.2}]
    path = results.save_rows(rows, tmp_path / "out.csv", script="pytest")
    text = path.read_text()
    assert text.startswith("# pulse_version=")
    assert "regret" in text.splitlines()[1]
    assert len(text.strip().splitlines()) == 4  # comment + header + 2 rows
