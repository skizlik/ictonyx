"""Tests for VariabilityStudyResults persistence — schema version and run_seeds."""

import pickle

import pandas as pd
import pytest

from ictonyx.runners import _ICTONYX_VERSION, VariabilityStudyResults


def _make_minimal_results(run_seeds=None):
    """Construct a VariabilityStudyResults with realistic field shapes."""
    return VariabilityStudyResults(
        all_runs_metrics=[
            pd.DataFrame({"epoch": [1, 2, 3], "val_accuracy": [0.8, 0.85, 0.88]}),
            pd.DataFrame({"epoch": [1, 2, 3], "val_accuracy": [0.82, 0.86, 0.89]}),
        ],
        final_metrics={"val_accuracy": [0.88, 0.89]},
        final_test_metrics=[],
        seed=42,
        run_seeds=run_seeds if run_seeds is not None else [12345, 67890],
    )


def test_save_writes_schema_version(tmp_path):
    results = _make_minimal_results()
    path = tmp_path / "r.pkl"
    results.save(str(path))

    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert payload["_schema_version"] == _ICTONYX_VERSION


def test_save_writes_run_seeds(tmp_path):
    results = _make_minimal_results(run_seeds=[111, 222, 333])
    path = tmp_path / "r.pkl"
    results.save(str(path))

    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert payload["run_seeds"] == [111, 222, 333]


def test_roundtrip_preserves_run_seeds(tmp_path):
    results = _make_minimal_results(run_seeds=[111, 222, 333])
    path = tmp_path / "r.pkl"
    results.save(str(path))
    loaded = VariabilityStudyResults.load(str(path))
    assert loaded.run_seeds == [111, 222, 333]


def test_load_warns_on_missing_schema_version(tmp_path):
    """Simulate a pre-v0.4.5 payload (no _schema_version key)."""
    path = tmp_path / "legacy.pkl"
    legacy_payload = {
        "all_runs_metrics": [pd.DataFrame({"val_accuracy": [0.9]})],
        "final_metrics": {"val_accuracy": [0.9]},
        "final_test_metrics": [],
        "seed": 7,
        # note: no _schema_version, no run_seeds
    }
    with open(path, "wb") as f:
        pickle.dump(legacy_payload, f)

    with pytest.warns(UserWarning, match="schema version"):
        loaded = VariabilityStudyResults.load(str(path))
    assert loaded.run_seeds == []  # back-compat: empty list when absent


def test_load_warns_on_version_mismatch(tmp_path):
    """Simulate a payload saved by a different ictonyx version."""
    path = tmp_path / "old.pkl"
    payload = {
        "_schema_version": "0.0.0",
        "all_runs_metrics": [pd.DataFrame({"val_accuracy": [0.9]})],
        "final_metrics": {"val_accuracy": [0.9]},
        "final_test_metrics": [],
        "seed": 7,
        "run_seeds": [],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.warns(UserWarning, match="0.0.0"):
        VariabilityStudyResults.load(str(path))


def test_load_no_warning_on_current_version(tmp_path):
    """Round-trip with the current version must not emit a UserWarning."""
    import warnings

    results = _make_minimal_results()
    path = tmp_path / "r.pkl"
    results.save(str(path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        VariabilityStudyResults.load(str(path))


def test_variability_study_populates_run_seeds():
    """End-to-end: a study run through the public API must populate run_seeds."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    import ictonyx as ix

    X, y = make_classification(n_samples=100, random_state=0)
    results = ix.variability_study(
        model=LogisticRegression,
        data=(X, y),
        runs=3,
        seed=42,
    )
    assert len(results.run_seeds) == 3
    assert all(isinstance(s, int) for s in results.run_seeds)
    assert len(set(results.run_seeds)) == 3
