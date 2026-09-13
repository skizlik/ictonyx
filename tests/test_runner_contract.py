"""Cross-path runner contract (Master Dev Guide v5 §2.3, T1 and T11).

Every execution path must deliver the same fit() kwargs and the same
per-run seeds. This test would have caught 0.3 and 1.5.
"""

import importlib.util

import numpy as np
import pytest
from _spy_wrappers import build_fails_on_run, build_named_lr_spy, build_recording_spy

import ictonyx as ix


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


X = np.random.RandomState(0).randn(60, 4)
y = (X[:, 0] > 0).astype(int)

PATHS = [
    pytest.param({}, id="standard"),
    pytest.param(
        {"use_process_isolation": True},
        id="isolated",
        marks=pytest.mark.skipif(not _has("cloudpickle"), reason="needs [isolation]"),
    ),
    pytest.param(
        {"use_parallel": True, "n_jobs": 2},
        id="parallel",
        marks=pytest.mark.skipif(not _has("joblib"), reason="needs joblib"),
    ),
]


@pytest.mark.parametrize("path", PATHS)
def test_run_seed_reaches_fit_and_matches_run_seeds(path):
    r = ix.variability_study(
        model=build_recording_spy, data=(X, y), runs=3, epochs=1, seed=11, verbose=False, **path
    )
    seen = [int(v) for v in r.get_metric_values("val_seed_seen")]
    assert seen == [r.get_run_seed(i) for i in r.run_ids]
    assert len(set(seen)) == 3


@pytest.mark.parametrize("path", PATHS)
def test_named_fit_kwarg_is_forwarded(path):
    r = ix.variability_study(
        model=build_named_lr_spy,
        data=(X, y),
        runs=2,
        epochs=1,
        seed=11,
        verbose=False,
        learning_rate=0.123,
        **path,
    )
    assert r.get_metric_values("val_lr_seen") == pytest.approx([0.123, 0.123])


def test_unaccepted_fit_kwarg_warns_and_is_not_forwarded_standard():
    # Standard mode only: build_fit_kwargs runs in-process, so the warning is observable.
    with pytest.warns(UserWarning, match="learning_rate"):
        r = ix.variability_study(
            model=build_recording_spy,
            data=(X, y),
            runs=2,
            epochs=1,
            seed=11,
            verbose=False,
            learning_rate=0.123,
        )
    assert r.get_metric_values("val_n_kwargs") == [4.0, 4.0]


@pytest.mark.parametrize("path", [p for p in PATHS if p.id != "standard"])
def test_unaccepted_fit_kwarg_is_not_forwarded_worker_paths(path):
    # Isolated and parallel run fit() in a child process; the warning is raised
    # there and does not propagate. Assert only the contract: nothing forwarded.
    r = ix.variability_study(
        model=build_recording_spy,
        data=(X, y),
        runs=2,
        epochs=1,
        seed=11,
        verbose=False,
        learning_rate=0.123,
        **path,
    )
    assert r.get_metric_values("val_n_kwargs") == [4.0, 4.0]


def test_results_invariants_after_failure():
    import _spy_wrappers as m

    m.FAIL_SEED = int(np.random.SeedSequence(11).spawn(4)[1].generate_state(1)[0])
    r = ix.variability_study(
        model=build_fails_on_run, data=(X, y), runs=4, epochs=1, seed=11, verbose=False
    )
    assert r.failed_runs == [2] and r.run_ids == [1, 3, 4]
    assert len(r.run_seeds) == r.n_runs + len(r.failed_runs)
