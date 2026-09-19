"""v0.4.10: checkpoint provenance.

Closes v12 1.29, 1.33, 1.38, 2.15.
"""

import os

import numpy as np
import pandas as pd
import pytest
from _builders import _InterruptsAfter

from ictonyx.analysis import align_paired
from ictonyx.api import _get_model_builder
from ictonyx.config import ModelConfig
from ictonyx.data import ArraysDataHandler
from ictonyx.runners import ExperimentRunner, VariabilityStudyResults

RUNS = 5


def _run(X, y, seed, ckpt=None, interrupt_after=None, **kw):
    _InterruptsAfter.calls = 0
    _InterruptsAfter.stop_after = interrupt_after
    runner = ExperimentRunner(
        _get_model_builder(_InterruptsAfter),
        ArraysDataHandler(X, y, val_split=0.3),
        ModelConfig({}),
        verbose=False,
        seed=seed,
    )
    return runner.run_study(num_runs=RUNS, checkpoint_dir=None if ckpt is None else str(ckpt), **kw)


def test_resume_default_seed_adopts_checkpoint_seed(X, y, tmp_path):
    """v12 1.38: with seed=None the resumed runner must continue the SAME seed family."""
    r1 = _run(X, y, seed=None, ckpt=tmp_path, interrupt_after=2)
    assert r1.n_runs == 2 and (tmp_path / "checkpoint.pkl").exists()
    r2 = _run(X, y, seed=None, ckpt=tmp_path)
    assert r2.seed == r1.seed
    assert r2.run_seeds[:2] == r1.run_seeds[:2]
    assert r2.n_runs == RUNS and len(set(r2.run_seeds)) == RUNS


def test_resume_explicit_same_seed_is_silent(X, y, tmp_path, recwarn):
    _run(X, y, seed=11, ckpt=tmp_path, interrupt_after=2)
    r2 = _run(X, y, seed=11, ckpt=tmp_path)
    assert r2.seed == 11 and r2.n_runs == RUNS
    assert not [w for w in recwarn if "Mixed-seed" in str(w.message)]


def test_resume_explicit_different_seed_raises(X, y, tmp_path):
    _run(X, y, seed=1, ckpt=tmp_path, interrupt_after=2)
    with pytest.raises(ValueError, match="seed=1"):
        _run(X, y, seed=2, ckpt=tmp_path)
    assert (tmp_path / "checkpoint.pkl").exists()  # not silently restarted


def test_resume_ignore_seed_repairs_provenance(X, y, tmp_path):
    r1 = _run(X, y, seed=1, ckpt=tmp_path, interrupt_after=2)
    with pytest.warns(UserWarning, match="Mixed-seed"):
        r2 = _run(X, y, seed=2, ckpt=tmp_path, resume_ignore_seed=True)
    assert r2.seed is None
    assert r2.run_seeds[:2] == r1.run_seeds[:2]  # restored runs keep their real seeds
    assert r2.run_seeds[2:] != r1.run_seeds[2:]  # new runs come from seed 2


@pytest.mark.parametrize("explicit", [False, True], ids=["default_seed", "explicit_seed"])
def test_resumed_study_reproduces_uninterrupted_study(X, y, tmp_path, explicit):
    """R6 property: interrupt-then-resume equals one uninterrupted study, exactly."""
    interrupted = _run(X, y, seed=7 if explicit else None, ckpt=tmp_path, interrupt_after=2)
    full = _run(X, y, seed=interrupted.seed)  # same seed, no checkpoint, no interruption
    resumed = _run(X, y, seed=7 if explicit else None, ckpt=tmp_path)
    assert resumed.run_seeds == full.run_seeds
    assert resumed.get_metric_values("val_accuracy") == pytest.approx(
        full.get_metric_values("val_accuracy")
    )


def test_completed_checkpoint_is_retired(X, y, tmp_path):
    _run(X, y, seed=1, ckpt=tmp_path)
    assert not (tmp_path / "checkpoint.pkl").exists()
    assert (tmp_path / "checkpoint.done.pkl").exists()
    r = _run(X, y, seed=1, ckpt=tmp_path)  # trains again; does not replay
    assert r.n_runs == RUNS and _InterruptsAfter.calls == RUNS


def _mk(ids, vals):
    return VariabilityStudyResults(
        all_runs_metrics=[
            pd.DataFrame({"run_num": [i], "epoch": [1], "val_accuracy": [v]})
            for i, v in zip(ids, vals)
        ],
        final_metrics={"val_accuracy": list(vals)},
        final_test_metrics=[],
        seed=7,
        run_seeds=[100, 200, 300],
        failed_runs=[i for i in (1, 2, 3) if i not in ids],
        metric_run_ids={"val_accuracy": list(ids)},
    )


def test_json_round_trip_preserves_run_ids_and_pairs_correctly():
    """v12 2.15: [1,2] vs [1,3] must pair only run 1 after a JSON round trip."""
    a, b = _mk([1, 2], [0.1, 0.2]), _mk([1, 3], [0.1, 0.3])
    ra = VariabilityStudyResults.from_json(a.to_json())
    rb = VariabilityStudyResults.from_json(b.to_json())
    assert ra.run_ids == [1, 2] and rb.run_ids == [1, 3]
    assert ra.run_seeds == [100, 200, 300]
    assert ra.metric_run_ids == {"val_accuracy": [1, 2]}
    assert align_paired(ra, rb, "val_accuracy")[0] == [1]
