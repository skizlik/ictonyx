"""Regression tests for v0.4.12 (Master Dev Guide v19; implementation guide §4).

Each test names the register ID it guards. Unless marked as a preservation
row, every test here fails on 0.4.11.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import ictonyx as ix
from ictonyx import ArraysDataHandler, ExperimentRunner, ModelConfig
from ictonyx.core import ScikitLearnModelWrapper

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def wine():
    return load_wine(return_X_y=True)


def _builder_factory(state):
    """Builder that fails on call numbers in state['fail'] and raises
    KeyboardInterrupt on call state['ki']."""

    def builder(conf):
        state["n"] += 1
        if state["n"] in state["fail"]:
            raise RuntimeError("boom")
        if state.get("ki") and state["n"] == state["ki"]:
            raise KeyboardInterrupt
        return ScikitLearnModelWrapper(
            RandomForestClassifier(n_estimators=5, random_state=conf.get("run_seed"))
        )

    return builder


# ---- 2.108: runner reuse (commit 01) -------------------------------------------
def test_runner_reuse_resets_run_ids(wine):
    X, y = wine
    st = {"n": 0, "fail": {3}}
    runner = ExperimentRunner(
        _builder_factory(st),
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
    )
    runner.run_study(num_runs=5)
    st["fail"] = set()
    r2 = runner.run_study(num_runs=5)
    ids, vals = r2.get_metric_values("val_accuracy", with_run_ids=True)
    assert ids == [1, 2, 3, 4, 5] and len(vals) == 5
