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


# ---- 2.109: one metric accessor (commit 02) --------------------------------------
def test_get_metric_values_routes_test_prefix_with_run_ids(wine):
    X, y = wine
    r = ix.variability_study(DecisionTreeClassifier, data=(X, y), runs=4, seed=1, verbose=False)
    ids, vals = r.get_metric_values("test_accuracy", with_run_ids=True)
    assert ids == [1, 2, 3, 4] and len(vals) == 4


@pytest.mark.parametrize("fn", ["plot_run_distribution", "plot_run_strip"])
def test_default_metric_plots_return_a_figure(wine, fn):
    import matplotlib

    matplotlib.use("Agg")
    X, y = wine
    r = ix.variability_study(DecisionTreeClassifier, data=(X, y), runs=4, seed=1, verbose=False)
    assert getattr(ix.plotting, fn)(r, show=False) is not None


def test_readme_plot_call_on_default_study(wine):
    import matplotlib

    matplotlib.use("Agg")
    X, y = wine
    r = ix.variability_study(DecisionTreeClassifier, data=(X, y), runs=4, seed=1, verbose=False)
    ix.plot_variability_summary(results=r, metric="accuracy", show=False)


@pytest.mark.parametrize("k", [2, 3])
def test_compare_models_accepts_test_metric(wine, k):
    X, y = wine
    models = [RandomForestClassifier, ExtraTreesClassifier, RandomForestClassifier(max_depth=2)][:k]
    c = ix.compare_models(
        models, data=(X, y), runs=6, seed=0, verbose=False, metric="test_accuracy"
    )
    assert c.metric == "test_accuracy"


# ---- 2.110: resume retries failed runs (commit 03) -------------------------------
def test_resume_retries_failed_run_once(wine):
    X, y = wine
    d = tempfile.mkdtemp()
    st = {"n": 0, "fail": {3}, "ki": 5}
    ExperimentRunner(
        _builder_factory(st),
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
    ).run_study(num_runs=6, checkpoint_dir=d)
    st.update(n=100, fail=set(), ki=None)
    res = ExperimentRunner(
        _builder_factory(st),
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
    ).run_study(num_runs=6, checkpoint_dir=d)
    assert sorted(res.run_ids) == [1, 2, 3, 4, 5, 6]
    assert res.failed_runs == []
    assert res.retried_runs == [3]
    assert res.n_requested == 6
