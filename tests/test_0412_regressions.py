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


# ---- 2.107: every random_state in the estimator tree (commit 05) -----------------
@pytest.mark.parametrize(
    "model",
    [
        make_pipeline(StandardScaler(), DecisionTreeClassifier(splitter="random", random_state=0)),
        # A meta-estimator with its own seed fixed too, inside a Pipeline: 0.4.11 saw
        # no top-level random_state and left both fixed. (A bare BaggingClassifier is
        # NOT a guard: its top-level random_state was already overridden in 0.4.11.)
        make_pipeline(
            StandardScaler(),
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=0), n_estimators=3, random_state=0
            ),
        ),
    ],
    ids=["pipeline", "pipeline_bagging"],
)
def test_inner_random_state_is_overridden_per_run(wine, model):
    X, y = wine
    r = ix.variability_study(model, data=(X, y), runs=5, seed=1, verbose=False)
    assert len(set(r.get_metric_values("val_accuracy"))) > 1


# ---- 3.49 / 3.59 / 2.136: the interval contract (commit 06) ----------------------
def test_hl_default_is_percentile_and_never_zero_width():
    from ictonyx.bootstrap import bootstrap_hodges_lehmann_ci

    rng = np.random.default_rng(0)
    for s in range(30):
        a = (30 + rng.binomial(6, 0.5, 20)) / 36
        b = (29 + rng.binomial(6, 0.5, 20)) / 36
        r = bootstrap_hodges_lehmann_ci(a, b, n_bootstrap=1000, random_state=s)
        assert r.method == "percentile"
        assert np.isnan(r.ci_lower) or r.ci_upper > r.ci_lower


def test_bca_bias_term_counts_ties_as_half():
    from ictonyx.bootstrap import _midrank_prop_below

    boot = np.array([0.0] * 20 + [1.0] * 60 + [2.0] * 20)
    assert _midrank_prop_below(boot, 1.0) == pytest.approx(0.5)


def test_two_sample_acceleration_matches_multisample_formula():
    from scipy.stats import norm

    from ictonyx.bootstrap import _two_sample_bca_ci

    rng = np.random.default_rng(1)
    g1, g2 = rng.exponential(1, 30), rng.exponential(1, 8)

    def f(x, y):
        return float(x.mean() - y.mean())

    num = den = 0.0
    for jk in (
        np.array([f(np.delete(g1, i), g2) for i in range(30)]),
        np.array([f(g1, np.delete(g2, j)) for j in range(8)]),
    ):
        m = len(jk)
        u = (m - 1) * (jk.mean() - jk)
        num += (u**3).sum() / m**3
        den += (u**2).sum() / m**2
    a_ref = num / (6 * den**1.5)
    boot = np.linspace(-1, 1, 2001) + f(g1, g2)  # symmetric: z0 = 0
    lo, _ = _two_sample_bca_ci(g1, g2, f, boot, f(g1, g2), 0.05)
    z = norm.ppf(0.025)
    q = norm.cdf(z / (1 - a_ref * z))
    assert lo == pytest.approx(np.percentile(boot, 100 * q), abs=2e-3)


# ---- 3.48: constant paired differences are undefined (commit 07) -----------------
@pytest.mark.parametrize(
    "a,b,zero_var",
    [
        ([0.9167] * 20, [0.8889] * 20, ["a", "b"]),
        (list(np.linspace(0.8, 0.9, 20)), list(np.linspace(0.8, 0.9, 20) - 0.05), []),
        ([0.9] * 8, [0.9] * 8, ["a", "b"]),
    ],
    ids=["both_constant", "both_vary_offset", "identical"],
)
def test_paired_constant_difference_is_undefined(a, b, zero_var):
    from ictonyx.analysis import compare_two_models

    r = compare_two_models(pd.Series(a), pd.Series(b), paired=True)
    assert np.isnan(r.p_value)
    assert not r.is_significant()
    assert r.confidence_interval is None
    assert r.sample_sizes["effective_n"] == 1
    assert r.assumption_details["zero_variance_groups"] == zero_var


def test_paired_and_unpaired_agree_on_deterministic_pair():
    from ictonyx.analysis import compare_two_models

    a, b = pd.Series([0.9167] * 20), pd.Series([0.8889] * 20)
    assert np.isnan(compare_two_models(a, b, paired=True).p_value)
    assert np.isnan(compare_two_models(a, b, paired=False).p_value)


def test_compare_models_deterministic_pipelines_not_significant(wine):
    X, y = wine
    c = ix.compare_models(
        [
            make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0)),
            make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=1, random_state=0)),
        ],
        data=(X, y),
        runs=20,
        seed=42,
        verbose=False,
    )
    assert np.isnan(c.overall_test.p_value)
    assert c.significant_comparisons == []
