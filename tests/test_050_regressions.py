"""Regression tests for v0.5.0 (Master Dev Guide v20 §8.1, Part 1).

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


# ---- 3.50: NaN-aware correction (commit 08) --------------------------------------
@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh"])
def test_correction_excludes_nan_from_family(method):
    from ictonyx.analysis import apply_multiple_comparison_correction

    got, desc = apply_multiple_comparison_correction([0.01, float("nan"), 0.02, 0.04], method)
    ref, _ = apply_multiple_comparison_correction([0.01, 0.02, 0.04], method)
    assert np.isnan(got[1])
    assert [got[0], got[2], got[3]] == pytest.approx(ref)
    assert "excluded" in desc


# ---- 3.51: one metric-direction table (commit 09) --------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("neg_log_loss", "higher"),
        ("val_neg_mean_squared_error", "higher"),
        ("val_mcc", "higher"),
        ("val_kappa", "higher"),
        ("val_fbeta", "higher"),
        ("val_ece", "lower"),
        ("val_wer", "lower"),
        ("val_accuracy_loss", "lower"),  # preservation row: already "lower" in 0.4.11
    ],
)
def test_metric_direction_extended(name, expected):
    from ictonyx.analysis import metric_direction

    assert metric_direction(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("val_iou", "maximize"),
        ("val_dice", "maximize"),
        ("val_map", "maximize"),
        ("val_mcc", "maximize"),
        ("val_accuracy_loss", "minimize"),
        ("val_loss", "minimize"),  # preservation row
    ],
)
def test_tuner_direction_uses_metric_direction(name, expected):
    from ictonyx.tuning import _resolve_direction

    assert _resolve_direction("auto", name) == expected


def test_tuner_unknown_metric_requires_direction():
    from ictonyx.exceptions import ConfigurationError
    from ictonyx.tuning import _resolve_direction

    with pytest.raises(ConfigurationError):
        _resolve_direction("auto", "val_wobble_index")
    assert _resolve_direction("maximize", "val_wobble_index") == "maximize"


# ---- 2.129: one route for k = 2 unpaired (commit 10) -----------------------------
def test_unpaired_k2_uses_compare_two_models_guard(wine):
    X, y = wine
    c = ix.compare_models(
        [RandomForestClassifier, RandomForestClassifier(max_depth=2)],
        data=(X, y),
        runs=5,
        seed=0,
        verbose=False,
        paired=False,
    )
    assert c.overall_test.test_name == "Insufficient Data"


# ==== Part 2 ======================================================================


# ---- 2.129: compare_multiple_models k = 2 (commit 10b) ---------------------------
def test_compare_multiple_models_k2_matches_compare_two_models():
    from ictonyx.analysis import compare_multiple_models, compare_two_models

    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.80, 0.02, 12))
    b = pd.Series(rng.normal(0.78, 0.02, 12))
    m = compare_multiple_models({"a": a, "b": b}, metric="val_accuracy", random_state=0)
    t = compare_two_models(a, b, paired=False, metric="val_accuracy", random_state=0)
    assert m.overall_test.test_name == t.test_name
    assert m.overall_test.p_value == pytest.approx(t.p_value)
    assert m.overall_test.confidence_interval == pytest.approx(t.confidence_interval)


# ---- 2.111: interrupt semantics (commit 04) --------------------------------------
def test_interrupted_study_reports_requested_count(wine, recwarn):
    X, y = wine
    st = {"n": 0, "fail": set(), "ki": 4}
    runner = ExperimentRunner(
        _builder_factory(st),
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
    )
    res = runner.run_study(num_runs=10)
    assert any("interrupted after 3 of 10" in str(w.message) for w in recwarn)
    assert res.n_runs == 3
    assert res.n_requested == 10
    assert res.stopped_early == "interrupted"
    assert "Stopped early (interrupted)" in res.summarize()


def test_failure_rate_stop_is_recorded(wine):
    X, y = wine
    st = {"n": 0, "fail": set(range(2, 100))}  # first run succeeds, the rest fail
    runner = ExperimentRunner(
        _builder_factory(st),
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
    )
    res = runner.run_study(num_runs=10)
    assert res.stopped_early == "failure_rate"
    assert res.n_requested == 10
    assert res.n_runs + len(res.failed_runs) < 10


def test_interrupt_in_compare_models_stops_the_call(wine):
    X, y = wine
    n = {"c": 0}

    def a(conf):
        n["c"] += 1
        if n["c"] == 4:
            raise KeyboardInterrupt
        return ScikitLearnModelWrapper(
            RandomForestClassifier(n_estimators=5, random_state=conf.get("run_seed"))
        )

    def b(conf):
        pytest.fail("model B must not train after Ctrl-C in model A")

    with pytest.raises(KeyboardInterrupt):
        ix.compare_models([a, b], data=(X, y), runs=10, seed=0, verbose=False)


# ---- 2.120 / 3.50 / 3.53: conclusions and the forest label (commit 11) -----------
@pytest.mark.parametrize("method", ["student_t", "welch_t"])
def test_t_paths_have_direction_aware_conclusion(method):
    from ictonyx.analysis import compare_two_models

    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.90, 0.01, 20))
    b = pd.Series(rng.normal(0.85, 0.01, 20))
    r = compare_two_models(a, b, paired=False, test_method=method, metric="val_accuracy")
    assert "Model A outperforms Model B" in r.conclusion
    r2 = compare_two_models(a, b, paired=False, test_method=method, metric="val_loss")
    assert "Model B outperforms Model A" in r2.conclusion
    assert r.detailed_interpretation


def test_mw_conclusion_on_nan_corrected_p():
    from ictonyx.analysis import StatisticalTestResult, _generate_mann_whitney_conclusion

    r = StatisticalTestResult(test_name="x", statistic=float("nan"), p_value=float("nan"))
    r.corrected_p_value = float("nan")
    r.correction_method = "holm"
    text = _generate_mann_whitney_conclusion(r, 0.05)
    assert text.startswith("Undefined") and "p=nan" not in text


# ---- 3.56 / 3.42: scope of inference (commit 12) ---------------------------------
def test_every_harness_conclusion_is_scoped_to_split(wine):
    from ictonyx.analysis import compare_multiple_models, compare_two_models

    rng = np.random.default_rng(1)
    a, b, c = (pd.Series(rng.normal(m, 0.02, 20)) for m in (0.80, 0.78, 0.75))
    for r in (
        compare_two_models(a, b, paired=True, metric="val_accuracy"),
        compare_two_models(a, b, paired=False, metric="val_accuracy"),
        compare_two_models(a, b, paired=False, test_method="welch_t", metric="val_accuracy"),
    ):
        assert "on this split" in r.conclusion, r.conclusion
    m = compare_multiple_models({"a": a, "b": b, "c": c}, metric="val_accuracy")
    assert "on this split" in m.overall_test.conclusion
    assert all("on this split" in t.conclusion for t in m.pairwise_comparisons.values())
    X, y = wine
    s = ix.variability_study(RandomForestClassifier, data=(X, y), runs=10, seed=1, verbose=False)
    assert "on this split" in s.test_against_null(null_value=0.5).conclusion


def test_summarize_prints_evaluation_se_not_granularity_claim(wine):
    X, y = wine
    s = ix.variability_study(RandomForestClassifier, data=(X, y), runs=5, seed=1, verbose=False)
    text = s.summarize()
    assert "Evaluation-set sampling error" in text
    assert "no number of runs resolves finer" not in text


def test_constant_difference_stated_in_evaluation_examples(wine):
    X, y = wine
    c = ix.compare_models(
        [
            make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0)),
            make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=1, random_state=0)),
        ],
        data=(X, y),
        runs=6,
        seed=42,
        verbose=False,
    )
    assert "val examples)" in c.overall_test.conclusion


def test_run_order_check_is_a_diagnostic_not_an_assumption():
    from ictonyx.analysis import mann_whitney_test

    rng = np.random.default_rng(2)
    r = mann_whitney_test(pd.Series(rng.normal(0, 1, 20)), pd.Series(rng.normal(0, 1, 20)))
    assert "independence" not in r.assumptions_met
    assert "run_order_autocorrelation" in r.assumption_details


# ---- 3.55: test_above_chance is scoped (commit 13) -------------------------------
def test_above_chance_warns_conditional_scope(wine):
    X, y = wine
    s = ix.variability_study(RandomForestClassifier, data=(X, y), runs=10, seed=1, verbose=False)
    with pytest.warns(UserWarning, match="not a test of population accuracy"):
        r = s.test_above_chance()
    assert "on this evaluation set" in r.conclusion
    assert any("population accuracy" in w for w in r.warnings)


# ---- 3.62: one-sample wording (commit 14) ----------------------------------------
@pytest.mark.parametrize(
    "alt,verb", [("two-sided", "differ from"), ("greater", "sit above"), ("less", "sit below")]
)
def test_one_sample_wording_follows_alternative(alt, verb):
    from ictonyx.analysis import _wilcoxon_signed_rank_impl

    rng = np.random.default_rng(8)
    center = 0.9 if alt != "less" else 0.1
    r = _wilcoxon_signed_rank_impl(pd.Series(rng.normal(center, 0.01, 20)), 0.5, alt)
    assert verb in r.conclusion and "median" not in r.conclusion


# ---- 2.134 / 3.58: visible deprecations; "parametric" (commit 15) ----------------
def test_parametric_is_deprecated_with_visible_warning():
    from ictonyx.analysis import compare_two_models
    from ictonyx.exceptions import IctonyxFutureWarning

    rng = np.random.default_rng(9)
    a, b = pd.Series(rng.normal(0.8, 0.02, 20)), pd.Series(rng.normal(0.78, 0.02, 20))
    with pytest.warns(IctonyxFutureWarning, match="register 3.58") as rec:
        compare_two_models(a, b, paired=False, test_method="parametric")
    w = [x for x in rec if isinstance(x.message, IctonyxFutureWarning)][0].message
    assert isinstance(w, FutureWarning) and isinstance(w, UserWarning)


# ---- 2.113: grid keys and verbose (commit 16) ------------------------------------
def test_grid_accepts_list_and_dict_values(wine):
    from ictonyx.runners import GridStudyResults, run_grid_study

    X, y = wine

    def builder(conf):
        return ScikitLearnModelWrapper(
            RandomForestClassifier(n_estimators=5, random_state=conf.get("run_seed"))
        )

    grid = run_grid_study(
        builder,
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        {"tags": [[1, 2], [3]], "opts": [{"a": 1}]},
        num_runs=3,
        use_process_isolation=False,
        verbose=False,
        seed=0,
    )
    assert grid.n_configurations == 2
    assert grid.get_results_for_config({"tags": [1, 2], "opts": {"a": 1}}).n_runs == 3
    assert GridStudyResults._config_key({"lr": 0.1, "bs": 8}) == (("bs", 8), ("lr", 0.1))


def test_grid_verbose_false_is_quiet(wine, caplog):
    import logging

    from ictonyx.runners import run_grid_study

    X, y = wine

    def builder(conf):
        return ScikitLearnModelWrapper(
            RandomForestClassifier(n_estimators=5, random_state=conf.get("run_seed"))
        )

    # Pin the ictonyx logger at INFO so the result cannot depend on what earlier
    # tests left behind; only run_grid_study's own verbose handling may quiet it.
    with caplog.at_level(logging.INFO, logger="ictonyx"):
        run_grid_study(
            builder,
            ArraysDataHandler(X, y),
            ModelConfig({"epochs": 1}),
            {"x": [1, 2]},
            num_runs=3,
            use_process_isolation=False,
            verbose=False,
            seed=0,
        )
    noisy = [
        r for r in caplog.records if r.name.startswith("ictonyx") and r.levelno == logging.INFO
    ]
    assert noisy == [], [r.getMessage() for r in noisy][:3]


# ---- 2.115: paired deltas align on run id (commit 17) ----------------------------
def test_paired_deltas_align_on_run_id():
    import matplotlib

    matplotlib.use("Agg")
    from ictonyx.runners import VariabilityStudyResults

    def mk(ids, vals):
        return VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": list(vals)},
            final_test_metrics=[],
            seed=1,
            metric_run_ids={"val_accuracy": list(ids)},
            _run_ids=list(ids),
        )

    a = mk([1, 2, 4, 5], [0.90, 0.91, 0.92, 0.93])
    b = mk([1, 2, 3, 5], [0.80, 0.81, 0.99, 0.83])
    with pytest.warns(UserWarning, match="align_paired"):
        fig = ix.plotting.plot_paired_deltas(a, b, metric="val_accuracy", show=False)
    pts = fig.axes[0].collections[0].get_offsets()
    assert list(pts[:, 0]) == [1, 2, 5]  # run ids present in both studies
    assert list(pts[:, 1]) == pytest.approx([0.1, 0.1, 0.1])  # never run 4 against run 3


# ---- 2.119: persistence round-trips every field (commit 18) ----------------------
def _full_results():
    from ictonyx.runners import VariabilityStudyResults

    return VariabilityStudyResults(
        all_runs_metrics=[pd.DataFrame({"epoch": [1], "val_accuracy": [0.9], "run_num": [2]})],
        final_metrics={"val_accuracy": [0.9]},
        final_test_metrics=[{"accuracy": 0.8, "run_id": 2}],
        seed=7,
        run_seeds=[11, 12],
        failed_runs=[1],
        metric_run_ids={"val_accuracy": [2]},
        split_sizes={"train": 100, "val": 20, "test": 30},
        retried_runs=[1],
        num_runs_requested=2,
        stopped_early="interrupted",
        _run_ids=[2],
    )


def _assert_same(a, b, skip=()):
    import dataclasses

    for f in dataclasses.fields(a):
        if f.name in skip:
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name == "all_runs_metrics":
            assert len(va) == len(vb) and all(x.equals(y) for x, y in zip(va, vb))
        else:
            assert va == vb, f.name


def test_pickle_round_trip_preserves_every_field(tmp_path):
    from ictonyx.runners import VariabilityStudyResults

    r = _full_results()
    r.save(str(tmp_path / "r.pkl"))
    _assert_same(r, VariabilityStudyResults.load(str(tmp_path / "r.pkl")))


def test_json_round_trip_preserves_every_field_but_histories():
    from ictonyx.runners import VariabilityStudyResults

    r = _full_results()
    back = VariabilityStudyResults.from_json(r.to_json())
    _assert_same(r, back, skip=("all_runs_metrics",))


# ---- 2.112 / 2.121: isolation checks (commit 19) ---------------------------------
def test_isolation_rejects_unpicklable_data_up_front(wine):
    import threading

    X, y = wine

    class LockedHandler(ArraysDataHandler):
        def load(self, *a, **k):
            out = super().load(*a, **k)
            out["train_data"] = (out["train_data"], threading.Lock())
            return out

    with pytest.raises(ValueError, match="cannot be serialised"):
        ExperimentRunner(
            _module_level_builder,
            LockedHandler(X, y),
            ModelConfig({"epochs": 1}),
            seed=1,
            verbose=False,
            use_process_isolation=True,
        )


def test_isolated_first_run_error_is_named(wine, monkeypatch):
    from ictonyx.exceptions import ExperimentError

    X, y = wine
    runner = ExperimentRunner(
        _module_level_builder,
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        seed=1,
        verbose=False,
        use_process_isolation=True,
    )
    monkeypatch.setattr(
        runner.memory_manager,
        "run_isolated",
        lambda *a, **k: {"success": False, "error": "ValueError: bad config"},
    )
    with pytest.raises(ExperimentError, match="bad config"):
        runner.run_study(num_runs=3)


def _module_level_builder(conf):
    return ScikitLearnModelWrapper(
        RandomForestClassifier(n_estimators=5, random_state=conf.get("run_seed"))
    )


# ---- 3.61: provenance (commit 20) ------------------------------------------------
def test_result_records_scipy_provenance():
    import scipy

    from ictonyx.analysis import compare_two_models

    rng = np.random.default_rng(10)
    a, b = pd.Series(rng.normal(0.8, 0.02, 20)), pd.Series(rng.normal(0.78, 0.02, 20))
    for paired in (True, False):
        p = compare_two_models(a, b, paired=paired).provenance
        assert p["scipy"] == scipy.__version__ and p["n"] == 20 and "has_ties" in p
