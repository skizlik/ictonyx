"""v0.4.11 regression tests: the library must not say something untrue.

One test per finding (Master Dev Guide v16):

* 3.25 -- conclusion text agrees with ``is_significant()`` after correction
* 3.26 -- constant non-zero paired differences are decisive, not inconclusive
* 3.37 -- direction wording follows metric direction (loss vs accuracy)
* 3.40 -- unpaired path does not manufacture a p-value from two constant groups
* 3.36 -- README example does not fit a transform on the full dataset
* 3.32 -- ``required_runs`` docstrings give the right reason (ties, not spread)
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ictonyx import analysis as A

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 3.37
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("val_loss", "lower"),
        ("test_mae", "lower"),
        ("rmse", "lower"),
        ("accuracy", "higher"),
        ("val_f1_macro", "higher"),
        ("r2", "higher"),
        ("wobble", "unknown"),
        (None, "unknown"),
    ],
)
def test_metric_direction_table(name, expected):
    assert A.metric_direction(name) == expected


def _paired_fixture():
    rng = np.random.default_rng(7)
    a = pd.Series(rng.normal(0.30, 0.01, 20))  # A has the HIGHER values
    b = pd.Series(rng.normal(0.25, 0.01, 20))
    return a, b


def test_direction_wording_flips_for_loss_metrics():
    a, b = _paired_fixture()
    acc = A.paired_wilcoxon_test(a, b, metric="val_accuracy")
    loss = A.paired_wilcoxon_test(a, b, metric="val_loss")
    assert acc.is_significant() and loss.is_significant()
    # Same data, same effect size, opposite winner named.
    assert acc.effect_size == pytest.approx(loss.effect_size)
    assert "Model A outperforms" in acc.conclusion
    assert "Model B outperforms" in loss.conclusion
    assert "lower val_loss" in loss.conclusion


def test_unknown_metric_uses_neutral_wording():
    a, b = _paired_fixture()
    r = A.paired_wilcoxon_test(a, b)  # no metric
    assert "outperforms" not in r.conclusion
    assert "higher" in r.conclusion and "Model A" in r.conclusion
    r2 = A.paired_wilcoxon_test(a, b, metric="wobble_index")
    assert "outperforms" not in r2.conclusion


def test_forest_plot_colour_follows_metric_direction():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from ictonyx import plotting as P

    a, b = _paired_fixture()
    from ictonyx.analysis import ModelComparisonResults, compare_two_models

    def build(metric):
        # Same shape compare_models() returns for two models (paired, with CI).
        t = compare_two_models(a, b, paired=True, metric=metric)
        assert t.is_significant() and t.confidence_interval is not None
        return ModelComparisonResults(
            overall_test=t,
            raw_data={"A": a, "B": b},
            pairwise_comparisons={"A_vs_B": t},
            significant_comparisons=["A_vs_B"],
            correction_method="none",
            n_models=2,
            metric=metric,
        )

    cmp_acc, cmp_loss = build("val_accuracy"), build("val_loss")

    fig_acc = P.plot_comparison_forest(
        cmp_acc, baseline_model="B", metric="val_accuracy", show=False
    )
    fig_loss = P.plot_comparison_forest(cmp_loss, baseline_model="B", metric="val_loss", show=False)

    def bar_colour(fig):
        ax = fig.axes[0]
        # errorbar's line collection carries the ecolor
        cols = [c for c in ax.collections if hasattr(c, "get_color")]
        assert cols, "no errorbar drawn"
        return tuple(np.asarray(cols[0].get_color()[0]).round(3))

    from ictonyx import settings

    good = tuple(np.asarray(matplotlib.colors.to_rgba(settings.THEME["test"])).round(3))
    bad = tuple(np.asarray(matplotlib.colors.to_rgba(settings.THEME["significant"])).round(3))
    assert bar_colour(fig_acc) == good  # A above B on accuracy: better
    assert bar_colour(fig_loss) == bad  # A above B on loss: worse
    matplotlib.pyplot.close("all")


# --------------------------------------------------------------------------
# 3.26
# --------------------------------------------------------------------------
def test_constant_nonzero_paired_difference_is_decisive():
    a = pd.Series(np.linspace(0.80, 0.90, 20))
    b = a - 0.05  # A beats B by exactly 0.05 on every run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = A.paired_wilcoxon_test(a, b)
    assert "inconclusive" not in r.test_name.lower()
    assert r.p_value == pytest.approx(2 * 0.5**20)
    assert r.effect_size == pytest.approx(1.0)
    assert r.is_significant()
    assert any("constant" in w.lower() for w in r.warnings)

    # Sign is respected.
    r2 = A.paired_wilcoxon_test(b, a)
    assert r2.effect_size == pytest.approx(-1.0)


def test_all_zero_paired_differences_stay_undefined():
    a = pd.Series(np.linspace(0.80, 0.90, 20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = A.paired_wilcoxon_test(a, a.copy())
    assert np.isnan(r.p_value)
    assert not r.is_significant()


# --------------------------------------------------------------------------
# 3.40
# --------------------------------------------------------------------------
def test_unpaired_both_constant_is_undefined_like_paired():
    c1 = pd.Series([0.80] * 20)
    c2 = pd.Series([0.70] * 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = A.mann_whitney_test(c1, c2)
    assert np.isnan(r.p_value)
    assert not r.is_significant()
    assert "undefined" in r.test_name.lower() or "inconclusive" in r.test_name.lower()
    assert r.assumption_details.get("zero_variance_groups") == ["group1", "group2"]


def test_unpaired_one_constant_group_warns_but_tests():
    rng = np.random.default_rng(1)
    c = pd.Series([0.80] * 20)
    n = pd.Series(rng.normal(0.75, 0.02, 20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = A.mann_whitney_test(c, n)
    assert np.isfinite(r.p_value)
    assert r.assumption_details.get("zero_variance_groups") == ["group1"]
    assert any("deterministic" in w.lower() for w in r.warnings)
    # A constant series is not "autocorrelated".
    assert not any("autocorrelation" in w.lower() for w in r.warnings)


# --------------------------------------------------------------------------
# 3.25
# --------------------------------------------------------------------------
@pytest.mark.parametrize("correction", ["holm", "bonferroni", "fdr_bh"])
def test_conclusion_agrees_with_is_significant_after_correction(correction):
    """Four groups chosen so at least one pair is raw-significant but not
    corrected-significant. The sentence must follow the corrected verdict."""
    rng = np.random.default_rng(3)
    groups = {
        "a": pd.Series(rng.normal(0.0, 1, 20)),
        "b": pd.Series(rng.normal(0.9, 1, 20)),
        "c": pd.Series(rng.normal(0.4, 1, 20)),
        "d": pd.Series(rng.normal(0.4, 1, 20)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = A.compare_multiple_models(groups, correction_method=correction)

    flipped = [
        t
        for t in res.pairwise_comparisons.values()
        if t.corrected_p_value is not None and t.p_value < 0.05 <= t.corrected_p_value
    ]
    if correction != "fdr_bh":  # BH is not conservative enough to flip this fixture
        assert flipped, "fixture no longer produces a raw/corrected disagreement"

    for t in res.pairwise_comparisons.values():
        says_no = "no statistically significant" in t.conclusion.lower()
        assert says_no == (not t.is_significant()), t.conclusion
        # The printed p must be the one the verdict used.
        shown = float(re.search(r"p=([0-9.]+)", t.conclusion).group(1))
        assert shown == pytest.approx(t.corrected_p_value, abs=5e-5)
        assert t.detailed_interpretation.startswith(t.conclusion)


# --------------------------------------------------------------------------
# 3.32 -- documentation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "func",
    [A.required_runs, A.required_runs_paired, A.minimum_detectable_effect],
)
def test_power_docstrings_name_ties_not_spread(func):
    doc = func.__doc__
    assert "over-estimate" not in doc and "overestimates spread" not in doc
    assert "ties" in doc.lower()


# --------------------------------------------------------------------------
# 3.36 -- documentation, and 2.94 -- model naming
# --------------------------------------------------------------------------
def test_readme_example_has_no_pre_split_transform():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks
    for block in blocks:
        # A transform fitted on the whole dataset before Ictonyx splits it.
        assert not re.search(r"\.fit_transform\(\s*data\.data", block), block
        assert "StandardScaler().fit_transform(X)" not in block


def test_pipeline_and_partial_builders_are_named_by_estimator():
    from functools import partial

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ictonyx.api import _get_model_name

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    assert _get_model_name(pipe) == "Pipeline(LogisticRegression)"
    assert _get_model_name(partial(LogisticRegression, C=0.1)) == "LogisticRegression"
    assert _get_model_name(LogisticRegression) == "LogisticRegression"
