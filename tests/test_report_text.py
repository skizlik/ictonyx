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
