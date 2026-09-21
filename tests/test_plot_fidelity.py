"""v0.4.10 C21: plot_variability_summary defects; forest plot draws the computed interval."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import ictonyx.plotting as P
from ictonyx.analysis import StatisticalTestResult
from ictonyx.runners import VariabilityStudyResults


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    runs = [
        pd.DataFrame(
            {
                "run_num": [i] * 4,
                "epoch": [1, 2, 3, 4],
                "train_accuracy": rng.random(4),
                "val_accuracy": rng.random(4),
                "loss": rng.random(4),
            }
        )
        for i in range(1, 6)
    ]
    return VariabilityStudyResults(
        all_runs_metrics=runs,
        final_metrics={"val_accuracy": list(rng.random(5)), "loss": list(rng.random(5))},
        final_test_metrics=[],
        seed=1,
    )


@pytest.mark.parametrize("hist,box", [(True, True), (True, False), (False, True), (False, False)])
def test_summary_plot_every_allocated_axis_has_data(results, hist, box):
    fig = P.plot_variability_summary(
        results=results, metric="accuracy", show_histogram=hist, show_boxplot=box, show=False
    )
    assert len(fig.axes) == 1 + int(hist) + int(box)
    assert all(ax.has_data() for ax in fig.axes)


def test_kind_dispatch_forwards_metric(results, monkeypatch):
    seen = {}
    real = P.plot_run_distribution

    def spy(res, metric=None, **kw):
        seen["metric"] = metric
        return real(res, metric=metric, **kw)

    monkeypatch.setattr(P, "plot_run_distribution", spy)
    P.plot_variability_summary(results=results, kind="distribution", metric="loss", show=False)
    assert seen["metric"] == "loss"


def test_ragged_runs_keep_the_mean_line(results):
    full = P.plot_variability_summary(results=results, metric="accuracy", show=False)
    ragged = results
    ragged.all_runs_metrics[0] = ragged.all_runs_metrics[0].iloc[:2]  # one run stopped early
    fig = P.plot_variability_summary(results=ragged, metric="accuracy", show=False)
    assert len(fig.axes[0].lines) == len(full.axes[0].lines)
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert "Mean Val" in labels and "Mean Train" in labels


def test_positional_results_object_is_accepted(results):
    fig = P.plot_variability_summary(results, metric="accuracy", show=False)
    assert fig is not None and fig.axes[0].has_data()


@pytest.mark.parametrize("name", ["val_accuracy", "accuracy"])
def test_full_and_base_metric_names_both_resolve(results, name):
    fig = P.plot_variability_summary(results=results, metric=name, show=False)
    assert fig is not None and fig.axes[0].has_data()


def test_dpi_docstring_matches_default():
    import inspect

    assert inspect.signature(P.plot_variability_summary).parameters["dpi"].default == 300
    assert "default ``300``" in inspect.getdoc(P.plot_variability_summary)


def _forest_with(pairwise, raw):
    calls = []
    import matplotlib.axes

    real = matplotlib.axes.Axes.errorbar

    def spy(self, x, y, xerr=None, **kw):
        calls.append((float(x), np.asarray(xerr, dtype=float).ravel().tolist(), kw.get("ecolor")))
        return real(self, x, y, xerr=xerr, **kw)

    matplotlib.axes.Axes.errorbar = spy
    try:
        P.plot_comparison_forest(
            {"raw_data": raw, "pairwise_comparisons": pairwise}, "B", show=False
        )
    finally:
        matplotlib.axes.Axes.errorbar = real
    return calls


def test_forest_draws_the_computed_interval_at_its_point_estimate():
    r = StatisticalTestResult(test_name="t", statistic=0.0, p_value=0.01)
    r.confidence_interval = (0.1, 0.5)
    r.point_estimate = 0.2  # asymmetric: 0.1 below, 0.3 above
    calls = _forest_with({"A_vs_B": r}, {"A": [0.9, 0.8, 0.85], "B": [0.6, 0.65, 0.7]})
    ((center, xerr, ecolor),) = calls
    assert center == pytest.approx(0.2)
    assert xerr == pytest.approx([0.1, 0.3])  # [center - lo, hi - center]
    assert ecolor != "gray"  # corrected significance available and True


def test_forest_flips_orientation_when_key_is_reversed():
    r = StatisticalTestResult(test_name="t", statistic=0.0, p_value=0.5)
    r.confidence_interval = (-0.5, -0.1)  # B - A
    r.point_estimate = -0.3
    calls = _forest_with({"B_vs_A": r}, {"A": [0.9, 0.8, 0.85], "B": [0.6, 0.65, 0.7]})
    ((center, xerr, ecolor),) = calls
    assert center == pytest.approx(0.3)  # drawn as A - B
    assert xerr == pytest.approx([0.2, 0.2])
    assert ecolor == "gray"  # not significant


def test_forest_fallback_uses_welch_satterthwaite_df():
    from scipy import stats

    a = np.array([0.9, 0.8, 0.85, 0.7, 0.95])
    b = np.array([0.6, 0.65, 0.7, 0.62, 0.68, 0.71, 0.64])
    calls = _forest_with({}, {"A": a, "B": b})
    ((center, xerr, ecolor),) = calls
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    df_w = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    half = stats.t.ppf(0.975, df_w) * np.sqrt(v1 + v2)
    assert center == pytest.approx(a.mean() - b.mean())
    assert xerr == pytest.approx([half, half])
    assert ecolor == "gray"  # no corrected test: no colour claim
