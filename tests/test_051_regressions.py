"""Regression tests for v0.5.1.

Each test fails on 0.5.0.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import ictonyx as ix  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def noisy_study():
    # Noisy labels and a small forest: validation and test accuracies differ
    # run to run, so a histogram of one cannot pass for the other by chance.
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=4, flip_y=0.2, random_state=0
    )
    return ix.variability_study(
        RandomForestClassifier(n_estimators=5), data=(X, y), runs=6, seed=3, verbose=False
    )


def _record_histplots(monkeypatch):
    import ictonyx.plotting as P

    calls = []
    real = P.sns.histplot

    def spy(*args, **kwargs):
        data = args[0] if args else kwargs.get("y", kwargs.get("x"))
        calls.append((kwargs.get("label"), list(np.asarray(data, dtype=float))))
        return real(*args, **kwargs)

    monkeypatch.setattr(P.sns, "histplot", spy)
    return calls


# ---- histogram labelled "Validation" holds validation values ----------------------
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_variability_summary_histograms_match_their_labels(noisy_study, monkeypatch, orientation):
    r = noisy_study
    val = r.get_metric_values("val_accuracy")
    test = r.get_metric_values("test_accuracy")
    assert val != test  # precondition: the two series are distinguishable
    calls = _record_histplots(monkeypatch)
    fig = ix.plot_variability_summary(
        results=r, metric="accuracy", histogram_orientation=orientation, show=False
    )
    plt.close(fig)
    drawn = dict(calls)
    assert [label for label, _ in calls] == ["Validation", "Test"]
    assert drawn["Validation"] == pytest.approx(val)
    assert drawn["Test"] == pytest.approx(test)


def test_variability_summary_boxplot_uses_validation_values(noisy_study):
    r = noisy_study
    fig = ix.plot_variability_summary(
        results=r, metric="accuracy", show_histogram=False, show_boxplot=True, show=False
    )
    ax = fig.axes[-1]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Val", "Test"]
    # Each box's median line sits at the median of the series it summarises.
    medians = [line.get_ydata()[0] for line in ax.lines if len(set(line.get_ydata())) == 1]
    for key in ("val_accuracy", "test_accuracy"):
        m = np.median(r.get_metric_values(key))
        assert any(abs(m - v) < 1e-9 for v in medians), (key, m, medians)
    plt.close(fig)


def test_variability_summary_labels_a_non_validation_series_by_its_split(monkeypatch):
    # With no validation split the preferred series is test_*: the histogram
    # must call it "Test", and must not draw it a second time.
    X, y = make_classification(n_samples=200, random_state=0)
    r = ix.variability_study(
        RandomForestClassifier(n_estimators=5),
        data=ix.ArraysDataHandler(X, y, val_split=0.0, test_split=0.2),
        runs=3,
        seed=0,
        verbose=False,
    )
    assert "val_accuracy" not in r.final_metrics
    calls = _record_histplots(monkeypatch)
    plt.close(ix.plot_variability_summary(results=r, metric="accuracy", show=False))
    assert [label for label, _ in calls] == ["Test"]
    assert calls[0][1] == pytest.approx(r.get_metric_values("test_accuracy"))


# ---- README: citations and examples --------------------------------------------
def test_readme_cites_seed_and_benchmark_variance_sources():
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    measure = text[text.index("## What does Ictonyx measure?") : text.index("## Quick start")]
    for ref in (
        "Picard (2021)",
        "arXiv:2109.08203",
        "Bouthillier et al. (2021)",
        "arXiv:2103.03098",
    ):
        assert ref in measure, ref


def test_readme_examples_use_digits():
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    quick = text[text.index("## Quick start") : text.index("## Comparing two models")]
    compare = text[text.index("## Comparing two models") : text.index("## Process isolation")]
    for section in (quick, compare):
        assert "load_digits" in section
        assert "load_wine" not in section and "load_breast_cancer" not in section
    # The quick start no longer needs a BatchNormalization explanation.
    assert "BatchNormalization" not in quick
