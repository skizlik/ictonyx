"""v0.4.10 C18: the docstrings and README say what the inference is conditional on."""

import inspect
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

import ictonyx as ix
from ictonyx.analysis import check_convergence, compare_multiple_models, compare_two_models
from ictonyx.api import compare_models, compare_results
from ictonyx.bootstrap import bootstrap_mean_difference_ci
from ictonyx.runners import VariabilityStudyResults

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "obj,phrase",
    [
        (compare_models, "does **not** in general increase power"),
        (compare_models, "conditional on this fixed"),
        (compare_results, "conditional on this fixed"),
        (VariabilityStudyResults.test_against_null, "bounded by"),
        (VariabilityStudyResults.test_above_chance, "under the null hypothesis"),
        (VariabilityStudyResults.get_epoch_statistics, "pointwise"),
        (check_convergence, "accepts the\n    null"),
        (compare_multiple_models, "Holm\n    correction controls"),
        (bootstrap_mean_difference_ci, "Coverage note"),
    ],
)
def test_docstring_states_scope(obj, phrase):
    doc = " ".join(inspect.getdoc(obj).split())
    assert " ".join(phrase.split()) in doc


def test_no_internal_ids_in_readme():
    text = (ROOT / "README.md").read_text()
    for bad in ("v12 ", "v11 ", "v10 ", "v9 ", "IX-EVAL", "(closes", "Master Dev Guide"):
        assert bad not in text, bad


def test_readme_states_split_conditionality():
    text = (ROOT / "README.md").read_text()
    assert "on this train/validation split" in text
    assert "strong statistical significance" not in text
    assert "one evaluation sample" in text


def test_summarize_reports_split_sizes(X, y):
    r = ix.variability_study(LogisticRegression, data=(X, y), runs=2, seed=0, verbose=False)
    s = r.summarize()
    assert "Data split: train" in s
    assert "Metric granularity" in s
    assert r.split_sizes["train"] + r.split_sizes["val"] + r.split_sizes["test"] == len(X)
