"""v0.4.10 C7: 'auto' retired; compare_models and compare_results run the same unpaired test.

Closes v12 2.17, 2.70.
"""

import warnings

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import ictonyx as ix
from ictonyx.analysis import compare_two_models


def test_compare_models_public_path_emits_no_deprecation(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ix.compare_models(
            [LogisticRegression, DecisionTreeClassifier],
            data=(X, y),
            runs=6,
            seed=0,
            verbose=False,
        )


def test_auto_is_rejected():
    import pandas as pd

    with pytest.raises(ValueError, match="removed in v0.4.10"):
        compare_two_models(pd.Series([1.0, 2, 3]), pd.Series([2.0, 3, 4]), ci_target="auto")
    with pytest.raises(ValueError, match="removed in v0.4.10"):
        compare_two_models(pd.Series([1.0, 2, 3]), pd.Series([2.0, 3, 4]), test_method="auto")


@pytest.mark.slow
def test_unpaired_entry_points_agree(X, y):
    """compare_results(paired=False) and compare_models(paired=False) must run the same test."""
    ra = ix.variability_study(LogisticRegression, data=(X, y), runs=8, seed=3, verbose=False)
    rb = ix.variability_study(DecisionTreeClassifier, data=(X, y), runs=8, seed=3, verbose=False)
    via_results = ix.compare_results(ra, rb, metric="val_accuracy", paired=False)
    via_models = ix.compare_models(
        [LogisticRegression, DecisionTreeClassifier],
        data=(X, y),
        runs=8,
        seed=3,
        verbose=False,
        paired=False,
    )
    assert "Mann-Whitney" in via_results.overall_test.test_name
    assert "Mann-Whitney" in via_models.overall_test.test_name
    # Both sklearn models are deterministic under the fixed split, so both
    # paths return the undefined (NaN) result; they must still agree.
    assert via_results.overall_test.p_value == pytest.approx(
        via_models.overall_test.p_value, nan_ok=True
    )
