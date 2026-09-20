"""v0.4.10 C17: warn once, resolve the metric across all studies, honest bootstrap.

Closes v12 2.27 (warn-once), 2.29, 2.30.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import ictonyx as ix
from ictonyx.bootstrap import bootstrap_effect_size_ci
from ictonyx.exceptions import ConfigurationError


def test_runs_warning_fires_once_per_comparison(X, y, recwarn):
    ix.compare_models(
        [LogisticRegression, DecisionTreeClassifier], data=(X, y), runs=6, seed=0, verbose=False
    )
    n = sum("may be insufficient" in str(w.message) for w in recwarn)
    assert n == 1, f"expected one runs<20 warning, got {n}"


def test_suppress_switch_never_reaches_model_config(X, y):
    """The private kwarg is popped before ModelConfig sees it."""
    r = ix.variability_study(
        LogisticRegression, data=(X, y), runs=2, seed=0, verbose=False, _suppress_runs_warning=True
    )
    assert r.n_runs == 2  # would have raised/warned on a bogus constructor kwarg otherwise


def test_metric_resolved_across_all_studies(X, y):
    """A regressor first and a classifier second must not pick val_r2 (v12 2.30)."""
    y_f = y.astype(float)
    with pytest.raises(ConfigurationError, match="no metric common to all models"):
        ix.compare_models(
            [LinearRegression, DecisionTreeClassifier], data=(X, y_f), runs=4, seed=0, verbose=False
        )


def test_zero_variance_resamples_are_dropped_not_zeroed():
    """Glass's delta divides by group 2's SD alone. With one non-zero value in a
    group of 10, ~35% of resamples omit it and have zero variance; those must be
    dropped (NaN) rather than counted as an effect of 0 (v12 2.29)."""
    rng = np.random.default_rng(0)
    g1 = rng.normal(2.0, 0.5, 30)
    g2 = np.array([0.0] * 9 + [1.0])
    with pytest.warns(UserWarning, match="bootstrap replicates were non-finite"):
        ci = bootstrap_effect_size_ci(
            g1, g2, n_bootstrap=400, method="percentile", pooled=False, random_state=0
        )
    assert np.isfinite(ci.ci_lower) and np.isfinite(ci.ci_upper)
    assert ci.ci_lower > 0  # not dragged toward zero by fake d=0 replicates
