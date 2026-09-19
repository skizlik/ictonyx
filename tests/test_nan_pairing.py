"""v0.4.10 C8: list-wise NaN handling; per-metric run ids.

Closes v12 2.18, 2.19, 2.25.
"""

import numpy as np
import pandas as pd
import pytest
from _spy_wrappers import _Base

import ictonyx as ix
from ictonyx.analysis import friedman_test, paired_wilcoxon_test
from ictonyx.bootstrap import bootstrap_paired_difference_ci
from ictonyx.core import TrainingResult


def test_paired_bootstrap_drops_rows_listwise():
    """Reviewer-2 reproduction. Values are identical wherever both are observed,
    so the true paired difference is 0. Per-group NaN removal gave 0.2."""
    g1 = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
    g2 = [1.0, 2.0, 3.0, np.nan, 5.0, 6.0]
    with pytest.warns(UserWarning, match="dropped 2 pair"):
        ci = bootstrap_paired_difference_ci(g1, g2, n_bootstrap=200, random_state=0)
    assert ci.point_estimate == pytest.approx(0.0)


def test_friedman_drops_rows_listwise():
    """Three groups, one NaN each in different rows: only two complete rows remain."""
    groups = {
        "A": pd.Series([1.0, 2.0, np.nan, 4.0, 5.0]),
        "B": pd.Series([1.0, np.nan, 3.0, 4.0, 5.0]),
        "C": pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0]),
    }
    with pytest.warns(UserWarning, match="dropped 3 run"):
        r = friedman_test(groups)
    assert r.sample_sizes == {"A": 2, "B": 2, "C": 2}


def _friedman_p(a, b):
    return friedman_test({"a": a, "b": b, "c": a * 1.1 + 0.01}).p_value


def _paired_bootstrap_point(a, b):
    return bootstrap_paired_difference_ci(a, b, n_bootstrap=300, random_state=0).point_estimate


def _paired_wilcoxon_stat(a, b):
    return paired_wilcoxon_test(a, b).statistic


@pytest.mark.parametrize(
    "fn",
    [_friedman_p, _paired_bootstrap_point, _paired_wilcoxon_stat],
    ids=["friedman", "paired_bootstrap", "paired_wilcoxon"],
)
def test_pairing_invariant_to_nan_position(fn):
    """Moving a NaN between the two groups of the same row changes only which
    row is dropped, never how the remaining rows are paired (R6 property test)."""
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=12))
    b = a + 0.3 + rng.normal(scale=0.1, size=12)
    ref = fn(a.drop(3).reset_index(drop=True), b.drop(3).reset_index(drop=True))

    a_nan, b_nan = a.copy(), b.copy()
    a_nan[3] = np.nan
    assert fn(a_nan, b_nan) == pytest.approx(ref)

    a_nan, b_nan = a.copy(), b.copy()
    b_nan[3] = np.nan
    assert fn(a_nan, b_nan) == pytest.approx(ref)


class _ExtraColumnOnRun2(_Base):
    """History gains a 'val_extra' column only on the second fit() call.

    Standard mode only: the call counter is process-level state and does not
    survive into a spawned child. The per-metric bookkeeping is one shared
    method on both paths after C8, so the standard path exercises it.
    """

    _calls = 0

    def fit(self, train_data, validation_data=None, **kw):
        _ExtraColumnOnRun2._calls += 1
        hist = {"val_accuracy": [0.5], "loss": [0.1]}
        if _ExtraColumnOnRun2._calls == 2:
            hist["val_extra"] = [0.9]
        self.training_result = TrainingResult(history=hist, params={})


def _build_extra(cfg):
    return _ExtraColumnOnRun2(None)


def test_metric_run_ids_align(X, y):
    """A metric present in only one run maps to that run's id, not to position 1."""
    _ExtraColumnOnRun2._calls = 0
    r = ix.variability_study(_build_extra, data=(X, y), runs=3, seed=0, verbose=False)
    ids, vals = r.get_metric_values("val_extra", with_run_ids=True)
    assert ids == [2]
    assert vals == [0.9]
    ids, _ = r.get_metric_values("val_accuracy", with_run_ids=True)
    assert ids == [1, 2, 3]
    assert r.metric_run_ids["val_extra"] == [2]
