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
