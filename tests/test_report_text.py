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
