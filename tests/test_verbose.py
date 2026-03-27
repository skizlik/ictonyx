"""Regression tests for verbose parameter forwarding. Prevents silent
re-introduction of the v0.3.11 regression where verbose=False was ignored.

Why we test logger level, not output
-------------------------------------
The ictonyx logger has propagate=False and a StreamHandler bound to
sys.stdout at module-import time. No pytest capture fixture reliably
captures its output across all CI environments:
  - capsys: misses it (stale StreamHandler reference bypasses attribute redirect)
  - caplog: misses it (propagate=False blocks records from reaching root)
  - capfd:  misses it in some CI configurations (coverage instrumentation
            interferes with fd-level dup2 redirection)

The behavior under test is: verbose=True/False causes set_verbose() to be
called, which sets the ictonyx logger level to INFO or WARNING respectively.
Testing the logger level directly is exact, environment-independent, and
tests the actual mechanism the verbose flag controls.
"""

import logging
import warnings

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import ictonyx.settings as ict_settings
from ictonyx import api


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


class TestVerboseForwarding:
    def test_variability_study_verbose_false_sets_warning_level(self, small_df):
        """verbose=False must set the ictonyx logger to WARNING level.

        Regression guard for v0.3.11: verbose=False was ignored and the
        logger stayed at INFO, producing output that should have been suppressed.
        """
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api.variability_study(
                model=RandomForestClassifier,
                data=small_df,
                target_column="target",
                runs=3,
                verbose=False,
                seed=42,
            )
        assert ict_settings.logger.level == logging.WARNING, (
            f"verbose=False must set logger to WARNING, "
            f"got level {ict_settings.logger.level} ({logging.getLevelName(ict_settings.logger.level)}). "
            "set_verbose(False) may not be reaching the logger."
        )

    def test_variability_study_verbose_true_sets_info_level(self, small_df):
        """verbose=True must set the ictonyx logger to INFO level."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api.variability_study(
                model=RandomForestClassifier,
                data=small_df,
                target_column="target",
                runs=3,
                verbose=True,
                seed=42,
            )
        assert ict_settings.logger.level == logging.INFO, (
            f"verbose=True must set logger to INFO, "
            f"got level {ict_settings.logger.level} ({logging.getLevelName(ict_settings.logger.level)}). "
            "set_verbose(True) may not be reaching the logger."
        )

    def test_compare_models_verbose_false_sets_warning_level(self, small_df):
        """verbose=False in compare_models must set the ictonyx logger to WARNING level."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api.compare_models(
                models=[RandomForestClassifier, DecisionTreeClassifier],
                data=small_df,
                target_column="target",
                runs=3,
                verbose=False,
                seed=42,
            )
        assert ict_settings.logger.level == logging.WARNING, (
            f"verbose=False in compare_models must set logger to WARNING, "
            f"got {ict_settings.logger.level}."
        )
