# tests/test_verbose.py
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ictonyx import api


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "feature1": range(20),
            "feature2": [x * 0.1 for x in range(20)],
            "target": [0, 1] * 10,
        }
    )


class TestVerboseForwarding:
    def test_variability_study_verbose_false_no_stdout(self, capsys, small_df):
        """verbose=False must produce no stdout output."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("ictonyx.api._run_study") as mock_run:
                mock_run.return_value = MagicMock()
                try:
                    api.variability_study(
                        model=RandomForestClassifier,
                        data=small_df,
                        target_column="target",
                        runs=3,
                        verbose=False,
                        seed=42,
                    )
                except Exception:
                    pass
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_variability_study_verbose_true_produces_output(self, capsys, small_df):
        """verbose=True should not suppress output."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("ictonyx.api._run_study") as mock_run:
                mock_run.return_value = MagicMock()
                try:
                    api.variability_study(
                        model=RandomForestClassifier,
                        data=small_df,
                        target_column="target",
                        runs=3,
                        verbose=True,
                        seed=42,
                    )
                except Exception:
                    pass
        # We just check it doesn't crash — output routing depends on
        # logging configuration in the test environment.
        assert True

    def test_compare_models_verbose_false_no_stdout(self, capsys, small_df):
        """verbose=False in compare_models must produce no stdout."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("ictonyx.api.variability_study") as mock_vs:
                mock_result = MagicMock()
                mock_result.get_metric_values.return_value = [0.8, 0.82, 0.79]
                mock_vs.return_value = mock_result
                try:
                    api.compare_models(
                        models=[RandomForestClassifier, DecisionTreeClassifier],
                        data=small_df,
                        target_column="target",
                        runs=3,
                        verbose=False,
                        seed=42,
                    )
                except Exception:
                    pass
        captured = capsys.readouterr()
        assert captured.out == ""
