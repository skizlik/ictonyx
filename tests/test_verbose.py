"""Regression tests for verbose parameter forwarding. Prevents silent
re-introduction of the v0.3.11 regression where verbose=False was ignored."""

import warnings

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
    def test_variability_study_verbose_false_no_stdout(self, capsys, small_df):
        api.variability_study(
            model=RandomForestClassifier,
            data=small_df,
            target_column="target",
            runs=3,
            verbose=False,
            seed=42,
        )
        assert capsys.readouterr().out == ""

    def test_variability_study_verbose_true_produces_output(self, capsys, small_df):
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
        captured = capsys.readouterr()
        # tqdm writes to stderr; logging writes to stdout — either is acceptable
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_compare_models_verbose_false_no_stdout(self, capsys, small_df):
        api.compare_models(
            models=[RandomForestClassifier, DecisionTreeClassifier],
            data=small_df,
            target_column="target",
            runs=3,
            verbose=False,
            seed=42,
        )
        assert capsys.readouterr().out == ""
