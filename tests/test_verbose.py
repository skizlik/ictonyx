"""Regression tests for verbose parameter forwarding. Prevents silent
re-introduction of the v0.3.11 regression where verbose=False was ignored.

Implementation note on capture fixture choice
---------------------------------------------
ictonyx's logger (name="ictonyx") has propagate=False and a StreamHandler
attached to sys.stdout at module-import time.

- ``capsys`` replaces the sys.stdout *attribute* after import. The handler
  holds a reference to the original object, so capsys sees nothing.
- ``caplog`` captures from the root logger. With propagate=False, records
  never reach root, so caplog sees nothing.
- ``capfd`` captures at the OS file-descriptor level (fd 1 / fd 2). All
  writes to fd 1 are intercepted regardless of which Python object initiated
  them. This is the only fixture that reliably captures ictonyx logging output.
"""

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
    def test_variability_study_verbose_false_no_output(self, capfd, small_df):
        """verbose=False must produce no stdout output from the ictonyx logger.

        set_verbose(False) sets the logger level to WARNING. No INFO records
        are emitted, so nothing is written to fd 1.
        """
        api.variability_study(
            model=RandomForestClassifier,
            data=small_df,
            target_column="target",
            runs=3,
            verbose=False,
            seed=42,
        )
        captured = capfd.readouterr()
        assert captured.out == "", f"verbose=False produced stdout output: {captured.out[:300]!r}"

    def test_variability_study_verbose_true_produces_output(self, capfd, small_df):
        """verbose=True must produce output to stdout or stderr.

        set_verbose(True) sets the logger level to INFO. The study emits
        multiple INFO records ("Starting Variability Study", run summaries,
        etc.) via the StreamHandler on stdout. tqdm (if installed) writes
        a progress bar to stderr. Either constitutes passing output.
        """
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
        captured = capfd.readouterr()
        assert len(captured.out) > 0 or len(captured.err) > 0, (
            "verbose=True produced no output on stdout or stderr. "
            "Check that set_verbose(True) is called and the ictonyx "
            "logger level is set to INFO."
        )

    def test_compare_models_verbose_false_no_output(self, capfd, small_df):
        """verbose=False in compare_models must produce no stdout output."""
        api.compare_models(
            models=[RandomForestClassifier, DecisionTreeClassifier],
            data=small_df,
            target_column="target",
            runs=3,
            verbose=False,
            seed=42,
        )
        captured = capfd.readouterr()
        assert captured.out == "", (
            f"verbose=False in compare_models produced stdout output: " f"{captured.out[:300]!r}"
        )
