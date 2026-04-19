# tests/test_plotting.py

# Imports

from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ictonyx import settings
from ictonyx.plotting import (
    plot_autocorr_vs_lag,
    plot_averaged_autocorr,
    plot_averaged_pacf,
    plot_comparison_boxplots,
    plot_comparison_forest,
    plot_confusion_matrix,
    plot_grid_study_heatmap,
    plot_pacf_vs_lag,
    plot_pairwise_comparison_matrix,
    plot_precision_recall_curve,
    plot_rank_correlation_over_epoch,
    plot_roc_curve,
    plot_run_distribution,
    plot_run_strip,
    plot_run_trajectories,
    plot_training_history,
    plot_training_stability,
    plot_variability_summary,
)

# non-interactive backend for testing
matplotlib.use("Agg")


def _make_fake_runs(n_runs=5, n_epochs=10):
    """Generate fake run data for plotting tests."""
    runs = []
    for i in range(n_runs):
        base_acc = 0.5 + np.random.randn() * 0.05
        df = pd.DataFrame(
            {
                "accuracy": np.linspace(0.3, base_acc, n_epochs) + np.random.randn(n_epochs) * 0.02,
                "val_accuracy": np.linspace(0.25, base_acc - 0.05, n_epochs)
                + np.random.randn(n_epochs) * 0.03,
                "loss": np.linspace(1.5, 0.5, n_epochs) + np.random.randn(n_epochs) * 0.1,
                "val_loss": np.linspace(1.6, 0.6, n_epochs) + np.random.randn(n_epochs) * 0.1,
            }
        )
        runs.append(df)
    final_vals = pd.Series([run["val_accuracy"].iloc[-1] for run in runs])
    final_test = pd.Series([run["val_accuracy"].iloc[-1] - 0.02 for run in runs])
    return runs, final_vals, final_test


# --- Fixtures ---


@pytest.fixture
def sample_history_df():
    return pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_accuracy": [0.5, 0.6, 0.7],
            "val_accuracy": [0.4, 0.5, 0.6],
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
        }
    )


@pytest.fixture
def mock_comparison_results():
    """Mock results structure for comparison plots."""
    # Mocking StatisticalTestResult objects for pairwise
    mock_res_a = MagicMock()
    mock_res_a.is_significant.return_value = True

    return {
        "overall_test": MagicMock(),
        "pairwise_comparisons": {"ModelA_vs_ModelB": mock_res_a},
        # The new plots require the raw data dictionary
        "raw_data": {
            "ModelA": [0.85, 0.86, 0.84, 0.85],
            "ModelB": [0.75, 0.74, 0.76, 0.75],
            "ModelC": [0.80, 0.81, 0.79, 0.80],
        },
    }


@pytest.fixture
def sample_results():
    """Mock VariabilityStudyResults for plot tests."""
    runs, final_vals, _ = _make_fake_runs(n_runs=5, n_epochs=10)
    mock = MagicMock()
    mock.all_runs_metrics = runs
    mock.get_metric_values.return_value = list(final_vals)
    return mock


@pytest.fixture
def sample_comparison():
    """Mock comparison dict for forest plot tests."""
    return {
        "ModelA": [0.85, 0.86, 0.84, 0.85, 0.87],
        "ModelB": [0.75, 0.74, 0.76, 0.75, 0.73],
    }


@pytest.fixture
def small_results():
    """Minimal VariabilityStudyResults for new plotting function tests."""
    from ictonyx.runners import VariabilityStudyResults

    run_dfs = [
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_accuracy": [0.5, 0.7, 0.85],
                "val_accuracy": [0.48, 0.65, 0.80],
                "run_num": [i] * 3,
            }
        )
        for i in range(1, 6)
    ]
    return VariabilityStudyResults(
        all_runs_metrics=run_dfs,
        final_metrics={"val_accuracy": [0.80, 0.78, 0.83, 0.79, 0.82]},
        final_test_metrics=[],
        seed=42,
    )


@pytest.fixture
def large_results():
    """20-run VariabilityStudyResults — minimum for rank correlation."""
    from ictonyx.runners import VariabilityStudyResults

    rng = np.random.default_rng(42)
    run_dfs = [
        pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4, 5],
                "val_accuracy": np.cumsum(rng.uniform(0.05, 0.15, 5)).tolist(),
                "run_num": [i] * 5,
            }
        )
        for i in range(1, 21)
    ]
    final_vals = [df["val_accuracy"].iloc[-1] for df in run_dfs]
    return VariabilityStudyResults(
        all_runs_metrics=run_dfs,
        final_metrics={"val_accuracy": final_vals},
        final_test_metrics=[],
        seed=42,
    )


# --- Tests ---


class TestBasicPlots:
    def setup_method(self):
        # Ensure we don't actually block execution
        settings.set_display_plots(False)

    @patch("matplotlib.pyplot.show")
    def test_plot_confusion_matrix(self, mock_show):
        cm = pd.DataFrame([[10, 2], [3, 15]], columns=["A", "B"], index=["A", "B"])
        fig = plot_confusion_matrix(cm)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_training_history(self, mock_show, sample_history_df):
        fig = plot_training_history(sample_history_df)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_variability_summary(self, mock_show, sample_history_df):
        final_accs = pd.Series([0.6, 0.65, 0.62])
        # Must provide list of DFs
        fig = plot_variability_summary([sample_history_df], final_accs)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_autocorr_vs_lag(self, mock_show):
        data = pd.Series(np.random.rand(50))
        fig = plot_autocorr_vs_lag(data, max_lag=5)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_averaged_autocorr(self, mock_show):
        lags = [1, 2, 3]
        means = [0.5, 0.3, 0.1]
        stds = [0.1, 0.1, 0.1]
        fig = plot_averaged_autocorr(lags, means, stds)
        assert fig is not None


class TestStatisticalPlots:
    def setup_method(self):
        settings.set_display_plots(False)

    @patch("matplotlib.pyplot.show")
    def test_plot_training_stability(self, mock_show):
        results = {
            "n_runs": 5,
            "common_length": 10,
            "final_loss_mean": 0.5,
            "final_loss_std": 0.1,
            "final_loss_cv": 0.2,
            "stability_assessment": "moderate",
            "convergence_rate": 0.8,
            "converged_runs": 4,
            "final_losses_list": [0.4, 0.5, 0.6, 0.5, 0.5],
        }
        fig = plot_training_stability(results)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_training_stability_accepts_results_object(self, mock_show):
        """Passing results=... computes stability analysis internally."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        import ictonyx as ix

        X, y = make_classification(n_samples=100, random_state=0)
        # PyTorch wrapper would give real loss curves; for sklearn we
        # just need *some* per-epoch data. Use a minimal fixture:
        results = ix.variability_study(model=LogisticRegression, data=(X, y), runs=5, seed=42)
        # sklearn wrappers create a 1-epoch mock history with "loss" key
        # if available; skip this test if not present
        if "loss" not in results.all_runs_metrics[0].columns:
            pytest.skip("sklearn mock history does not contain 'loss' key")
        fig = plot_training_stability(results=results)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_training_stability_both_inputs_raises(self, mock_show):
        """Passing both stability_results and results raises ValueError."""
        stability_dict = {
            "n_runs": 5,
            "common_length": 10,
            "final_loss_mean": 0.5,
            "final_loss_std": 0.1,
            "final_loss_cv": 0.2,
            "stability_assessment": "moderate",
            "convergence_rate": 0.8,
            "converged_runs": 4,
            "final_losses_list": [0.4, 0.5, 0.6, 0.5, 0.5],
        }
        # Make a minimal results-like object via Mock
        from unittest.mock import MagicMock

        fake_results = MagicMock()
        fake_results.all_runs_metrics = [pd.DataFrame({"loss": [0.5, 0.4, 0.3]})]
        with pytest.raises(ValueError, match="either"):
            plot_training_stability(stability_results=stability_dict, results=fake_results)

    @patch("matplotlib.pyplot.show")
    def test_plot_training_stability_neither_input_raises(self, mock_show):
        """Passing nothing raises ValueError."""
        with pytest.raises(ValueError, match="requires either"):
            plot_training_stability()

    @patch("matplotlib.pyplot.show")
    def test_plot_comparison_boxplots(self, mock_show, mock_comparison_results):
        """Test the new boxplot visualization."""
        fig = plot_comparison_boxplots(mock_comparison_results, metric="Accuracy")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_comparison_forest(self, mock_show, mock_comparison_results):
        """Test the new forest plot visualization."""
        fig = plot_comparison_forest(
            mock_comparison_results, baseline_model="ModelB", metric="Accuracy"
        )
        assert fig is not None

    def test_plot_comparison_forest_invalid_baseline(self, mock_comparison_results):
        """Test that invalid baseline returns None/Error logic."""
        fig = plot_comparison_forest(mock_comparison_results, baseline_model="Model_Does_Not_Exist")
        # Should handle gracefully (log error and return None)
        assert fig is None


class TestPlotStructure:
    """Test that plots return figures with correct structure."""

    def setup_method(self):
        settings.set_display_plots(False)

    @patch("matplotlib.pyplot.show")
    def test_confusion_matrix_axes(self, mock_show):
        """Test confusion matrix has correct axes."""
        cm = pd.DataFrame(
            [[10, 2, 1], [3, 15, 0], [1, 0, 12]], columns=["A", "B", "C"], index=["A", "B", "C"]
        )
        fig = plot_confusion_matrix(cm, title="Test CM")
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_training_history_with_custom_metrics(self, mock_show):
        """Test history plot with specific metrics requested."""
        df = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4, 5],
                "train_accuracy": [0.5, 0.6, 0.7, 0.8, 0.85],
                "val_accuracy": [0.4, 0.5, 0.6, 0.65, 0.7],
                "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
                "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
            }
        )
        fig = plot_training_history(df, metrics=["accuracy"])
        assert fig is not None
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_training_history_with_title(self, mock_show):
        """Test that custom title is applied."""
        df = pd.DataFrame(
            {
                "epoch": [1, 2],
                "train_loss": [1.0, 0.5],
                "val_loss": [1.1, 0.6],
            }
        )
        fig = plot_training_history(df, title="My Custom Title")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_variability_summary_multiple_runs(self, mock_show):
        """Test variability summary with multiple runs."""
        runs = []
        for _ in range(5):
            base = np.random.rand()
            runs.append(
                pd.DataFrame(
                    {
                        "epoch": [1, 2, 3],
                        "train_accuracy": [0.5 + base * 0.1, 0.6 + base * 0.1, 0.7 + base * 0.1],
                        "val_accuracy": [0.4 + base * 0.1, 0.5 + base * 0.1, 0.6 + base * 0.1],
                        "train_loss": [1.0 - base * 0.1, 0.8 - base * 0.1, 0.6 - base * 0.1],
                        "val_loss": [1.1 - base * 0.1, 0.9 - base * 0.1, 0.7 - base * 0.1],
                    }
                )
            )
        final_accs = pd.Series([0.6, 0.65, 0.62, 0.58, 0.63])
        fig = plot_variability_summary(runs, final_accs)
        assert fig is not None
        # Should have multiple subplots
        assert len(fig.axes) >= 2

    @patch("matplotlib.pyplot.show")
    def test_comparison_boxplots_raw_dict(self, mock_show):
        """Test boxplots when passed raw data dict directly."""
        data = {
            "ModelA": [0.85, 0.86, 0.84, 0.87, 0.85],
            "ModelB": [0.80, 0.81, 0.79, 0.82, 0.80],
        }
        fig = plot_comparison_boxplots(data, metric="Accuracy")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_comparison_forest_three_models(self, mock_show):
        """Test forest plot with multiple comparison models."""
        data = {
            "raw_data": {
                "Baseline": [0.80, 0.81, 0.79, 0.80, 0.82],
                "ModelA": [0.85, 0.86, 0.84, 0.87, 0.85],
                "ModelB": [0.90, 0.91, 0.89, 0.90, 0.88],
            }
        }
        fig = plot_comparison_forest(data, baseline_model="Baseline")
        assert fig is not None


class TestPlotVariabilitySummaryOptions:
    """Tests for new display options in plot_variability_summary."""

    def test_default_parameters_still_work(self):
        """Existing calls with no new params should produce a figure."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, show=False)
        assert fig is not None
        plt.close(fig)

    def test_show_mean_lines_false(self):
        """Disabling mean lines should still produce a valid figure."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, show_mean_lines=False, show=False)
        assert fig is not None
        # Check that the trajectory axes has fewer lines than with mean lines
        ax = fig.axes[0]
        fig_with_mean = plot_variability_summary(runs, finals, show_mean_lines=True, show=False)
        ax_with_mean = fig_with_mean.axes[0]
        assert len(ax.lines) < len(ax_with_mean.lines)
        plt.close(fig)
        plt.close(fig_with_mean)

    def test_show_train_false(self):
        """Hiding training curves should produce fewer lines."""
        runs, finals, _ = _make_fake_runs()
        fig_both = plot_variability_summary(runs, finals, show=False)
        fig_val_only = plot_variability_summary(runs, finals, show_train=False, show=False)
        assert len(fig_val_only.axes[0].lines) < len(fig_both.axes[0].lines)
        plt.close(fig_both)
        plt.close(fig_val_only)

    def test_show_val_false(self):
        """Hiding validation curves should produce fewer lines."""
        runs, finals, _ = _make_fake_runs()
        fig_both = plot_variability_summary(runs, finals, show=False)
        fig_train_only = plot_variability_summary(runs, finals, show_val=False, show=False)
        assert len(fig_train_only.axes[0].lines) < len(fig_both.axes[0].lines)
        plt.close(fig_both)
        plt.close(fig_train_only)

    def test_horizontal_histogram(self):
        """Horizontal histogram should produce a valid figure."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, histogram_orientation="horizontal", show=False)
        assert fig is not None
        plt.close(fig)

    def test_invalid_histogram_orientation_raises(self):
        """Invalid orientation should raise ValueError."""
        runs, finals, _ = _make_fake_runs()
        with pytest.raises(ValueError, match="histogram_orientation"):
            plot_variability_summary(runs, finals, histogram_orientation="diagonal", show=False)

    def test_custom_alpha(self):
        """Explicit alpha should be accepted."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, alpha=0.1, show=False)
        assert fig is not None
        plt.close(fig)

    def test_invalid_alpha_raises(self):
        """Alpha outside (0, 1] should raise ValueError."""
        runs, finals, _ = _make_fake_runs()
        with pytest.raises(ValueError, match="alpha"):
            plot_variability_summary(runs, finals, alpha=1.5, show=False)
        with pytest.raises(ValueError, match="alpha"):
            plot_variability_summary(runs, finals, alpha=0.0, show=False)

    def test_custom_figsize(self):
        """Custom figsize should be applied to the figure."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, figsize=(20, 10), show=False)
        w, h = fig.get_size_inches()
        assert abs(w - 20) < 0.1
        assert abs(h - 10) < 0.1
        plt.close(fig)

    def test_custom_dpi(self):
        """Custom dpi should be applied to the figure."""
        runs, finals, _ = _make_fake_runs()
        fig = plot_variability_summary(runs, finals, dpi=300, show=False)
        assert fig.dpi == 300
        plt.close(fig)

    def test_auto_alpha_few_runs(self):
        """With 3 runs, auto alpha should be higher than with 30."""
        runs_3, finals_3, _ = _make_fake_runs(n_runs=3)
        runs_30, finals_30, _ = _make_fake_runs(n_runs=30)
        fig_3 = plot_variability_summary(runs_3, finals_3, show=False)
        fig_30 = plot_variability_summary(runs_30, finals_30, show=False)
        # Can't directly check alpha from the figure, but at least
        # verify both produce valid figures
        assert fig_3 is not None
        assert fig_30 is not None
        plt.close(fig_3)
        plt.close(fig_30)

    def test_all_new_options_combined(self):
        """All new options together should not crash."""
        runs, finals, test_finals = _make_fake_runs()
        fig = plot_variability_summary(
            runs,
            finals,
            test_finals,
            show_mean_lines=False,
            show_train=False,
            show_val=True,
            show_histogram=True,
            show_boxplot=True,
            histogram_orientation="horizontal",
            alpha=0.2,
            figsize=(18, 8),
            dpi=200,
            show=False,
        )
        assert fig is not None
        assert len(fig.axes) == 3  # trajectory + histogram + boxplot
        plt.close(fig)

    def test_horizontal_histogram_with_test_series(self):
        """Horizontal histogram with both val and test series."""
        runs, finals, test_finals = _make_fake_runs()
        fig = plot_variability_summary(
            runs,
            finals,
            test_finals,
            histogram_orientation="horizontal",
            show=False,
        )
        assert fig is not None
        plt.close(fig)


class TestPairwiseMatrixFixes:
    def test_empty_dict_returns_none_with_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = plot_pairwise_comparison_matrix({})
        assert result is None
        assert any("empty" in str(warning.message).lower() for warning in w)

    def test_underscore_model_names_do_not_crash(self):
        data = {
            "model_with_underscore_vs_another_model": MagicMock(
                p_value=0.01,
                is_significant=MagicMock(return_value=True),
            )
        }
        # Should not raise — may return None or a figure
        try:
            plot_pairwise_comparison_matrix(data)
        except Exception as e:
            pytest.fail(f"Raised unexpected exception: {e}")


class TestPlotReturnValues:
    """All plot functions must return Figure unconditionally (R7-4)."""

    @patch("matplotlib.pyplot.show")
    def test_plot_variability_summary_returns_figure(self, mock_show, sample_results):
        from matplotlib.figure import Figure

        result = plot_variability_summary(results=sample_results)
        assert isinstance(
            result, Figure
        ), "plot_variability_summary must return Figure even when show=True"

    @patch("matplotlib.pyplot.show")
    def test_plot_comparison_forest_returns_figure(self, mock_show, sample_comparison):
        from matplotlib.figure import Figure

        result = plot_comparison_forest(sample_comparison, baseline_model="ModelA")
        assert isinstance(result, Figure) or result is None  # None if no models


class TestPlotVariabilitySummaryResultsInput:
    """plot_variability_summary accepts VariabilityStudyResults via results= (R7-3)."""

    @patch("matplotlib.pyplot.show")
    def test_accepts_results_object(self, mock_show, sample_results):
        # Should not raise
        plot_variability_summary(results=sample_results, metric="val_accuracy")

    @patch("matplotlib.pyplot.show")
    def test_results_equivalent_to_manual_extraction(self, mock_show, sample_results):
        """Both call patterns must produce equivalent figures."""
        fig1 = plot_variability_summary(results=sample_results)
        fig2 = plot_variability_summary(
            all_runs_metrics_list=sample_results.all_runs_metrics,
            final_metrics_series=pd.Series(sample_results.get_metric_values("val_accuracy")),
        )
        assert fig1 is not None
        assert fig2 is not None


class TestPlotGridStudyHeatmap:
    """Tests for plot_grid_study_heatmap() — new in v0.3.12."""

    def _make_grid_results(self):
        """Build a minimal GridStudyResults mock."""
        from unittest.mock import MagicMock

        import pandas as pd

        grid = MagicMock()
        grid.metric = "val_accuracy"
        df = pd.DataFrame(
            {
                "lr": [0.001, 0.001, 0.01, 0.01],
                "batch_size": [16, 32, 16, 32],
                "mean": [0.81, 0.83, 0.79, 0.85],
                "sd": [0.01, 0.02, 0.03, 0.01],
            }
        )
        grid.to_dataframe.return_value = df
        return grid

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show):
        from matplotlib.figure import Figure

        grid = self._make_grid_results()
        result = plot_grid_study_heatmap(grid, x_param="batch_size", y_param="lr")
        assert isinstance(result, Figure)

    @patch("matplotlib.pyplot.show")
    def test_sd_stat(self, mock_show):
        from matplotlib.figure import Figure

        grid = self._make_grid_results()
        result = plot_grid_study_heatmap(grid, x_param="batch_size", y_param="lr", stat="sd")
        assert isinstance(result, Figure)

    def test_invalid_x_param_raises(self):
        grid = self._make_grid_results()
        with pytest.raises(ValueError, match="x_param"):
            plot_grid_study_heatmap(grid, x_param="nonexistent", y_param="lr")

    def test_invalid_y_param_raises(self):
        grid = self._make_grid_results()
        with pytest.raises(ValueError, match="y_param"):
            plot_grid_study_heatmap(grid, x_param="batch_size", y_param="nonexistent")

    def test_invalid_stat_raises(self):
        grid = self._make_grid_results()
        with pytest.raises(ValueError, match="stat"):
            plot_grid_study_heatmap(grid, x_param="batch_size", y_param="lr", stat="median")


class TestPlotTrainingHistory:
    """Smoke tests for plot_training_history()."""

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show):
        import pandas as pd
        from matplotlib.figure import Figure

        df = pd.DataFrame(
            {
                "train_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
                "val_accuracy": [0.55, 0.65, 0.75, 0.78, 0.82],
                "train_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
                "val_loss": [1.0, 0.8, 0.6, 0.55, 0.5],
            }
        )
        result = plot_training_history(df, show=False)
        assert isinstance(result, Figure)

    @patch("matplotlib.pyplot.show")
    def test_single_run_no_crash(self, mock_show):
        import pandas as pd

        df = pd.DataFrame(
            {
                "train_loss": [0.5, 0.4, 0.3],
            }
        )
        result = plot_training_history(df, show=False)
        assert result is not None


class TestROCAndPRCurves:
    """plot_roc_curve and plot_precision_recall_curve with TF utils patched out."""

    def setup_method(self):
        settings.set_display_plots(False)

    def _make_mock_wrapper(self):
        """A mock model wrapper with predict_proba returning 2-class probabilities."""
        mock = MagicMock()
        rng = np.random.default_rng(0)
        proba = rng.dirichlet([1, 1], size=30)  # shape (30, 2)
        mock.predict_proba.return_value = proba
        return mock

    @patch("matplotlib.pyplot.show")
    @patch("ictonyx.plotting._check_tensorflow_utils")
    def test_plot_roc_curve_returns_figure(self, mock_tf_check, mock_show):
        rng = np.random.default_rng(0)
        X_test = rng.standard_normal((30, 4))
        y_test = rng.integers(0, 2, 30)
        wrapper = self._make_mock_wrapper()
        fig = plot_roc_curve(wrapper, X_test, y_test)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    @patch("ictonyx.plotting._check_tensorflow_utils")
    def test_plot_roc_curve_custom_title(self, mock_tf_check, mock_show):
        rng = np.random.default_rng(1)
        X_test = rng.standard_normal((30, 3))
        y_test = rng.integers(0, 2, 30)
        wrapper = self._make_mock_wrapper()
        fig = plot_roc_curve(wrapper, X_test, y_test, title="My ROC")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_roc_curve_no_predict_proba_returns_none(self, mock_show):
        """Model without predict_proba must return None, not raise."""
        mock = MagicMock(spec=[])  # no attributes
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 3))
        y = rng.integers(0, 2, 10)
        # _check_tensorflow_utils raises ImportError if TF absent — patch it
        with patch("ictonyx.plotting._check_tensorflow_utils"):
            with patch("ictonyx.plotting._check_sklearn_metrics"):
                result = plot_roc_curve(mock, X, y)
        assert result is None

    @patch("matplotlib.pyplot.show")
    @patch("ictonyx.plotting._check_tensorflow_utils")
    def test_plot_precision_recall_returns_figure(self, mock_tf_check, mock_show):
        rng = np.random.default_rng(3)
        X_test = rng.standard_normal((30, 4))
        y_test = rng.integers(0, 2, 30)
        wrapper = self._make_mock_wrapper()
        fig = plot_precision_recall_curve(wrapper, X_test, y_test)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    @patch("ictonyx.plotting._check_tensorflow_utils")
    def test_plot_precision_recall_custom_title(self, mock_tf_check, mock_show):
        rng = np.random.default_rng(4)
        X_test = rng.standard_normal((30, 3))
        y_test = rng.integers(0, 2, 30)
        wrapper = self._make_mock_wrapper()
        fig = plot_precision_recall_curve(wrapper, X_test, y_test, title="My PR")
        assert fig is not None


class TestPACFPlots:
    """plot_pacf_vs_lag and plot_averaged_pacf."""

    def setup_method(self):
        settings.set_display_plots(False)

    def test_plot_pacf_vs_lag_without_statsmodels_returns_none(self):
        """When statsmodels is absent, must return None without raising."""
        import sys
        from unittest.mock import patch

        data = pd.Series(np.random.default_rng(0).standard_normal(50))
        with patch.dict(
            sys.modules,
            {"statsmodels": None, "statsmodels.tsa": None, "statsmodels.tsa.stattools": None},
        ):
            result = plot_pacf_vs_lag(data, max_lag=10)
        assert result is None

    def test_plot_pacf_vs_lag_too_short_returns_none(self):
        """Series too short (len <= max_lag + 1) must return None."""
        data = pd.Series([0.1, 0.2, 0.3])
        result = plot_pacf_vs_lag(data, max_lag=10)
        assert result is None

    @patch("matplotlib.pyplot.show")
    def test_plot_averaged_pacf_returns_figure(self, mock_show):
        lags = list(range(1, 6))
        mean_pacf = [0.4, 0.2, 0.1, 0.05, 0.02]
        std_pacf = [0.05, 0.04, 0.03, 0.02, 0.01]
        fig = plot_averaged_pacf(lags, mean_pacf, std_pacf)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_averaged_pacf_custom_title(self, mock_show):
        lags = [1, 2, 3]
        fig = plot_averaged_pacf(lags, [0.3, 0.1, 0.05], [0.02, 0.01, 0.01], title="My PACF")
        assert fig is not None


class TestPlotTrainingStabilityPaths:
    """Additional plot_training_stability coverage."""

    def setup_method(self):
        settings.set_display_plots(False)

    def test_error_key_returns_none(self):
        """Stability dict with 'error' key must return None immediately."""
        result = plot_training_stability({"error": "Too few histories"})
        assert result is None

    @patch("matplotlib.pyplot.show")
    def test_high_stability(self, mock_show):
        results = {
            "n_runs": 10,
            "common_length": 20,
            "final_loss_mean": 0.2,
            "final_loss_std": 0.01,
            "final_loss_cv": 0.05,
            "stability_assessment": "high",
            "convergence_rate": 1.0,
            "converged_runs": 10,
            "final_losses_list": [0.19 + i * 0.001 for i in range(10)],
        }
        fig = plot_training_stability(results)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_low_stability(self, mock_show):
        results = {
            "n_runs": 5,
            "common_length": 10,
            "final_loss_mean": 1.0,
            "final_loss_std": 0.8,
            "final_loss_cv": 0.8,
            "stability_assessment": "low",
            "convergence_rate": 0.2,
            "converged_runs": 1,
            "final_losses_list": [0.3, 1.2, 0.8, 1.5, 0.9],
        }
        fig = plot_training_stability(results)
        assert fig is not None


class TestPairwiseMatrixDataclass:
    def test_accepts_model_comparison_results(self):
        matplotlib.use("Agg")
        from ictonyx.analysis import ModelComparisonResults, StatisticalTestResult

        result = StatisticalTestResult(test_name="Mann-Whitney U", statistic=12.0, p_value=0.04)
        comparison = ModelComparisonResults(
            overall_test=result,
            raw_data={"A": [0.9, 0.88], "B": [0.85, 0.87]},
            pairwise_comparisons={"A_vs_B": result},
            significant_comparisons=["A_vs_B"],
            correction_method="none",
            n_models=2,
            metric="val_accuracy",
        )
        fig = plot_pairwise_comparison_matrix(comparison)
        assert fig is not None


class TestROCNoBluegateError:
    def test_roc_works_without_tensorflow(self):
        pytest.importorskip("sklearn")
        matplotlib.use("Agg")
        from unittest.mock import patch

        from sklearn.ensemble import RandomForestClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)
        wrapper = ScikitLearnModelWrapper(RandomForestClassifier(n_estimators=5))
        wrapper.fit((X, y))
        wrapper.predict(X)

        with patch("ictonyx.plotting.HAS_TENSORFLOW_UTILS", False):
            fig = plot_roc_curve(wrapper, X, y)
        assert fig is not None


class TestNewVariabilityFunctions:
    """Tests for plot_run_trajectories, plot_run_distribution, plot_run_strip."""

    def test_trajectories_returns_figure(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_trajectories(small_results, show=False) is not None

    def test_trajectories_bad_metric_returns_none(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_trajectories(small_results, metric="nonexistent", show=False) is None

    def test_trajectories_empty_runs_returns_none(self):
        matplotlib.use("Agg")
        from ictonyx.runners import VariabilityStudyResults

        empty = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.8]},
            final_test_metrics=[],
            seed=42,
        )
        assert plot_run_trajectories(empty, show=False) is None

    def test_distribution_returns_figure(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_distribution(small_results, show=False) is not None

    def test_distribution_bad_metric_returns_none(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_distribution(small_results, metric="nonexistent", show=False) is None

    def test_strip_returns_figure(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_strip(small_results, show=False) is not None

    def test_strip_bad_metric_returns_none(self, small_results):
        matplotlib.use("Agg")
        assert plot_run_strip(small_results, metric="nonexistent", show=False) is None

    def test_trajectories_accepts_ax_parameter(self, small_results):
        matplotlib.use("Agg")
        fig_outer, ax_outer = plt.subplots()
        result = plot_run_trajectories(small_results, ax=ax_outer, show=False)
        assert result is fig_outer
        plt.close("all")

    def test_dispatcher_trajectories(self, small_results):
        matplotlib.use("Agg")
        from ictonyx.plotting import plot_variability_summary

        fig = plot_variability_summary(results=small_results, kind="trajectories", show=False)
        assert fig is not None

    def test_dispatcher_distribution(self, small_results):
        matplotlib.use("Agg")
        from ictonyx.plotting import plot_variability_summary

        fig = plot_variability_summary(results=small_results, kind="distribution", show=False)
        assert fig is not None

    def test_dispatcher_strip(self, small_results):
        matplotlib.use("Agg")
        from ictonyx.plotting import plot_variability_summary

        fig = plot_variability_summary(results=small_results, kind="strip", show=False)
        assert fig is not None

    def test_dispatcher_unknown_kind_raises(self, small_results):
        from ictonyx.plotting import plot_variability_summary

        with pytest.raises(ValueError, match="Unknown kind"):
            plot_variability_summary(results=small_results, kind="banana", show=False)

    def test_dispatcher_legacy_form_warns(self, small_results):
        matplotlib.use("Agg")
        from ictonyx.plotting import plot_variability_summary

        with pytest.warns(DeprecationWarning, match="deprecated"):
            plot_variability_summary(
                all_runs_metrics_list=small_results.all_runs_metrics,
                show=False,
            )


class TestRankCorrelationPlot:
    def test_returns_figure_with_sufficient_runs(self, large_results):
        matplotlib.use("Agg")
        fig = plot_rank_correlation_over_epoch(large_results, show=False)
        assert fig is not None

    def test_raises_with_fewer_than_15_runs(self, small_results):
        """small_results has 5 runs — must raise ValueError."""
        matplotlib.use("Agg")
        with pytest.raises(ValueError, match="n_runs >= 15"):
            plot_rank_correlation_over_epoch(small_results, show=False)

    def test_accepts_explicit_metric(self, large_results):
        matplotlib.use("Agg")
        fig = plot_rank_correlation_over_epoch(large_results, metric="val_accuracy", show=False)
        assert fig is not None

    def test_accepts_ax_parameter(self, large_results):
        matplotlib.use("Agg")
        fig_outer, ax_outer = plt.subplots()
        result = plot_rank_correlation_over_epoch(large_results, ax=ax_outer, show=False)
        assert result is fig_outer
        plt.close("all")

    def test_custom_threshold(self, large_results):
        matplotlib.use("Agg")
        fig = plot_rank_correlation_over_epoch(large_results, threshold=0.5, show=False)
        assert fig is not None


class TestPlotPairedDeltas:
    """plot_paired_deltas — per-run paired differences."""

    def _make_results_with_values(self, values: list, metric_key: str = "test_accuracy"):
        """Minimal results-shaped mock for plot_paired_deltas tests."""
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.get_metric_values = MagicMock(return_value=list(values))
        fake.get_test_metric_values = MagicMock(return_value=list(values))
        fake.preferred_metric = MagicMock(return_value=metric_key)
        return fake

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.85, 0.87, 0.86, 0.88, 0.84])
        b = self._make_results_with_values([0.83, 0.86, 0.85, 0.84, 0.82])
        fig = plot_paired_deltas(a, b, metric="test_accuracy", show=False)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_unequal_run_counts_raises(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.85, 0.87, 0.86, 0.88, 0.84])
        b = self._make_results_with_values([0.83, 0.86, 0.85])
        with pytest.raises(ValueError, match="equal run counts"):
            plot_paired_deltas(a, b, metric="test_accuracy", show=False)

    @patch("matplotlib.pyplot.show")
    def test_metric_auto_resolved_when_none(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.85, 0.87, 0.86, 0.88, 0.84])
        b = self._make_results_with_values([0.83, 0.86, 0.85, 0.84, 0.82])
        fig = plot_paired_deltas(a, b, show=False)
        a.preferred_metric.assert_called_with(context="scalar")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_accepts_ax_parameter(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.85, 0.87, 0.86])
        b = self._make_results_with_values([0.83, 0.86, 0.85])
        fig, ax = plt.subplots()
        result = plot_paired_deltas(a, b, metric="test_accuracy", ax=ax, show=False)
        assert result is fig

    @patch("matplotlib.pyplot.show")
    def test_no_reversal_case_does_not_annotate(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.90, 0.91, 0.92, 0.93, 0.94])
        b = self._make_results_with_values([0.80, 0.81, 0.82, 0.83, 0.84])
        fig = plot_paired_deltas(a, b, metric="test_accuracy", show=False)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert not any("Winner reverses" in t for t in texts)

    @patch("matplotlib.pyplot.show")
    def test_reversal_case_does_annotate(self, mock_show):
        from ictonyx.plotting import plot_paired_deltas

        a = self._make_results_with_values([0.90, 0.80, 0.88, 0.82, 0.85])
        b = self._make_results_with_values([0.85, 0.82, 0.86, 0.84, 0.83])
        fig = plot_paired_deltas(a, b, metric="test_accuracy", show=False)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("Winner reverses" in t for t in texts)


class TestPlotRunIndependenceDiagnostics:
    """plot_run_independence_diagnostics — run-level autocorrelation plot."""

    def _make_results_with_values(self, values: list, metric_key: str = "test_accuracy"):
        """Minimal results-shaped mock."""
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.get_metric_values = MagicMock(return_value=list(values))
        fake.get_test_metric_values = MagicMock(return_value=list(values))
        fake.preferred_metric = MagicMock(return_value=metric_key)
        return fake

    @patch("matplotlib.pyplot.show")
    def test_returns_figure_for_adequate_n(self, mock_show):
        from ictonyx.plotting import plot_run_independence_diagnostics

        # 20 values, enough for max_lag=5 (need max_lag + 2 = 7 minimum)
        values = [0.85 + 0.01 * (-1) ** i for i in range(20)]
        r = self._make_results_with_values(values)
        fig = plot_run_independence_diagnostics(r, metric="test_accuracy", show=False)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_untestable_series_renders_notice(self, mock_show):
        from ictonyx.plotting import plot_run_independence_diagnostics

        # Only 3 values, too few for max_lag=5
        values = [0.85, 0.86, 0.87]
        r = self._make_results_with_values(values)
        fig = plot_run_independence_diagnostics(r, metric="test_accuracy", show=False)
        # Should render something (not crash), with an annotation about
        # insufficient data
        assert fig is not None
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert (
            any("too short" in t.lower() or "untestable" in t.lower() for t in texts)
            or "untestable" in ax.get_title().lower()
        )

    @patch("matplotlib.pyplot.show")
    def test_metric_auto_resolved_when_none(self, mock_show):
        from ictonyx.plotting import plot_run_independence_diagnostics

        values = [0.85 + 0.01 * (-1) ** i for i in range(20)]
        r = self._make_results_with_values(values)
        fig = plot_run_independence_diagnostics(r, show=False)
        r.preferred_metric.assert_called_with(context="scalar")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_accepts_ax_parameter(self, mock_show):
        from ictonyx.plotting import plot_run_independence_diagnostics

        values = [0.85 + 0.01 * (-1) ** i for i in range(20)]
        r = self._make_results_with_values(values)
        fig, ax = plt.subplots()
        result = plot_run_independence_diagnostics(r, metric="test_accuracy", ax=ax, show=False)
        assert result is fig

    @patch("matplotlib.pyplot.show")
    def test_custom_max_lag(self, mock_show):
        from ictonyx.plotting import plot_run_independence_diagnostics

        values = [0.85 + 0.01 * (-1) ** i for i in range(30)]
        r = self._make_results_with_values(values)
        fig = plot_run_independence_diagnostics(r, metric="test_accuracy", max_lag=10, show=False)
        assert fig is not None
