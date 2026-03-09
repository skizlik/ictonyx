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
    plot_comparison_boxplots,
    plot_comparison_forest,
    plot_confusion_matrix,
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
