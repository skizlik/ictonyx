# tests/test_plotting.py

# Imports

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


from ictonyx.plotting import (
    plot_confusion_matrix,
    plot_training_history,
    plot_variability_summary,
    plot_autocorr_vs_lag,
    plot_averaged_autocorr,
    plot_training_stability,
    plot_comparison_boxplots,  # New
    plot_comparison_forest  # New
)
from ictonyx import settings


# --- Fixtures ---

@pytest.fixture
def sample_history_df():
    return pd.DataFrame({
        'epoch': [1, 2, 3],
        'train_accuracy': [0.5, 0.6, 0.7],
        'val_accuracy': [0.4, 0.5, 0.6],
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7]
    })


@pytest.fixture
def mock_comparison_results():
    """Mock results structure for comparison plots."""
    # Mocking StatisticalTestResult objects for pairwise
    mock_res_a = MagicMock()
    mock_res_a.is_significant.return_value = True

    return {
        'overall_test': MagicMock(),
        'pairwise_comparisons': {
            'ModelA_vs_ModelB': mock_res_a
        },
        # The new plots require the raw data dictionary
        'raw_data': {
            'ModelA': [0.85, 0.86, 0.84, 0.85],
            'ModelB': [0.75, 0.74, 0.76, 0.75],
            'ModelC': [0.80, 0.81, 0.79, 0.80]
        }
    }


# --- Tests ---

class TestBasicPlots:
    def setup_method(self):
        # Ensure we don't actually block execution
        settings.set_display_plots(False)

    @patch("matplotlib.pyplot.show")
    def test_plot_confusion_matrix(self, mock_show):
        cm = pd.DataFrame([[10, 2], [3, 15]], columns=['A', 'B'], index=['A', 'B'])
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
            'n_runs': 5,
            'common_length': 10,
            'final_loss_mean': 0.5,
            'final_loss_std': 0.1,
            'final_loss_cv': 0.2,
            'stability_assessment': 'moderate',
            'convergence_rate': 0.8,
            'converged_runs': 4,
            'final_losses_list': [0.4, 0.5, 0.6, 0.5, 0.5]
        }
        fig = plot_training_stability(results)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_comparison_boxplots(self, mock_show, mock_comparison_results):
        """Test the new boxplot visualization."""
        fig = plot_comparison_boxplots(mock_comparison_results, metric='Accuracy')
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_comparison_forest(self, mock_show, mock_comparison_results):
        """Test the new forest plot visualization."""
        fig = plot_comparison_forest(
            mock_comparison_results,
            baseline_model='ModelB',
            metric='Accuracy'
        )
        assert fig is not None

    def test_plot_comparison_forest_invalid_baseline(self, mock_comparison_results):
        """Test that invalid baseline returns None/Error logic."""
        fig = plot_comparison_forest(
            mock_comparison_results,
            baseline_model='Model_Does_Not_Exist'
        )
        # Should handle gracefully (log error and return None)
        assert fig is None