"""Test experiment runners."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import BaseModelWrapper, TrainingResult
from ictonyx.runners import ExperimentRunner, VariabilityStudyResults, run_variability_study


class MockModel(BaseModelWrapper):
    """Mock model for testing."""

    def __init__(self, config):
        super().__init__(None, "mock")
        self.config = config
        self.fit_count = 0

    def _cleanup_implementation(self):
        pass

    def fit(self, train_data, validation_data=None, **kwargs):
        self.fit_count += 1
        epochs = kwargs.get("epochs", 5)
        base_acc = 0.8 + np.random.random() * 0.1
        self.training_result = TrainingResult(
            history={
                "train_accuracy": np.linspace(0.5, base_acc, epochs).tolist(),
                "val_accuracy": np.linspace(0.45, base_acc - 0.05, epochs).tolist(),
            }
        )

    def predict(self, data, **kwargs):
        return np.zeros(len(data))

    def predict_proba(self, data, **kwargs):
        n = len(data)
        return np.random.rand(n, 2)

    def evaluate(self, data, **kwargs):
        return {"accuracy": 0.85 + np.random.random() * 0.05}

    def assess(self, true_labels):
        return {"accuracy": 0.85}

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        return cls(ModelConfig())


class MockDataHandler:
    """Mock data handler for testing."""

    @property
    def data_type(self):
        return "mock"

    @property
    def return_format(self):
        return "split_arrays"

    def load(self, **kwargs):
        X_train = np.random.rand(80, 10)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(0, 2, 20)

        return {"train_data": (X_train, y_train), "val_data": (X_val, y_val), "test_data": None}


class TestExperimentRunner:
    """Test ExperimentRunner class."""

    def test_runner_initialization(self):
        """Test creating experiment runner."""

        def model_builder(config):
            return MockModel(config)

        config = ModelConfig({"epochs": 5})
        data_handler = MockDataHandler()

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=data_handler,
            model_config=config,
            verbose=False,
        )

        assert runner.model_builder is model_builder
        assert runner.model_config is config
        assert runner.train_data is not None
        assert runner.val_data is not None

    def test_single_run(self):
        """Test running single training iteration."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 3}),
            verbose=False,
        )

        result = runner._run_single_fit(run_id=1, epochs=3)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "train_accuracy" in result.columns
        assert len(result) == 3  # 3 epochs

    def test_full_study(self):
        """Test running full variability study."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=False,
        )

        results = runner.run_study(num_runs=3, epochs_per_run=2)

        assert isinstance(results, VariabilityStudyResults)
        assert results.n_runs == 3
        assert "val_accuracy" in results.final_metrics
        assert len(results.final_metrics["val_accuracy"]) == 3

    def test_failure_handling(self):
        """Test handling of failed runs."""

        def failing_model_builder(config):
            if np.random.random() > 0.2:
                raise ValueError("Random failure")
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=failing_model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig(),
            verbose=False,
        )

        results = runner.run_study(num_runs=5, stop_on_failure_rate=0.8)

        # Verify the runner tracked everything consistently.
        total_attempted = results.n_runs + len(runner.failed_runs)
        assert total_attempted >= 1
        assert total_attempted <= 5
        assert results.n_runs == len(results.all_runs_metrics)

    def test_summary_stats(self):
        """Test getting summary statistics."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig(),
            verbose=False,
        )

        runner.run_study(num_runs=2)

        stats = runner.get_summary_stats()

        assert "total_runs" in stats
        assert "successful_runs" in stats
        assert stats["successful_runs"] == 2

    def test_seed_reproducibility(self):
        """Test that same seed produces identical results."""

        def model_builder(config):
            return MockModel(config)

        def run_with_seed(seed):
            runner = ExperimentRunner(
                model_builder=model_builder,
                data_handler=MockDataHandler(),
                model_config=ModelConfig({"epochs": 3}),
                seed=seed,
                verbose=False,
            )
            results = runner.run_study(num_runs=3, epochs_per_run=3)
            return results.final_metrics["val_accuracy"]

        # Same seed → same results
        run_a = run_with_seed(42)
        run_b = run_with_seed(42)
        assert run_a == run_b, f"Same seed produced different results: {run_a} vs {run_b}"

        # Different seed → different results
        run_c = run_with_seed(99)
        assert run_a != run_c, "Different seeds produced identical results"

    def test_seed_stored_in_results(self):
        """Test that seed is stored in VariabilityStudyResults."""

        def model_builder(config):
            return MockModel(config)

        # Explicit seed
        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            seed=42,
            verbose=False,
        )
        results = runner.run_study(num_runs=2)
        assert results.seed == 42

        # Auto-generated seed
        runner2 = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=False,
        )
        results2 = runner2.run_study(num_runs=2)
        assert results2.seed is not None
        assert isinstance(results2.seed, int)

    def test_seed_in_summary(self):
        """Test that seed appears in summarize() output."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"a": [1]})],
            final_metrics={"val_accuracy": [0.8]},
            final_test_metrics=[],
            seed=42,
        )
        summary = results.summarize()
        assert "Seed: 42" in summary


class TestVariabilityStudyResults:
    """Test VariabilityStudyResults class."""

    def test_results_creation(self):
        """Test creating results object."""
        df1 = pd.DataFrame(
            {"epoch": [1, 2], "train_accuracy": [0.6, 0.8], "val_accuracy": [0.5, 0.75]}
        )

        results = VariabilityStudyResults(
            all_runs_metrics=[df1],
            final_metrics={"val_accuracy": [0.75], "train_accuracy": [0.8]},
            final_test_metrics=[],
        )

        assert results.n_runs == 1
        assert len(results.final_metrics["val_accuracy"]) == 1

    def test_get_metric_values(self):
        """Test extracting metric values."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={
                "val_accuracy": [0.7, 0.8],
                "val_loss": [0.3, 0.2],
                "val_f1": [0.65, 0.75],
            },
            final_test_metrics=[],
        )

        assert results.get_metric_values("val_accuracy") == [0.7, 0.8]
        assert results.get_metric_values("val_f1") == [0.65, 0.75]

        with pytest.raises(KeyError, match="not found"):
            results.get_metric_values("nonexistent")

    def test_get_final_metrics(self):
        """Test extracting final metrics per run from DataFrames."""
        df1 = pd.DataFrame({"val_accuracy": [0.5, 0.6, 0.7], "val_loss": [0.5, 0.4, 0.3]})
        df2 = pd.DataFrame({"val_accuracy": [0.6, 0.7, 0.8], "val_loss": [0.4, 0.3, 0.2]})

        results = VariabilityStudyResults(
            all_runs_metrics=[df1, df2],
            final_metrics={"val_accuracy": [0.7, 0.8]},
            final_test_metrics=[],
        )

        final = results.get_final_metrics("val_accuracy")
        assert len(final) == 2
        assert final["run_1"] == 0.7
        assert final["run_2"] == 0.8

    def test_get_available_metrics(self):
        """Test listing available metrics."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.7], "val_loss": [0.3], "train_accuracy": [0.8]},
            final_test_metrics=[],
        )

        available = results.get_available_metrics()
        assert "train_accuracy" in available
        assert "val_accuracy" in available
        assert "val_loss" in available

    def test_to_dataframe(self):
        """Test converting results to DataFrame."""
        df1 = pd.DataFrame({"train_accuracy": [0.6, 0.8], "val_accuracy": [0.5, 0.75]})

        results = VariabilityStudyResults(
            all_runs_metrics=[df1],
            final_metrics={"val_accuracy": [0.75]},
            final_test_metrics=[{"test_acc": 0.72}],
        )

        summary_df = results.to_dataframe()

        assert len(summary_df) == 1
        assert "final_train_accuracy" in summary_df.columns
        assert "test_test_acc" in summary_df.columns
        assert summary_df.iloc[0]["final_val_accuracy"] == 0.75

    def test_summarize(self):
        """Test summary string generation."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"a": [1]})],
            final_metrics={"val_accuracy": [0.8, 0.85, 0.82]},
            final_test_metrics=[],
        )

        summary = results.summarize()

        assert "Successful runs: 1" in summary
        assert "val_accuracy:" in summary
        assert "Mean:" in summary


class TestConvenienceFunction:
    """Test run_variability_study convenience function."""

    def test_run_variability_study(self):
        """Test the main convenience function."""

        def model_builder(config):
            return MockModel(config)

        results = run_variability_study(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            num_runs=2,
            verbose=False,
        )

        assert isinstance(results, VariabilityStudyResults)
        assert results.n_runs == 2
        assert len(results.all_runs_metrics) == 2

    def test_collects_all_metrics(self):
        """Test that runner collects all metrics, not just accuracy."""

        class MultiMetricModel(BaseModelWrapper):
            def __init__(self, config):
                super().__init__(None, "multi")

            def _cleanup_implementation(self):
                pass

            def fit(self, train_data, validation_data=None, **kwargs):
                epochs = kwargs.get("epochs", 3)
                self.training_result = TrainingResult(
                    history={
                        "train_accuracy": np.linspace(0.5, 0.9, epochs).tolist(),
                        "train_loss": np.linspace(1.0, 0.2, epochs).tolist(),
                        "val_accuracy": np.linspace(0.4, 0.85, epochs).tolist(),
                        "val_loss": np.linspace(0.9, 0.25, epochs).tolist(),
                        "val_f1": np.linspace(0.3, 0.8, epochs).tolist(),
                    }
                )

            def predict(self, data, **kwargs):
                return np.zeros(len(data))

            def predict_proba(self, data, **kwargs):
                return np.random.rand(len(data), 2)

            def evaluate(self, data, **kwargs):
                return {"accuracy": 0.85, "f1": 0.8}

            def assess(self, true_labels):
                return {"accuracy": 0.85}

            def save_model(self, path):
                pass

            @classmethod
            def load_model(cls, path):
                return cls(ModelConfig())

        runner = ExperimentRunner(
            model_builder=lambda c: MultiMetricModel(c),
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 3}),
            verbose=False,
        )

        results = runner.run_study(num_runs=3, epochs_per_run=3)

        assert "val_accuracy" in results.final_metrics
        assert "val_loss" in results.final_metrics
        assert "val_f1" in results.final_metrics
        assert len(results.final_metrics["val_f1"]) == 3

    def test_seed_passthrough(self):
        """Test that seed passes through the convenience function."""

        def model_builder(config):
            return MockModel(config)

        results = run_variability_study(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            num_runs=2,
            seed=123,
            verbose=False,
        )

        assert results.seed == 123


class TestVariabilityStudyResultsExtended:
    """Extended tests for VariabilityStudyResults edge cases."""

    def test_to_dataframe_empty(self):
        """Test to_dataframe with no runs."""
        results = VariabilityStudyResults(
            all_runs_metrics=[], final_metrics={}, final_test_metrics=[]
        )
        df = results.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_multiple_runs_with_test_metrics(self):
        """Test to_dataframe with test metrics included."""
        df1 = pd.DataFrame({"val_accuracy": [0.7, 0.8], "val_loss": [0.3, 0.2]})
        df2 = pd.DataFrame({"val_accuracy": [0.75, 0.85], "val_loss": [0.25, 0.15]})

        results = VariabilityStudyResults(
            all_runs_metrics=[df1, df2],
            final_metrics={"val_accuracy": [0.8, 0.85]},
            final_test_metrics=[
                {"accuracy": 0.78, "f1": 0.75},
                {"accuracy": 0.82, "f1": 0.80},
            ],
        )
        summary = results.to_dataframe()
        assert len(summary) == 2
        assert "test_accuracy" in summary.columns
        assert "test_f1" in summary.columns
        assert summary.iloc[0]["test_accuracy"] == 0.78

    def test_get_final_metrics_missing_metric(self):
        """Test get_final_metrics with a metric not in the data."""
        df1 = pd.DataFrame({"val_accuracy": [0.8]})
        results = VariabilityStudyResults(
            all_runs_metrics=[df1], final_metrics={"val_accuracy": [0.8]}, final_test_metrics=[]
        )
        # Asking for a metric that doesn't exist in the DataFrames
        final = results.get_final_metrics("nonexistent")
        assert final == {}

    def test_summarize_multiple_metrics(self):
        """Test summarize with several metrics."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"a": [1]})],
            final_metrics={
                "val_accuracy": [0.80, 0.85, 0.82],
                "val_loss": [0.20, 0.15, 0.18],
                "train_accuracy": [0.90, 0.92, 0.91],
            },
            final_test_metrics=[],
            seed=99,
        )
        summary = results.summarize()
        assert "val_accuracy" in summary
        assert "val_loss" in summary
        assert "train_accuracy" in summary
        assert "Seed: 99" in summary
        assert "Mean:" in summary
        assert "Std:" in summary

    def test_n_runs(self):
        """Test n_runs property."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
            final_metrics={},
            final_test_metrics=[],
        )
        assert results.n_runs == 3

    def test_get_available_metrics_empty(self):
        """Test get_available_metrics with no metrics."""
        results = VariabilityStudyResults(
            all_runs_metrics=[], final_metrics={}, final_test_metrics=[]
        )
        assert results.get_available_metrics() == []


class TestGetSummaryStatsExtended:
    """Extended tests for ExperimentRunner.get_summary_stats."""

    def test_summary_stats_with_metrics(self):
        """Test summary stats include per-metric statistics."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 3}),
            verbose=False,
        )
        runner.run_study(num_runs=5, epochs_per_run=3)
        stats = runner.get_summary_stats()

        assert stats["total_runs"] == 5
        assert stats["successful_runs"] == 5
        assert stats["failed_runs"] == 0
        assert stats["failure_rate"] == 0.0
        assert "val_accuracy_mean" in stats
        assert "val_accuracy_std" in stats
        assert "val_accuracy_min" in stats
        assert "val_accuracy_max" in stats
        assert stats["val_accuracy_std"] >= 0
        assert stats["val_accuracy_min"] <= stats["val_accuracy_max"]
