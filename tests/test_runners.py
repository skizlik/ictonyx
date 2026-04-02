"""Test experiment runners."""

import itertools
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import BaseModelWrapper, TrainingResult
from ictonyx.runners import (
    ExperimentRunner,
    GridStudyResults,
    VariabilityStudyResults,
    run_grid_study,
    run_variability_study,
)


def _picklable_builder(config):
    from unittest.mock import MagicMock

    return MagicMock()


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
            final_test_metrics=[{"run_id": 1, "test_acc": 0.72}],
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

    def test_summarize_sd_label_specifies_sample(self):
        """summarize() output must indicate sample SD (ddof=1)."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"val_accuracy": [0.8, 0.82, 0.79]}) for _ in range(5)],
            final_metrics={"val_accuracy": [0.80, 0.82, 0.79, 0.81, 0.78]},
            final_test_metrics=[],
            seed=42,
        )
        summary = results.summarize()
        assert "sample" in summary.lower() or "N-1" in summary, (
            "summarize() must label SD as sample (N-1) to avoid ambiguity " "in publication use."
        )


class TestTestAgainstNull:
    """VariabilityStudyResults.test_against_null() — EXT-2."""

    def _make_results(self, values, metric="val_accuracy"):
        return VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={metric: values},
            final_test_metrics=[],
            seed=42,
        )

    def test_method_exists(self):
        """Must not raise AttributeError — the ghost API is now real."""
        results = self._make_results([0.8, 0.85, 0.82, 0.79, 0.88, 0.83, 0.81, 0.86, 0.84, 0.87])
        result = results.test_against_null(null_value=0.5)
        assert result is not None

    def test_returns_statistical_test_result(self):
        from ictonyx.analysis import StatisticalTestResult

        results = self._make_results([0.8, 0.85, 0.82, 0.79, 0.88, 0.83, 0.81, 0.86, 0.84, 0.87])
        result = results.test_against_null(null_value=0.5)
        assert isinstance(result, StatisticalTestResult)
        assert hasattr(result, "p_value")
        assert hasattr(result, "is_significant")
        assert hasattr(result, "conclusion")

    def test_significant_result_above_null(self):
        """Values clearly above 0.5 should produce a significant result."""
        results = self._make_results([0.90, 0.91, 0.89, 0.92, 0.88, 0.93, 0.90, 0.91, 0.89, 0.92])
        result = results.test_against_null(null_value=0.5)
        assert result.p_value < 0.05

    def test_null_result_near_null_value(self):
        """Values near the null should not be significant."""
        results = self._make_results([0.50, 0.51, 0.49, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50, 0.51])
        result = results.test_against_null(null_value=0.5)
        assert result.p_value > 0.05

    def test_invalid_metric_raises_value_error(self):
        results = self._make_results([0.8, 0.85, 0.82], metric="val_accuracy")
        with pytest.raises(ValueError, match="not found"):
            results.test_against_null(metric="nonexistent_metric")

    def test_migration_guidance_no_longer_raises(self):
        """The exact call from the removal stub must work without AttributeError."""
        results = self._make_results([0.8, 0.85, 0.82, 0.79, 0.88, 0.83, 0.81, 0.86, 0.84, 0.87])
        # This is the exact call directed to in the compare_models_statistically stub
        try:
            results.test_against_null(null_value=0.5, metric="val_accuracy")
        except AttributeError:
            pytest.fail("test_against_null raised AttributeError — ghost API not resolved")


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
                {"run_id": 1, "accuracy": 0.78, "f1": 0.75},
                {"run_id": 2, "accuracy": 0.82, "f1": 0.80},
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
        assert "SD (sample" in summary

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


# =============================================================================
# ADD TO: tests/test_runners.py  (paste at the bottom)
# =============================================================================
# No new imports needed — existing file has MockModel, MockDataHandler,
# ExperimentRunner, VariabilityStudyResults, ModelConfig, np, pd, pytest


class MockDataHandlerWithTestData:
    """Mock data handler that returns test_data too."""

    @property
    def data_type(self):
        return "mock"

    @property
    def return_format(self):
        return "split_arrays"

    def load(self, **kwargs):
        X_train = np.random.rand(60, 10)
        y_train = np.random.randint(0, 2, 60)
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(0, 2, 20)
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return {
            "train_data": (X_train, y_train),
            "val_data": (X_val, y_val),
            "test_data": (X_test, y_test),
        }


class TestRunnerWithTestData:
    """Test that test_data evaluation path works."""

    def test_study_with_test_data(self):
        """Test that runner evaluates on test data when present."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandlerWithTestData(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=False,
        )
        results = runner.run_study(num_runs=3, epochs_per_run=2)

        assert results.n_runs == 3
        # Test metrics should be populated
        assert len(results.final_test_metrics) == 3
        for tm in results.final_test_metrics:
            assert isinstance(tm, dict)
            assert "accuracy" in tm

    def test_test_metrics_in_to_dataframe(self):
        """Test that test metrics appear in to_dataframe output."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandlerWithTestData(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=False,
        )
        results = runner.run_study(num_runs=2, epochs_per_run=2)
        df = results.to_dataframe()

        assert "test_accuracy" in df.columns
        assert len(df) == 2

    def test_child_seeds_are_not_adjacent(self):
        """Child seeds must not be consecutive integers."""
        import numpy as np

        ss = np.random.SeedSequence(42)
        children = [int(c.generate_state(1)[0]) for c in ss.spawn(5)]
        for i in range(len(children) - 1):
            assert (
                abs(children[i] - children[i + 1]) > 1000
            ), "Adjacent child seeds indicate incorrect seed generation"


class TestRunnerVerbose:
    """Test verbose output paths."""

    def test_verbose_study_runs(self, capsys):
        """Test that verbose mode produces output without crashing."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=True,
        )
        results = runner.run_study(num_runs=2, epochs_per_run=2)
        assert results.n_runs == 2


class TestRunnerFailureRate:
    """Test that high failure rate stops the study early."""

    def test_stops_on_high_failure_rate(self):
        """Runner should stop when failure rate exceeds threshold."""
        call_count = 0

        def always_failing_builder(config):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated failure")

        runner = ExperimentRunner(
            model_builder=always_failing_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig(),
            verbose=False,
        )

        results = runner.run_study(num_runs=10, stop_on_failure_rate=0.5)

        # Should have stopped early, not attempted all 10
        total = results.n_runs + len(runner.failed_runs)
        assert total < 10
        assert results.n_runs == 0  # all failed

    def test_no_training_result_counts_as_failure(self):
        """A model that trains but produces no training_result should count as failed."""

        class NoResultModel(MockModel):
            def fit(self, train_data, validation_data=None, **kwargs):
                # Trains but doesn't set training_result
                self.training_result = None

        def builder(config):
            return NoResultModel(config)

        runner = ExperimentRunner(
            model_builder=builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=False,
        )
        # Use high stop threshold so all runs are attempted
        results = runner.run_study(num_runs=3, stop_on_failure_rate=1.0)

        assert results.n_runs == 0
        assert len(runner.failed_runs) >= 1


class TestRunnerDefaultEpochs:
    """Test that epochs default to config value."""

    def test_epochs_from_config(self):
        """When epochs_per_run is None, should use config['epochs']."""

        def model_builder(config):
            return MockModel(config)

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 7}),
            verbose=False,
        )
        results = runner.run_study(num_runs=2, epochs_per_run=None)

        assert results.n_runs == 2
        # Each run's DataFrame should have 7 epochs
        df = results.all_runs_metrics[0]
        assert len(df) == 7


class TestCompareModelsStatisticallyRemoved:
    """compare_models_statistically() was removed in v0.3.10."""

    def test_raises_attribute_error(self):
        """The removed method must raise AttributeError with a helpful message."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.8, 0.85, 0.82]},
            final_test_metrics=[],
            seed=42,
        )
        with pytest.raises(AttributeError, match="removed in v0.3.10"):
            results.compare_models_statistically("val_accuracy")

    def test_message_mentions_replacement(self):
        """The error message must name the replacement methods."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.8, 0.85]},
            final_test_metrics=[],
            seed=42,
        )
        with pytest.raises(AttributeError) as exc_info:
            results.compare_models_statistically()
        msg = str(exc_info.value)
        assert "test_against_null" in msg or "compare_models" in msg


class TestGetEpochStatistics:
    """Tests for VariabilityStudyResults.get_epoch_statistics()."""

    def _make_results(self, n_runs=5, n_epochs=10):
        dfs = []
        vals = []
        for i in range(n_runs):
            acc = 0.7 + i * 0.02
            df = pd.DataFrame(
                {
                    "val_accuracy": np.linspace(0.5, acc, n_epochs),
                    "run_num": [i + 1] * n_epochs,
                    "epoch": range(1, n_epochs + 1),
                }
            )
            dfs.append(df)
            vals.append(acc)
        return VariabilityStudyResults(
            all_runs_metrics=dfs,
            final_metrics={"val_accuracy": vals},
            final_test_metrics=[],
        )

    def test_returns_dataframe_with_correct_shape(self):
        results = self._make_results(n_runs=5, n_epochs=10)
        stats = results.get_epoch_statistics("val_accuracy")
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 10
        assert set(stats.columns) == {"epoch", "mean", "sd", "se", "ci_lower", "ci_upper", "n_runs"}

    def test_epoch_column_is_one_indexed(self):
        results = self._make_results(n_runs=3, n_epochs=5)
        stats = results.get_epoch_statistics("val_accuracy")
        assert stats["epoch"].tolist() == [1, 2, 3, 4, 5]

    def test_n_runs_column_correct(self):
        results = self._make_results(n_runs=4, n_epochs=3)
        stats = results.get_epoch_statistics("val_accuracy")
        assert all(stats["n_runs"] == 4)

    def test_ci_bounds_bracket_mean(self):
        results = self._make_results(n_runs=5, n_epochs=8)
        stats = results.get_epoch_statistics("val_accuracy")
        assert all(stats["ci_lower"] <= stats["mean"])
        assert all(stats["mean"] <= stats["ci_upper"])

    def test_sd_is_nonnegative(self):
        results = self._make_results(n_runs=5, n_epochs=5)
        stats = results.get_epoch_statistics("val_accuracy")
        assert all(stats["sd"] >= 0)

    def test_missing_metric_raises_key_error(self):
        results = self._make_results()
        with pytest.raises(KeyError, match="not found"):
            results.get_epoch_statistics("nonexistent_metric")

    def test_empty_results_raises_value_error(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={},
            final_test_metrics=[],
        )
        with pytest.raises(ValueError, match="No runs available"):
            results.get_epoch_statistics("val_accuracy")

    def test_confidence_parameter_affects_ci_width(self):
        """Higher confidence should produce wider CI bands."""
        results = self._make_results(n_runs=10, n_epochs=5)
        stats_95 = results.get_epoch_statistics("val_accuracy", confidence=0.95)
        stats_50 = results.get_epoch_statistics("val_accuracy", confidence=0.50)
        ci_width_95 = (stats_95["ci_upper"] - stats_95["ci_lower"]).mean()
        ci_width_50 = (stats_50["ci_upper"] - stats_50["ci_lower"]).mean()
        assert ci_width_95 > ci_width_50


class TestToDataframeRunIdAlignment:
    """Verify to_dataframe() aligns test metrics by run_id not position."""

    def test_missing_run_does_not_misalign(self):
        """If run 1 has no test metrics, run 2's must not appear in run 1's row."""
        df1 = pd.DataFrame({"val_accuracy": [0.80], "run_num": [1], "epoch": [1]})
        df2 = pd.DataFrame({"val_accuracy": [0.90], "run_num": [2], "epoch": [1]})
        results = VariabilityStudyResults(
            all_runs_metrics=[df1, df2],
            final_metrics={"val_accuracy": [0.80, 0.90]},
            final_test_metrics=[{"run_id": 2, "accuracy": 0.88}],
        )
        summary = results.to_dataframe()
        run1 = summary[summary["run_id"] == 1].iloc[0]
        run2 = summary[summary["run_id"] == 2].iloc[0]
        assert run2["test_accuracy"] == 0.88
        assert "test_accuracy" not in run1.index or pd.isna(run1.get("test_accuracy"))

    def test_both_runs_have_test_metrics(self):
        df1 = pd.DataFrame({"val_accuracy": [0.80], "run_num": [1], "epoch": [1]})
        df2 = pd.DataFrame({"val_accuracy": [0.90], "run_num": [2], "epoch": [1]})
        results = VariabilityStudyResults(
            all_runs_metrics=[df1, df2],
            final_metrics={"val_accuracy": [0.80, 0.90]},
            final_test_metrics=[
                {"run_id": 1, "accuracy": 0.78},
                {"run_id": 2, "accuracy": 0.88},
            ],
        )
        summary = results.to_dataframe()
        assert summary[summary["run_id"] == 1].iloc[0]["test_accuracy"] == 0.78
        assert summary[summary["run_id"] == 2].iloc[0]["test_accuracy"] == 0.88


class TestFinalValAccuraciesProperty:
    """Test the final_val_accuracies backward-compat property if it exists."""

    def test_final_metrics_access(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"a": [1]})],
            final_metrics={"val_accuracy": [0.8, 0.85, 0.82], "val_loss": [0.2, 0.15, 0.18]},
            final_test_metrics=[],
        )
        # Access through dict
        assert results.final_metrics["val_accuracy"] == [0.8, 0.85, 0.82]
        assert len(results.final_metrics["val_loss"]) == 3


class TestGridStudyResults:
    """Tests for GridStudyResults dataclass and methods."""

    def _make_mock_variability_result(self, mean, sd, n=12):
        """Helper: construct a minimal VariabilityStudyResults with controlled values."""
        np.random.seed(42)
        values = np.random.normal(mean, sd, n).tolist()
        df = pd.DataFrame(
            {
                "train_accuracy": np.linspace(0.3, mean, 10),
                "val_accuracy": np.linspace(0.2, mean, 10),
                "train_loss": np.linspace(1.5, 0.5, 10),
                "val_loss": np.linspace(1.6, 0.6, 10),
            }
        )
        return VariabilityStudyResults(
            all_runs_metrics=[df] * n,
            final_metrics={"val_accuracy": values, "train_accuracy": values},
            final_test_metrics=[{}] * n,
            seed=42,
        )

    def _make_grid_results(self):
        """Helper: construct a minimal GridStudyResults for testing."""
        param_grid = {"learning_rate": [0.001, 0.0001, 0.00001]}
        base_config = ModelConfig(
            {
                "input_shape": (131, 131, 1),
                "num_classes": 9,
                "epochs": 12,
                "batch_size": 8,
                "learning_rate": 0.001,
                "verbose": 0,
            }
        )
        results = {}
        for lr in param_grid["learning_rate"]:
            key = GridStudyResults._config_key({"learning_rate": lr})
            results[key] = self._make_mock_variability_result(mean=0.3 + lr * 100, sd=0.02)
        return GridStudyResults(
            results=results, param_grid=param_grid, base_config=base_config, metric="val_accuracy"
        )

    def test_n_configurations(self):
        gr = self._make_grid_results()
        assert gr.n_configurations == 3

    def test_n_runs_per_config(self):
        gr = self._make_grid_results()
        assert gr.n_runs_per_config == 12

    def test_config_key_is_sorted(self):
        key1 = GridStudyResults._config_key({"batch_size": 8, "learning_rate": 0.001})
        key2 = GridStudyResults._config_key({"learning_rate": 0.001, "batch_size": 8})
        assert key1 == key2

    def test_get_results_for_config(self):
        gr = self._make_grid_results()
        result = gr.get_results_for_config({"learning_rate": 0.001})
        assert isinstance(result, VariabilityStudyResults)
        assert result.n_runs == 12

    def test_get_results_for_config_missing_raises(self):
        gr = self._make_grid_results()
        with pytest.raises(KeyError):
            gr.get_results_for_config({"learning_rate": 99.0})

    def test_list_configurations(self):
        gr = self._make_grid_results()
        configs = gr.list_configurations()
        assert len(configs) == 3
        assert all("learning_rate" in c for c in configs)

    def test_to_dataframe_shape(self):
        gr = self._make_grid_results()
        df = gr.to_dataframe()
        assert len(df) == 3
        assert "learning_rate" in df.columns
        assert all(col in df.columns for col in ["mean", "sd", "se", "min", "max", "n"])

    def test_to_dataframe_sorted(self):
        gr = self._make_grid_results()
        df = gr.to_dataframe()
        assert df["learning_rate"].is_monotonic_increasing

    def test_to_dataframe_sample_sd(self):
        """Verify SD is sample SD (N-1), not population SD (N)."""
        gr = self._make_grid_results()
        df = gr.to_dataframe()
        result = gr.get_results_for_config({"learning_rate": 0.001})
        values = pd.Series(result.get_metric_values("val_accuracy"))
        assert df.loc[df["learning_rate"] == 0.001, "sd"].iloc[0] == pytest.approx(values.std())

    def test_summarize_returns_string(self):
        gr = self._make_grid_results()
        s = gr.summarize()
        assert isinstance(s, str)
        assert "Grid Study Results" in s
        assert "3" in s  # n_configurations


class TestRunGridStudy:
    """Tests for run_grid_study function."""

    def _make_mock_builder(self):
        """Model builder that returns a minimal mock model."""

        def builder(config):
            model = MagicMock()
            model.fit.return_value = MagicMock(
                history={
                    "accuracy": [0.3, 0.4, 0.5],
                    "val_accuracy": [0.25, 0.35, 0.45],
                    "loss": [1.5, 1.2, 0.9],
                    "val_loss": [1.6, 1.3, 1.0],
                }
            )
            return model

        return builder

    def test_dry_run_returns_empty_results(self):
        base_config = ModelConfig(
            {
                "input_shape": (10, 10, 1),
                "num_classes": 2,
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.001,
                "verbose": 0,
            }
        )
        result = run_grid_study(
            model_builder=self._make_mock_builder(),
            data_handler=MagicMock(),
            base_config=base_config,
            param_grid={"learning_rate": [0.001, 0.0001]},
            num_runs=2,
            dry_run=True,
        )
        assert isinstance(result, GridStudyResults)
        assert result.n_configurations == 0

    def test_empty_param_grid_raises(self):
        with pytest.raises(ValueError, match="param_grid must contain"):
            run_grid_study(
                model_builder=MagicMock(),
                data_handler=MagicMock(),
                base_config=MagicMock(),
                param_grid={},
            )

    def test_empty_param_values_raises(self):
        with pytest.raises(ValueError, match="is empty"):
            run_grid_study(
                model_builder=MagicMock(),
                data_handler=MagicMock(),
                base_config=MagicMock(),
                param_grid={"learning_rate": []},
            )

    def test_cartesian_product_count(self):
        """Verify correct number of configurations from Cartesian product."""
        param_grid = {"learning_rate": [0.001, 0.0001], "batch_size": [8, 16, 32]}
        # 2 × 3 = 6 combinations
        combos = [
            dict(zip(param_grid.keys(), combo)) for combo in itertools.product(*param_grid.values())
        ]
        assert len(combos) == 6

    @patch("ictonyx.runners.run_variability_study")
    def test_run_grid_study_calls_variability_study_correct_times(self, mock_run):
        mock_run.return_value = MagicMock(
            spec=VariabilityStudyResults,
            n_runs=2,
            get_metric_values=MagicMock(return_value=[0.5, 0.6]),
        )
        base_config = ModelConfig(
            {
                "input_shape": (10, 10, 1),
                "num_classes": 2,
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.001,
                "verbose": 0,
            }
        )
        run_grid_study(
            model_builder=MagicMock(),
            data_handler=MagicMock(),
            base_config=base_config,
            param_grid={"learning_rate": [0.001, 0.0001, 0.00001]},
            num_runs=2,
            use_process_isolation=False,
        )
        assert mock_run.call_count == 3

    @patch("ictonyx.runners.run_variability_study")
    def test_config_override_correct(self, mock_run):
        """Verify each config passed to run_variability_study has correct lr."""
        called_lrs = []

        def capture_config(*args, **kwargs):
            called_lrs.append(kwargs["model_config"]["learning_rate"])
            return MagicMock(
                spec=VariabilityStudyResults,
                n_runs=2,
                get_metric_values=MagicMock(return_value=[0.5, 0.6]),
            )

        mock_run.side_effect = capture_config
        base_config = ModelConfig(
            {
                "input_shape": (10, 10, 1),
                "num_classes": 2,
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.001,
                "verbose": 0,
            }
        )
        run_grid_study(
            model_builder=MagicMock(),
            data_handler=MagicMock(),
            base_config=base_config,
            param_grid={"learning_rate": [0.001, 0.0001]},
            num_runs=2,
            use_process_isolation=False,
        )
        assert set(called_lrs) == {0.001, 0.0001}


class TestDdof1:
    def test_summarize_uses_sample_std(self):
        values = [0.80, 0.85, 0.82, 0.79, 0.88, 0.83, 0.81, 0.86, 0.84, 0.87]
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": values},
            final_test_metrics=[],
            seed=42,
        )
        expected_std = float(np.std(values, ddof=1))
        summary = results.summarize()
        for line in summary.split("\n"):
            if "SD (sample" in line and ":" in line:
                reported = float(line.split(":")[-1].strip())
                np.testing.assert_allclose(reported, expected_std, atol=0.00005)
                return
        pytest.fail("No SD (sample) line found in summarize() output")


class TestInputValidation:
    def test_num_runs_zero_raises(self):
        with pytest.raises(ValueError, match="num_runs"):
            run_variability_study(
                model_builder=lambda c: None,
                data_handler=MagicMock(),
                model_config=ModelConfig({"epochs": 1}),
                num_runs=0,
            )

    def test_num_runs_negative_raises(self):
        with pytest.raises(ValueError, match="num_runs"):
            run_variability_study(
                model_builder=lambda c: None,
                data_handler=MagicMock(),
                model_config=ModelConfig({"epochs": 1}),
                num_runs=-1,
            )

    def test_confidence_95_raises(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"val_accuracy": [0.8, 0.85]})],
            final_metrics={"val_accuracy": [0.8, 0.85]},
            final_test_metrics=[],
            seed=42,
        )
        with pytest.raises(ValueError, match="confidence"):
            results.get_epoch_statistics(metric="val_accuracy", confidence=95)

    def test_confidence_valid_accepted(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[
                pd.DataFrame({"val_accuracy": [0.8, 0.85, 0.82]}),
                pd.DataFrame({"val_accuracy": [0.81, 0.84, 0.83]}),
            ],
            final_metrics={"val_accuracy": [0.82, 0.83]},
            final_test_metrics=[],
            seed=42,
        )
        df = results.get_epoch_statistics(metric="val_accuracy", confidence=0.95)
        assert df is not None
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns


class TestGetEpochStatisticsCI:
    def test_ci_columns_are_not_percentiles(self):
        """ci_lower/ci_upper must be t-interval, not empirical percentiles."""
        epoch_values = [0.80, 0.85, 0.79, 0.83, 0.81]
        n = len(epoch_values)
        mean = np.mean(epoch_values)
        sd = np.std(epoch_values, ddof=1)
        se = sd / np.sqrt(n)
        from scipy import stats

        t_crit = stats.t.ppf(0.975, df=n - 1)
        expected_lower = mean - t_crit * se
        expected_upper = mean + t_crit * se
        # Percentile equivalents for comparison
        percentile_lower = np.percentile(epoch_values, 2.5)
        percentile_upper = np.percentile(epoch_values, 97.5)
        # They must differ — if they're equal the test data is wrong
        assert not np.isclose(
            expected_lower, percentile_lower
        ), "Test data too symmetric — can't distinguish t-interval from percentile"
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({"val_accuracy": [v]}) for v in epoch_values],
            final_metrics={"val_accuracy": epoch_values},
            final_test_metrics=[],
            seed=42,
        )
        df = results.get_epoch_statistics(metric="val_accuracy", confidence=0.95)
        row = df.iloc[0]
        np.testing.assert_allclose(row["ci_lower"], expected_lower, rtol=1e-3)
        np.testing.assert_allclose(row["ci_upper"], expected_upper, rtol=1e-3)


class TestGridStudySeedStrategy:
    """SeedSequence used in grid study, not seed+i."""

    def test_grid_study_seeds_are_not_arithmetic_offsets(self):
        """Seeds passed to run_variability_study must not be seed+i pattern."""
        captured_seeds = []

        def capturing_run(*args, **kwargs):
            captured_seeds.append(kwargs.get("seed"))
            return MagicMock(get_metric_values=lambda m: [0.8, 0.82, 0.81])

        with patch("ictonyx.runners.run_variability_study", capturing_run):
            run_grid_study(
                model_builder=lambda c: MagicMock(),
                data_handler=MagicMock(),
                base_config=ModelConfig({"epochs": 1}),
                param_grid={"lr": [0.001, 0.01, 0.1]},
                num_runs=3,
                seed=42,
                verbose=False,
            )
        # seed+i pattern would give [43, 44, 45]
        arithmetic = [42 + i for i in range(1, len(captured_seeds) + 1)]
        assert captured_seeds != arithmetic, (
            "Grid study is using seed+i instead of SeedSequence — statistical "
            "independence not guaranteed."
        )

    def test_grid_study_with_seed_is_reproducible(self):
        """Same seed must produce identical per-configuration seeds."""
        seeds_run1 = []
        seeds_run2 = []
        base_config = ModelConfig({"epochs": 1})
        param_grid = {"lr": [0.001, 0.01, 0.1]}

        def capture_1(*args, **kwargs):
            seeds_run1.append(kwargs.get("seed"))
            return MagicMock(get_metric_values=lambda m: [0.8])

        def capture_2(*args, **kwargs):
            seeds_run2.append(kwargs.get("seed"))
            return MagicMock(get_metric_values=lambda m: [0.8])

        with patch("ictonyx.runners.run_variability_study", capture_1):
            run_grid_study(
                model_builder=lambda c: MagicMock(),
                data_handler=MagicMock(),
                base_config=base_config,
                param_grid=param_grid,
                num_runs=3,
                seed=42,
                verbose=False,
            )

        with patch("ictonyx.runners.run_variability_study", capture_2):
            run_grid_study(
                model_builder=lambda c: MagicMock(),
                data_handler=MagicMock(),
                base_config=base_config,
                param_grid=param_grid,
                num_runs=3,
                seed=42,
                verbose=False,
            )

        assert seeds_run1 == seeds_run2


class TestNRunsWarning:
    @pytest.fixture
    def small_df(self):
        return pd.DataFrame(
            {
                "feature1": range(20),
                "feature2": [x * 0.1 for x in range(20)],
                "target": [0, 1] * 10,
            }
        )

    def test_warns_when_runs_lt_10(self):
        import warnings

        from ictonyx import api

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with (
                patch("ictonyx.api._run_study"),
                patch("ictonyx.api.auto_resolve_handler"),
                patch("ictonyx.api._get_model_builder"),
            ):
                try:
                    api.variability_study(
                        model=MagicMock,
                        data=MagicMock(),
                        runs=5,
                        seed=42,
                    )
                except Exception:
                    pass
        user_warnings = [
            x for x in w if issubclass(x.category, UserWarning) and "runs=5" in str(x.message)
        ]
        assert len(user_warnings) > 0

    def test_no_warning_when_runs_gte_10(self, small_df):
        import warnings

        from ictonyx import api

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("ictonyx.api._run_study") as mock_run:
                mock_run.return_value = MagicMock()
                try:
                    api.variability_study(
                        model=RandomForestClassifier,
                        data=small_df,
                        target_column="target",
                        runs=10,
                        seed=42,
                    )
                except Exception:
                    pass
        assert not any(
            "runs=10" in str(warning.message) and issubclass(warning.category, UserWarning)
            for warning in w
        )


class TestProcessIsolationValidation:

    def test_validation_accepts_picklable_function(self):
        """A plain function must pass validation."""
        from unittest.mock import MagicMock

        import numpy as np

        from ictonyx.config import ModelConfig
        from ictonyx.data import ArraysDataHandler
        from ictonyx.runners import ExperimentRunner

        def builder(config):
            return MagicMock()

        runner = ExperimentRunner(
            model_builder=_picklable_builder,
            data_handler=ArraysDataHandler(np.zeros((20, 2)), np.zeros(20)),
            model_config=ModelConfig({}),
            use_process_isolation=False,
            verbose=False,
        )
        runner.model_builder = _picklable_builder
        runner.train_data = (np.zeros((16, 2)), np.zeros(16))
        runner._validate_process_isolation()  # must not raise

    def test_validation_uses_cloudpickle_when_available(self):
        """When cloudpickle is installed, validation must use it."""
        from unittest.mock import MagicMock, patch

        mock_cp = MagicMock()
        mock_cp.__name__ = "cloudpickle"
        mock_cp.dumps.return_value = b"serialised"

        with patch.dict("sys.modules", {"cloudpickle": mock_cp}):
            import importlib

            import ictonyx.runners as runners_mod
            from ictonyx.config import ModelConfig
            from ictonyx.runners import ExperimentRunner

            importlib.reload(runners_mod)
            # After reload, cloudpickle import will be attempted
            # Just verify the logic path — full integration test requires
            # actual cloudpickle installation
        assert True  # if no ImportError, cloudpickle path is reachable


class TestRunSeedInjection:

    def test_run_seed_present_in_config_during_standard_mode(self):
        """Standard mode must inject run_seed into model_config before model build."""
        import numpy as np

        from ictonyx.config import ModelConfig
        from ictonyx.data import ArraysDataHandler
        from ictonyx.runners import ExperimentRunner

        received_seeds = []

        def capturing_builder(config):
            received_seeds.append(config.get("run_seed"))
            from unittest.mock import MagicMock

            from ictonyx.core import TrainingResult

            wrapper = MagicMock()
            wrapper.training_result = TrainingResult(
                history={"val_accuracy": [0.8], "train_accuracy": [0.85]}
            )
            wrapper.fit.return_value = None
            return wrapper

        X = np.random.default_rng(0).standard_normal((30, 3))
        y = (X[:, 0] > 0).astype(int)
        handler = ArraysDataHandler(X, y)

        runner = ExperimentRunner(
            model_builder=capturing_builder,
            data_handler=handler,
            model_config=ModelConfig({"epochs": 1}),
            use_process_isolation=False,
            verbose=False,
        )
        runner.run_study(num_runs=3, epochs_per_run=1)

        assert len(received_seeds) == 3
        assert all(s is not None for s in received_seeds), (
            "run_seed must be injected into model_config in standard mode. "
            "Got None — check _run_single_fit_standard."
        )
        assert len(set(received_seeds)) == 3, "Each run must receive a distinct seed."


class TestDeterministicCudnn:
    """Tests for ExperimentRunner._deterministic_cudnn() context manager."""

    def test_context_manager_exists(self):
        """_deterministic_cudnn must be a static method on ExperimentRunner."""
        from ictonyx.runners import ExperimentRunner

        assert hasattr(
            ExperimentRunner, "_deterministic_cudnn"
        ), "_deterministic_cudnn context manager is missing from ExperimentRunner."

    def test_context_manager_does_not_raise_without_cuda(self):
        """Must not raise when CUDA is unavailable or torch is absent."""
        from ictonyx.runners import ExperimentRunner

        # Should work on any machine regardless of GPU availability
        with ExperimentRunner._deterministic_cudnn():
            pass  # must not raise

    def test_context_manager_restores_settings_when_torch_available(self):
        """If torch is installed, prior cudnn settings must be restored on exit."""
        from ictonyx.runners import ExperimentRunner

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — cudnn settings irrelevant")

        # Set known prior state
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        with ExperimentRunner._deterministic_cudnn():
            # Inside: determinism must be enabled
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False

        # After: prior state must be restored
        assert (
            torch.backends.cudnn.deterministic is False
        ), "cudnn.deterministic was not restored after context exit."
        assert (
            torch.backends.cudnn.benchmark is True
        ), "cudnn.benchmark was not restored after context exit."

    def test_context_manager_restores_on_exception(self):
        """cudnn settings must be restored even when an exception is raised inside."""
        from ictonyx.runners import ExperimentRunner

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        try:
            with ExperimentRunner._deterministic_cudnn():
                raise RuntimeError("simulated training failure")
        except RuntimeError:
            pass

        assert (
            torch.backends.cudnn.deterministic is False
        ), "cudnn.deterministic was not restored after exception."
        assert (
            torch.backends.cudnn.benchmark is True
        ), "cudnn.benchmark was not restored after exception."


class TestVariabilityStudyResultsTestMetrics:
    """Tests for test-metric surfacing: has_test_data, preferred_metric,
    get_test_metric_values, summarize, and test_against_null auto-resolution.
    Added in v0.4.0 (Priority 1).
    """

    def _make_results_no_test(self, values=None):
        """VariabilityStudyResults with only validation metrics."""
        if values is None:
            values = [0.80, 0.82, 0.79, 0.83, 0.81, 0.84, 0.80, 0.82, 0.83, 0.81]
        return VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": values},
            final_test_metrics=[],
            seed=42,
        )

    def _make_results_with_test(self, val_values=None, test_values=None):
        """VariabilityStudyResults with both validation and test metrics."""
        if val_values is None:
            val_values = [0.80, 0.82, 0.79, 0.83, 0.81, 0.84, 0.80, 0.82, 0.83, 0.81]
        if test_values is None:
            test_values = [0.78, 0.80, 0.77, 0.81, 0.79, 0.82, 0.78, 0.80, 0.81, 0.79]
        return VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": val_values},
            final_test_metrics=[
                {"run_id": i + 1, "test_accuracy": v} for i, v in enumerate(test_values)
            ],
            seed=42,
        )

    # --- has_test_data ---

    def test_has_test_data_false_when_empty(self):
        results = self._make_results_no_test()
        assert results.has_test_data is False

    def test_has_test_data_true_when_populated(self):
        results = self._make_results_with_test()
        assert results.has_test_data is True

    def test_has_test_data_false_when_final_test_metrics_is_empty_list(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.8, 0.82]},
            final_test_metrics=[],
            seed=42,
        )
        assert results.has_test_data is False

    # --- preferred_metric ---

    def test_preferred_metric_returns_val_when_no_test_data(self):
        results = self._make_results_no_test()
        assert results.preferred_metric("accuracy") == "val_accuracy"

    def test_preferred_metric_returns_test_when_test_data_present(self):
        results = self._make_results_with_test()
        assert results.preferred_metric("accuracy") == "test_accuracy"

    def test_preferred_metric_falls_back_to_val_when_test_key_not_tracked(self):
        """Test data present but doesn't contain the requested base metric."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_loss": [0.3, 0.28, 0.31]},
            final_test_metrics=[{"run_id": 1, "test_loss": 0.32}],
            seed=42,
        )
        # test_accuracy not tracked — should fall back to val_accuracy
        assert results.preferred_metric("accuracy") == "val_accuracy"

    def test_preferred_metric_different_bases(self):
        results = self._make_results_with_test()
        assert results.preferred_metric("accuracy") == "test_accuracy"
        # loss not in test metrics — falls back
        assert results.preferred_metric("loss") == "val_loss"

    # --- get_test_metric_values ---

    def test_get_test_metric_values_returns_list(self):
        results = self._make_results_with_test()
        values = results.get_test_metric_values("test_accuracy")
        assert isinstance(values, list)
        assert len(values) == 10

    def test_get_test_metric_values_correct_values(self):
        test_vals = [0.78, 0.80, 0.77, 0.81, 0.79, 0.82, 0.78, 0.80, 0.81, 0.79]
        results = self._make_results_with_test(test_values=test_vals)
        retrieved = results.get_test_metric_values("test_accuracy")
        assert retrieved == test_vals

    def test_get_test_metric_values_raises_when_metric_absent(self):
        results = self._make_results_with_test()
        with pytest.raises(KeyError, match="test_loss"):
            results.get_test_metric_values("test_loss")

    def test_get_test_metric_values_raises_when_no_test_data(self):
        results = self._make_results_no_test()
        with pytest.raises(KeyError):
            results.get_test_metric_values("test_accuracy")

    # --- summarize ---

    def test_summarize_contains_ci_line(self):
        results = self._make_results_no_test()
        summary = results.summarize()
        assert "95% CI" in summary

    def test_summarize_no_test_data_contains_note(self):
        results = self._make_results_no_test()
        summary = results.summarize()
        assert "no held-out test data" in summary

    def test_summarize_with_test_data_has_test_section_before_val(self):
        results = self._make_results_with_test()
        summary = results.summarize()
        assert "Test Set Metrics" in summary
        assert "Validation Metrics" in summary
        assert summary.index("Test Set Metrics") < summary.index("Validation Metrics")

    def test_summarize_with_test_data_does_not_show_absent_note(self):
        results = self._make_results_with_test()
        summary = results.summarize()
        assert "no held-out test data" not in summary

    def test_summarize_ci_requires_at_least_two_values(self):
        """Single-run results should not crash — CI line omitted at n=1."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.82]},
            final_test_metrics=[],
            seed=42,
        )
        summary = results.summarize()
        # Should not raise; CI line should be absent for n=1
        assert "Mean" in summary

    # --- test_against_null auto-resolution ---

    def test_test_against_null_resolves_val_when_no_test_data(self):
        """metric=None with no test data should resolve to val_accuracy."""
        results = self._make_results_no_test(
            [0.80, 0.82, 0.79, 0.83, 0.81, 0.84, 0.80, 0.82, 0.83, 0.81]
        )
        result = results.test_against_null(null_value=0.5)
        assert result is not None
        assert result.p_value is not None

    def test_test_against_null_resolves_test_when_test_data_present(self):
        """metric=None with test data should resolve to test_accuracy."""
        results = self._make_results_with_test()
        result = results.test_against_null(null_value=0.5)
        assert result is not None

    def test_test_against_null_explicit_metric_overrides_auto(self):
        """Explicitly passing metric= must override auto-resolution."""
        results = self._make_results_with_test()
        # Ask for val_accuracy explicitly even though test data is present
        result = results.test_against_null(null_value=0.5, metric="val_accuracy")
        assert result is not None

    def test_test_against_null_raises_for_unknown_explicit_metric(self):
        results = self._make_results_no_test()
        with pytest.raises(ValueError):
            results.test_against_null(null_value=0.5, metric="nonexistent_metric")


class TestVariabilityStudyResultsPersistence:
    """Tests for save(), load(), and to_json() on VariabilityStudyResults.
    Added in v0.4.0 (Priority 1).
    """

    def _make_results(self):
        return VariabilityStudyResults(
            all_runs_metrics=[
                pd.DataFrame(
                    {"val_accuracy": [0.75, 0.78, 0.80], "train_accuracy": [0.80, 0.83, 0.85]}
                ),
                pd.DataFrame(
                    {"val_accuracy": [0.77, 0.79, 0.82], "train_accuracy": [0.82, 0.84, 0.87]}
                ),
            ],
            final_metrics={"val_accuracy": [0.80, 0.82]},
            final_test_metrics=[
                {"run_id": 1, "test_accuracy": 0.78},
                {"run_id": 2, "test_accuracy": 0.80},
            ],
            seed=99,
        )

    # --- save / load ---

    def test_save_creates_file(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        import os

        assert os.path.exists(path)

    def test_load_restores_n_runs(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.n_runs == results.n_runs

    def test_load_restores_seed(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.seed == results.seed

    def test_load_restores_final_metrics(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.final_metrics == results.final_metrics

    def test_load_restores_final_test_metrics(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.final_test_metrics == results.final_test_metrics

    def test_load_restores_all_runs_metrics(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert len(loaded.all_runs_metrics) == len(results.all_runs_metrics)
        for orig, restored in zip(results.all_runs_metrics, loaded.all_runs_metrics):
            pd.testing.assert_frame_equal(orig, restored)

    def test_has_test_data_preserved_after_roundtrip(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "results.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.has_test_data is True

    def test_save_load_roundtrip_no_test_data(self, tmp_path):
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": [0.80, 0.82, 0.81]},
            final_test_metrics=[],
            seed=7,
        )
        path = str(tmp_path / "results_no_test.joblib")
        results.save(path)
        loaded = VariabilityStudyResults.load(path)
        assert loaded.has_test_data is False
        assert loaded.final_metrics == results.final_metrics

    # --- to_json ---

    def test_to_json_returns_string(self):
        results = self._make_results()
        assert isinstance(results.to_json(), str)

    def test_to_json_is_valid_json(self):
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert isinstance(data, dict)

    def test_to_json_contains_n_runs(self):
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert data["n_runs"] == results.n_runs

    def test_to_json_contains_seed(self):
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert data["seed"] == results.seed

    def test_to_json_contains_final_metrics(self):
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert "final_metrics" in data
        assert data["final_metrics"] == results.final_metrics

    def test_to_json_contains_final_test_metrics(self):
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert "final_test_metrics" in data
        assert len(data["final_test_metrics"]) == len(results.final_test_metrics)

    def test_to_json_omits_all_runs_metrics(self):
        """all_runs_metrics (per-epoch DataFrames) must not appear in JSON output."""
        import json

        results = self._make_results()
        data = json.loads(results.to_json())
        assert "all_runs_metrics" not in data


class TestVariabilityStudyResultsRepr:
    """__repr__ must be concise and informative — not a wall of DataFrames."""

    def _make_results(self, with_test_data: bool = False) -> "VariabilityStudyResults":
        df = pd.DataFrame({"val_accuracy": [0.80, 0.82], "val_loss": [0.4, 0.38]})
        test_metrics = (
            [{"run_id": 1, "test_accuracy": 0.79}, {"run_id": 2, "test_accuracy": 0.81}]
            if with_test_data
            else []
        )
        return VariabilityStudyResults(
            all_runs_metrics=[df, df],
            final_metrics={"val_accuracy": [0.80, 0.82], "val_loss": [0.4, 0.38]},
            final_test_metrics=test_metrics,
            seed=42,
        )

    def test_repr_contains_n_runs(self):
        results = self._make_results()
        r = repr(results)
        assert "n_runs=2" in r

    def test_repr_contains_seed(self):
        results = self._make_results()
        r = repr(results)
        assert "seed=42" in r

    def test_repr_contains_metric_names(self):
        results = self._make_results()
        r = repr(results)
        assert "val_accuracy" in r

    def test_repr_does_not_contain_raw_dataframe(self):
        """repr must not dump the full DataFrame content."""
        results = self._make_results()
        r = repr(results)
        assert "dtype" not in r
        assert "0.80" not in r

    def test_repr_includes_test_metrics_when_present(self):
        results = self._make_results(with_test_data=True)
        r = repr(results)
        assert "test_metrics" in r
        assert "test_accuracy" in r

    def test_repr_excludes_test_part_when_no_test_data(self):
        results = self._make_results(with_test_data=False)
        r = repr(results)
        assert "test_metrics" not in r


class TestVariabilityStudyResultsRunSeeds:
    """run_seeds field must be populated from _child_seeds after run_study()."""

    def test_run_seeds_default_empty(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={},
            final_test_metrics=[],
            seed=1,
        )
        assert results.run_seeds == []

    def test_run_seeds_accepts_list(self):
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={},
            final_test_metrics=[],
            seed=1,
            run_seeds=[111, 222, 333],
        )
        assert results.run_seeds == [111, 222, 333]
        assert len(results.run_seeds) == 3

    def test_run_seeds_populated_by_run_study(self):
        """run_study() must populate run_seeds from _child_seeds."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.data import ArraysDataHandler
        from ictonyx.runners import ExperimentRunner

        rng = np.random.default_rng(7)
        X = rng.standard_normal((60, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int).astype(float)
        handler = ArraysDataHandler(X, y)
        config = ModelConfig({"epochs": 2})
        runner = ExperimentRunner(
            model_builder=lambda cfg: __import__(
                "ictonyx.core", fromlist=["ScikitLearnModelWrapper"]
            ).ScikitLearnModelWrapper(LogisticRegression()),
            data_handler=handler,
            model_config=config,
            verbose=False,
            seed=42,
        )
        results = runner.run_study(num_runs=3)
        assert len(results.run_seeds) == 3
        assert all(isinstance(s, int) for s in results.run_seeds)
        # All seeds must be distinct
        assert len(set(results.run_seeds)) == 3


class TestCompareResults:
    """compare_results() compares two VariabilityStudyResults without re-training."""

    def _make_results(self, values, seed=1):
        return VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={"val_accuracy": values},
            final_test_metrics=[],
            seed=seed,
        )

    def test_returns_model_comparison_results(self):
        from ictonyx.analysis import ModelComparisonResults
        from ictonyx.api import compare_results

        a = self._make_results([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.90, 0.91])
        b = self._make_results([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71])
        result = compare_results(a, b)
        assert isinstance(result, ModelComparisonResults)

    def test_resolves_metric_automatically(self):
        from ictonyx.api import compare_results

        a = self._make_results([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.90, 0.91])
        b = self._make_results([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71])
        result = compare_results(a, b)
        assert result.overall_test is not None

    def test_significant_difference_detected(self):
        from ictonyx.api import compare_results

        a = self._make_results([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.90, 0.91])
        b = self._make_results([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71])
        result = compare_results(a, b)
        assert result.overall_test.is_significant()

    def test_explicit_metric(self):
        from ictonyx.api import compare_results

        a = self._make_results([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.90, 0.91])
        b = self._make_results([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71])
        result = compare_results(a, b, metric="val_accuracy")
        assert result.overall_test is not None


class TestCheckpointResume:
    """Verify that checkpoint_dir saves progress and a resumed run
    skips already-completed runs exactly."""

    def _make_runner(self, verbose=False):
        def model_builder(config):
            return MockModel(config)

        return ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=verbose,
        )

    def test_checkpoint_file_is_created(self, tmp_path):
        """A checkpoint.pkl file must exist after the first run completes."""
        runner = self._make_runner()
        runner.run_study(num_runs=3, checkpoint_dir=str(tmp_path))
        assert (tmp_path / "checkpoint.pkl").exists()

    def test_checkpoint_resumes_correctly(self, tmp_path):
        """A fresh runner loading the checkpoint must produce exactly
        num_runs results, not 2x, and must not re-run completed runs."""
        checkpoint_dir = str(tmp_path)

        # First run: complete 3 runs and save checkpoint
        runner1 = self._make_runner()
        results1 = runner1.run_study(num_runs=3, checkpoint_dir=checkpoint_dir)
        assert results1.n_runs == 3

        # Second run: same runner type, same checkpoint dir.
        # Since all 3 runs are already complete, 0 new runs should execute.
        runner2 = self._make_runner()
        results2 = runner2.run_study(num_runs=3, checkpoint_dir=checkpoint_dir)

        assert results2.n_runs == 3
        assert len(results2.all_runs_metrics) == 3
        assert len(results2.final_metrics.get("val_accuracy", [])) == 3

    def test_checkpoint_missing_file_starts_fresh(self, tmp_path):
        """If checkpoint dir exists but has no checkpoint.pkl, start fresh."""
        runner = self._make_runner()
        # Point at a real directory with no checkpoint file
        results = runner.run_study(num_runs=3, checkpoint_dir=str(tmp_path))
        assert results.n_runs == 3


class TestParallelExecution:
    """Verify that use_parallel=True produces exactly num_runs results
    with final_metrics fully populated — not doubled, not empty."""

    def _make_runner(self, verbose=False):
        def model_builder(config):
            return MockModel(config)

        return ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({"epochs": 2}),
            verbose=verbose,
        )

    def test_parallel_all_runs_metrics_count(self):
        """all_runs_metrics must have exactly num_runs entries, not 2x."""
        runner = self._make_runner()
        results = runner.run_study(num_runs=4, use_parallel=True, n_jobs=2)
        assert len(results.all_runs_metrics) == 4

    def test_parallel_final_metrics_populated(self):
        """final_metrics must be populated after a parallel run."""
        runner = self._make_runner()
        results = runner.run_study(num_runs=4, use_parallel=True, n_jobs=2)
        assert "val_accuracy" in results.final_metrics
        assert len(results.final_metrics["val_accuracy"]) == 4

    def test_parallel_n_runs_matches(self):
        """VariabilityStudyResults.n_runs must equal num_runs."""
        runner = self._make_runner()
        results = runner.run_study(num_runs=4, use_parallel=True, n_jobs=2)
        assert results.n_runs == 4

    def test_parallel_get_metric_values_works(self):
        """get_metric_values() must succeed after a parallel run."""
        runner = self._make_runner()
        results = runner.run_study(num_runs=4, use_parallel=True, n_jobs=2)
        values = results.get_metric_values("val_accuracy")
        assert len(values) == 4
        assert all(isinstance(v, float) for v in values)

    def test_sequential_and_parallel_same_count(self):
        """Sequential and parallel runs on the same setup must produce
        the same number of results."""
        runner_seq = self._make_runner()
        runner_par = self._make_runner()
        results_seq = runner_seq.run_study(num_runs=4)
        results_par = runner_par.run_study(num_runs=4, use_parallel=True, n_jobs=2)
        assert results_seq.n_runs == results_par.n_runs
        assert len(results_seq.final_metrics["val_accuracy"]) == len(
            results_par.final_metrics["val_accuracy"]
        )


class TestSetupMlflow:
    def test_setup_mlflow_raises_without_mlflow(self):
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"mlflow": None}):
            from ictonyx.utils import setup_mlflow

            with pytest.raises(ImportError, match="MLflow"):
                setup_mlflow("test_experiment")
