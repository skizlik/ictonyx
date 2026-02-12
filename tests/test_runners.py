"""Test experiment runners."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from ictonyx.runners import (
    ExperimentRunner,
    VariabilityStudyResults,
    run_variability_study
)
from ictonyx.config import ModelConfig
from ictonyx.core import BaseModelWrapper, TrainingResult


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
        epochs = kwargs.get('epochs', 5)
        base_acc = 0.8 + np.random.random() * 0.1
        self.training_result = TrainingResult(history={
            'train_accuracy': np.linspace(0.5, base_acc, epochs).tolist(),
            'val_accuracy': np.linspace(0.45, base_acc - 0.05, epochs).tolist(),
        })
    
    def predict(self, data, **kwargs):
        return np.zeros(len(data))
    
    def predict_proba(self, data, **kwargs):
        n = len(data)
        return np.random.rand(n, 2)
    
    def evaluate(self, data, **kwargs):
        return {'accuracy': 0.85 + np.random.random() * 0.05}
    
    def assess(self, true_labels):
        return {'accuracy': 0.85}
    
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
        
        return {
            'train_data': (X_train, y_train),
            'val_data': (X_val, y_val),
            'test_data': None
        }


class TestExperimentRunner:
    """Test ExperimentRunner class."""
    
    def test_runner_initialization(self):
        """Test creating experiment runner."""
        
        def model_builder(config):
            return MockModel(config)
        
        config = ModelConfig({'epochs': 5})
        data_handler = MockDataHandler()
        
        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=data_handler,
            model_config=config,
            verbose=False
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
            model_config=ModelConfig({'epochs': 3}),
            verbose=False
        )
        
        result = runner._run_single_fit(run_id=1, epochs=3)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'train_accuracy' in result.columns
        assert len(result) == 3  # 3 epochs
    
    def test_full_study(self):
        """Test running full variability study."""
        
        def model_builder(config):
            return MockModel(config)
        
        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig({'epochs': 2}),
            verbose=False
        )

        results = runner.run_study(
            num_runs=3,
            epochs_per_run=2
        )

        assert isinstance(results, VariabilityStudyResults)
        assert results.n_runs == 3
        assert 'val_accuracy' in results.final_metrics
        assert len(results.final_metrics['val_accuracy']) == 3

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
            verbose=False
        )

        results = runner.run_study(
            num_runs=5,
            stop_on_failure_rate=0.8
        )

        # Some runs may fail — that's the point. Verify the runner
        # tracked everything correctly regardless of outcome.
        assert results.n_runs + len(runner.failed_runs) == 5
        assert results.n_runs == len(results.all_runs_metrics)

    def test_summary_stats(self):
        """Test getting summary statistics."""
        
        def model_builder(config):
            return MockModel(config)
        
        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockDataHandler(),
            model_config=ModelConfig(),
            verbose=False
        )
        
        runner.run_study(num_runs=2)
        
        stats = runner.get_summary_stats()
        
        assert 'total_runs' in stats
        assert 'successful_runs' in stats
        assert stats['successful_runs'] == 2


class TestVariabilityStudyResults:
    """Test VariabilityStudyResults class."""

    def test_results_creation(self):
        """Test creating results object."""
        df1 = pd.DataFrame({
            'epoch': [1, 2],
            'train_accuracy': [0.6, 0.8],
            'val_accuracy': [0.5, 0.75]
        })

        results = VariabilityStudyResults(
            all_runs_metrics=[df1],
            final_metrics={'val_accuracy': [0.75], 'train_accuracy': [0.8]},
            final_test_metrics=[]
        )

        assert results.n_runs == 1
        assert len(results.final_metrics['val_accuracy']) == 1

    def test_get_metric_values(self):
        """Test extracting metric values."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={
                'val_accuracy': [0.7, 0.8],
                'val_loss': [0.3, 0.2],
                'val_f1': [0.65, 0.75]
            },
            final_test_metrics=[]
        )

        assert results.get_metric_values('val_accuracy') == [0.7, 0.8]
        assert results.get_metric_values('val_f1') == [0.65, 0.75]

        with pytest.raises(KeyError, match="not found"):
            results.get_metric_values('nonexistent')

    def test_get_final_metrics(self):
        """Test extracting final metrics per run from DataFrames."""
        df1 = pd.DataFrame({
            'val_accuracy': [0.5, 0.6, 0.7],
            'val_loss': [0.5, 0.4, 0.3]
        })
        df2 = pd.DataFrame({
            'val_accuracy': [0.6, 0.7, 0.8],
            'val_loss': [0.4, 0.3, 0.2]
        })

        results = VariabilityStudyResults(
            all_runs_metrics=[df1, df2],
            final_metrics={'val_accuracy': [0.7, 0.8]},
            final_test_metrics=[]
        )

        final = results.get_final_metrics('val_accuracy')
        assert len(final) == 2
        assert final['run_1'] == 0.7
        assert final['run_2'] == 0.8

    def test_get_available_metrics(self):
        """Test listing available metrics."""
        results = VariabilityStudyResults(
            all_runs_metrics=[],
            final_metrics={'val_accuracy': [0.7], 'val_loss': [0.3], 'train_accuracy': [0.8]},
            final_test_metrics=[]
        )

        available = results.get_available_metrics()
        assert 'train_accuracy' in available
        assert 'val_accuracy' in available
        assert 'val_loss' in available

    def test_to_dataframe(self):
        """Test converting results to DataFrame."""
        df1 = pd.DataFrame({
            'train_accuracy': [0.6, 0.8],
            'val_accuracy': [0.5, 0.75]
        })

        results = VariabilityStudyResults(
            all_runs_metrics=[df1],
            final_metrics={'val_accuracy': [0.75]},
            final_test_metrics=[{'test_acc': 0.72}]
        )

        summary_df = results.to_dataframe()

        assert len(summary_df) == 1
        assert 'final_train_accuracy' in summary_df.columns
        assert 'test_test_acc' in summary_df.columns
        assert summary_df.iloc[0]['final_val_accuracy'] == 0.75

    def test_summarize(self):
        """Test summary string generation."""
        results = VariabilityStudyResults(
            all_runs_metrics=[pd.DataFrame({'a': [1]})],
            final_metrics={'val_accuracy': [0.8, 0.85, 0.82]},
            final_test_metrics=[]
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
            model_config=ModelConfig({'epochs': 2}),
            num_runs=2,
            verbose=False
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
                epochs = kwargs.get('epochs', 3)
                self.training_result = TrainingResult(history={
                    'train_accuracy': np.linspace(0.5, 0.9, epochs).tolist(),
                    'train_loss': np.linspace(1.0, 0.2, epochs).tolist(),
                    'val_accuracy': np.linspace(0.4, 0.85, epochs).tolist(),
                    'val_loss': np.linspace(0.9, 0.25, epochs).tolist(),
                    'val_f1': np.linspace(0.3, 0.8, epochs).tolist(),
                })
            def predict(self, data, **kwargs):
                return np.zeros(len(data))
            def predict_proba(self, data, **kwargs):
                return np.random.rand(len(data), 2)
            def evaluate(self, data, **kwargs):
                return {'accuracy': 0.85, 'f1': 0.8}
            def assess(self, true_labels):
                return {'accuracy': 0.85}
            def save_model(self, path):
                pass
            @classmethod
            def load_model(cls, path):
                return cls(ModelConfig())

        runner = ExperimentRunner(
            model_builder=lambda c: MultiMetricModel(c),
            data_handler=MockDataHandler(),
            model_config=ModelConfig({'epochs': 3}),
            verbose=False
        )

        results = runner.run_study(num_runs=3, epochs_per_run=3)

        assert 'val_accuracy' in results.final_metrics
        assert 'val_loss' in results.final_metrics
        assert 'val_f1' in results.final_metrics
        assert len(results.final_metrics['val_f1']) == 3