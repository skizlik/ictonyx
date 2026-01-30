# tests/test_api.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from ictonyx import api, ModelConfig
from ictonyx.core import BaseModelWrapper

# --- Fixtures (Reusable Data) ---

@pytest.fixture
def sample_df():
    """Creates a minimal valid DataFrame."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def sample_arrays():
    """Creates minimal valid (X, y) arrays."""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    return X, y

@pytest.fixture
def dummy_model_func():
    """A simple function that acts as a model builder."""
    def builder(config=None):
        mock = MagicMock(spec=BaseModelWrapper)
        mock.fit.return_value = None
        mock.predict.return_value = np.array([0, 1])
        return mock
    return builder

@pytest.fixture
def dummy_model_func_2():
    """A second dummy model builder with a different name."""
    def builder_2(config=None):
        mock = MagicMock(spec=BaseModelWrapper)
        mock.fit.return_value = None
        mock.predict.return_value = np.array([0, 1])
        return mock
    return builder_2

@pytest.fixture
def mock_runner_results():
    """Creates a fake VariabilityStudyResults object."""
    mock_results = MagicMock()
    mock_results.get_final_metrics.return_value = {'run_1': 0.95, 'run_2': 0.96}
    return mock_results


# --- Tests for variability_study ---

@patch("ictonyx.api._run_study")  # FIX: Updated name from _run_study_internal
def test_variability_study_dataframe(mock_run, sample_df, dummy_model_func, mock_runner_results):
    """Test standard usage with DataFrame."""
    # Arrange
    mock_run.return_value = mock_runner_results

    # Act
    results = api.variability_study(
        model=dummy_model_func,
        data=sample_df,
        target_column='target',
        runs=3,
        epochs=5,
        learning_rate=0.01  # Extra kwarg for config
    )

    # Assert
    assert results == mock_runner_results
    
    # Verify _run_study was called
    mock_run.assert_called_once()
    
    # Verify Config construction
    call_args = mock_run.call_args
    config_arg = call_args.kwargs['model_config']
    
    assert isinstance(config_arg, ModelConfig)
    assert config_arg['epochs'] == 5
    assert config_arg['learning_rate'] == 0.01  # Verify kwargs passthrough

@patch("ictonyx.api._run_study") # FIX: Updated name
def test_variability_study_arrays(mock_run, sample_arrays, dummy_model_func, mock_runner_results):
    """Test usage with numpy tuples."""
    mock_run.return_value = mock_runner_results
    
    api.variability_study(
        model=dummy_model_func,
        data=sample_arrays,
        runs=2
    )
    
    # Verify DataHandler was resolved correctly
    passed_handler = mock_run.call_args.kwargs['data_handler']
    assert passed_handler.data_type == 'arrays'

def test_variability_study_invalid_data():
    """Test that invalid data raises appropriate errors."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        api.variability_study(lambda x: x, data=12345)

def test_variability_study_missing_target(sample_df):
    """Test missing target column error for DataFrames."""
    with pytest.raises(ValueError, match="target_column is required"):
        api.variability_study(lambda x: x, data=sample_df)


# --- Tests for compare_models ---

@patch("ictonyx.api._stat_compare")
@patch("ictonyx.api.variability_study")
def test_compare_models_flow(mock_var_study, mock_stat_compare, sample_df, dummy_model_func, dummy_model_func_2, mock_runner_results):
    """Test that compare_models correctly orchestrates multiple studies."""
    # Arrange
    mock_var_study.return_value = mock_runner_results
    mock_stat_compare.return_value = {'overall_test': 'PASSED'}
    
    models = [dummy_model_func, dummy_model_func_2]
    
    # Act
    result = api.compare_models(
        models=models,
        data=sample_df,
        target_column='target',
        runs=5,
        metric='val_accuracy'
    )
    
    # Assert
    # Should run study once for each model
    assert mock_var_study.call_count == 2
    
    # Should call stats comparison once
    mock_stat_compare.assert_called_once()
    
    # Check that it extracted the metrics correctly
    stats_call_args = mock_stat_compare.call_args[0][0]
    assert len(stats_call_args) == 2
    assert isinstance(stats_call_args[list(stats_call_args.keys())[0]], pd.Series)

@patch("ictonyx.api.variability_study")
def test_compare_models_insufficient_data(mock_var_study, sample_df):
    """Test error when models fail to return metrics."""
    # Arrange: Study returns empty metrics
    empty_results = MagicMock()
    empty_results.get_final_metrics.return_value = {}
    mock_var_study.return_value = empty_results
    
    # Act
    result = api.compare_models(
        models=[lambda x: x, lambda x: x],
        data=sample_df,
        target_column='target'
    )
    
    # Assert
    assert 'error' in result
    assert result['error'] == 'Insufficient valid results for comparison'


# --- Tests for Model Wrapping Helpers ---

class DummyClassifier:
    """A mock class that looks like an sklearn estimator."""
    def fit(self, X, y): pass
    def predict(self, X): pass

def test_get_model_builder_class():
    """Test wrapping a raw class."""
    # FIX: Updated function name to _get_model_builder
    wrapper = api._get_model_builder(DummyClassifier)
    model_instance = wrapper(ModelConfig({}))
    assert isinstance(model_instance, BaseModelWrapper)

def test_get_model_builder_instance():
    """Test wrapping an instance."""
    instance = DummyClassifier()
    # FIX: Updated function name
    wrapper = api._get_model_builder(instance)
    model_instance = wrapper(ModelConfig({}))
    assert isinstance(model_instance, BaseModelWrapper)

def test_get_model_builder_invalid():
    """Test rejection of invalid model inputs."""
    with pytest.raises(ValueError, match="Invalid model input"):
        # FIX: Updated function name
        api._get_model_builder("not a model")
