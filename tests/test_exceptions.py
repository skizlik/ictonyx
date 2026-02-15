"""Test custom exception classes."""

import numpy as np
import pandas as pd
import pytest

from ictonyx.exceptions import (
    ConfigurationError,
    DataValidationError,
    ExperimentError,
    IctonyxError,
    ModelError,
    StatisticalTestError,
    validate_not_empty,
    validate_splits,
    validate_statistical_input,
)


class TestIctonyxError:
    """Test base exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = IctonyxError("Test error")
        assert str(error) == "Test error"
        assert error.timestamp is not None

    def test_error_with_context(self):
        """Test error with context information."""
        error = IctonyxError("Test error", context={"key": "value"})
        assert error.context == {"key": "value"}
        assert "key: value" in error.get_context_summary()

    def test_add_context(self):
        """Test adding context after creation."""
        error = IctonyxError("Test")
        error.add_context("run_id", 5)
        error.add_context("epoch", 10)

        assert error.context["run_id"] == 5
        assert error.context["epoch"] == 10


class TestDataValidationError:
    """Test data validation errors."""

    def test_empty_dataset_error(self):
        """Test empty dataset error factory."""
        error = DataValidationError.for_empty_dataset("train_data", expected_size=100)

        assert "train_data" in str(error)
        assert error.data_info["dataset_name"] == "train_data"
        assert error.data_info["expected_min_size"] == 100

    def test_splits_error(self):
        """Test data splits error factory."""
        error = DataValidationError.for_splits(0.3, 0.8)

        assert "0.3" in str(error)
        assert "0.8" in str(error)
        assert error.data_info["sum"] == 1.1
        assert error.validation_rule == "test_split + val_split must be < 1.0"


class TestModelError:
    """Test model-related errors."""

    def test_training_failed(self):
        """Test training failure error."""
        error = ModelError.training_failed(model_type="CNN", epoch=5, error_details="OOM")

        assert "CNN" in str(error)
        assert "epoch 5" in str(error)
        assert error.model_info["failed_at_epoch"] == 5
        assert error.operation == "training"

    def test_prediction_failed(self):
        """Test prediction failure error."""
        error = ModelError.prediction_failed(
            model_type="XGBoost", input_shape=(100, 20), error_details="Shape mismatch"
        )

        assert "XGBoost" in str(error)
        assert error.model_info["input_shape"] == (100, 20)
        assert error.operation == "prediction"


class TestExperimentError:
    """Test experiment errors."""

    def test_high_failure_rate(self):
        """Test high failure rate error."""
        error = ExperimentError.high_failure_rate(failed_runs=7, total_runs=10, threshold=0.5)

        assert "7/10" in str(error)
        assert "70.0%" in str(error)
        assert error.failure_count == 7

    def test_resource_exhausted(self):
        """Test resource exhaustion error."""
        error = ExperimentError.resource_exhausted("GPU memory", run_id=3)

        assert "GPU memory" in str(error)
        assert "run 3" in str(error)
        assert error.run_id == 3


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_splits(self):
        """Test split validation."""
        # Valid splits
        validate_splits(0.2, 0.1)  # Should pass
        validate_splits(0.0, 0.0)  # Should pass

        # Invalid: sum >= 1
        with pytest.raises(DataValidationError):
            validate_splits(0.5, 0.5)

        # Invalid: negative
        with pytest.raises(DataValidationError):
            validate_splits(-0.1, 0.2)

        with pytest.raises(DataValidationError):
            validate_splits(0.2, -0.1)

        # Invalid: >= 1
        with pytest.raises(DataValidationError):
            validate_splits(1.0, 0.0)

    def test_validate_not_empty(self):
        """Test empty data validation."""
        # Valid data
        validate_not_empty([1, 2, 3], "test_data")
        validate_not_empty(pd.DataFrame({"a": [1, 2]}), "df")

        # Empty list
        with pytest.raises(DataValidationError) as exc_info:
            validate_not_empty([], "empty_list")
        assert "empty_list" in str(exc_info.value)

        # Empty dataframe
        with pytest.raises(DataValidationError):
            validate_not_empty(pd.DataFrame(), "empty_df")

        # Insufficient size
        with pytest.raises(DataValidationError):
            validate_not_empty([1, 2], "small_data", min_size=5)

    def test_validate_statistical_input(self):
        """Test statistical input validation."""
        # Valid inputs
        validate_statistical_input(pd.Series([1, 2, 3, 4]), "test")
        validate_statistical_input(np.array([1, 2, 3, 4]), "test")
        validate_statistical_input([1, 2, 3, 4], "test")

        # Invalid type
        with pytest.raises(StatisticalTestError):
            validate_statistical_input({"a": 1}, "test")

        # All NaN
        with pytest.raises(StatisticalTestError):
            validate_statistical_input(pd.Series([np.nan, np.nan]), "test")

        # Insufficient valid data
        with pytest.raises(StatisticalTestError):
            validate_statistical_input([1, np.nan], "test", min_samples=3)


class TestStatisticalTestError:
    """Test StatisticalTestError factory methods."""

    def test_insufficient_data(self):
        error = StatisticalTestError.insufficient_data(
            test_name="Mann-Whitney U", sample_size=3, minimum_required=6
        )
        assert "Mann-Whitney" in str(error)
        assert error.test_info["minimum_samples"] == 6

    def test_invalid_distribution(self):
        error = StatisticalTestError.invalid_distribution(
            test_name="t-test", issue="non-normal distribution"
        )
        assert "t-test" in str(error)
        assert "non-normal" in str(error)


class TestConfigurationError:
    """Test ConfigurationError factory methods."""

    def test_invalid_parameter(self):
        error = ConfigurationError.invalid_parameter(
            param_name="epochs", param_value=-5, valid_range=(1, 1000)
        )
        assert "epochs" in str(error)
        assert error.config_info["parameter"] == "epochs"
        assert error.config_info["provided_value"] == -5
        assert error.config_info["valid_range"] == (1, 1000)


class TestIctonyxErrorExtended:
    """Extended tests for base exception."""

    def test_str_with_context(self):
        error = IctonyxError("Something broke", context={"run": 3, "epoch": 5})
        s = str(error)
        assert "Something broke" in s
        assert "run" in s

    def test_empty_context_summary(self):
        error = IctonyxError("No context")
        assert error.get_context_summary() == "No additional context available."

    def test_chained_add_context(self):
        error = IctonyxError("Error")
        result = error.add_context("a", 1)
        assert result is error  # returns self for chaining
        assert error.context["a"] == 1

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from IctonyxError."""
        assert issubclass(DataValidationError, IctonyxError)
        assert issubclass(ModelError, IctonyxError)
        assert issubclass(ExperimentError, IctonyxError)
        assert issubclass(StatisticalTestError, IctonyxError)
        assert issubclass(ConfigurationError, IctonyxError)

    def test_data_validation_is_value_error(self):
        """Test that DataValidationError is also a ValueError."""
        assert issubclass(DataValidationError, ValueError)

    def test_statistical_test_is_value_error(self):
        """Test that StatisticalTestError is also a ValueError."""
        assert issubclass(StatisticalTestError, ValueError)
