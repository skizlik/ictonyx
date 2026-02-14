# ictonyx/exceptions.py
"""
Custom exception classes for the ictonyx library.

This module defines all exception types used throughout ictonyx,
providing clear error categorization, context information, and
better debugging experience.
"""

import traceback
from typing import Any, Dict, List, Optional, Union


class IctonyxError(Exception):
    """
    Base exception class for ictonyx library.

    Provides common functionality for all ictonyx exceptions including
    context tracking and enhanced error reporting.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = None

        # Capture timestamp when error occurs
        import datetime

        self.timestamp = datetime.datetime.now()

    def add_context(self, key: str, value: Any) -> "IctonyxError":
        """Add contextual information to the exception."""
        self.context[key] = value
        return self

    def get_context_summary(self) -> str:
        """Get a formatted summary of the error context."""
        if not self.context:
            return "No additional context available."

        lines = ["Error Context:"]
        for key, value in self.context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            return f"{base_msg}\n{self.get_context_summary()}"
        return base_msg


class DataValidationError(IctonyxError, ValueError):
    """
    Raised when data validation fails.

    Inherits from ValueError for compatibility with existing error handling
    that expects ValueError for invalid data.
    """

    def __init__(
        self,
        message: str,
        data_info: Optional[Dict[str, Any]] = None,
        validation_rule: Optional[str] = None,
    ):
        context = {}
        if data_info:
            context.update(data_info)
        if validation_rule:
            context["validation_rule"] = validation_rule

        super().__init__(message, context)
        self.data_info = data_info or {}
        self.validation_rule = validation_rule

    @classmethod
    def for_splits(cls, test_split: float, val_split: float) -> "DataValidationError":
        """Factory method for split validation errors."""
        return cls(
            f"Invalid data splits: test_split={test_split}, val_split={val_split}",
            data_info={
                "test_split": test_split,
                "val_split": val_split,
                "sum": test_split + val_split,
            },
            validation_rule="test_split + val_split must be < 1.0",
        )

    @classmethod
    def for_empty_dataset(
        cls, dataset_name: str, expected_size: Optional[int] = None
    ) -> "DataValidationError":
        """Factory method for empty dataset errors."""
        context = {"dataset_name": dataset_name}
        if expected_size:
            context["expected_min_size"] = expected_size

        return cls(
            f"Dataset '{dataset_name}' is empty or has insufficient data",
            data_info=context,
            validation_rule="Dataset must contain at least some valid samples",
        )


class ModelError(IctonyxError):
    """
    Raised when model-related operations fail.

    Includes information about the model state and operation that failed.
    """

    def __init__(
        self,
        message: str,
        model_info: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
        model_state: Optional[str] = None,
    ):
        context = {}
        if model_info:
            context.update(model_info)
        if operation:
            context["failed_operation"] = operation
        if model_state:
            context["model_state"] = model_state

        super().__init__(message, context)
        self.model_info = model_info or {}
        self.operation = operation
        self.model_state = model_state

    @classmethod
    def training_failed(
        cls, model_type: str, epoch: Optional[int] = None, error_details: Optional[str] = None
    ) -> "ModelError":
        """Factory method for training failures."""
        context = {"model_type": model_type}
        if epoch is not None:
            context["failed_at_epoch"] = epoch
        if error_details:
            context["underlying_error"] = error_details

        message = f"Model training failed for {model_type}"
        if epoch is not None:
            message += f" at epoch {epoch}"

        return cls(message, model_info=context, operation="training", model_state="training")

    @classmethod
    def prediction_failed(
        cls,
        model_type: str,
        input_shape: Optional[tuple] = None,
        error_details: Optional[str] = None,
    ) -> "ModelError":
        """Factory method for prediction failures."""
        context = {"model_type": model_type}
        if input_shape:
            context["input_shape"] = input_shape
        if error_details:
            context["underlying_error"] = error_details

        return cls(
            f"Model prediction failed for {model_type}",
            model_info=context,
            operation="prediction",
            model_state="trained",
        )


class ExperimentError(IctonyxError):
    """
    Raised when experiment execution fails.

    Tracks experiment state, run information, and failure patterns.
    """

    def __init__(
        self,
        message: str,
        experiment_info: Optional[Dict[str, Any]] = None,
        run_id: Optional[int] = None,
        stage: Optional[str] = None,
        failure_count: Optional[int] = None,
    ):
        context = {}
        if experiment_info:
            context.update(experiment_info)
        if run_id is not None:
            context["run_id"] = run_id
        if stage:
            context["failure_stage"] = stage
        if failure_count is not None:
            context["total_failures"] = failure_count

        super().__init__(message, context)
        self.experiment_info = experiment_info or {}
        self.run_id = run_id
        self.stage = stage
        self.failure_count = failure_count

    @classmethod
    def high_failure_rate(
        cls, failed_runs: int, total_runs: int, threshold: float = 0.5
    ) -> "ExperimentError":
        """Factory method for high failure rate errors."""
        failure_rate = failed_runs / total_runs if total_runs > 0 else 1.0

        return cls(
            f"Experiment stopped due to high failure rate: {failed_runs}/{total_runs} "
            f"runs failed ({failure_rate:.1%} > {threshold:.1%} threshold)",
            experiment_info={
                "failed_runs": failed_runs,
                "total_runs": total_runs,
                "failure_rate": failure_rate,
                "threshold": threshold,
            },
            failure_count=failed_runs,
            stage="experiment_monitoring",
        )

    @classmethod
    def resource_exhausted(
        cls, resource_type: str, run_id: Optional[int] = None
    ) -> "ExperimentError":
        """Factory method for resource exhaustion errors."""
        message = f"Experiment failed due to {resource_type} exhaustion"
        if run_id:
            message += f" during run {run_id}"

        return cls(
            message,
            experiment_info={"resource_type": resource_type},
            run_id=run_id,
            stage="resource_management",
        )


class StatisticalTestError(IctonyxError, ValueError):
    """
    Raised when statistical test prerequisites aren't met.

    Provides detailed information about what validation failed
    and suggestions for remediation.
    """

    def __init__(
        self,
        message: str,
        test_info: Optional[Dict[str, Any]] = None,
        data_summary: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        context = {}
        if test_info:
            context.update(test_info)
        if data_summary:
            context["data_summary"] = data_summary
        if suggestion:
            context["suggestion"] = suggestion

        super().__init__(message, context)
        self.test_info = test_info or {}
        self.data_summary = data_summary or {}
        self.suggestion = suggestion

    @classmethod
    def insufficient_data(
        cls, test_name: str, sample_size: int, minimum_required: int = 3
    ) -> "StatisticalTestError":
        """Factory method for insufficient data errors."""
        return cls(
            f"Insufficient data for {test_name}: got {sample_size} samples, "
            f"need at least {minimum_required}",
            test_info={"test_name": test_name, "minimum_samples": minimum_required},
            data_summary={"actual_samples": sample_size},
            suggestion=f"Collect more data points or use a different statistical test "
            f"that works with smaller samples",
        )

    @classmethod
    def invalid_distribution(
        cls, test_name: str, issue: str, data_properties: Optional[Dict[str, Any]] = None
    ) -> "StatisticalTestError":
        """Factory method for distribution assumption violations."""
        return cls(
            f"Statistical test {test_name} assumptions violated: {issue}",
            test_info={"test_name": test_name, "assumption_violation": issue},
            data_summary=data_properties or {},
            suggestion="Consider using a non-parametric alternative or transform the data",
        )


class ConfigurationError(IctonyxError):
    """
    Raised when configuration validation fails.

    Helps users understand what configuration options are invalid
    and provides guidance on valid alternatives.
    """

    def __init__(
        self,
        message: str,
        config_info: Optional[Dict[str, Any]] = None,
        invalid_params: Optional[List[str]] = None,
        valid_options: Optional[Dict[str, Any]] = None,
    ):
        context = {}
        if config_info:
            context.update(config_info)
        if invalid_params:
            context["invalid_parameters"] = invalid_params
        if valid_options:
            context["valid_options"] = valid_options

        super().__init__(message, context)
        self.config_info = config_info or {}
        self.invalid_params = invalid_params or []
        self.valid_options = valid_options or {}

    @classmethod
    def invalid_parameter(
        cls,
        param_name: str,
        param_value: Any,
        valid_options: Optional[List[Any]] = None,
        valid_range: Optional[tuple] = None,
    ) -> "ConfigurationError":
        """Factory method for invalid parameter errors."""
        message = f"Invalid value for parameter '{param_name}': {param_value}"

        context = {"parameter": param_name, "provided_value": param_value}
        suggestion_parts = []

        if valid_options:
            context["valid_options"] = valid_options
            suggestion_parts.append(f"Valid options: {valid_options}")

        if valid_range:
            context["valid_range"] = valid_range
            suggestion_parts.append(f"Valid range: {valid_range[0]} to {valid_range[1]}")

        if suggestion_parts:
            message += f". {'; '.join(suggestion_parts)}"

        return cls(message, config_info=context, invalid_params=[param_name])


# Utility functions for common error patterns
def validate_not_empty(data, name: str, min_size: int = 1):
    """Utility function to validate data is not empty."""
    if hasattr(data, "__len__") and len(data) < min_size:
        raise DataValidationError.for_empty_dataset(name, min_size)
    elif hasattr(data, "empty") and data.empty:
        raise DataValidationError.for_empty_dataset(name, min_size)


def validate_splits(test_split: float, val_split: float):
    """Utility function to validate train/val/test splits."""
    if test_split < 0 or val_split < 0:
        raise DataValidationError(
            "Split values cannot be negative",
            data_info={"test_split": test_split, "val_split": val_split},
        )

    if test_split >= 1.0 or val_split >= 1.0:
        raise DataValidationError(
            "Split values must be less than 1.0",
            data_info={"test_split": test_split, "val_split": val_split},
        )

    if test_split + val_split >= 1.0:
        raise DataValidationError.for_splits(test_split, val_split)


def validate_statistical_input(data, name: str, min_samples: int = 3):
    """Utility function to validate data for statistical tests."""
    import numpy as np
    import pandas as pd

    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise StatisticalTestError(f"{name} must be pandas Series, numpy array, or list")

    # Convert to pandas Series for consistent handling
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Check for empty data
    if data.empty:
        raise StatisticalTestError.insufficient_data("statistical test", 0, min_samples)

    # Check for all NaN
    if data.isna().all():
        raise StatisticalTestError(f"{name} contains only missing values")

    # Check for sufficient valid data
    valid_data = data.dropna()
    if len(valid_data) < min_samples:
        raise StatisticalTestError.insufficient_data(
            "statistical test", len(valid_data), min_samples
        )
