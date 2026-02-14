# ictonyx/core.py

import gc
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from .memory import get_memory_manager
from .settings import logger

# Optional TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.utils import Sequence

    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    KerasModel = None
    Sequence = None
    TENSORFLOW_AVAILABLE = False

# Optional scikit-learn imports
try:
    from sklearn.base import BaseEstimator

    SKLEARN_AVAILABLE = True
except ImportError:
    BaseEstimator = None
    SKLEARN_AVAILABLE = False

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    PYTORCH_AVAILABLE = False


@dataclass
class TrainingResult:
    """Standardized training output from any model wrapper.

    Every wrapper's fit() method must produce one of these.
    Replaces the Keras-specific .history.history probing pattern.

    Attributes:
        history: Dict mapping metric names to lists of per-epoch values.
                 Keys follow the convention: 'loss', 'accuracy', 'val_loss',
                 'val_accuracy', or any custom metric name.
        params: Training parameters (epochs, batch_size, etc.)
    """

    history: Dict[str, list]
    params: Dict[str, Any] = field(default_factory=dict)


# Abstract Base Class for ModelWrappers


class BaseModelWrapper(ABC):
    """
    A base class for wrapping a machine learning model.
    It provides a standardized interface for common analysis tasks.
    """

    def __init__(self, model: Any, model_id: str = ""):
        self.model = model
        self.model_id = model_id
        self.training_result: Optional[TrainingResult] = None
        self.predictions: Optional[np.ndarray] = None
        self._resource_manager = get_memory_manager(use_process_isolation=False)

    def __repr__(self) -> str:
        model_type = self.model.__class__.__name__
        is_trained = "Yes" if self.training_result is not None else "No"
        return f"Ictonyx BaseModelWrapper(id='{self.model_id}', type='{model_type}', is_trained={is_trained})"

    def __del__(self):
        """Automatic cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def cleanup(self):
        """
        Clean up resources used by this model.
        """
        try:
            self._cleanup_implementation()
            # Use new resource manager cleanup
            cleanup_result = self._resource_manager.cleanup()

            # Report significant cleanup events
            if cleanup_result.memory_freed_mb and cleanup_result.memory_freed_mb > 10:
                logger.info(f"Model cleanup freed {cleanup_result.memory_freed_mb:.1f}MB")

            if cleanup_result.errors:
                logger.warning(f"Model cleanup had {len(cleanup_result.errors)} warnings")

        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    @abstractmethod
    def fit(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs):
        """
        Trains (fits) the encapsulated model on the provided data.

        This is an abstract method that must be implemented by concrete subclasses
        (e.g., KerasModelWrapper, ScikitLearnModelWrapper). The implementation
        is responsible for handling its specific data format (e.g.,
        tf.data.Dataset vs. numpy arrays) and storing the training
        history (if any) in the `self.training_result` attribute.

        Args:
            train_data (Any): The data to train the model on. The required
                format is defined by the subclass (e.g., a tuple (X, y)
                or a tf.data.Dataset).
            validation_data (Optional[Any], optional): The data to use for
                validation during training. The format should be compatible
                with `train_data`. Defaults to None.
            **kwargs: Additional, framework-specific arguments to be passed
                to the model's underlying `.fit()` method (e.g., `epochs`,
                `batch_size`, `callbacks`).
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generates predictions for the given input data.

        This abstract method must be implemented by subclasses. It is
        responsible for running the model's prediction logic.

        The implementation should:
        1.  Generate raw predictions from `self.model`.
        2.  For classification models, convert probabilities or logits into
            final class indices (e.g., `np.argmax(..., axis=1)`).
        3.  For regression models, return the predicted values directly.
        4.  Store the final predictions in the `self.predictions` attribute.
        5.  Return the final predictions.

        Args:
            data (np.ndarray): The input data to generate predictions for.
            **kwargs: Additional, framework-specific arguments to be passed
                to the model's underlying `.predict()` method.

        Returns:
            np.ndarray: An array of predictions. For classification, this
            should be class indices (integers). For regression, this
            should be the predicted values (floats).
        """
        pass

    @abstractmethod
    def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generates class probability predictions for the given input data.

        This abstract method is only applicable for classification models.
        It must be implemented by subclasses that support probability outputs.

        The implementation should:
        1.  Generate raw probability scores from `self.model`.
        2.  Ensure the output is a 2D array of shape `(n_samples, n_classes)`.
        3.  For binary models with a single sigmoid output, this means
            stacking `(1-p, p)` to create the 2D array.
        4.  For multi-class models, this is typically the direct output
            of a softmax layer.

        Args:
            data (np.ndarray): The input data to generate probabilities for.
            **kwargs: Additional, framework-specific arguments to be passed
                to the model's underlying `.predict_proba()` method.

        Returns:
            np.ndarray: A 2D array of shape `(n_samples, n_classes)`
            containing the probability for each class.

        Raises:
            NotImplementedError: If the underlying model does not support
            probability predictions (e.g., SVM with linear kernel).
            ValueError: If called on a regression model.
        """
        pass

    @abstractmethod
    def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the encapsulated model on a given dataset (e.g., test set).

        This abstract method must be implemented by subclasses. Unlike
        `predict`, the `data` argument here is expected to contain both
        features and true labels in whatever format the underlying
        framework requires (e.g., a tuple (X, y) or a tf.data.Dataset).

        Args:
            data (Any): The data to evaluate the model on, including
                features and true labels.
            **kwargs: Additional, framework-specific arguments to be passed
                to the model's underlying `.evaluate()` method.

        Returns:
            Dict[str, Any]: A dictionary of metrics, where keys are the
            metric names (e.g., 'loss', 'accuracy') and values are the
            scalar metric values.
        """
        pass

    @abstractmethod
    def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Provides a quick assessment of model performance using stored predictions.

        This method is typically called *after* `predict()` has been run.
        It compares the stored `self.predictions` against a set of true
        labels to calculate simple, high-level metrics (like accuracy).

        This differs from `evaluate()`, which often runs the model's
        built-in evaluation method on a dataset.

        Args:
            true_labels (np.ndarray): A 1D array of the true ground-truth
                labels, corresponding to the stored `self.predictions`.

        Returns:
            Dict[str, float]: A simple dictionary of one or more
            assessment metrics, e.g., {'accuracy': 0.95}.

        Raises:
            ValueError: If `predict()` has not been called yet and
                `self.predictions` is None.
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """
        Saves the encapsulated model to a specified path.

        This abstract method must be implemented by subclasses. It is
        responsible for serializing the *inner* model (e.g., the Keras
        model or scikit-learn estimator) and saving it to disk in the
        appropriate format (e.g., HDF5, pickle).

        Args:
            path (str): The file path (or directory path, depending on
                the framework) where the model should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> "BaseModelWrapper":
        """
        Loads a model from a file and wraps it in a new class instance.

        This abstract class method must be implemented by subclasses. It is
        responsible for deserializing a model from disk and returning a
        new, fully-functional instance of the *wrapper* class (e.g.,
        `KerasModelWrapper(loaded_keras_model)`).

        Args:
            path (str): The file path (or directory path) from which to
                load the model.

        Returns:
            BaseModelWrapper: A new instance of the concrete wrapper class
            (e.g., `KerasModelWrapper`) containing the loaded model.
        """
        pass

    @abstractmethod
    def _cleanup_implementation(self):
        """Framework-specific cleanup implementation (must be implemented by subclasses)."""
        pass

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report (available on all model wrappers)."""
        from .memory import get_memory_info

        return get_memory_info()

    def check_memory_and_cleanup_if_needed(self) -> Optional[Dict[str, Any]]:
        """Check memory usage and cleanup if thresholds exceeded."""
        # This method doesn't exist in new manager, just do cleanup
        cleanup_result = self._resource_manager.cleanup()
        if cleanup_result.memory_freed_mb and cleanup_result.memory_freed_mb > 50:
            return {"cleaned": True, "freed_mb": cleanup_result.memory_freed_mb}
        return None


# --- CONCRETE IMPLEMENTATIONS ---

# Concrete class for TensorFlow/Keras models

if TENSORFLOW_AVAILABLE:

    class KerasModelWrapper(BaseModelWrapper):
        def __init__(self, model: KerasModel, model_id: str = ""):
            super().__init__(model, model_id)

        def _cleanup_implementation(self):
            """TensorFlow/Keras specific cleanup."""
            # Delete the model reference first
            if hasattr(self, "model"):
                try:
                    del self.model
                except Exception:
                    pass

            # Perform TensorFlow cleanup
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
            except Exception:
                pass

            # Basic garbage collection
            import gc

            gc.collect()

        def _cleanup_model_references(self):
            """Clear model references."""
            if hasattr(self, "model"):
                del self.model

        def _cleanup_tensorflow_session(self):
            """Clear TensorFlow session and memory."""
            tf.keras.backend.clear_session()
            gc.collect()

        def fit(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs):
            """
            Fits the encapsulated Keras model.

            This implementation handles three common Keras data formats:
            1.  A `tf.data.Dataset` object.
            2.  A `tf.keras.utils.Sequence` object (a data generator).
            3.  A tuple of `(X, y)` numpy arrays.

            The training metrics are stored in `self.training_result`
            as a `TrainingResult` object.

            Args:
                train_data (Any): The training data. Can be a `tf.data.Dataset`,
                    a `Sequence`, or a tuple of (X, y) numpy arrays.
                validation_data (Optional[Any], optional): The validation data.
                    Format must match `train_data`. Defaults to None.
                **kwargs: Additional arguments (e.g., `epochs`, `batch_size`,
                    `callbacks`) passed directly to the `model.fit()` call.

            Raises:
                TypeError: If `train_data` is not in one of the three
                    supported formats.
            """
            if isinstance(train_data, (tf.data.Dataset, Sequence)):
                keras_history = self.model.fit(
                    train_data, validation_data=validation_data, **kwargs
                )
            elif isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
                keras_history = self.model.fit(
                    x=X_train, y=y_train, validation_data=validation_data, **kwargs
                )
            else:
                raise TypeError(
                    "train_data must be a tuple of (X, y), a tf.data.Dataset, or a Sequence."
                )

            self.training_result = TrainingResult(
                history=dict(keras_history.history), params=getattr(keras_history, "params", {})
            )

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates predictions and stores them.

            For classification models, returns class indices as integers.
            For regression models, returns predictions as floats.

            Args:
                data: Input data for prediction
                **kwargs: Additional arguments passed to model.predict()

            Returns:
                np.ndarray: Predicted class labels (classification) or values (regression)
            """
            raw_predictions = self.model.predict(data, **kwargs)

            # Handle edge cases
            if np.any(np.isnan(raw_predictions)):
                raise ValueError("Model returned NaN predictions. Check your input data and model.")

            # Ensure 2D shape for consistent handling
            if raw_predictions.ndim == 1:
                raw_predictions = raw_predictions.reshape(-1, 1)

            n_samples, n_outputs = raw_predictions.shape

            # Determine if this is classification or regression
            is_classification = self._is_classification_model()

            if not is_classification:
                # Regression: return raw predictions, maintain original shape
                self.predictions = raw_predictions.flatten() if n_outputs == 1 else raw_predictions
            else:
                # Classification: return class predictions as integers
                if n_outputs == 1:
                    # Binary classification with single output (sigmoid)
                    if not np.all((raw_predictions >= 0) & (raw_predictions <= 1)):
                        raise ValueError(
                            "Binary classification model output not in [0,1] range. "
                            "Ensure final layer uses sigmoid activation."
                        )
                    self.predictions = (raw_predictions.flatten() > 0.5).astype(int)
                else:
                    # Multi-class classification (softmax)
                    self.predictions = np.argmax(raw_predictions, axis=1).astype(int)

            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates probability scores for each class.

            Only works for classification models.

            Args:
                data: Input data for prediction
                **kwargs: Additional arguments passed to model.predict()

            Returns:
                np.ndarray: Class probabilities with shape (n_samples, n_classes)
                           Always returns 2D array, even for binary classification
            """
            if not self._is_classification_model():
                raise ValueError(
                    "predict_proba() is only available for classification models. "
                    "This appears to be a regression model."
                )

            raw_predictions = self.model.predict(data, **kwargs)

            # Handle edge cases
            if np.any(np.isnan(raw_predictions)):
                raise ValueError("Model returned NaN predictions. Check your input data and model.")

            # Ensure 2D shape
            if raw_predictions.ndim == 1:
                raw_predictions = raw_predictions.reshape(-1, 1)

            n_samples, n_outputs = raw_predictions.shape

            if n_outputs == 1:
                # Binary classification with sigmoid output
                prob_positive = raw_predictions.flatten()

                # Validate probabilities are in valid range
                if not np.all((prob_positive >= 0) & (prob_positive <= 1)):
                    raise ValueError(
                        "Binary classification model output not in [0,1] range. "
                        "Ensure final layer uses sigmoid activation."
                    )

                prob_negative = 1.0 - prob_positive
                return np.column_stack([prob_negative, prob_positive])
            else:
                # Multi-class with softmax output
                # Validate probabilities sum to approximately 1
                row_sums = np.sum(raw_predictions, axis=1)
                if not np.allclose(row_sums, 1.0, rtol=1e-3):
                    raise ValueError(
                        "Multi-class model output doesn't sum to 1 across classes. "
                        "Ensure final layer uses softmax activation."
                    )

                return raw_predictions

        def _is_classification_model(self) -> bool:
            """
            Determine if this is a classification model by examining the loss function.

            Returns:
                bool: True if classification, False if regression
            """
            try:
                # Try to get the loss function
                if not hasattr(self.model, "loss") or self.model.loss is None:
                    # Model not compiled, assume classification (most common in this library)
                    return True

                loss = self.model.loss

                # Handle different loss function representations
                if hasattr(loss, "__name__"):
                    loss_name = loss.__name__.lower()
                elif hasattr(loss, "name"):
                    loss_name = loss.name.lower()
                elif isinstance(loss, str):
                    loss_name = loss.lower()
                else:
                    loss_name = str(loss).lower()

                # Classification loss functions (comprehensive list)
                classification_indicators = [
                    "categorical_crossentropy",
                    "sparse_categorical_crossentropy",
                    "binary_crossentropy",
                    "categorical_hinge",
                    "sparse_categorical_hinge",
                    "hinge",
                    "squared_hinge",
                    "focal_loss",
                    "crossentropy",
                ]

                # Regression loss functions
                regression_indicators = [
                    "mean_squared_error",
                    "mse",
                    "mean_absolute_error",
                    "mae",
                    "mean_absolute_percentage_error",
                    "mape",
                    "huber_loss",
                    "huber",
                    "log_cosh",
                    "logcosh",
                ]

                # Check for regression first (more specific)
                if any(reg_loss in loss_name for reg_loss in regression_indicators):
                    return False

                # Check for classification
                if any(cls_loss in loss_name for cls_loss in classification_indicators):
                    return True

                # If we can't determine, assume classification (safer default for this library)
                return True

            except (AttributeError, TypeError):
                # Fallback: assume classification
                return True

        def _get_num_classes(self) -> int:
            """
            Helper method to determine the number of classes from model output.

            Returns:
                int: Number of classes the model predicts
            """
            try:
                output_shape = self.model.output_shape
                if isinstance(output_shape, tuple):
                    # Single output
                    return 2 if output_shape[-1] == 1 else output_shape[-1]
                elif isinstance(output_shape, list):
                    # Multiple outputs - use first output
                    return 2 if output_shape[0][-1] == 1 else output_shape[0][-1]
                else:
                    # Fallback
                    return 2
            except (AttributeError, IndexError):
                # If we can't determine from model structure, assume binary
                return 2

        def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
            """
            Evaluates the encapsulated Keras model on a given dataset.

            This implementation handles three common Keras data formats:
            1.  A `tf.data.Dataset` object.
            2.  A `tf.keras.utils.Sequence` object (a data generator).
            3.  A tuple of `(X, y)` numpy arrays.

            It retrieves the metric names from `model.metrics_names` and
            zips them with the results from `model.evaluate()` to create
            a human-readable dictionary.

            Args:
                data (Any): The data to evaluate on, including features
                    and true labels. Can be a `tf.data.Dataset`,
                    a `Sequence`, or a tuple of (X, y) numpy arrays.
                **kwargs: Additional arguments passed directly to the
                    `model.evaluate()` call.

            Returns:
                Dict[str, Any]: A dictionary of metrics, where keys are the
                metric names (e.g., 'loss', 'accuracy') and values are the
                scalar metric values.

            Raises:
                TypeError: If `data` is not in one of the three
                    supported formats.
            """
            if isinstance(data, (tf.data.Dataset, Sequence)):
                results = self.model.evaluate(data, **kwargs)
            elif isinstance(data, tuple) and len(data) == 2:
                X_test, y_test = data
                results = self.model.evaluate(X_test, y_test, **kwargs)
            else:
                raise TypeError(
                    "Evaluation data must be a tuple of (X, y), a tf.data.Dataset, or a Sequence."
                )
            metric_names = self.model.metrics_names
            return dict(zip(metric_names, results))

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")
            accuracy = accuracy_score(true_labels, self.predictions)
            return {"accuracy": accuracy}

        def save_model(self, path: str):
            self.model.save(path)
            logger.info(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str) -> "KerasModelWrapper":
            loaded_model = tf.keras.models.load_model(path)
            return cls(loaded_model)

else:
    KerasModelWrapper = None

# Concrete class for Scikit-learn models
if SKLEARN_AVAILABLE:

    class ScikitLearnModelWrapper(BaseModelWrapper):
        def __init__(self, model: BaseEstimator, model_id: str = ""):
            super().__init__(model, model_id)

        def _cleanup_implementation(self):
            """Scikit-learn specific cleanup (minimal)."""
            import gc

            gc.collect()

        def fit(
            self,
            train_data: Union[Tuple[np.ndarray, np.ndarray], Any],
            validation_data: Optional[Any] = None,
            epochs: Optional[int] = None,  # Accept but ignore
            batch_size: Optional[int] = None,  # Accept but ignore
            verbose: Optional[int] = None,  # Accept but ignore
            **kwargs,
        ):
            """
            Fits the encapsulated scikit-learn model and creates a mock history.

            This implementation is designed for compatibility with the
            Ictonyx `ExperimentRunner`, which expects an epoch-based history.
            Since scikit-learn models train in a single call, this method:
            1.  Fits the model using `model.fit()`.
            2.  Calculates training accuracy (and validation accuracy if
                `validation_data` is provided).
            3.  Creates a `TrainingResult` containing a single "epoch"
                with these metrics (e.g., `{'accuracy': [0.95],
                'val_accuracy': [0.92]}`).
            4.  Stores the result in `self.training_result`.

            Args:
                train_data: The training data, must be a tuple of
                    `(X_train, y_train)` numpy arrays.
                validation_data: Optional tuple of `(X_val, y_val)`
                    numpy arrays. If not provided, validation metrics in the
                    mock history will be populated as a copy of the
                    training metrics.
                epochs: **Ignored.** Accepted for API compatibility with Keras.
                batch_size: **Ignored.** Accepted for API compatibility.
                verbose: **Ignored.** Accepted for API compatibility.
                **kwargs: Additional fit parameters (like `sample_weight`)
                    that will be passed directly to the scikit-learn
                    model's `.fit()` method.

            Raises:
                ValueError: If `train_data` is not a tuple of (X, y).
            """
            if isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data

                # Filter out Keras-specific kwargs that sklearn doesn't understand
                sklearn_kwargs = {}
                for key, value in kwargs.items():
                    # Only pass through parameters that sklearn models typically accept
                    if key in ["sample_weight", "check_input", "X_idx_sorted"]:
                        sklearn_kwargs[key] = value
                    # Silently ignore other parameters (like validation_split, callbacks, etc.)

                # Fit the model (sklearn models ignore epochs/batch_size/verbose from signature)
                self.model.fit(X_train, y_train, **sklearn_kwargs)

                # Calculate training accuracy
                train_accuracy = self.model.score(X_train, y_train)

                # Create mock history dictionary (sklearn trains in one "epoch")
                history_dict = {
                    "accuracy": [train_accuracy],  # Training accuracy
                    "loss": [1.0 - train_accuracy],  # Mock loss as 1 - accuracy
                }

                # Add validation metrics if validation data provided
                if (
                    validation_data is not None
                    and isinstance(validation_data, tuple)
                    and len(validation_data) == 2
                ):
                    X_val, y_val = validation_data
                    val_accuracy = self.model.score(X_val, y_val)
                    history_dict["val_accuracy"] = [val_accuracy]
                    history_dict["val_loss"] = [1.0 - val_accuracy]
                else:
                    # ExperimentRunner expects validation metrics - use training metrics as fallback
                    history_dict["val_accuracy"] = [train_accuracy]
                    history_dict["val_loss"] = [1.0 - train_accuracy]

                self.training_result = TrainingResult(history=history_dict)

            else:
                raise ValueError(
                    "ScikitLearnModelWrapper.fit() expects train_data as a tuple of (X_train, y_train)"
                )

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            self.predictions = self.model.predict(data, **kwargs)
            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(data, **kwargs)
            else:
                raise NotImplementedError(
                    "The scikit-learn model does not have a predict_proba method."
                )

        def evaluate(
            self, data: Union[Tuple[np.ndarray, np.ndarray], Any], **kwargs
        ) -> Dict[str, Any]:
            """Evaluate sklearn model, returning all applicable metrics."""
            X_test, y_test = data
            y_pred = self.model.predict(X_test)

            metrics: Dict[str, Any] = {"accuracy": accuracy_score(y_test, y_pred)}

            try:
                from sklearn.metrics import f1_score, precision_score, recall_score

                average = "binary" if len(np.unique(y_test)) == 2 else "weighted"
                metrics["precision"] = precision_score(
                    y_test, y_pred, average=average, zero_division=0
                )
                metrics["recall"] = recall_score(y_test, y_pred, average=average, zero_division=0)
                metrics["f1"] = f1_score(y_test, y_pred, average=average, zero_division=0)
            except Exception:
                pass

            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                if not hasattr(self.model, "predict_proba"):
                    metrics["r2"] = r2_score(y_test, y_pred)
                    metrics["mse"] = mean_squared_error(y_test, y_pred)
                    metrics["mae"] = mean_absolute_error(y_test, y_pred)
            except Exception:
                pass

            return metrics

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")
            accuracy = accuracy_score(true_labels, self.predictions)
            return {"accuracy": accuracy}

        def save_model(self, path: str):
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str) -> "ScikitLearnModelWrapper":
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)
            return cls(loaded_model)

else:
    ScikitLearnModelWrapper = None

# Concrete class for PyTorch models
if PYTORCH_AVAILABLE:

    class PyTorchModelWrapper(BaseModelWrapper):
        """Wrapper for PyTorch nn.Module models.

        Provides a complete training loop, metric tracking, and device
        management. Designed to integrate seamlessly with Ictonyx's
        ExperimentRunner and variability study pipeline.

        The optimizer is specified as a class + params dict (not an instance)
        because a fresh optimizer must be bound to each new model's
        parameters when the ExperimentRunner rebuilds the model per run.

        Args:
            model: A PyTorch nn.Module instance.
            criterion: A PyTorch loss function instance (e.g., nn.CrossEntropyLoss()).
            optimizer_class: Optimizer class (e.g., torch.optim.Adam). Default: Adam.
            optimizer_params: Dict of optimizer kwargs (e.g., {'lr': 0.001}).
                Default: {'lr': 0.001}.
            device: 'cuda', 'cpu', or 'auto'. Default: 'auto' (uses CUDA if available).
            task: 'classification' or 'regression'. Default: 'classification'.
                Controls which metrics are computed during training and evaluation.
            model_id: Optional string identifier.

        Example:
            >>> model = nn.Sequential(nn.Linear(10, 2))
            >>> wrapper = PyTorchModelWrapper(
            ...     model,
            ...     criterion=nn.CrossEntropyLoss(),
            ...     optimizer_class=torch.optim.Adam,
            ...     optimizer_params={'lr': 0.001}
            ... )
            >>> wrapper.fit((X_train, y_train), validation_data=(X_val, y_val), epochs=10)
        """

        def __init__(
            self,
            model: "nn.Module",
            criterion: Any = None,
            optimizer_class: Any = None,
            optimizer_params: Optional[Dict[str, Any]] = None,
            device: str = "auto",
            task: str = "classification",
            model_id: str = "",
        ):
            super().__init__(model, model_id)
            self.criterion = criterion or nn.CrossEntropyLoss()
            self.optimizer_class = optimizer_class or torch.optim.Adam
            self.optimizer_params = optimizer_params or {"lr": 0.001}
            self.task = task

            # Resolve device
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model.to(self.device)

        def _cleanup_implementation(self):
            """PyTorch-specific cleanup."""
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.cpu()
                    del self.model
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        def _to_tensor(self, data: np.ndarray, dtype=None) -> "torch.Tensor":
            """Convert numpy array to tensor on the correct device."""
            if dtype is None:
                # Float for features, long for classification labels, float for regression labels
                dtype = torch.float32
            return torch.tensor(data, dtype=dtype).to(self.device)

        def _make_dataloader(
            self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
        ) -> "DataLoader":
            """Create a DataLoader from numpy arrays."""
            X_t = self._to_tensor(X, dtype=torch.float32)
            if self.task == "classification":
                y_t = self._to_tensor(y, dtype=torch.long)
            else:
                y_t = self._to_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_t, y_t)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        def fit(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs):
            """
            Trains the PyTorch model with a standard training loop.

            Accepts (X, y) numpy array tuples or PyTorch DataLoaders. Tracks
            loss and accuracy (classification) or MSE (regression) per epoch,
            storing results in self.training_result as a TrainingResult.

            Args:
                train_data: Tuple of (X, y) numpy arrays, or a PyTorch DataLoader.
                validation_data: Optional tuple of (X_val, y_val) or DataLoader.
                **kwargs: Accepts 'epochs' (default 10), 'batch_size' (default 32),
                    'verbose' (default 0). Other kwargs are ignored for
                    compatibility with ExperimentRunner.
            """
            epochs = kwargs.get("epochs", 10)
            batch_size = kwargs.get("batch_size", 32)

            # Build DataLoaders
            if isinstance(train_data, DataLoader):
                train_loader = train_data
            elif isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
                train_loader = self._make_dataloader(X_train, y_train, batch_size, shuffle=True)
            else:
                raise TypeError(
                    "train_data must be a tuple of (X, y) numpy arrays or a PyTorch DataLoader."
                )

            val_loader = None
            if validation_data is not None:
                if isinstance(validation_data, DataLoader):
                    val_loader = validation_data
                elif isinstance(validation_data, tuple) and len(validation_data) == 2:
                    X_val, y_val = validation_data
                    val_loader = self._make_dataloader(X_val, y_val, batch_size, shuffle=False)

            # Create optimizer bound to this model's parameters
            optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)

            # History tracking
            history = {
                "loss": [],
            }
            if self.task == "classification":
                history["accuracy"] = []

            if val_loader is not None:
                history["val_loss"] = []
                if self.task == "classification":
                    history["val_accuracy"] = []
                else:
                    history["val_mse"] = []

            # Training loop
            for epoch in range(epochs):
                # --- Train phase ---
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)

                    # Squeeze for regression with single output
                    if self.task == "regression" and outputs.dim() > 1 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)

                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)
                    total += X_batch.size(0)

                    if self.task == "classification":
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == y_batch).sum().item()

                epoch_loss = running_loss / total
                history["loss"].append(epoch_loss)

                if self.task == "classification":
                    epoch_acc = correct / total
                    history["accuracy"].append(epoch_acc)

                # --- Validation phase ---
                if val_loader is not None:
                    val_loss, val_metric = self._evaluate_loader(val_loader)
                    history["val_loss"].append(val_loss)
                    if self.task == "classification":
                        history["val_accuracy"].append(val_metric)
                    else:
                        history["val_mse"].append(val_metric)

            self.training_result = TrainingResult(
                history=history,
                params={"epochs": epochs, "batch_size": batch_size, "device": str(self.device)},
            )

        def _evaluate_loader(self, loader: "DataLoader") -> Tuple[float, float]:
            """Evaluate model on a DataLoader. Returns (loss, metric).

            For classification, metric is accuracy. For regression, metric is MSE.
            """
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch in loader:
                    outputs = self.model(X_batch)

                    if self.task == "regression" and outputs.dim() > 1 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)

                    loss = self.criterion(outputs, y_batch)
                    running_loss += loss.item() * X_batch.size(0)
                    total += X_batch.size(0)

                    if self.task == "classification":
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == y_batch).sum().item()

            avg_loss = running_loss / total
            if self.task == "classification":
                metric = correct / total
            else:
                metric = avg_loss  # MSE is the loss itself for regression

            return avg_loss, metric

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates predictions from numpy input.

            For classification, returns class indices (integers).
            For regression, returns predicted values (floats).

            Args:
                data: Input numpy array of shape (n_samples, n_features).

            Returns:
                np.ndarray: Predictions.
            """
            self.model.eval()
            X_t = self._to_tensor(data, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(X_t)

            if self.task == "classification":
                _, predicted = torch.max(outputs, 1)
                self.predictions = predicted.cpu().numpy()
            else:
                if outputs.dim() > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                self.predictions = outputs.cpu().numpy()

            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates class probability predictions.

            Applies softmax to the model's raw logits. Only valid for
            classification models.

            Args:
                data: Input numpy array.

            Returns:
                np.ndarray: Probabilities of shape (n_samples, n_classes).

            Raises:
                ValueError: If task is 'regression'.
            """
            if self.task != "classification":
                raise ValueError("predict_proba() is only available for classification models.")

            self.model.eval()
            X_t = self._to_tensor(data, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(X_t)
                probabilities = torch.softmax(outputs, dim=1)

            return probabilities.cpu().numpy()

        def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
            """
            Evaluates the model on labeled data.

            Args:
                data: Tuple of (X, y) numpy arrays, or a DataLoader.

            Returns:
                Dict with 'loss' and task-specific metrics:
                - Classification: 'accuracy', 'correct', 'total'
                - Regression: 'mse'
            """
            if isinstance(data, DataLoader):
                loader = data
            elif isinstance(data, tuple) and len(data) == 2:
                X_test, y_test = data
                loader = self._make_dataloader(X_test, y_test, batch_size=256, shuffle=False)
            else:
                raise TypeError("Evaluation data must be a tuple of (X, y) or a DataLoader.")

            loss, metric = self._evaluate_loader(loader)
            metrics: Dict[str, Any] = {"loss": loss}

            if self.task == "classification":
                metrics["accuracy"] = metric
            else:
                metrics["mse"] = metric

            return metrics

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            """Quick assessment using stored predictions."""
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")

            if self.task == "classification":
                accuracy = float(np.mean(self.predictions == true_labels))
                return {"accuracy": accuracy}
            else:
                mse = float(np.mean((self.predictions - true_labels) ** 2))
                return {"mse": mse}

        def save_model(self, path: str):
            """Save model state dict to disk."""
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "task": self.task,
                    "device": str(self.device),
                },
                path,
            )
            logger.info(f"PyTorch model saved to {path}")

        @classmethod
        def load_model(
            cls, path: str, model: "nn.Module", task: str = "classification"
        ) -> "PyTorchModelWrapper":
            """
            Load model from a state dict checkpoint.  Return model wrapper.
            """
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            wrapper = cls(model, task=checkpoint.get("task", task))
            return wrapper

else:
    PyTorchModelWrapper = None
