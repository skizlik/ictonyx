# ictonyx/core.py

import gc
import os
import pickle
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import ModelError
from .memory import get_memory_manager
from .settings import logger

# Optional sklearn metrics (fallback provided for accuracy_score)
try:
    from sklearn.metrics import accuracy_score

    _HAS_SKLEARN_METRICS = True
except ImportError:
    _HAS_SKLEARN_METRICS = False

    def accuracy_score(y_true, y_pred):
        """Numpy fallback when sklearn is not installed."""
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# Optional TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.utils import Sequence

    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    KerasModel = None  # type: ignore[assignment]
    Sequence = None  # type: ignore[assignment]
    TENSORFLOW_AVAILABLE = False

# Optional scikit-learn imports
try:
    from sklearn.base import BaseEstimator

    SKLEARN_AVAILABLE = True
except ImportError:
    BaseEstimator = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TensorDataset = None  # type: ignore[assignment,misc]
    PYTORCH_AVAILABLE = False

# HuggingFace Transformers imports
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers import set_seed as _hf_set_seed

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def _regression_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute r², MSE, and MAE for regression predictions.

    Args:
        preds: Predicted values, shape (n,).
        labels: True values, shape (n,).

    Returns:
        Dict with keys ``'r2'``, ``'mse'``, ``'rmse'``, ``'mae'``.
        ``r2`` is ``nan`` when all labels are identical.
    """
    preds = np.asarray(preds, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
    ss_res = float(np.sum((labels - preds) ** 2))
    mse = float(np.mean((labels - preds) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(labels - preds))),
    }


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
    """Abstract base class for wrapping ML models with a common interface.

    All model wrappers (Keras, scikit-learn, PyTorch) inherit from this
    class and implement its abstract methods. The wrapper provides:

    * A consistent ``fit`` / ``predict`` / ``evaluate`` / ``assess`` API.
    * Automatic resource cleanup via :meth:`cleanup`.
    * A standardized :class:`TrainingResult` output from ``fit()``.
    * Memory tracking via :class:`~ictonyx.memory.MemoryManager`.

    Subclasses must implement: :meth:`fit`, :meth:`predict`,
    :meth:`predict_proba`, :meth:`evaluate`, :meth:`assess`,
    :meth:`save_model`, :meth:`load_model`, and
    :meth:`_cleanup_implementation`.

    Args:
        model: The underlying model object (Keras Model, sklearn estimator,
            PyTorch Module, etc.).
        model_id: An optional string identifier for logging and display.
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
        """Automatic cleanup when object is destroyed.

        Skips cleanup during interpreter shutdown. At shutdown time, modules
        such as ``tensorflow`` and ``torch`` may be partially unloaded.
        Calling framework teardown (e.g. ``tf.keras.backend.clear_session()``)
        in that state can trigger out-of-order shutdown sequences.
        """

        if sys.is_finalizing():
            return
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
        """Run cleanup and report if significant memory was freed.

        Returns a summary dict if more than 50 MB was freed, otherwise ``None``.
        """
        cleanup_result = self._resource_manager.cleanup()
        if cleanup_result.memory_freed_mb and cleanup_result.memory_freed_mb > 50:
            return {"cleaned": True, "freed_mb": cleanup_result.memory_freed_mb}
        return None


# --- CONCRETE IMPLEMENTATIONS ---

# Concrete class for TensorFlow/Keras models

if TENSORFLOW_AVAILABLE:

    class KerasModelWrapper(BaseModelWrapper):
        """Wrapper for Keras / TensorFlow models.

        Handles the three common Keras data formats (``tf.data.Dataset``,
        ``tf.keras.utils.Sequence``, and ``(X, y)`` numpy tuples), stores
        training history as a :class:`TrainingResult`, and performs
        TensorFlow session cleanup on :meth:`cleanup`.

        Args:
            model: A compiled ``tf.keras.Model`` instance.
            model_id: Optional string identifier for logging.
            task: ``'classification'``, ``'regression'``, or ``None``
                (default). When ``None``, the task is inferred from the
                compiled loss function. Set this explicitly whenever you
                use a custom loss function, an uncompiled model, or a
                loss name not in the built-in recognition list.

        Raises:
            ValueError: If *task* is not one of the three accepted values.
        """

        def __init__(
            self,
            model: KerasModel,
            model_id: str = "",
            task: Optional[str] = None,
            clear_session: bool = True,
        ):
            """
            Args:
                model: A compiled ``tf.keras.Model``.
                model_id: Optional label for logging.
                task: ``'classification'``, ``'regression'``, or ``None``
                    (auto-detect from loss function).
                clear_session: If ``True`` (default), call
                    ``tf.keras.backend.clear_session()`` at the start of
                    each ``fit()`` call. This releases GPU memory from
                    previous runs and prevents accumulation across a
                    variability study. Set to ``False`` only if you manage
                    the session lifecycle externally.
            """
            if task is not None and task not in ("classification", "regression"):
                raise ValueError(
                    f"task must be 'classification', 'regression', or None "
                    f"(auto-detect from loss), got '{task}'."
                )
            super().__init__(model, model_id)
            self.task: Optional[str] = task
            self.clear_session = clear_session

        def _cleanup_implementation(self):
            """TensorFlow/Keras specific cleanup."""
            # Clear the TF session before deleting the model reference.
            # Reversing this order can leave dangling references in the session.
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
            except Exception:
                pass
            if hasattr(self, "model"):
                try:
                    del self.model
                except Exception:
                    pass
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
            if self.clear_session:
                try:
                    import tensorflow as tf

                    tf.keras.backend.clear_session()
                except Exception:
                    pass

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
                # Regression: return raw predicted values as floats.
                self.predictions = raw_predictions.flatten().astype(float)
                return self.predictions  # type: ignore[return-value]
            else:
                if n_outputs == 1:
                    self.predictions = (raw_predictions.flatten() >= 0.5).astype(int)
                else:
                    self.predictions = np.argmax(raw_predictions, axis=1).astype(int)
                return self.predictions  # type: ignore[return-value]

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
                # Use atol, not rtol: checking whether probabilities sum to 1
                # is an absolute (not relative) tolerance question.
                if not np.allclose(row_sums, 1.0, atol=1e-3):
                    raise ValueError(
                        "Multi-class model output doesn't sum to 1 across classes. "
                        "Ensure final layer uses softmax activation."
                    )

                return raw_predictions

        def _is_classification_model(self) -> bool:
            """Determine whether this is a classification or regression model.

            If the wrapper was created with an explicit ``task`` argument,
            that value is used directly and no inspection is performed.
            Otherwise the method inspects the compiled loss function.

            Returns:
                ``True`` for classification, ``False`` for regression.

            Raises:
                ValueError: If the task cannot be determined — i.e. the
                    model is uncompiled, uses a custom loss, or the loss
                    name is not in the built-in recognition list — and no
                    explicit ``task`` was provided at construction.
            """
            # Honour an explicit task set at construction time.
            if self.task is not None:
                return self.task == "classification"

            # --- Step 1: extract the loss name. ---
            # This accesses model attributes and is the only step that can
            # raise AttributeError or TypeError from unexpected model state.
            try:
                if not hasattr(self.model, "loss") or self.model.loss is None:
                    raise ValueError(
                        "Cannot determine task type: the Keras model has not been "
                        "compiled (model.loss is None). Either compile the model "
                        "before wrapping it, or pass task='classification' or "
                        "task='regression' to KerasModelWrapper()."
                    )

                loss = self.model.loss

                if hasattr(loss, "__name__"):
                    loss_name = loss.__name__.lower()
                elif hasattr(loss, "name"):
                    loss_name = loss.name.lower()
                elif isinstance(loss, str):
                    loss_name = loss.lower()
                else:
                    loss_name = str(loss).lower()

            except (AttributeError, TypeError) as exc:
                raise ValueError(
                    "Cannot determine task type: failed to inspect model.loss "
                    f"({exc}). Pass task='classification' or task='regression' "
                    "to KerasModelWrapper()."
                ) from exc

            # --- Step 2: match the loss name against known indicators. ---
            # Pure string logic; no exceptions expected from here on.
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

            # Check regression first — names are more specific than classification ones.
            if any(reg in loss_name for reg in regression_indicators):
                return False

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

            if any(cls in loss_name for cls in classification_indicators):
                return True

            # Unknown loss — refuse to guess silently.
            raise ValueError(
                f"Cannot determine task type from loss function '{loss_name}'. "
                "Pass task='classification' or task='regression' to "
                "KerasModelWrapper() to set the task type explicitly."
            )

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
            # TF 2.x returns a plain float for loss-only models (no additional metrics).
            if not hasattr(results, "__iter__"):
                results = [results]
            return dict(zip(metric_names, results))

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            """Assess stored predictions against true labels.

            For classification models: returns ``{'accuracy': float}``.
            For regression models: returns ``{'r2': float, 'mse': float, 'rmse': float, 'mae': float}``.

            Task type is determined via :meth:`_is_classification_model`, which
            respects the explicit ``task`` parameter if set at construction.

            Args:
                true_labels: 1-D array of ground-truth labels or values,
                    corresponding to the stored ``self.predictions``.

            Returns:
                Dict of metric name → float.

            Raises:
                ValueError: If :meth:`predict` has not been called yet.
                ValueError: If task type cannot be determined (uncompiled model
                    with no explicit ``task``).
            """
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")

            if self._is_classification_model():
                return {"accuracy": float(accuracy_score(true_labels, self.predictions))}

            # Regression: pure NumPy, no additional imports required.
            preds = np.asarray(self.predictions, dtype=float)
            labels = np.asarray(true_labels, dtype=float)
            return _regression_metrics(preds, labels)

        def save_model(self, path: str):
            self.model.save(path)
            logger.info(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str, task: Optional[str] = None) -> "KerasModelWrapper":
            """Load a saved Keras model and return a wrapped instance.

            Args:
                path: Path written by :meth:`save_model` (SavedModel
                    directory, ``.keras``, or ``.h5`` file).
                task: ``'classification'``, ``'regression'``, or ``None``
                    (auto-detect from loss). Pass explicitly when using
                    custom loss functions.

            Returns:
                :class:`KerasModelWrapper` wrapping the loaded model.
            """
            loaded_model = tf.keras.models.load_model(path)
            return cls(loaded_model, task=task)

else:
    KerasModelWrapper = None  # type: ignore[assignment,misc]

# Concrete class for Scikit-learn models
if SKLEARN_AVAILABLE:

    class ScikitLearnModelWrapper(BaseModelWrapper):
        """Wrapper for scikit-learn estimators.

        Provides a consistent ``fit`` / ``predict`` / ``evaluate`` / ``assess``
        interface over any scikit-learn estimator, making it compatible with
        :class:`~ictonyx.runners.ExperimentRunner` and the variability study
        pipeline.

        Task type (classification vs. regression) is inferred automatically after
        ``fit()`` by inspecting the estimator for ``predict_proba`` or
        ``classes_``. It can also be set explicitly via the ``task`` parameter
        if auto-detection is unreliable for a custom estimator.

        Because scikit-learn models train in a single call rather than
        epoch-by-epoch, ``fit()`` produces a one-entry mock history
        (``{'accuracy': [score], 'val_accuracy': [score]}`` for classifiers,
        or ``{'r2': [score], 'val_r2': [score]}`` for regressors) so that
        downstream metric-aggregation code operates uniformly across all
        wrapper types.

        Args:
            model: A scikit-learn estimator instance (any object implementing
                ``fit`` and ``predict``).
            model_id: Optional string identifier for logging and display.
            task: ``'classification'``, ``'regression'``, or ``None``
                (default). When ``None``, the task is inferred after the
                first ``fit()`` call. Set this explicitly for custom
                estimators that do not expose ``predict_proba`` or
                ``classes_``.

        Example::

            from sklearn.ensemble import RandomForestClassifier
            import ictonyx as ix

            wrapper = ix.ScikitLearnModelWrapper(
                RandomForestClassifier(n_estimators=100),
                model_id="rf_baseline",
            )
            wrapper.fit((X_train, y_train), validation_data=(X_val, y_val))
            wrapper.predict(X_test)
            print(wrapper.assess(y_test))
        """

        _SKLEARN_FIT_KWARGS: frozenset = frozenset(
            {
                "sample_weight",
                "eval_set",  # XGBoost/LightGBM (ahead of v0.5.0 wrappers)
                "cat_features",  # CatBoost
            }
        )

        _KNOWN_IGNORED_KWARGS: frozenset = frozenset(
            {
                "epochs",
                "batch_size",
                "verbose",
                "validation_split",
                "callbacks",
                "shuffle",
                "steps_per_epoch",
            }
        )

        def __init__(self, model: BaseEstimator, model_id: str = "", task: Optional[str] = None):
            super().__init__(model, model_id)
            self.task: Optional[str] = task

        def _cleanup_implementation(self):
            """Scikit-learn specific cleanup (minimal)."""

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

                # Warn on deprecated explicit kwargs
                for _dep_kwarg, _dep_val in [
                    ("epochs", epochs),
                    ("batch_size", batch_size),
                    ("verbose", verbose),
                ]:
                    if _dep_val is not None:
                        warnings.warn(
                            f"Passing '{_dep_kwarg}' to ScikitLearnModelWrapper.fit() "
                            "is deprecated and will raise TypeError in v0.5.0. "
                            "sklearn models ignore this parameter; remove it from your call.",
                            DeprecationWarning,
                            stacklevel=2,
                        )

                sklearn_kwargs = {}
                for key, value in kwargs.items():
                    if key in self._SKLEARN_FIT_KWARGS:
                        sklearn_kwargs[key] = value
                    elif key not in self._KNOWN_IGNORED_KWARGS:
                        warnings.warn(
                            f"ScikitLearnModelWrapper.fit() received unrecognized "
                            f"keyword argument '{key}={value!r}'. This argument has "
                            f"been ignored. If it is a valid sklearn fit() parameter, "
                            f"add it to _SKLEARN_FIT_KWARGS in ScikitLearnModelWrapper.",
                            UserWarning,
                            stacklevel=2,
                        )

                # Fit the model (sklearn models ignore epochs/batch_size/verbose from signature)
                self.model.fit(X_train, y_train, **sklearn_kwargs)
                self._n_classes = int(len(np.unique(y_train)))

                # Detect classifier vs regressor
                is_classifier = hasattr(self.model, "predict_proba") or hasattr(
                    self.model, "classes_"
                )

                # Calculate training score (accuracy for classifiers, R² for regressors)
                train_score = self.model.score(X_train, y_train)

                # Calculate validation score only when real validation data was provided.
                # Do NOT fabricate val_score = train_score when validation_data is None.
                # A fabricated val_accuracy is indistinguishable from a real one downstream
                # and silently corrupts any comparison or summary that reads val_accuracy.
                has_val_data = (
                    validation_data is not None
                    and isinstance(validation_data, tuple)
                    and len(validation_data) == 2
                )

                if is_classifier:
                    history_dict: Dict[str, list] = {"accuracy": [train_score]}
                    if has_val_data:
                        if validation_data is None:
                            raise RuntimeError(
                                "has_val_data is True but validation_data is None. "
                                "This is a bug — please report it."
                            )
                        X_val, y_val = validation_data
                        val_score = self.model.score(X_val, y_val)
                        history_dict["val_accuracy"] = [val_score]
                else:
                    history_dict = {"r2": [train_score]}
                    if has_val_data:
                        if validation_data is None:
                            raise RuntimeError(
                                "has_val_data is True but validation_data is None. "
                                "This is a bug — please report it."
                            )
                        X_val, y_val = validation_data
                        val_score = self.model.score(X_val, y_val)
                        history_dict["val_r2"] = [val_score]

                self.training_result = TrainingResult(history=history_dict)
                self.task = "classification" if is_classifier else "regression"

            else:
                raise ValueError(
                    "ScikitLearnModelWrapper.fit() expects train_data as a tuple of (X_train, y_train)"
                )

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            raw = self.model.predict(data, **kwargs)
            if raw is None:
                raise ModelError(
                    "sklearn model.predict() returned None. "
                    "This is unexpected behaviour from the underlying estimator.",
                    operation="predict",
                )
            self.predictions = raw
            return self.predictions  # type: ignore[return-value]

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
            """Evaluate the model on a test set, returning all applicable metrics.

            For classifiers (models with ``predict_proba`` or ``classes_``):
            returns accuracy, precision, recall, and F1. For regressors:
            returns R², MSE, RMSE, and MAE.

            Multiclass classification uses weighted averaging for precision,
            recall, and F1. Binary classification uses binary averaging.

            Args:
                data: A tuple of ``(X_test, y_test)`` numpy arrays.
                **kwargs: Unused, accepted for API compatibility.

            Returns:
                Dict mapping metric names to float values. Keys depend on
                model type — classifiers get ``'accuracy'``, ``'precision'``,
                ``'recall'``, ``'f1'``; regressors get ``'r2'``, ``'mse'``, ``'rmse'``,
                ``'mae'``.
            """
            X_test, y_test = data
            y_pred = self.model.predict(X_test)

            metrics: Dict[str, Any] = {}

            # Classification metrics only apply to discrete targets
            is_classifier = hasattr(self.model, "predict_proba") or hasattr(self.model, "classes_")
            if is_classifier:
                metrics["accuracy"] = accuracy_score(y_test, y_pred)

                try:
                    from sklearn.metrics import f1_score, precision_score, recall_score

                    # Use the class count from training, not from the test batch.
                    # A small or imbalanced test batch may be missing some classes,
                    # causing np.unique(y_test) to return fewer classes than the model
                    # actually knows about and selecting the wrong averaging strategy.
                    # getattr fallback: if evaluate() is called before fit() (e.g. on a
                    # freshly loaded model), fall back to the test-set count.
                    n_classes = getattr(self, "_n_classes", len(np.unique(y_test)))
                    average = "binary" if n_classes == 2 else "weighted"
                    metrics["precision"] = precision_score(
                        y_test, y_pred, average=average, zero_division=0
                    )
                    metrics["recall"] = recall_score(
                        y_test, y_pred, average=average, zero_division=0
                    )
                    metrics["f1"] = f1_score(y_test, y_pred, average=average, zero_division=0)
                except Exception as exc:
                    logger.warning(f"Could not compute precision/recall/f1: {exc}")

            if not is_classifier:
                try:
                    metrics.update(
                        _regression_metrics(
                            np.asarray(y_pred, dtype=float),
                            np.asarray(y_test, dtype=float),
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Could not compute regression metrics: {exc}")

            return metrics

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            """Assess stored predictions against true labels.

            For classification: returns ``{'accuracy': float}``.
            For regression models: returns ``{'r2': float, 'mse': float, 'rmse': float, 'mae': float}``,
            consistent with :class:`ScikitLearnModelWrapper` and
            :class:`KerasModelWrapper`.

            Args:
                true_labels: 1-D array of ground-truth labels or values,
                    corresponding to the stored ``self.predictions``.

            Returns:
                Dict of metric name → float.

            Raises:
                ValueError: If :meth:`predict` has not been called yet.
            """
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")

            if self.task is not None:
                is_classifier = self.task == "classification"
            else:
                is_classifier = hasattr(self.model, "predict_proba") or hasattr(
                    self.model, "classes_"
                )

            if is_classifier:
                return {"accuracy": float(accuracy_score(true_labels, self.predictions))}

            # Regression: match ScikitLearnModelWrapper metric set exactly.
            preds = np.asarray(self.predictions, dtype=float)
            labels = np.asarray(true_labels, dtype=float)
            return _regression_metrics(preds, labels)

        def save_model(self, path: str):
            """Save the sklearn model using joblib.
            Warning:
                Only load files from **trusted sources**. Joblib files
                from untrusted sources can execute arbitrary code.
            Args:
                path: Destination file path (e.g. ``'model.joblib'``).
            """
            try:
                import joblib

                joblib.dump(self.model, path)
            except ImportError:
                import pickle

                with open(path, "wb") as f:
                    pickle.dump(self.model, f)
                logger.warning(
                    "joblib not available — model saved with pickle. "
                    "Install joblib for faster, more efficient serialization."
                )
            logger.info(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str) -> "ScikitLearnModelWrapper":
            """Load a saved scikit-learn model.
            Warning:
                Only load files from **trusted sources**. Serialized files
                from untrusted sources can execute arbitrary code.
            Args:
                path: Path written by :meth:`save_model`.
            Returns:
                ScikitLearnModelWrapper wrapping the loaded model.
            """
            try:
                import joblib

                loaded_model = joblib.load(path)
            except ImportError:
                import pickle

                with open(path, "rb") as f:
                    loaded_model = pickle.load(f)
                logger.warning(
                    "joblib not available — model loaded with pickle. "
                    "Only load files from trusted sources; pickle files "
                    "can execute arbitrary code on deserialization. "
                    "Install joblib for safer serialization: pip install joblib"
                )
            return cls(loaded_model)

else:
    ScikitLearnModelWrapper = None  # type: ignore[assignment,misc]

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
            history: Dict[str, list] = {
                "loss": [],
            }
            if self.task == "classification":
                history["accuracy"] = []

            if val_loader is not None:
                history["val_loss"] = []
                if self.task == "classification":
                    history["val_accuracy"] = []

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
                        reg_metrics = self._compute_epoch_regression_metrics(val_loader)
                        for k, v in reg_metrics.items():
                            history.setdefault(k, []).append(v)

            self.training_result = TrainingResult(
                history=history,
                params={"epochs": epochs, "batch_size": batch_size, "device": str(self.device)},
            )

        def _compute_epoch_regression_metrics(self, loader: "DataLoader") -> Dict[str, float]:
            """Compute full regression metrics over a DataLoader (val use only).

            Returns dict with ``'val_mse'``, ``'val_rmse'``, ``'val_mae'``,
            ``'val_r2'``, and ``'val_loss'``.
            """
            self.model.eval()
            all_preds: list = []
            all_labels: list = []
            running_loss = 0.0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch in loader:
                    outputs = self.model(X_batch)
                    if outputs.dim() > 1 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                    loss = self.criterion(outputs, y_batch)
                    running_loss += loss.item() * X_batch.size(0)
                    total += X_batch.size(0)
                    all_preds.extend(outputs.cpu().numpy().tolist())
                    all_labels.extend(y_batch.cpu().numpy().tolist())

            metrics = _regression_metrics(np.array(all_preds), np.array(all_labels))
            avg_loss = running_loss / total
            return {
                "val_loss": avg_loss,
                "val_mse": metrics["mse"],
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_r2": metrics["r2"],
            }

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

            if torch.any(torch.isnan(outputs)):
                raise ValueError(
                    "PyTorchModelWrapper.predict() returned NaN outputs. "
                    "Check model inputs for NaN/Inf values, or inspect the "
                    "training run for gradient explosion."
                )

            if self.task == "classification":
                _, predicted = torch.max(outputs, 1)
                self.predictions = predicted.cpu().numpy()
                return self.predictions  # type: ignore[return-value]
            else:
                if outputs.dim() > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                self.predictions = outputs.cpu().numpy()
                return self.predictions  # type: ignore[return-value]

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

            # Use direct children (not recursive modules()) — reliable for common architectures.
            # modules()[-1] returns the deepest leaf of the last sub-branch for ResNets and
            # other nested designs, which is not the output layer.
            _children = list(self.model.children())
            if _children and isinstance(_children[-1], (torch.nn.Softmax, torch.nn.LogSoftmax)):
                warnings.warn(
                    f"PyTorchModelWrapper.predict_proba(): the model's final "
                    f"layer is {type(_children[-1]).__name__}, but predict_proba() "
                    "also applies softmax. This double-application produces "
                    "incorrectly squashed probabilities. Remove the final "
                    "softmax layer and return raw logits instead.",
                    UserWarning,
                    stacklevel=2,
                )

            self.model.eval()
            X_t = self._to_tensor(data, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(X_t)

                # Single-output sigmoid: binary classifier with one output neuron.
                # softmax on (n, 1) produces all-ones rather than probabilities.
                if outputs.dim() > 1 and outputs.shape[-1] == 1:
                    p_pos = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
                    return np.column_stack([1.0 - p_pos, p_pos])

                # Multi-output softmax (standard multi-class case).
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
            """Assess stored predictions against ground-truth labels.

            Computes metrics from ``self.predictions``, which must have been
            populated by a prior call to :meth:`predict`. Does not run the
            model or require input data.

            For classification: returns ``{'accuracy': float}``.
            For regression: returns ``{'r2': float, 'mse': float, 'rmse': float, 'mae': float}``.

            Args:
                true_labels: 1-D array of ground-truth class indices (classification)
                    or target values (regression), aligned with the data passed to
                    the preceding :meth:`predict` call.

            Returns:
                Dict of metric name → float. Keys depend on task type — see above.

            Raises:
                ValueError: If :meth:`predict` has not been called yet
                    (``self.predictions`` is ``None``).
                ValueError: If ``self.predictions`` contains NaN values.
                ValueError: If ``self.task`` is ``None`` and task type cannot be
                    determined. Set ``task='classification'`` or ``task='regression'``
                    at construction to avoid ambiguity.
            """
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")

            if np.any(np.isnan(self.predictions)):
                raise ValueError(
                    "Stored predictions contain NaN. Check the preceding predict() "
                    "call for model instability or bad input data."
                )

            if self.task is None:
                raise ValueError(
                    "Cannot determine task type: self.task is None. "
                    "Pass task='classification' or task='regression' at construction."
                )

            if self.task == "classification":
                # accuracy_score at module top-level has a numpy fallback for sklearn-free environments.
                return {"accuracy": float(accuracy_score(true_labels, self.predictions))}

            preds = np.asarray(self.predictions, dtype=float)
            labels = np.asarray(true_labels, dtype=float)
            return _regression_metrics(preds, labels)

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
            cls,
            path: str,
            model: Optional["nn.Module"] = None,
            task: str = "classification",
            weights_only: bool = True,
        ) -> "PyTorchModelWrapper":
            """Load model weights from a state-dict checkpoint.

            .. note::
                PyTorch state dicts store **weights only**, not the model
                architecture. The architecture must be supplied via ``model``
                and must match the one used when :meth:`save_model` was called.

            Args:
                path: Path to the ``.pt`` checkpoint written by
                    :meth:`save_model`.
                model: A ``torch.nn.Module`` instance whose architecture
                    matches the saved checkpoint. **Required** — raises
                    ``ValueError`` if omitted.
                task: ``'classification'`` or ``'regression'``. Overridden
                    by the value stored in the checkpoint when present.
                weights_only: If ``True`` (default), restricts deserialization
                    to tensors and primitives — safe for all checkpoints written
                    by :meth:`save_model`. Set to ``False`` only for legacy
                    checkpoints from untrusted sources. Default ``True``.

            Returns:
                :class:`PyTorchModelWrapper` with weights loaded from *path*.

            Raises:
                ValueError: If *model* is ``None``.

            Example::

                net = MyNet()   # same architecture used at training time
                wrapper = PyTorchModelWrapper.load_model("run1.pt", model=net)
            """
            if model is None:
                raise ValueError(
                    "PyTorchModelWrapper.load_model() requires the 'model' argument: "
                    "a torch.nn.Module instance whose architecture matches the saved "
                    "checkpoint. PyTorch state dicts contain weights only — the "
                    "architecture cannot be inferred from the file.\n\n"
                    "Example:\n"
                    "    net = MyNetwork()  # same architecture used during training\n"
                    "    wrapper = PyTorchModelWrapper.load_model('run1.pt', model=net)"
                )
            checkpoint = torch.load(path, map_location="cpu", weights_only=weights_only)
            model.load_state_dict(checkpoint["model_state_dict"])
            wrapper = cls(model, task=checkpoint.get("task", task))
            return wrapper

else:
    PyTorchModelWrapper = None  # type: ignore[assignment,misc]


if HUGGINGFACE_AVAILABLE:

    class _IctonxMetricsCallback(TrainerCallback):
        """Captures per-epoch eval metrics into a buffer dict."""

        def __init__(self, buffer: dict):
            self._buf = buffer

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                for k, v in metrics.items():
                    clean = k.replace("eval_", "val_").replace("train_", "")
                    if isinstance(v, float):
                        self._buf.setdefault(clean, []).append(v)

    class HuggingFaceModelWrapper(BaseModelWrapper):
        """Wrapper for HuggingFace text-classification models.

        Fine-tunes any ``AutoModelForSequenceClassification`` using the
        ``transformers.Trainer`` API. Captures per-epoch validation metrics
        into :class:`~ictonyx.runners.TrainingResult` in the format expected
        by :class:`~ictonyx.runners.ExperimentRunner`.

        Per-run seeding (v0.4.7+): the runner threads a ``run_seed`` through
        ``fit_kwargs`` generated via ``SeedSequence.spawn()``. Each call to
        ``fit()`` pops ``run_seed`` from ``kwargs`` and applies it via
        ``transformers.set_seed()`` plus ``TrainingArguments(seed=run_seed,
        data_seed=run_seed)`` so that weight initialisation, batch ordering,
        and dropout are all deterministic per run. Pre-v0.4.7, the wrapper
        read from ``self.model_config`` (never populated) and silently fell
        back to 42 on every run, causing variability studies to produce
        identical metrics across all runs.

        .. note::
            ``load_best_model_at_end`` is always ``False``. Setting it to
            ``True`` causes Trainer to silently replace the final model weights
            with an earlier checkpoint, invalidating run-to-run comparison.

        .. note::
            The wrapper does not override ``lr_scheduler_type``, so it inherits
            the HF default of ``'linear'`` — a linear decay from the peak
            learning rate (``learning_rate``) to zero over the course of
            training, with an initial warmup controlled by ``warmup_ratio``.
            This is the standard behavior for transformer fine-tuning; to use
            a different schedule (e.g., cosine), subclass
            ``HuggingFaceModelWrapper`` and override the ``fit()`` method
            with a custom ``TrainingArguments`` call.

        Args:
            model_name_or_path: HuggingFace Hub model identifier or local path.
                The model is **not** downloaded until ``fit()`` is called.
            task: Task type. Currently only ``'text-classification'`` is
                supported.
            num_labels: Number of output classes. Inferred from dataset
                labels during ``fit()`` when ``None``.
            tokenizer_name_or_path: Tokenizer identifier. Defaults to the
                same value as ``model_name_or_path``.
            max_length: Maximum token sequence length. Default 128.
            device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'cuda:0'``.
                ``'auto'`` selects CUDA > MPS > CPU.
            model_id: Optional label for logging.
            **model_kwargs: Forwarded to ``from_pretrained()``.

        Example::

            # model_kwargs is forwarded to the wrapper's __init__; additional
            # kwargs in model_kwargs beyond the wrapper's init signature
            # are forwarded to from_pretrained() via **model_kwargs.
            results = ix.variability_study(
                model=ix.HuggingFaceModelWrapper,
                model_kwargs={
                    "model_name_or_path": "distilbert-base-uncased",
                    "num_labels": 4,
                },
                data=(texts_train, labels_train),
                validation_data=(texts_val, labels_val),
                runs=20,
                epochs=3,
                batch_size=16,
                learning_rate=2e-5,
                seed=42,
                verbose=False,  # disables Trainer mid-epoch log chatter
            )
        """

        def __init__(
            self,
            model_name_or_path: str,
            task: str = "text-classification",
            num_labels: Optional[int] = None,
            tokenizer_name_or_path: Optional[str] = None,
            max_length: int = 128,
            device: str = "auto",
            model_id: str = "",
            **model_kwargs,
        ):
            super().__init__(model=None, model_id=model_id)
            self.model_name_or_path = model_name_or_path
            self.task = task
            self.num_labels = num_labels
            self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
            self.max_length = max_length
            self._model_kwargs = model_kwargs
            self.tokenizer: Optional[Any] = None
            self._tmp_dir: Optional[str] = None

            if device == "auto":
                import torch as _torch

                if _torch.cuda.is_available():
                    self._device_str = "cuda"
                elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    self._device_str = "mps"
                else:
                    self._device_str = "cpu"
            else:
                self._device_str = device

        def _tokenize(self, texts: list, labels=None):
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized. Call fit() first.")
            from datasets import Dataset as _HFDataset

            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            data = dict(encoding)
            if labels is not None:
                data["labels"] = list(labels)
            return _HFDataset.from_dict(data)

        def _resolve_data(self, data):
            try:
                from datasets import Dataset as _HFDataset

                if isinstance(data, _HFDataset):
                    return data
            except ImportError:
                pass
            if isinstance(data, tuple) and len(data) == 2:
                texts, labels = data
                return self._tokenize(list(texts), list(labels))
            raise TypeError(
                "HuggingFaceModelWrapper expects data as "
                "(List[str], List[int]) or a datasets.Dataset."
            )

        def _compute_metrics(self, eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float(np.mean(preds == labels))}

        def fit(
            self,
            train_data,
            validation_data=None,
            epochs: int = 3,
            batch_size: int = 16,
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01,
            warmup_ratio: float = 0.1,
            gradient_accumulation_steps: int = 1,
            fp16: bool = False,
            logging_steps: int = 50,
            **kwargs,
        ) -> None:
            """Fine-tune the model for text classification.

            Args:
                train_data: ``(List[str], List[int])`` or
                    ``datasets.Dataset`` with a ``labels`` column.
                validation_data: Same format. Required for val metrics.
                epochs: Training epochs. Default 3.
                batch_size: Per-device batch size. Default 16.
                learning_rate: AdamW learning rate. Default 2e-5.
                weight_decay: AdamW weight decay. Default 0.01.
                warmup_ratio: Fraction of total steps for LR warmup.
                    Default 0.1.
                gradient_accumulation_steps: Default 1.
                fp16: Mixed-precision training (requires GPU). Default
                    ``False``.
                logging_steps: Trainer log frequency. Default 50.
            """
            import gc
            import tempfile

            # ── Seed control ─────────────────────────────────────────────
            # Per-run seed is threaded via fit_kwargs by the runner (X-40 fix,
            # v0.4.7). The prior implementation read from self.model_config,
            # which is never populated on the wrapper — so every run used the
            # fallback 42, producing bit-for-bit identical metrics across all
            # runs in a variability study.
            run_seed = kwargs.pop("run_seed", None)
            if run_seed is None:
                run_seed = 42
                warnings.warn(
                    "HuggingFaceModelWrapper.fit() received no run_seed; "
                    "using default 42. All runs will produce identical results. "
                    "This indicates a runner-wrapper integration bug.",
                    UserWarning,
                    stacklevel=2,
                )
            _hf_set_seed(run_seed)

            # ── Verbose control ──────────────────────────────────────────
            # Threaded via fit_kwargs by the runner (same pattern as run_seed).
            # Translates to disable_tqdm in TrainingArguments + transformers
            # global logging level. Prior to v0.4.7, verbose=False in
            # variability_study produced 50+ lines of stdout per run.
            verbose_flag = bool(kwargs.pop("verbose", True))
            if not verbose_flag:
                # Quiet the transformers library root logger for this run.
                try:
                    from transformers import logging as _hf_logging

                    _hf_logging.set_verbosity_error()
                except ImportError:
                    pass

            # ── Tokenizer + data ─────────────────────────────────────────
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            train_dataset = self._resolve_data(train_data)
            eval_dataset = (
                self._resolve_data(validation_data) if validation_data is not None else None
            )

            # ── Infer num_labels ─────────────────────────────────────────
            num_labels = self.num_labels
            if num_labels is None:
                if "labels" in train_dataset.features:
                    num_labels = int(np.max(train_dataset["labels"])) + 1
                else:
                    raise ValueError(
                        "num_labels must be specified or inferable from a "
                        "'labels' column in the dataset."
                    )

            # ── Load model ───────────────────────────────────────────────
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=num_labels,
                **self._model_kwargs,
            )

            # ── Temp dir ─────────────────────────────────────────────────
            # Deleted in _cleanup_implementation().
            # bert-base-uncased is ~440 MB; 20 runs without cleanup = 8.8 GB.
            self._tmp_dir = tempfile.mkdtemp(prefix="ictonyx_hf_")

            # ── TrainingArguments ─────────────────────────────────────────
            training_args = TrainingArguments(
                output_dir=self._tmp_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=fp16,
                logging_steps=logging_steps,
                disable_tqdm=not verbose_flag,
                log_level="error" if not verbose_flag else "passive",
                eval_strategy="epoch" if eval_dataset is not None else "no",
                save_strategy="no",
                load_best_model_at_end=False,  # MUST be False — see class docstring
                seed=run_seed,
                data_seed=run_seed,
                report_to="none",
                use_cpu=(self._device_str == "cpu"),
            )

            # ── Callback + Trainer ────────────────────────────────────────
            history_buffer: dict = {}
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics if eval_dataset else None,
                callbacks=[_IctonxMetricsCallback(history_buffer)],
            )
            trainer.train()

            # Capture final train loss from Trainer log history
            for entry in reversed(trainer.state.log_history):
                if "train_loss" in entry:
                    history_buffer.setdefault("loss", []).append(entry["train_loss"])
                    break
                elif "loss" in entry and "eval_loss" not in entry:
                    history_buffer.setdefault("loss", []).append(entry["loss"])
                    break

                # Align history lengths — val metrics have one value per epoch
                # but train loss is captured as a single final value.
                # Mismatched lengths cause DataFrame construction to fail silently.
            n_epochs = max((len(v) for v in history_buffer.values()), default=0)
            if "loss" in history_buffer and len(history_buffer["loss"]) == 1:
                history_buffer["loss"] = history_buffer["loss"] * n_epochs

            self.training_result = TrainingResult(
                history=history_buffer,
                params={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": run_seed,
                    "model": self.model_name_or_path,
                },
            )

        def predict(self, data, **kwargs) -> np.ndarray:
            """Generate class predictions.

            Args:
                data: ``(texts, labels)`` tuple, ``List[str]``, or
                    ``datasets.Dataset``. Labels are ignored.

            Returns:
                Integer class indices, shape ``(n_samples,)``.

            Raises:
                RuntimeError: If called before ``fit()``.
            """
            import torch as _torch

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("HuggingFaceModelWrapper.predict() called before fit().")
            if isinstance(data, tuple):
                texts = list(data[0])
            elif isinstance(data, list):
                texts = data
            else:
                texts = list(data)

            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self._device_str) for k, v in encoding.items()}
            self.model.to(self._device_str)
            self.model.eval()
            with _torch.no_grad():
                outputs = self.model(**encoding)
            self.predictions = np.argmax(outputs.logits.cpu().numpy(), axis=-1).astype(int)
            return self.predictions  # type: ignore[return-value]

        def predict_proba(self, data, **kwargs) -> np.ndarray:
            """Generate class probabilities via softmax."""
            import torch as _torch

            if isinstance(data, tuple):
                texts = list(data[0])
            elif isinstance(data, list):
                texts = data
            else:
                texts = list(data)

            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized. Call fit() first.")

            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self._device_str) for k, v in encoding.items()}
            self.model.to(self._device_str)
            self.model.eval()
            with _torch.no_grad():
                outputs = self.model(**encoding)
            return _torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        def evaluate(self, data, **kwargs) -> dict:
            """Evaluate on labeled data.

            Args:
                data: ``(texts, labels)`` tuple.

            Returns:
                Dict with ``'accuracy'`` key.
            """
            if not (isinstance(data, tuple) and len(data) == 2):
                raise TypeError("HuggingFaceModelWrapper.evaluate() expects (texts, labels).")
            texts, labels = data
            preds = self.predict(list(texts))
            return {"accuracy": float(accuracy_score(np.array(list(labels)), preds))}

        def assess(self, true_labels: np.ndarray) -> dict:
            """Assess stored predictions against ground-truth labels."""
            if self.predictions is None:
                raise ValueError("Call predict() before assess().")
            return {"accuracy": float(accuracy_score(true_labels, self.predictions))}

        def save_model(self, path: str) -> None:
            """Save fine-tuned model and tokenizer in HuggingFace format."""
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
            logger.info(f"HuggingFace model saved to {path}")

        @classmethod
        def load_model(
            cls,
            path: str,
            task: str = "text-classification",
            **kwargs,
        ) -> "HuggingFaceModelWrapper":
            """Load a saved fine-tuned model."""
            wrapper = cls(model_name_or_path=path, task=task, **kwargs)
            wrapper.tokenizer = AutoTokenizer.from_pretrained(path)
            wrapper.model = AutoModelForSequenceClassification.from_pretrained(path)
            return wrapper

        def _cleanup_implementation(self) -> None:
            """Release GPU memory and delete temporary checkpoint directory."""
            import gc
            import shutil

            import torch as _torch

            if self._tmp_dir and os.path.exists(self._tmp_dir):
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
                self._tmp_dir = None
            if self.model is not None:
                try:
                    self.model.cpu()
                    del self.model
                    self.model = None
                except Exception:
                    pass
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            gc.collect()

else:
    HuggingFaceModelWrapper = None  # type: ignore[assignment,misc]
    HUGGINGFACE_AVAILABLE = False
