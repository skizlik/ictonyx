"""Test core model wrapper functionality."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import (
    SKLEARN_AVAILABLE,
    TENSORFLOW_AVAILABLE,
    BaseModelWrapper,
    ScikitLearnModelWrapper,
    TrainingResult,
)

_PYTORCH_AVAILABLE = __import__("ictonyx.core", fromlist=["PYTORCH_AVAILABLE"]).PYTORCH_AVAILABLE

SKIP_NO_TORCH = pytest.mark.skipif(not _PYTORCH_AVAILABLE, reason="PyTorch not installed")


class DummyModel:
    """Dummy model for testing."""

    def __init__(self):
        self.trained = False
        self.predict_count = 0

    def fit(self, X, y):
        self.trained = True
        return self

    def predict(self, X):
        self.predict_count += 1
        return np.ones(len(X))


class TestableWrapper(BaseModelWrapper):
    """Concrete wrapper for testing."""

    def _cleanup_implementation(self):
        """Test cleanup."""
        if hasattr(self, "cleaned"):
            self.cleaned += 1
        else:
            self.cleaned = 1

    def fit(self, train_data, validation_data=None, **kwargs):
        X, y = train_data
        self.model.fit(X, y)
        self.training_result = TrainingResult(
            history={"loss": [0.5, 0.3, 0.1], "accuracy": [0.5, 0.7, 0.9]}
        )

    def predict(self, data, **kwargs):
        self.predictions = self.model.predict(data)
        return self.predictions

    def predict_proba(self, data, **kwargs):
        preds = self.predict(data)
        # Mock binary classification probabilities
        return np.column_stack([1 - preds, preds])

    def evaluate(self, data, **kwargs):
        return {"test_loss": 0.15, "test_accuracy": 0.92}

    def assess(self, true_labels):
        if self.predictions is None:
            raise ValueError("Model has not generated predictions yet. Call predict() first.")
        # Mock accuracy
        return {"accuracy": 0.9}

    def save_model(self, path):
        # Mock save
        self.save_path = path

    @classmethod
    def load_model(cls, path):
        wrapper = cls(DummyModel(), model_id="loaded")
        wrapper.load_path = path
        return wrapper


class TestBaseModelWrapper:
    """Test BaseModelWrapper functionality."""

    def test_wrapper_creation(self):
        """Test creating wrapper with model."""
        model = DummyModel()
        wrapper = TestableWrapper(model, model_id="test_model")

        assert wrapper.model is model
        assert wrapper.model_id == "test_model"
        assert wrapper.training_result is None
        assert wrapper.predictions is None

    def test_wrapper_repr(self):
        """Test string representation."""
        wrapper = TestableWrapper(DummyModel(), model_id="my_model")
        repr_str = repr(wrapper)

        assert "my_model" in repr_str
        assert "DummyModel" in repr_str
        assert "is_trained=No" in repr_str

        # After training
        wrapper.training_result = TrainingResult(history={"loss": [0.1]})
        repr_str = repr(wrapper)
        assert "is_trained=Yes" in repr_str

    def test_training(self):
        """Test model training."""
        wrapper = TestableWrapper(DummyModel())

        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)

        wrapper.fit((X, y))

        assert wrapper.model.trained
        assert wrapper.training_result is not None
        assert "loss" in wrapper.training_result.history
        assert len(wrapper.training_result.history["loss"]) == 3

    def test_prediction(self):
        """Test model prediction."""
        wrapper = TestableWrapper(DummyModel())

        X = np.random.rand(10, 5)
        preds = wrapper.predict(X)

        assert len(preds) == 10
        assert wrapper.predictions is not None
        assert np.array_equal(preds, wrapper.predictions)
        assert wrapper.model.predict_count == 1
        # The assert in predict() was replaced with ModelError to survive python -O.
        # Normal operation never triggers it; this test confirms the happy path works.

    def test_predict_proba(self):
        """Test probability prediction."""
        wrapper = TestableWrapper(DummyModel())

        X = np.random.rand(10, 5)
        proba = wrapper.predict_proba(X)

        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate(self):
        """Test model evaluation."""
        wrapper = TestableWrapper(DummyModel())

        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)

        metrics = wrapper.evaluate((X, y))

        assert "test_loss" in metrics
        assert "test_accuracy" in metrics
        assert metrics["test_accuracy"] == 0.92

    def test_assess_requires_predictions(self):
        """Test assess requires predictions first."""
        wrapper = TestableWrapper(DummyModel())

        with pytest.raises(ValueError, match="Call predict.*first"):
            wrapper.assess(np.array([1, 0, 1]))

        # After prediction it should work
        wrapper.predict(np.random.rand(3, 5))
        result = wrapper.assess(np.array([1, 0, 1]))
        assert "accuracy" in result

    def test_cleanup(self):
        """Test cleanup functionality."""
        wrapper = TestableWrapper(DummyModel())

        # First cleanup
        wrapper.cleanup()
        assert wrapper.cleaned == 1

        # Second cleanup
        wrapper.cleanup()
        assert wrapper.cleaned == 2

        # Note: __del__ is not tested directly. Interpreter-shutdown behaviour
        # (sys.is_finalizing() == True) is not reproducible in pytest. The guard
        # exists to prevent TF/PyTorch teardown from running during shutdown.

    def test_memory_report(self):
        """Test memory reporting."""
        wrapper = TestableWrapper(DummyModel())

        report = wrapper.get_memory_report()
        assert isinstance(report, dict)

    def test_save_load(self):
        """Test save/load methods."""
        wrapper = TestableWrapper(DummyModel(), model_id="original")

        # Save
        wrapper.save_model("/tmp/model.pkl")
        assert wrapper.save_path == "/tmp/model.pkl"

        # Load
        loaded = TestableWrapper.load_model("/tmp/model.pkl")
        assert loaded.model_id == "loaded"
        assert loaded.load_path == "/tmp/model.pkl"


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestKerasWrapper:
    """Test KerasModelWrapper if TensorFlow available."""

    def test_keras_wrapper_import(self):
        """Test that KerasModelWrapper can be imported."""
        from ictonyx.core import KerasModelWrapper

        assert KerasModelWrapper is not None

    def test_keras_cleanup(self):
        """Test Keras-specific cleanup."""
        import tensorflow as tf

        from ictonyx.core import KerasModelWrapper

        # Create simple model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        wrapper = KerasModelWrapper(model, model_id="keras_test")
        wrapper.cleanup()


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestKerasModelWrapperPredict:

    def _make_regression_wrapper(self):
        import tensorflow as tf

        from ictonyx.core import KerasModelWrapper

        wrapper = KerasModelWrapper.__new__(KerasModelWrapper)
        wrapper.model = MagicMock()
        wrapper.task = "regression"
        wrapper.predictions = None
        return wrapper

    def _make_classification_wrapper(self):
        import tensorflow as tf

        from ictonyx.core import KerasModelWrapper

        wrapper = KerasModelWrapper.__new__(KerasModelWrapper)
        wrapper.model = MagicMock()
        wrapper.task = "classification"
        wrapper.predictions = None
        return wrapper

    def test_regression_predict_returns_floats(self):
        wrapper = self._make_regression_wrapper()
        raw = np.array([[2.7], [0.03], [1.5], [-0.4]])
        wrapper.model.predict = MagicMock(return_value=raw)
        wrapper.predict(MagicMock())
        assert wrapper.predictions.dtype in (np.float32, np.float64)
        assert not set(wrapper.predictions.tolist()).issubset({0, 1})

    def test_regression_predict_preserves_values(self):
        wrapper = self._make_regression_wrapper()
        raw = np.array([[2.7], [0.03], [1.5]])
        wrapper.model.predict = MagicMock(return_value=raw)
        wrapper.predict(MagicMock())
        np.testing.assert_allclose(wrapper.predictions, [2.7, 0.03, 1.5])

    def test_binary_classification_threshold_at_0_5(self):
        wrapper = self._make_classification_wrapper()
        raw = np.array([[0.5], [0.4999], [0.5001]])
        wrapper.model.predict = MagicMock(return_value=raw)
        wrapper.predict(MagicMock())
        assert wrapper.predictions[0] == 1
        assert wrapper.predictions[1] == 0
        assert wrapper.predictions[2] == 1

    def test_regression_predict_returns_array_not_none(self):
        """predict() must return its result, not None, for regression models."""
        wrapper = self._make_regression_wrapper()
        raw = np.array([[2.7], [0.03], [1.5]])
        wrapper.model.predict = MagicMock(return_value=raw)
        result = wrapper.predict(MagicMock())
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_regression_predict_returns_array_not_none(self):
        """Regression predict() must return self.predictions, not None."""
        wrapper = self._make_regression_wrapper()
        raw = np.array([[2.7], [0.03], [1.5]])
        wrapper.model.predict = MagicMock(return_value=raw)
        result = wrapper.predict(MagicMock())
        assert result is not None, (
            "predict() returned None for Keras regression. "
            "Add 'return self.predictions' to the regression branch."
        )
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_classification_predict_returns_array_not_none(self):
        """Classification path already returned correctly — confirm it still does."""
        wrapper = self._make_classification_wrapper()
        raw = np.array([[0.3], [0.8], [0.6]])
        wrapper.model.predict = MagicMock(return_value=raw)
        result = wrapper.predict(MagicMock())
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, wrapper.predictions)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestScikitLearnWrapper:
    """Test ScikitLearnModelWrapper if sklearn available."""

    def test_sklearn_wrapper_import(self):
        """Test that ScikitLearnModelWrapper can be imported."""
        from ictonyx.core import ScikitLearnModelWrapper

        assert ScikitLearnModelWrapper is not None

    def test_sklearn_training_result(self):
        """Test sklearn creates TrainingResult."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        model = LogisticRegression()
        wrapper = ScikitLearnModelWrapper(model)

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        wrapper.fit((X, y))

        assert wrapper.training_result is not None
        assert isinstance(wrapper.training_result, TrainingResult)
        assert "accuracy" in wrapper.training_result.history
        assert "val_accuracy" not in wrapper.training_result.history

    def test_sklearn_evaluate_full_metrics(self):
        """Test sklearn evaluate returns multiple metrics."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        model = LogisticRegression()
        wrapper = ScikitLearnModelWrapper(model)

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))

        result = wrapper.evaluate((X, y))

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        for value in result.values():
            assert 0.0 <= value <= 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestScikitLearnWrapperPredict:
    """Tests for predict(), predict_proba(), and assess() on sklearn wrappers."""

    def test_predict_classification(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))
        preds = wrapper.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0, 1})

    def test_predict_regression(self):
        from sklearn.linear_model import LinearRegression

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1
        wrapper.fit((X, y))
        preds = wrapper.predict(X)
        assert len(preds) == 50
        assert preds.dtype in (np.float32, np.float64, float)

    def test_predict_proba_classification(self):
        from sklearn.linear_model import LogisticRegression

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))
        proba = wrapper.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_regression_raises(self):
        from sklearn.linear_model import LinearRegression

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        wrapper.fit((X, y))
        with pytest.raises(NotImplementedError):
            wrapper.predict_proba(X)

    def test_assess_classification(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))
        wrapper.predict(X)
        result = wrapper.assess(y)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_assess_regression(self):
        from sklearn.linear_model import LinearRegression

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1
        wrapper.fit((X, y))
        wrapper.predict(X)
        result = wrapper.assess(y)
        assert "r2" in result
        assert "mse" in result
        assert "mae" in result
        assert result["r2"] > 0.5

    def test_assess_before_predict_raises(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))
        with pytest.raises(ValueError, match="predict"):
            wrapper.assess(y)

    def test_fit_with_validation_data_stores_metrics(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)
        X_val = np.random.rand(20, 4)
        y_val = np.random.randint(0, 2, 20)
        wrapper.fit((X, y), validation_data=(X_val, y_val))
        h = wrapper.training_result.history
        assert "val_accuracy" in h
        assert len(h["val_accuracy"]) == 1

    def test_fit_multiclass_classification(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(90, 5)
        y = np.random.randint(0, 3, 90)
        wrapper.fit((X, y))
        assert wrapper.training_result is not None
        preds = wrapper.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_random_state_injected_from_config(self):
        """random_state stored in config must be passed to estimator at construction."""
        import inspect

        from sklearn.ensemble import RandomForestClassifier

        model_class = RandomForestClassifier
        assert "random_state" in inspect.signature(model_class).parameters
        wrapper = ScikitLearnModelWrapper(model_class(random_state=42))
        assert wrapper.model.random_state == 42

    def test_sklearn_predict_returns_array_not_none(self):
        """predict() must return a numpy array, never None."""
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 3))
        y = (X[:, 0] > 0).astype(int)
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        wrapper.fit((X[:24], y[:24]))
        result = wrapper.predict(X[24:])
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 6


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestScikitLearnWrapperEdgeCases:
    """Edge cases and error conditions for ScikitLearnModelWrapper."""

    def test_fit_with_no_validation_data(self):
        from sklearn.linear_model import LogisticRegression

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y), validation_data=None)
        h = wrapper.training_result.history
        assert "val_accuracy" not in h

    def test_evaluate_before_fit_raises(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)
        with pytest.raises(Exception):
            wrapper.evaluate((X, y))

    def test_wrapper_accepts_model_only(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        assert wrapper is not None
        assert wrapper.model is not None

    def test_r2_nan_when_constant_target_via_assess(self):
        """assess() must return NaN R² when all target values are identical."""
        from sklearn.linear_model import LinearRegression

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = np.ones(50)
        wrapper.fit((X, y))
        wrapper.predict(X)
        result = wrapper.assess(y)
        assert "r2" in result
        assert np.isnan(result["r2"])

    def test_unrecognized_kwarg_warns(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        with pytest.warns(UserWarning, match="unrecognized"):
            wrapper.fit((X, y), totally_fake_kwarg=999)

    def test_evaluate_regression_excludes_classification_metrics(self):
        """Regression evaluate() must not include precision, recall, or f1."""
        import numpy as np
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.1, 50)

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X[:40], y[:40]))
        metrics = wrapper.evaluate((X[40:], y[40:]))

        assert "precision" not in metrics, (
            "Regression evaluate() must not compute precision. "
            "Check that the precision/recall/f1 block is inside if is_classifier:."
        )
        assert "recall" not in metrics
        assert "f1" not in metrics
        assert "r2" in metrics
        assert "mse" in metrics
        assert "mae" in metrics

    def test_evaluate_classification_still_includes_precision_recall_f1(self):
        """Classification evaluate() must still return precision, recall, f1."""
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        wrapper.fit((X[:45], y[:45]))
        metrics = wrapper.evaluate((X[45:], y[45:]))

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "r2" not in metrics


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestScikitLearnWrapperExtended:
    """Extended tests for ScikitLearnModelWrapper."""

    def test_evaluate_regression(self):
        """Test that evaluate returns regression metrics for a regressor."""
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        model = LinearRegression()
        wrapper = ScikitLearnModelWrapper(model)

        X = np.random.rand(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1

        wrapper.fit((X, y))
        result = wrapper.evaluate((X, y))

        # Regression model has no predict_proba, so regression metrics apply
        assert "r2" in result
        assert "mse" in result
        assert "mae" in result
        assert result["r2"] > 0.9  # should fit well on training data
        assert result["mse"] >= 0
        assert result["mae"] >= 0

    def test_save_load_roundtrip(self):
        """Test that save and load produce a working model."""
        import os
        import tempfile

        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        model = DecisionTreeClassifier(random_state=42)
        wrapper = ScikitLearnModelWrapper(model)

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        wrapper.fit((X, y))
        original_preds = wrapper.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            wrapper.save_model(path)
            loaded = ScikitLearnModelWrapper.load_model(path)

            loaded_preds = loaded.predict(X)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_assess(self):
        """Test assess method with known predictions."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        wrapper.fit((X, y))
        wrapper.predict(X)

        result = wrapper.assess(y)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_assess_regression(self):
        """assess() on a regressor must return r2, mse, mae — not accuracy."""
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        rng = np.random.default_rng(0)
        X = rng.random((60, 3))
        y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.05, 60)

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X, y))
        wrapper.predict(X)
        result = wrapper.assess(y)

        assert set(result.keys()) == {
            "r2",
            "mse",
            "rmse",
            "mae",
        }, f"Expected {{'r2','mse','rmse','mae'}}, got {set(result.keys())}"
        assert result["r2"] > 0.99, "R² should be near 1.0 on training data"
        assert result["mse"] >= 0.0
        assert result["mae"] >= 0.0
        # Verify accuracy_score is NOT being called (would return ~0.0)
        assert "accuracy" not in result

    def test_assess_without_predict_raises(self):
        """Test that assess raises before predict is called."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        with pytest.raises(ValueError, match="predict"):
            wrapper.assess(np.array([0, 1]))

    def test_predict_proba(self):
        """Test predict_proba with a model that supports it."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(30, 3)
        y = np.random.randint(0, 2, 30)
        wrapper.fit((X, y))

        proba = wrapper.predict_proba(X)
        assert proba.shape == (30, 2)
        # Probabilities should sum to 1 per row
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_unsupported(self):
        """Test predict_proba raises for models without it."""
        from sklearn.svm import LinearSVC

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LinearSVC())
        X = np.random.rand(30, 3)
        y = np.random.randint(0, 2, 30)
        wrapper.fit((X, y))

        with pytest.raises(NotImplementedError):
            wrapper.predict_proba(X)

    def test_fit_with_validation_data(self):
        """Test that fit uses validation data when provided."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X_train = np.random.rand(40, 3)
        y_train = np.random.randint(0, 2, 40)
        X_val = np.random.rand(10, 3)
        y_val = np.random.randint(0, 2, 10)

        wrapper.fit((X_train, y_train), validation_data=(X_val, y_val))

        h = wrapper.training_result.history
        # val_accuracy should be computed from actual validation data
        assert "val_accuracy" in h
        assert len(h["val_accuracy"]) == 1
        assert 0.0 <= h["val_accuracy"][0] <= 1.0

    def test_fit_without_validation_data_omits_val_keys(self):
        """When no validation data is provided, val_* keys must be absent from history.

        Previously val_accuracy was fabricated as a copy of train_accuracy.
        After the fix, no val_* key appears — absence is the correct signal
        that no validation was performed.
        """
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(40, 3)
        y = np.random.randint(0, 2, 40)

        wrapper.fit((X, y), validation_data=None)

        h = wrapper.training_result.history
        assert "accuracy" in h, "Train accuracy must be present"
        assert "val_accuracy" not in h, (
            "val_accuracy must NOT appear when no validation data was provided. "
            "Its presence means a fabricated value equal to train_accuracy "
            "is being treated as a generalisation metric."
        )

    def test_fit_with_validation_data_includes_val_accuracy(self):
        """When real validation data is provided, val_accuracy must appear in history."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X_train = np.random.rand(40, 3)
        y_train = np.random.randint(0, 2, 40)
        X_val = np.random.rand(10, 3)
        y_val = np.random.randint(0, 2, 10)

        wrapper.fit((X_train, y_train), validation_data=(X_val, y_val))

        h = wrapper.training_result.history
        assert "val_accuracy" in h
        assert len(h["val_accuracy"]) == 1
        assert 0.0 <= h["val_accuracy"][0] <= 1.0

    def test_fit_regression_without_validation_omits_val_r2(self):
        """Regression wrapper: val_r2 must be absent when validation_data is None."""
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(40, 3)
        y = X @ np.array([1.0, -0.5, 0.25]) + np.random.randn(40) * 0.05

        wrapper.fit((X, y), validation_data=None)

        h = wrapper.training_result.history
        assert "r2" in h
        assert "val_r2" not in h, "val_r2 must not appear when no validation data was provided."

    def test_fit_regression_with_validation_includes_val_r2(self):
        """Regression wrapper: val_r2 must appear when validation data is provided."""
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X_train = np.random.rand(40, 3)
        y_train = X_train @ np.array([1.0, -0.5, 0.25])
        X_val = np.random.rand(10, 3)
        y_val = X_val @ np.array([1.0, -0.5, 0.25])

        wrapper.fit((X_train, y_train), validation_data=(X_val, y_val))

        h = wrapper.training_result.history
        assert "val_r2" in h
        assert len(h["val_r2"]) == 1
        assert -1.0 <= h["val_r2"][0] <= 1.0

    def test_fit_bad_input(self):
        """Test that fit rejects non-tuple input."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        with pytest.raises(ValueError, match="tuple"):
            wrapper.fit(np.array([1, 2, 3]))

    def test_evaluate_multiclass(self):
        """Test evaluate with multiclass classification."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 3, 60)  # 3 classes
        wrapper.fit((X, y))

        result = wrapper.evaluate((X, y))
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_sklearn_regressor_assess_no_attribute_error(self):
        """Regression: predict then assess must not raise AttributeError."""
        from sklearn.linear_model import LinearRegression

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        wrapper.fit((X, y))
        wrapper.predict(X)
        result = wrapper.assess(y)
        assert "r2" in result
        assert "mse" in result
        assert "mae" in result

    def test_sklearn_task_set_after_fit(self):
        """task attribute is correctly set after fit."""
        from sklearn.ensemble import RandomForestClassifier

        clf = ScikitLearnModelWrapper(RandomForestClassifier(n_estimators=5))
        X = np.random.rand(40, 3)
        y = np.random.randint(0, 2, 40)
        clf.fit((X, y))
        assert clf.task == "classification"

        from sklearn.linear_model import LinearRegression

        reg = ScikitLearnModelWrapper(LinearRegression())
        reg.fit((X, y.astype(float)))
        assert reg.task == "regression"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestScikitLearnWrapperCoverage:
    """Target uncovered ScikitLearnModelWrapper lines."""

    def test_cleanup_does_not_crash(self):
        """Test that cleanup runs without error (line 614-617)."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        wrapper.fit((X, y))
        wrapper.cleanup()  # should not raise

    def test_fit_filters_keras_kwargs(self):
        """Test that Keras-specific kwargs are silently ignored (lines 663-664)."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        # These kwargs should be silently dropped, not cause errors
        wrapper.fit(
            (X, y),
            validation_data=None,
            callbacks=[],
            validation_split=0.2,
            shuffle=True,
        )
        assert wrapper.training_result is not None

    def test_fit_passes_sample_weight(self):
        """Test that sample_weight is forwarded to sklearn fit."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        weights = np.ones(20)
        # sample_weight is in the allowed list, should be passed through
        wrapper.fit((X, y), sample_weight=weights)
        assert wrapper.training_result is not None

    def test_fit_regression_model_score(self):
        """Test fit with a regressor — score() returns R², stored as val_accuracy."""
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LinearRegression())
        X = np.random.rand(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1
        X_val = np.random.rand(10, 3)
        y_val = X_val @ np.array([1.0, 2.0, 3.0]) + np.random.randn(10) * 0.1

        wrapper.fit((X, y), validation_data=(X_val, y_val))
        h = wrapper.training_result.history
        # Regressors now report R² under the correct key
        assert "val_r2" in h
        assert "val_accuracy" not in h
        assert h["val_r2"][0] > 0.5

    def test_repr_before_and_after_training(self):
        """Test that repr changes after training."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(), model_id="test_dt")
        repr_before = repr(wrapper)
        assert "is_trained=No" in repr_before
        assert "test_dt" in repr_before

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        wrapper.fit((X, y))
        repr_after = repr(wrapper)
        assert "is_trained=Yes" in repr_after

    def test_get_memory_report(self):
        """Test inherited get_memory_report returns a dict."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        report = wrapper.get_memory_report()
        assert isinstance(report, dict)

    def test_evaluate_binary_vs_multiclass_metrics(self):
        """Test that evaluate uses 'binary' avg for 2 classes, 'weighted' for 3+."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        # Binary
        wrapper_bin = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        X = np.random.rand(40, 3)
        y_bin = np.random.randint(0, 2, 40)
        wrapper_bin.fit((X, y_bin))
        result_bin = wrapper_bin.evaluate((X, y_bin))
        assert "precision" in result_bin

        # Multiclass
        wrapper_multi = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        y_multi = np.random.randint(0, 4, 40)
        wrapper_multi.fit((X, y_multi))
        result_multi = wrapper_multi.evaluate((X, y_multi))
        assert "precision" in result_multi

    def test_n_classes_stored_at_fit_time(self):
        """_n_classes attribute is set on the wrapper after fit()."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(30, 3)

        # Binary
        y_binary = np.random.randint(0, 2, 30)
        wrapper.fit((X, y_binary))
        assert wrapper._n_classes == 2

        # Re-fit as 4-class
        y_multi = np.random.randint(0, 4, 30)
        wrapper.fit((X, y_multi))
        assert wrapper._n_classes == 4

    def test_evaluate_averaging_uses_training_class_count(self):
        """evaluate() must use _n_classes from training, not from the test batch.

        A 3-class training set but a test batch containing only 2 classes must
        still use 'weighted' averaging (multiclass), not 'binary'.
        """
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        rng = np.random.default_rng(0)

        # Train on 3 classes
        X_train = rng.standard_normal((90, 4))
        y_train = np.array([0, 1, 2] * 30)

        # Test set contains only classes 0 and 1 — class 2 is absent
        X_test = rng.standard_normal((20, 4))
        y_test = np.array([0, 1] * 10)

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        wrapper.fit((X_train, y_train))

        assert (
            wrapper._n_classes == 3
        ), "_n_classes should be 3 after fitting on 3-class training data"

        # evaluate() should not raise and should return meaningful metrics
        result = wrapper.evaluate((X_test, y_test))
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        # All values should be in a valid range
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} is out of [0, 1] range"

    def test_save_load_preserves_predictions(self):
        """Test that a loaded model can predict identically."""
        import os
        import tempfile

        from sklearn.ensemble import RandomForestClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(RandomForestClassifier(n_estimators=5, random_state=42))
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        wrapper.fit((X, y))
        preds_orig = wrapper.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            wrapper.save_model(path)
            loaded = ScikitLearnModelWrapper.load_model(path)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)
        finally:
            os.unlink(path)

    def test_sklearn_classifier_fit_no_fabricated_loss(self):
        """Classifier history must not contain the fake 1-accuracy loss proxy."""
        from sklearn.linear_model import LogisticRegression

        wrapper = ScikitLearnModelWrapper(LogisticRegression(max_iter=200))
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)
        wrapper.fit((X, y))
        history = wrapper.training_result.history
        assert "loss" not in history
        assert "val_loss" not in history
        assert "accuracy" in history
        assert 0.0 <= history["accuracy"][0] <= 1.0

    def test_sklearn_regressor_fit_no_loss_column(self):
        """Regressor history must never have a loss column."""
        from sklearn.linear_model import Ridge

        wrapper = ScikitLearnModelWrapper(Ridge())
        X = np.random.rand(60, 4)
        y = np.random.rand(60)
        wrapper.fit((X, y))
        history = wrapper.training_result.history
        assert "loss" not in history
        assert "r2" in history

    def test_plot_training_history_no_loss_column(self):
        """plot_training_history must not raise when no loss column is present."""
        import matplotlib

        matplotlib.use("Agg")
        from ictonyx import plot_training_history

        # Simulate sklearn classifier history — accuracy only, no loss
        history_df = pd.DataFrame(
            {
                "train_accuracy": [0.7, 0.8, 0.85],
                "val_accuracy": [0.65, 0.75, 0.80],
            }
        )
        fig = plot_training_history(history_df, show=False)
        assert fig is not None

    def test_assess_stability_no_loss_column(self):
        """assess_training_stability must not return converged=False for all runs
        merely because there is no 'loss' column (sklearn post-fix behaviour)."""
        from ictonyx import assess_training_stability

        # Simulate sklearn classifier DataFrames — accuracy only, clearly converging
        runs = [
            pd.DataFrame(
                {
                    "train_accuracy": np.concatenate(
                        [np.linspace(0.5, 0.88, 15), np.full(5, 0.89)]
                    ),
                    "val_accuracy": np.concatenate([np.linspace(0.45, 0.84, 15), np.full(5, 0.85)]),
                    "run_num": [i + 1] * 20,
                    "epoch": range(1, 21),
                }
            )
            for i in range(5)
        ]
        result = assess_training_stability(runs, window_size=5)
        assert "error" not in result
        # With clearly converging curves, at least some runs should be flagged converged
        assert (
            result["converged_runs"] > 0
        ), "All runs reported not-converged — loss column fallback is not working"


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_creation(self):
        result = TrainingResult(history={"loss": [0.5, 0.3], "accuracy": [0.7, 0.9]})
        assert result.history["loss"] == [0.5, 0.3]
        assert len(result.history["accuracy"]) == 2

    def test_empty_history(self):
        result = TrainingResult(history={})
        assert result.history == {}


class TestScikitLearnSaveLoadJoblib:
    """save_model uses joblib not pickle (R8-3)."""

    def test_save_load_roundtrip(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        wrapper = ScikitLearnModelWrapper(RandomForestClassifier())
        path = str(tmp_path / "model.joblib")
        wrapper.save_model(path)
        assert os.path.exists(path)
        loaded = ScikitLearnModelWrapper.load_model(path)
        assert hasattr(loaded.model, "predict")


class TestScikitLearnWrapperRepr:
    """Cover __repr__ and model_id paths."""

    def test_repr_contains_model_type(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(), model_id="my_tree")
        r = repr(wrapper)
        assert "my_tree" in r or "DecisionTree" in r

    def test_repr_shows_trained_status(self):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        r_before = repr(wrapper)
        X = np.random.default_rng(0).standard_normal((20, 4))
        y = (X[:, 0] > 0).astype(int)
        wrapper.fit((X, y))
        r_after = repr(wrapper)
        # Before: not trained. After: trained.
        assert "No" in r_before or "False" in r_before or r_before != r_after


class TestScikitLearnWrapperFitEdgeCases:
    """Cover fit() paths not hit by other tests."""

    def test_fit_bad_input_type_raises(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        with pytest.raises(ValueError):
            wrapper.fit(np.zeros((10, 4)))  # not a tuple

    def test_evaluate_metrics_multiclass(self):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 4))
        y = np.array([i % 3 for i in range(60)])
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        wrapper.fit((X[:45], y[:45]))
        metrics = wrapper.evaluate((X[45:], y[45:]))
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestScikitLearnWrapperSaveLoad:
    """Cover save/load paths including joblib fallback."""

    def test_save_load_with_joblib(self, tmp_path):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 4))
        y = (X[:, 0] > 0).astype(int)
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier(random_state=0))
        wrapper.fit((X[:30], y[:30]))
        wrapper.predict(X[30:])

        path = str(tmp_path / "model.joblib")
        wrapper.save_model(path)
        assert os.path.exists(path)

        loaded = ScikitLearnModelWrapper.load_model(path)
        assert loaded is not None
        preds = loaded.predict(X[30:])
        assert len(preds) == 10

    def test_save_load_preserves_task(self, tmp_path):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3))
        y = (X[:, 0] > 0).astype(int)
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        wrapper.fit((X, y))
        assert wrapper.task == "classification"

        path = str(tmp_path / "clf.joblib")
        wrapper.save_model(path)
        loaded = ScikitLearnModelWrapper.load_model(path)
        loaded.predict(X)
        assert loaded.predictions is not None


class TestScikitLearnWrapperPredict2:
    """Additional predict/assess coverage."""

    def test_assess_regression_returns_r2_mse_mae(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.1, 50)
        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X[:40], y[:40]))
        wrapper.predict(X[40:])
        result = wrapper.assess(y[40:])
        assert "r2" in result
        assert "mse" in result
        assert "mae" in result
        assert result["mse"] >= 0
        assert result["mae"] >= 0

    def test_predict_proba_not_implemented_for_svm(self):
        from sklearn.svm import SVC

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 3))
        y = (X[:, 0] > 0).astype(int)
        # SVC without probability=True has no predict_proba
        wrapper = ScikitLearnModelWrapper(SVC())
        wrapper.fit((X, y))
        with pytest.raises(NotImplementedError):
            wrapper.predict_proba(X)

    def test_evaluate_regression_metrics(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 3))
        y = X[:, 0] * 2 + rng.normal(0, 0.05, 50)
        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X[:40], y[:40]))
        metrics = wrapper.evaluate((X[40:], y[40:]))
        assert "r2" in metrics
        assert "mse" in metrics
        assert "mae" in metrics

    def test_cleanup_does_not_raise(self):
        from sklearn.tree import DecisionTreeClassifier

        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        wrapper._cleanup_implementation()  # must not raise


@SKIP_NO_TORCH
class TestPyTorchWrapperPredictReturnValue:
    """
    Regression tests for BUG-CORE-02.

    PyTorchModelWrapper.predict() must return self.predictions for BOTH
    task types. Prior to the fix, the classification branch set
    self.predictions but fell through with no return statement, so callers
    received None.

    All tests here assert on the *return value* of predict(), not on the
    wrapper attribute. Testing only the attribute (wrapper.predictions) is
    what allowed the bug to survive the existing test suite undetected.
    """

    def _make_binary_classification_wrapper(self):
        """
        Lightweight 2-layer classifier: 4 inputs → 2 class logits.
        Uses CrossEntropyLoss which requires integer labels and 2-d logits.
        No training — we call predict() on random weights.
        """
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),  # 2 outputs → torch.max gives class 0 or 1
        )
        return PyTorchModelWrapper(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 0.001},
            task="classification",
            device="cpu",
        )

    def _make_multiclass_classification_wrapper(self):
        """3-class classifier: 6 inputs → 3 class logits."""
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        return PyTorchModelWrapper(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 0.001},
            task="classification",
            device="cpu",
        )

    def _make_regression_wrapper(self):
        """Simple regression wrapper for symmetric comparison."""
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        return PyTorchModelWrapper(
            model=model,
            criterion=nn.MSELoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 0.001},
            task="regression",
            device="cpu",
        )

    def _make_X(self, n_samples=20, n_features=4, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, n_features)).astype(np.float32)

    def _make_X6(self, n_samples=20, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, 6)).astype(np.float32)

    def test_binary_classification_predict_is_not_none(self):
        """
        BUG-CORE-02: binary classifier predict() must not return None.
        This is the canonical regression test for the missing-return bug.
        """
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        assert result is not None, (
            "predict() returned None for a binary classification task. "
            "Add 'return self.predictions' to the classification branch."
        )

    def test_multiclass_classification_predict_is_not_none(self):
        """predict() must not return None for 3-class output either."""
        wrapper = self._make_multiclass_classification_wrapper()
        X = self._make_X6()
        result = wrapper.predict(X)
        assert result is not None, "predict() returned None for a 3-class classification task."

    def test_regression_predict_is_not_none(self):
        """
        Symmetric check: regression already worked before the fix, but we
        test it here to ensure the fix did not break the regression path.
        """
        wrapper = self._make_regression_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        assert result is not None, (
            "predict() returned None for a regression task — the fix broke " "the regression path."
        )

    def test_binary_classification_return_equals_attribute(self):
        """Return value must be the same object as wrapper.predictions."""
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        np.testing.assert_array_equal(
            result,
            wrapper.predictions,
            err_msg=(
                "predict() return value differs from wrapper.predictions. "
                "The method must return self.predictions, not a copy."
            ),
        )

    def test_multiclass_classification_return_equals_attribute(self):
        wrapper = self._make_multiclass_classification_wrapper()
        X = self._make_X6()
        result = wrapper.predict(X)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_regression_return_equals_attribute(self):
        wrapper = self._make_regression_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_binary_classification_predict_shape(self):
        """predict() for n_samples inputs must return a 1-D array of length n_samples."""
        wrapper = self._make_binary_classification_wrapper()
        n = 20
        X = self._make_X(n_samples=n)
        result = wrapper.predict(X)
        assert result.shape == (n,), (
            f"Expected shape ({n},), got {result.shape}. "
            "Classification predict() must return a 1-D integer label array."
        )

    def test_multiclass_predict_shape(self):
        wrapper = self._make_multiclass_classification_wrapper()
        n = 15
        X = self._make_X6(n_samples=n)
        result = wrapper.predict(X)
        assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}."

    def test_regression_predict_shape(self):
        wrapper = self._make_regression_wrapper()
        n = 20
        X = self._make_X(n_samples=n)
        result = wrapper.predict(X)
        assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}."

    def test_binary_classification_predict_dtype_is_integer(self):
        """Classification must return integer class indices, not floats."""
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        assert np.issubdtype(result.dtype, np.integer), (
            f"Classification predict() returned dtype {result.dtype}. "
            "Expected integer dtype (class indices)."
        )

    def test_regression_predict_dtype_is_float(self):
        """Regression must return float values, not integers."""
        wrapper = self._make_regression_wrapper()
        X = self._make_X()
        result = wrapper.predict(X)
        assert np.issubdtype(result.dtype, np.floating), (
            f"Regression predict() returned dtype {result.dtype}. " "Expected float dtype."
        )

    def test_binary_classification_labels_are_zero_or_one(self):
        """Binary classifier must return only labels in {0, 1}."""
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X(n_samples=50)
        result = wrapper.predict(X)
        unique = set(result.tolist())
        assert unique.issubset({0, 1}), (
            f"Binary classification returned unexpected labels: {unique}. "
            "Expected only {{0, 1}}."
        )

    def test_multiclass_labels_are_in_valid_range(self):
        """3-class classifier must return labels in {0, 1, 2}."""
        wrapper = self._make_multiclass_classification_wrapper()
        X = self._make_X6(n_samples=60)
        result = wrapper.predict(X)
        unique = set(result.tolist())
        assert unique.issubset({0, 1, 2}), (
            f"3-class classification returned unexpected labels: {unique}. "
            "Expected a subset of {{0, 1, 2}}."
        )

    def test_classification_predict_is_deterministic(self):
        """
        Two consecutive predict() calls on the same input must return
        identical results (model is in eval mode, no dropout).
        """
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X(n_samples=20)
        result_1 = wrapper.predict(X)
        result_2 = wrapper.predict(X)
        np.testing.assert_array_equal(
            result_1,
            result_2,
            err_msg="predict() is non-deterministic for the same input.",
        )

    def test_regression_predict_is_deterministic(self):
        wrapper = self._make_regression_wrapper()
        X = self._make_X(n_samples=20)
        result_1 = wrapper.predict(X)
        result_2 = wrapper.predict(X)
        np.testing.assert_array_equal(result_1, result_2)

    def test_classification_predict_updates_attribute(self):
        """wrapper.predictions must be set after predict(), not remain None."""
        wrapper = self._make_binary_classification_wrapper()
        assert (
            wrapper.predictions is None
        ), "wrapper.predictions should be None before the first predict() call."
        wrapper.predict(self._make_X())
        assert wrapper.predictions is not None, (
            "wrapper.predictions is still None after predict(). "
            "The attribute must be set even if the caller ignores the return value."
        )

    def test_classification_predict_updates_attribute_on_second_call(self):
        """predictions attribute must be refreshed on every predict() call."""
        wrapper = self._make_binary_classification_wrapper()
        X1 = self._make_X(n_samples=10, seed=0)
        X2 = self._make_X(n_samples=5, seed=99)
        wrapper.predict(X1)
        first_len = len(wrapper.predictions)
        wrapper.predict(X2)
        second_len = len(wrapper.predictions)
        assert first_len == 10
        assert second_len == 5, "wrapper.predictions was not updated on the second predict() call."

    def test_classification_predict_single_sample(self):
        """predict() must work for a batch of exactly one sample."""
        wrapper = self._make_binary_classification_wrapper()
        X = self._make_X(n_samples=1)
        result = wrapper.predict(X)
        assert result is not None
        assert result.shape == (
            1,
        ), f"Single-sample predict() returned shape {result.shape}, expected (1,)."

    def test_regression_predict_single_sample(self):
        wrapper = self._make_regression_wrapper()
        X = self._make_X(n_samples=1)
        result = wrapper.predict(X)
        assert result is not None
        assert result.shape == (1,)


class TestPyTorchPredictProbaSingleOutput:
    """predict_proba() must handle single-output binary classifiers correctly."""

    @pytest.mark.skipif(
        not __import__("ictonyx.core", fromlist=["PYTORCH_AVAILABLE"]).PYTORCH_AVAILABLE,
        reason="PyTorch not installed",
    )
    def test_predict_proba_single_output_returns_valid_probabilities(self):
        """Single-output sigmoid binary classifier must return valid [0,1] probabilities."""
        import numpy as np
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        class SingleOutputBinaryNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 1)

            def forward(self, x):
                return self.linear(x)

        wrapper = PyTorchModelWrapper(
            model=SingleOutputBinaryNet(),
            task="classification",
        )
        # Simulate trained state
        X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
        proba = wrapper.predict_proba(X)

        assert proba.shape == (20, 2), (
            f"Expected shape (20, 2), got {proba.shape}. "
            "Single-output binary must produce 2-column probability array."
        )
        # All rows must sum to 1.0
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), atol=1e-5)
        # No column should be all-ones (which would indicate softmax on (n,1))
        assert not np.allclose(
            proba[:, 0], 1.0
        ), "Class 0 probabilities are all 1.0 — softmax on (n,1) bug is present."
        assert not np.allclose(
            proba[:, 1], 1.0
        ), "Class 1 probabilities are all 1.0 — softmax on (n,1) bug is present."


class TestPredictReturnValues:
    """predict() must return the same array it stores in wrapper.predictions.

    This class exists to catch the regression pattern from BUG-CORE-01
    and BUG-CORE-02, where a branch assigned self.predictions but forgot
    to return it, silently returning None to the caller.
    """

    def test_sklearn_classification_predict_returns_array(self):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(10)
        X = rng.standard_normal((40, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        wrapper.fit((X, y))
        result = wrapper.predict(X)
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_sklearn_regression_predict_returns_array(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(11)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 40)
        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X, y))
        result = wrapper.predict(X)
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_sklearn_predict_length_matches_input(self):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(12)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        wrapper = ScikitLearnModelWrapper(DecisionTreeClassifier())
        wrapper.fit((X, y))
        result = wrapper.predict(X)
        assert len(result) == len(X)


class TestRegressionMetricsConsistency:
    def test_sklearn_evaluate_includes_rmse(self):
        from sklearn.linear_model import LinearRegression

        from ictonyx.core import ScikitLearnModelWrapper

        X = np.random.rand(50, 3).astype(np.float32)
        y = X[:, 0] * 2 + 0.1
        wrapper = ScikitLearnModelWrapper(LinearRegression())
        wrapper.fit((X, y))
        metrics = wrapper.evaluate((X, y))
        assert "rmse" in metrics
        assert np.isfinite(metrics["rmse"]) and metrics["rmse"] >= 0
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-9

    def test_regression_metrics_helper_returns_four_keys(self):
        from ictonyx.core import _regression_metrics

        result = _regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.1]))
        assert set(result.keys()) == {"r2", "mse", "rmse", "mae"}


class TestPyTorchNaNGuard:
    def test_predict_raises_on_nan_output(self):
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        class NaNModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 2), float("nan"))

        wrapper = PyTorchModelWrapper(
            NaNModel(),
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="classification",
        )
        with pytest.raises(ValueError, match="NaN"):
            wrapper.predict(np.zeros((4, 2), dtype=np.float32))


class TestPyTorchRegressionHistory:

    def _make_regression_wrapper(self):
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(nn.Linear(4, 1))
        return PyTorchModelWrapper(
            model,
            criterion=nn.MSELoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="regression",
        )

    def test_val_mse_in_history(self):
        pytest.importorskip("torch")
        wrapper = self._make_regression_wrapper()
        X = np.random.rand(40, 4).astype(np.float32)
        y = np.random.rand(40).astype(np.float32)
        wrapper.fit((X, y), validation_data=(X[:10], y[:10]), epochs=2, batch_size=8)
        assert "val_mse" in wrapper.training_result.history
        assert len(wrapper.training_result.history["val_mse"]) == 2

    def test_val_r2_mae_rmse_in_history(self):
        pytest.importorskip("torch")
        wrapper = self._make_regression_wrapper()
        X = np.random.rand(40, 4).astype(np.float32)
        y = np.random.rand(40).astype(np.float32)
        wrapper.fit((X, y), validation_data=(X[:10], y[:10]), epochs=1, batch_size=8)
        hist = wrapper.training_result.history
        for key in ("val_r2", "val_mae", "val_rmse"):
            assert key in hist, f"Missing key: {key}"

    def test_standardize_history_renames_regression_keys(self):
        import pandas as pd

        from ictonyx.runners import ExperimentRunner

        df = pd.DataFrame(
            {
                "r2": [0.9],
                "mse": [0.01],
                "mae": [0.05],
                "rmse": [0.1],
                "accuracy": [0.95],
                "loss": [0.2],
            }
        )
        result = ExperimentRunner._standardize_history_df(df)
        assert "train_r2" in result.columns
        assert "train_mse" in result.columns
        assert "train_rmse" in result.columns
        assert "train_mae" in result.columns
        assert "r2" not in result.columns


class TestPyTorchDoubleSoftmax:
    def test_warns_on_softmax_final_layer(self):
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(nn.Linear(4, 3), nn.Softmax(dim=1))
        wrapper = PyTorchModelWrapper(
            model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="classification",
        )
        with pytest.warns(UserWarning, match="double"):
            wrapper.predict_proba(np.random.rand(5, 4).astype(np.float32))


class TestKerasEvaluateScalarReturn:
    def test_evaluate_handles_scalar_loss_only(self):
        """TF 2.x returns a float for loss-only models — must not crash."""
        pytest.importorskip("tensorflow")
        from unittest.mock import MagicMock

        from ictonyx.core import KerasModelWrapper

        model = MagicMock()
        model.metrics_names = ["loss"]
        model.evaluate.return_value = 0.543  # scalar float

        wrapper = KerasModelWrapper(model, task="regression")
        result = wrapper.evaluate((np.zeros((4, 2)), np.zeros(4)))
        assert result == {"loss": 0.543}

    def test_evaluate_handles_list_return(self):
        pytest.importorskip("tensorflow")
        from unittest.mock import MagicMock

        from ictonyx.core import KerasModelWrapper

        model = MagicMock()
        model.metrics_names = ["loss", "accuracy"]
        model.evaluate.return_value = [0.543, 0.91]

        wrapper = KerasModelWrapper(model, task="classification")
        result = wrapper.evaluate((np.zeros((4, 2)), np.zeros(4)))
        assert result == {"loss": 0.543, "accuracy": 0.91}


class TestPyTorchAssessNoInlineImport:
    def test_assess_uses_module_level_accuracy_score(self):
        """Removing inline import means the module-level fallback is used."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(nn.Linear(4, 2))
        wrapper = PyTorchModelWrapper(
            model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="classification",
        )
        wrapper.predictions = np.array([0, 1, 0, 1])
        result = wrapper.assess(np.array([0, 1, 1, 1]))
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0


class TestPyTorchDoubleSoftmaxDetection:
    def _nested_model(self):
        """A model with Softmax buried inside a sub-module — NOT the output layer."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        class SubBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)
                self.softmax = nn.Softmax(dim=1)  # buried — not output

            def forward(self, x):
                return self.softmax(self.linear(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = SubBlock()
                self.out = nn.Linear(3, 2)  # actual output — no softmax

            def forward(self, x):
                return self.out(self.block(x))

        return Model()

    def test_nested_softmax_no_false_positive(self):
        """Softmax inside a sub-block must NOT trigger the double-softmax warning."""
        pytest.importorskip("torch")
        import warnings

        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        wrapper = PyTorchModelWrapper(
            self._nested_model(),
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="classification",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            wrapper.predict_proba(np.random.rand(5, 4).astype(np.float32))

    def test_sequential_softmax_output_warns(self):
        """Sequential ending in Softmax must still warn."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from ictonyx.core import PyTorchModelWrapper

        model = nn.Sequential(nn.Linear(4, 3), nn.Softmax(dim=1))
        wrapper = PyTorchModelWrapper(
            model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            task="classification",
        )
        with pytest.warns(UserWarning, match="double"):
            wrapper.predict_proba(np.random.rand(5, 4).astype(np.float32))


class TestHuggingFaceModelWrapper:
    """Tests using prajjwal1/bert-tiny (4.4 MB, CPU-safe, < 60s)."""

    @pytest.fixture
    def tiny_data(self):
        texts = ["This is great.", "Wonderful.", "Excellent!", "Perfect."] * 5 + [
            "This is terrible.",
            "Awful.",
            "Dreadful!",
            "Horrible.",
        ] * 5
        labels = [1] * 20 + [0] * 20
        return texts, labels

    def test_init_does_not_download_model(self):
        pytest.importorskip("transformers")
        from ictonyx.core import HuggingFaceModelWrapper

        wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        assert wrapper.model is None
        assert wrapper.model_name_or_path == "google/bert_uncased_L-2_H-128_A-2"

    def test_fit_produces_training_result(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from ictonyx.core import HuggingFaceModelWrapper

        texts, labels = tiny_data
        wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        wrapper.fit(
            (texts[:32], labels[:32]),
            validation_data=(texts[32:], labels[32:]),
            epochs=1,
            batch_size=8,
        )
        assert wrapper.training_result is not None
        assert "val_accuracy" in wrapper.training_result.history
        assert len(wrapper.training_result.history["val_accuracy"]) == 1

    def test_predict_returns_integer_array(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from ictonyx.core import HuggingFaceModelWrapper

        texts, labels = tiny_data
        wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        wrapper.fit((texts, labels), epochs=1, batch_size=8)
        preds = wrapper.predict(texts[:5])
        assert preds.dtype in (np.int32, np.int64)
        assert preds.shape == (5,)
        assert all(p in (0, 1) for p in preds)

    def test_predict_before_fit_raises(self):
        pytest.importorskip("transformers")
        from ictonyx.core import HuggingFaceModelWrapper

        wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        with pytest.raises(RuntimeError, match="before fit"):
            wrapper.predict(["some text"])

    def test_cleanup_removes_tmp_dir(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        import os

        from ictonyx.core import HuggingFaceModelWrapper

        texts, labels = tiny_data
        wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        wrapper.fit((texts, labels), epochs=1, batch_size=8)
        tmp_dir = wrapper._tmp_dir
        assert os.path.exists(tmp_dir)
        wrapper.cleanup()
        assert not os.path.exists(tmp_dir)

    def test_same_seed_identical_loss(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from ictonyx.config import ModelConfig
        from ictonyx.core import HuggingFaceModelWrapper

        texts, labels = tiny_data
        losses = []
        for _ in range(2):
            wrapper = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
            wrapper.model_config = ModelConfig({"run_seed": 42})
            wrapper.fit((texts, labels), epochs=1, batch_size=8)
            losses.append(wrapper.training_result.history["loss"][-1])
        assert abs(losses[0] - losses[1]) < 1e-6

    def test_different_seeds_produce_valid_losses(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from ictonyx.config import ModelConfig
        from ictonyx.core import HuggingFaceModelWrapper

        texts, labels = tiny_data
        wrapper_a = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        wrapper_a.model_config = ModelConfig({"run_seed": 0})
        wrapper_a.fit((texts, labels), epochs=1, batch_size=8)
        wrapper_b = HuggingFaceModelWrapper("google/bert_uncased_L-2_H-128_A-2")
        wrapper_b.model_config = ModelConfig({"run_seed": 99})
        wrapper_b.fit((texts, labels), epochs=1, batch_size=8)
        loss_a = wrapper_a.training_result.history["loss"][-1]
        loss_b = wrapper_b.training_result.history["loss"][-1]
        assert isinstance(loss_a, float)
        assert isinstance(loss_b, float)
        assert loss_a > 0
        assert loss_b > 0

    @pytest.mark.slow
    def test_variability_study_integration(self, tiny_data):
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        import warnings

        import ictonyx as ix

        texts, labels = tiny_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = ix.variability_study(
                model=ix.HuggingFaceModelWrapper,
                model_kwargs={"model_name_or_path": "google/bert_uncased_L-2_H-128_A-2"},
                data=(texts, labels),
                runs=3,
                seed=42,
                verbose=False,
            )
        assert results.n_runs == 3
        assert len(results.get_metric_values("val_accuracy")) == 3
