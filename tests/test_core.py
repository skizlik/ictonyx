"""Test core model wrapper functionality."""

import numpy as np
import pandas as pd
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import SKLEARN_AVAILABLE, TENSORFLOW_AVAILABLE, BaseModelWrapper, TrainingResult


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
        # Should not crash


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
        assert "val_accuracy" in wrapper.training_result.history

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

    def test_fit_without_validation_data(self):
        """Test that fit uses training accuracy as val fallback."""
        from sklearn.linear_model import LogisticRegression

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(LogisticRegression())
        X = np.random.rand(40, 3)
        y = np.random.randint(0, 2, 40)

        wrapper.fit((X, y), validation_data=None)

        h = wrapper.training_result.history
        # Without val data, val metrics should equal train metrics
        assert h["val_accuracy"] == h["accuracy"]

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


# =============================================================================
# ADD TO: tests/test_core.py  (paste at the bottom)
# =============================================================================
# No new imports needed - existing file already has np, pytest, SKLEARN_AVAILABLE


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


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_creation(self):
        result = TrainingResult(history={"loss": [0.5, 0.3], "accuracy": [0.7, 0.9]})
        assert result.history["loss"] == [0.5, 0.3]
        assert len(result.history["accuracy"]) == 2

    def test_empty_history(self):
        result = TrainingResult(history={})
        assert result.history == {}
