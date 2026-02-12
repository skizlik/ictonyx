"""Test core model wrapper functionality."""
import pytest
import numpy as np
import pandas as pd
from ictonyx.core import BaseModelWrapper, TrainingResult, TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
from ictonyx.config import ModelConfig


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
        if hasattr(self, 'cleaned'):
            self.cleaned += 1
        else:
            self.cleaned = 1
    
    def fit(self, train_data, validation_data=None, **kwargs):
        X, y = train_data
        self.model.fit(X, y)
        self.training_result = TrainingResult(history={
            'loss': [0.5, 0.3, 0.1],
            'accuracy': [0.5, 0.7, 0.9]
        })
    
    def predict(self, data, **kwargs):
        self.predictions = self.model.predict(data)
        return self.predictions
    
    def predict_proba(self, data, **kwargs):
        preds = self.predict(data)
        # Mock binary classification probabilities
        return np.column_stack([1 - preds, preds])
    
    def evaluate(self, data, **kwargs):
        return {'test_loss': 0.15, 'test_accuracy': 0.92}
    
    def assess(self, true_labels):
        if self.predictions is None:
            raise ValueError("Model has not generated predictions yet. Call predict() first.")
        # Mock accuracy
        return {'accuracy': 0.9}
    
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
        wrapper.training_result = TrainingResult(history={'loss': [0.1]})
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
        assert 'loss' in wrapper.training_result.history
        assert len(wrapper.training_result.history['loss']) == 3
    
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
        
        assert 'test_loss' in metrics
        assert 'test_accuracy' in metrics
        assert metrics['test_accuracy'] == 0.92
    
    def test_assess_requires_predictions(self):
        """Test assess requires predictions first."""
        wrapper = TestableWrapper(DummyModel())
        
        with pytest.raises(ValueError, match="Call predict.*first"):
            wrapper.assess(np.array([1, 0, 1]))
        
        # After prediction it should work
        wrapper.predict(np.random.rand(3, 5))
        result = wrapper.assess(np.array([1, 0, 1]))
        assert 'accuracy' in result
    
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
        from ictonyx.core import KerasModelWrapper
        import tensorflow as tf
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
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
        from ictonyx.core import ScikitLearnModelWrapper
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        wrapper = ScikitLearnModelWrapper(model)

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        wrapper.fit((X, y))

        assert wrapper.training_result is not None
        assert isinstance(wrapper.training_result, TrainingResult)
        assert 'accuracy' in wrapper.training_result.history
        assert 'val_accuracy' in wrapper.training_result.history

    def test_sklearn_evaluate_full_metrics(self):
        """Test sklearn evaluate returns multiple metrics."""
        from ictonyx.core import ScikitLearnModelWrapper
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        wrapper = ScikitLearnModelWrapper(model)

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))

        result = wrapper.evaluate((X, y))

        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        for value in result.values():
            assert 0.0 <= value <= 1.0
