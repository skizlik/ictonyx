"""Test PyTorch model wrapper functionality."""
import pytest
import numpy as np
import pandas as pd

# Skip entire module if PyTorch not installed
torch = pytest.importorskip("torch")
nn = torch.nn

from ictonyx.core import PyTorchModelWrapper, TrainingResult, PYTORCH_AVAILABLE
from ictonyx.config import ModelConfig


# --- Test fixtures ---

def make_classification_data(n_samples=200, n_features=10, n_classes=3):
    """Generate random classification data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n_samples).astype(np.int64)
    split = int(n_samples * 0.8)
    return (X[:split], y[:split]), (X[split:], y[split:])


def make_regression_data(n_samples=200, n_features=5):
    """Generate random regression data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features).astype(np.float32)
    y = (X @ w + rng.randn(n_samples).astype(np.float32) * 0.1)
    split = int(n_samples * 0.8)
    return (X[:split], y[:split]), (X[split:], y[split:])


def make_classifier(n_features=10, n_classes=3):
    """Create a simple classification network."""
    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes)
    )


def make_regressor(n_features=5):
    """Create a simple regression network."""
    return nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )


# --- Wrapper creation ---

class TestPyTorchWrapperCreation:
    """Test PyTorchModelWrapper initialization."""

    def test_basic_creation(self):
        model = make_classifier()
        wrapper = PyTorchModelWrapper(model, model_id="test")
        assert wrapper.model is model
        assert wrapper.model_id == "test"
        assert wrapper.task == 'classification'
        assert wrapper.training_result is None

    def test_default_optimizer(self):
        wrapper = PyTorchModelWrapper(make_classifier())
        assert wrapper.optimizer_class is torch.optim.Adam
        assert wrapper.optimizer_params == {'lr': 0.001}

    def test_custom_optimizer(self):
        wrapper = PyTorchModelWrapper(
            make_classifier(),
            optimizer_class=torch.optim.SGD,
            optimizer_params={'lr': 0.01, 'momentum': 0.9}
        )
        assert wrapper.optimizer_class is torch.optim.SGD
        assert wrapper.optimizer_params['momentum'] == 0.9

    def test_device_auto(self):
        wrapper = PyTorchModelWrapper(make_classifier(), device='auto')
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert str(wrapper.device) == expected

    def test_device_explicit_cpu(self):
        wrapper = PyTorchModelWrapper(make_classifier(), device='cpu')
        assert str(wrapper.device) == 'cpu'

    def test_regression_task(self):
        wrapper = PyTorchModelWrapper(
            make_regressor(),
            criterion=nn.MSELoss(),
            task='regression'
        )
        assert wrapper.task == 'regression'


# --- Training (classification) ---

class TestPyTorchClassification:
    """Test classification training, prediction, and evaluation."""

    def test_fit_creates_training_result(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(
            make_classifier(),
            criterion=nn.CrossEntropyLoss()
        )
        wrapper.fit(train, validation_data=val, epochs=3, batch_size=32)

        assert wrapper.training_result is not None
        assert isinstance(wrapper.training_result, TrainingResult)

    def test_history_keys_classification(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, validation_data=val, epochs=5, batch_size=32)

        h = wrapper.training_result.history
        assert 'loss' in h
        assert 'accuracy' in h
        assert 'val_loss' in h
        assert 'val_accuracy' in h
        assert len(h['loss']) == 5
        assert len(h['val_accuracy']) == 5

    def test_history_without_validation(self):
        train, _ = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        h = wrapper.training_result.history
        assert 'loss' in h
        assert 'accuracy' in h
        assert 'val_loss' not in h
        assert 'val_accuracy' not in h

    def test_training_improves_loss(self):
        train, val = make_classification_data(n_samples=500)
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, validation_data=val, epochs=20, batch_size=32)

        losses = wrapper.training_result.history['loss']
        # First epoch loss should be higher than last
        assert losses[0] > losses[-1]

    def test_predict_returns_class_indices(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        X_val, y_val = val
        preds = wrapper.predict(X_val)

        assert preds.shape == y_val.shape
        assert preds.dtype in (np.int64, np.int32)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_predict_stores_predictions(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        preds = wrapper.predict(val[0])
        assert wrapper.predictions is not None
        assert np.array_equal(preds, wrapper.predictions)

    def test_predict_proba_shape(self):
        train, val = make_classification_data(n_classes=3)
        wrapper = PyTorchModelWrapper(make_classifier(n_classes=3), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        proba = wrapper.predict_proba(val[0])
        assert proba.shape == (len(val[0]), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        metrics = wrapper.evaluate(val)
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_assess_classification(self):
        train, val = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=3)

        wrapper.predict(val[0])
        result = wrapper.assess(val[1])
        assert 'accuracy' in result

    def test_assess_requires_predict_first(self):
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        with pytest.raises(ValueError, match="predict"):
            wrapper.assess(np.array([0, 1, 2]))


# --- Training (regression) ---

class TestPyTorchRegression:
    """Test regression training, prediction, and evaluation."""

    def test_regression_training(self):
        train, val = make_regression_data()
        wrapper = PyTorchModelWrapper(
            make_regressor(),
            criterion=nn.MSELoss(),
            task='regression'
        )
        wrapper.fit(train, validation_data=val, epochs=5, batch_size=32)

        h = wrapper.training_result.history
        assert 'loss' in h
        assert 'val_loss' in h
        assert 'val_mse' in h

    def test_regression_predict(self):
        train, val = make_regression_data()
        wrapper = PyTorchModelWrapper(
            make_regressor(), criterion=nn.MSELoss(), task='regression'
        )
        wrapper.fit(train, epochs=5)

        preds = wrapper.predict(val[0])
        assert preds.shape == val[1].shape
        assert preds.dtype == np.float32

    def test_regression_evaluate(self):
        train, val = make_regression_data()
        wrapper = PyTorchModelWrapper(
            make_regressor(), criterion=nn.MSELoss(), task='regression'
        )
        wrapper.fit(train, epochs=5)

        metrics = wrapper.evaluate(val)
        assert 'loss' in metrics
        assert 'mse' in metrics

    def test_regression_assess(self):
        train, val = make_regression_data()
        wrapper = PyTorchModelWrapper(
            make_regressor(), criterion=nn.MSELoss(), task='regression'
        )
        wrapper.fit(train, epochs=5)
        wrapper.predict(val[0])

        result = wrapper.assess(val[1])
        assert 'mse' in result

    def test_predict_proba_regression_raises(self):
        wrapper = PyTorchModelWrapper(
            make_regressor(), criterion=nn.MSELoss(), task='regression'
        )
        with pytest.raises(ValueError, match="classification"):
            wrapper.predict_proba(np.random.rand(10, 5).astype(np.float32))


# --- DataLoader input ---

class TestPyTorchDataLoader:
    """Test that DataLoader inputs work directly."""

    def test_fit_with_dataloader(self):
        train_data, _ = make_classification_data()
        X_t = torch.tensor(train_data[0])
        y_t = torch.tensor(train_data[1])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=32, shuffle=True
        )

        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(loader, epochs=3)

        assert wrapper.training_result is not None
        assert len(wrapper.training_result.history['loss']) == 3

    def test_invalid_train_data_raises(self):
        wrapper = PyTorchModelWrapper(make_classifier())
        with pytest.raises(TypeError, match="tuple"):
            wrapper.fit("not valid data", epochs=1)


# --- Cleanup and save/load ---

class TestPyTorchUtilities:
    """Test cleanup, save, load."""

    def test_cleanup(self):
        wrapper = PyTorchModelWrapper(make_classifier())
        wrapper.cleanup()
        # Should not crash

    def test_save_load(self, tmp_path):
        train, _ = make_classification_data()
        wrapper = PyTorchModelWrapper(make_classifier(), criterion=nn.CrossEntropyLoss())
        wrapper.fit(train, epochs=2)

        save_path = str(tmp_path / "model.pt")
        wrapper.save_model(save_path)

        checkpoint = PyTorchModelWrapper.load_model(save_path)
        assert 'model_state_dict' in checkpoint
        assert checkpoint['task'] == 'classification'

    def test_repr(self):
        wrapper = PyTorchModelWrapper(make_classifier(), model_id="my_net")
        r = repr(wrapper)
        assert "my_net" in r


# --- Integration with ExperimentRunner ---

class TestPyTorchRunnerIntegration:
    """Test that PyTorchModelWrapper works with ExperimentRunner."""

    def test_full_variability_study(self):
        from ictonyx.runners import ExperimentRunner, VariabilityStudyResults
        from ictonyx.config import ModelConfig

        train, val = make_classification_data(n_samples=200)

        class MockPyTorchDataHandler:
            @property
            def data_type(self):
                return "arrays"
            @property
            def return_format(self):
                return "split_arrays"
            def load(self, **kwargs):
                return {
                    'train_data': train,
                    'val_data': val,
                    'test_data': None
                }

        def model_builder(config):
            return PyTorchModelWrapper(
                make_classifier(),
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_params={'lr': 0.01},
            )

        runner = ExperimentRunner(
            model_builder=model_builder,
            data_handler=MockPyTorchDataHandler(),
            model_config=ModelConfig({'epochs': 5, 'batch_size': 32}),
            seed=42,
            verbose=False
        )

        results = runner.run_study(num_runs=3, epochs_per_run=5)

        assert isinstance(results, VariabilityStudyResults)
        assert results.n_runs == 3
        assert 'val_accuracy' in results.final_metrics
        assert len(results.final_metrics['val_accuracy']) == 3

    def test_seed_reproducibility(self):
        from ictonyx.runners import ExperimentRunner
        from ictonyx.config import ModelConfig

        train, val = make_classification_data(n_samples=100)

        class DataHandler:
            @property
            def data_type(self):
                return "arrays"
            @property
            def return_format(self):
                return "split_arrays"
            def load(self, **kwargs):
                return {'train_data': train, 'val_data': val, 'test_data': None}

        def model_builder(config):
            return PyTorchModelWrapper(
                make_classifier(),
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_params={'lr': 0.01}
            )

        def run_with_seed(seed):
            runner = ExperimentRunner(
                model_builder=model_builder,
                data_handler=DataHandler(),
                model_config=ModelConfig({'epochs': 3, 'batch_size': 32}),
                seed=seed,
                verbose=False
            )
            return runner.run_study(num_runs=2, epochs_per_run=3)

        r1 = run_with_seed(42)
        r2 = run_with_seed(42)

        # Same seed → same val_accuracy values
        assert r1.final_metrics['val_accuracy'] == r2.final_metrics['val_accuracy']

    def test_convenience_function(self):
        from ictonyx.runners import run_variability_study, VariabilityStudyResults
        from ictonyx.config import ModelConfig

        train, val = make_classification_data(n_samples=100)

        class DataHandler:
            @property
            def data_type(self):
                return "arrays"
            @property
            def return_format(self):
                return "split_arrays"
            def load(self, **kwargs):
                return {'train_data': train, 'val_data': val, 'test_data': None}

        def model_builder(config):
            return PyTorchModelWrapper(
                make_classifier(),
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_params={'lr': 0.01}
            )

        results = run_variability_study(
            model_builder=model_builder,
            data_handler=DataHandler(),
            model_config=ModelConfig({'epochs': 3}),
            num_runs=2,
            seed=99,
            verbose=False
        )

        assert isinstance(results, VariabilityStudyResults)
        assert results.n_runs == 2
        assert results.seed == 99
