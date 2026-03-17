"""Tests for HyperparameterTuner."""

import numpy as np
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import BaseModelWrapper, TrainingResult

try:
    from ictonyx.tuning import HyperparameterTuner, create_search_space

    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

pytest.importorskip("shap", reason="shap not installed")

pytestmark = pytest.mark.skipif(not HAS_HYPEROPT, reason="hyperopt not installed")


class SimpleRegressionWrapper(BaseModelWrapper):
    def __init__(self, config):
        super().__init__(None, "test")
        self.config = config

    def fit(self, train_data, validation_data=None, **kwargs):
        lr = self.config.get("learning_rate", 0.01)
        r2 = min(0.9, lr * 100)
        self.training_result = TrainingResult(history={"r2": [r2], "val_r2": [r2 - 0.05]})

    def predict(self, data, **kwargs):
        return np.zeros(len(data))

    def predict_proba(self, data, **kwargs):
        raise NotImplementedError

    def evaluate(self, data, **kwargs):
        return {"r2": 0.8}

    def assess(self, true_labels):
        return {"r2": 0.8}

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        return cls(ModelConfig())

    def _cleanup_implementation(self):
        pass


class SimpleClassificationWrapper(BaseModelWrapper):
    def __init__(self, config):
        super().__init__(None, "test_clf")
        self.config = config

    def fit(self, train_data, validation_data=None, **kwargs):
        acc = min(0.99, self.config.get("learning_rate", 0.1) * 5)
        self.training_result = TrainingResult(
            history={"accuracy": [acc], "val_accuracy": [acc - 0.03]}
        )

    def predict(self, data, **kwargs):
        return np.zeros(len(data))

    def predict_proba(self, data, **kwargs):
        return np.ones((len(data), 2)) * 0.5

    def evaluate(self, data, **kwargs):
        return {"accuracy": 0.9}

    def assess(self, true_labels):
        return {"accuracy": 0.9}

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        return cls(ModelConfig())

    def _cleanup_implementation(self):
        pass


@pytest.fixture
def regression_handler():
    from ictonyx.data import ArraysDataHandler

    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    return ArraysDataHandler(X, y)


def test_tuner_init(regression_handler):
    config = ModelConfig({"learning_rate": 0.01, "epochs": 2})
    tuner = HyperparameterTuner(
        model_builder=lambda c: SimpleRegressionWrapper(c),
        data_handler=regression_handler,
        model_config=config,
        metric="val_r2",
    )
    assert tuner.metric == "val_r2"


def test_tuner_rejects_empty_param_space(regression_handler):
    config = ModelConfig({"learning_rate": 0.01})
    tuner = HyperparameterTuner(lambda c: SimpleRegressionWrapper(c), regression_handler, config)
    with pytest.raises(ValueError, match="non-empty"):
        tuner.tune({}, max_evals=2)


def test_tuner_rejects_invalid_max_evals(regression_handler):
    from hyperopt import hp

    config = ModelConfig({"learning_rate": 0.01})
    tuner = HyperparameterTuner(lambda c: SimpleRegressionWrapper(c), regression_handler, config)
    with pytest.raises(ValueError, match="max_evals"):
        tuner.tune({"learning_rate": hp.uniform("lr", 0.001, 0.1)}, max_evals=0)


def test_r2_is_negated_for_minimization(regression_handler):
    """REGRESSION TEST for bug fix: r2 must be negated so tuner maximises it."""
    from hyperopt import hp

    config = ModelConfig({"learning_rate": 0.01, "epochs": 1})
    tuner = HyperparameterTuner(
        lambda c: SimpleRegressionWrapper(c), regression_handler, config, metric="val_r2"
    )
    result = tuner.tune({"learning_rate": hp.uniform("lr", 0.001, 0.5)}, max_evals=5)
    # best val_r2 should be positive — tuner found a good lr, not the worst one
    assert result["best_metric_value"] > 0


def test_accuracy_best_value_positive(regression_handler):
    """Accuracy is maximize-better; returned best_metric_value must be positive."""
    from hyperopt import hp

    from ictonyx.data import ArraysDataHandler

    X = np.random.rand(60, 3)
    y = np.random.randint(0, 2, 60).astype(float)
    handler = ArraysDataHandler(X, y)
    config = ModelConfig({"learning_rate": 0.1, "epochs": 1})
    tuner = HyperparameterTuner(
        lambda c: SimpleClassificationWrapper(c), handler, config, metric="val_accuracy"
    )
    result = tuner.tune({"learning_rate": hp.uniform("lr", 0.01, 0.2)}, max_evals=3)
    assert result["best_metric_value"] > 0
