"""Tests for HyperparameterTuner.

v0.4.10: this file was gated on shap (unrelated) and referenced a class that
never existed, so it had never run on any machine (v12 2.73). It now requires
optuna (the only backend; Hyperopt was removed in v0.5.0).
"""

import numpy as np
import pytest

from ictonyx.config import ModelConfig
from ictonyx.core import BaseModelWrapper, TrainingResult
from ictonyx.tuning import HyperparameterTuner, _resolve_direction

pytest.importorskip("optuna", reason="optuna not installed")


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
    import optuna

    config = ModelConfig({"learning_rate": 0.01})
    tuner = HyperparameterTuner(
        lambda c: SimpleRegressionWrapper(c), regression_handler, config, metric="val_r2"
    )
    with pytest.raises(ValueError, match="max_evals"):
        tuner.tune(
            {"learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1)}, max_evals=0
        )


class TestResolveDirection:
    """_resolve_direction() replaces the two former copies of the metric heuristic."""

    @pytest.mark.parametrize("metric", ["val_loss", "train_loss", "loss", "mse", "mae"])
    def test_minimize_metrics(self, metric):
        assert _resolve_direction("auto", metric) == "minimize"

    @pytest.mark.parametrize(
        "metric", ["val_accuracy", "accuracy", "r2", "val_r2", "f1", "val_f1", "auc"]
    )
    def test_maximize_metrics(self, metric):
        assert _resolve_direction("auto", metric) == "maximize"

    def test_explicit_direction_wins(self):
        assert _resolve_direction("minimize", "val_accuracy") == "minimize"

    def test_auto_without_metric_raises(self):
        with pytest.raises(ValueError, match="direction"):
            _resolve_direction("auto", None)


class TestTuningImportErrors:
    """ImportError paths do not require hyperopt to be installed."""

    def test_tune_without_optuna_raises_import_error(self):
        """The Hyperopt fallback was removed in v0.5.0 (promise ledger #10)."""
        from unittest.mock import MagicMock, patch

        from ictonyx.data import ArraysDataHandler

        X = np.zeros((20, 2))
        y = np.zeros(20)
        with patch("ictonyx.tuning.HAS_OPTUNA", False):
            tuner = HyperparameterTuner(
                model_builder=MagicMock(),
                data_handler=ArraysDataHandler(X, y),
                model_config=ModelConfig({}),
            )
            with pytest.raises(ImportError, match="Optuna"):
                tuner.tune({"x": 1}, max_evals=1)

    def test_create_search_space_is_a_tombstone(self):
        from ictonyx.exceptions import RemovedAPIError
        from ictonyx.tuning import create_search_space

        with pytest.raises(RemovedAPIError, match="Optuna"):
            create_search_space()


class TestStabilityWeightValidation:
    """stability_weight outside [0, 1] must raise ValueError."""

    def _make_tuner(self, stability_weight):
        from unittest.mock import MagicMock

        return HyperparameterTuner(
            model_builder=lambda cfg: MagicMock(),
            data_handler=MagicMock(),
            model_config=ModelConfig({}),
            stability_weight=stability_weight,
        )

    def test_zero_is_valid(self):
        self._make_tuner(0.0)

    def test_one_is_valid(self):
        self._make_tuner(1.0)

    def test_midpoint_is_valid(self):
        self._make_tuner(0.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="stability_weight"):
            self._make_tuner(-0.1)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="stability_weight"):
            self._make_tuner(1.1)
