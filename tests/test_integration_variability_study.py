"""Integration: variability_study() end-to-end per wrapper family."""

import os
import tempfile

import numpy as np
import pytest

import ictonyx as ix
from ictonyx import VariabilityStudyResults

FEATURES = ix.get_feature_availability()


@pytest.fixture
def cls_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10).astype(np.float32)
    y_clean = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    flip_mask = rng.rand(200) < 0.1
    y = y_clean.copy()
    y[flip_mask] = 1 - y[flip_mask]
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10).astype(np.float32)
    y = (X @ rng.randn(10)).astype(np.float32)
    return X, y


def _assert_study_valid(results, expected_runs=3, metric="val_accuracy", check_variance=True):
    assert results.n_runs == expected_runs, f"Expected {expected_runs} runs, got {results.n_runs}"
    vals = results.get_metric_values(metric)
    assert len(vals) == expected_runs, f"Expected {expected_runs} metric values, got {len(vals)}"
    if check_variance:
        assert (
            len(set(round(v, 8) for v in vals)) > 1
        ), f"All {expected_runs} runs identical — seed not reaching model"
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        results.save(f.name)
        loaded = VariabilityStudyResults.load(f.name)
        assert loaded.n_runs == expected_runs
        assert loaded.get_metric_values(metric) == vals
        os.unlink(f.name)


class TestSklearn:
    def test_classification(self, cls_data):
        from sklearn.ensemble import RandomForestClassifier

        results = ix.variability_study(
            model=RandomForestClassifier,
            data=cls_data,
            runs=3,
            seed=42,
            verbose=False,
            n_estimators=10,
        )
        _assert_study_valid(results)

    def test_regression(self, reg_data):
        from sklearn.linear_model import Ridge

        results = ix.variability_study(
            model=Ridge,
            data=reg_data,
            runs=3,
            seed=42,
            verbose=False,
        )
        _assert_study_valid(results, metric="val_r2", check_variance=False)

    def test_kwargs_class_seed_injection(self, cls_data):
        """BUG-DS2-SEED: class with **kwargs must receive random_state."""

        class KwargsEstimator:
            def __init__(self, **kwargs):
                self._rng = np.random.RandomState(kwargs.get("random_state", 0))

            def fit(self, X, y):
                self.w_ = self._rng.randn(X.shape[1])
                return self

            def predict(self, X):
                return (X @ self.w_ > 0).astype(int)

            def score(self, X, y):
                return float(np.mean(self.predict(X) == y))

        import inspect

        assert "random_state" not in inspect.signature(KwargsEstimator).parameters

        results = ix.variability_study(
            model=KwargsEstimator,
            data=cls_data,
            runs=3,
            seed=42,
            verbose=False,
        )
        available = results.get_available_metrics()
        metric = "val_accuracy" if "val_accuracy" in available else available[0]
        _assert_study_valid(results, metric=metric)


@pytest.mark.skipif(not FEATURES.get("pytorch_support"), reason="PyTorch not available")
class TestPyTorch:
    def test_classification(self, cls_data):
        import torch.nn as nn

        def build(config):
            m = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
            return ix.PyTorchModelWrapper(
                m,
                criterion=nn.CrossEntropyLoss(),
                task="classification",
                optimizer_params={"lr": 0.01},
            )

        results = ix.variability_study(
            model=build,
            data=cls_data,
            runs=3,
            epochs=3,
            seed=42,
            verbose=False,
        )
        _assert_study_valid(results)

    def test_regression(self, reg_data):
        """BUG-048-1: regression val_loss duplication crashed here."""
        import torch.nn as nn

        def build(config):
            m = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
            return ix.PyTorchModelWrapper(
                m,
                criterion=nn.MSELoss(),
                task="regression",
                optimizer_params={"lr": 0.01},
            )

        results = ix.variability_study(
            model=build,
            data=reg_data,
            runs=3,
            epochs=3,
            seed=42,
            verbose=False,
        )
        _assert_study_valid(results, metric="val_loss", check_variance=False)
