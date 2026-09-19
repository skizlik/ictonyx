"""v0.4.10 C10: the tuner runs under the fit() contract, seeded and reproducible.

Closes v12 0.8, 1.31, 2.31, 2.36, 2.37, 2.44.
"""

import inspect

import numpy as np
import pytest
from _spy_wrappers import RecordingSpy
from sklearn.ensemble import RandomForestClassifier

import ictonyx.tuning
from ictonyx.config import ModelConfig
from ictonyx.core import ScikitLearnModelWrapper
from ictonyx.data import ArraysDataHandler
from ictonyx.exceptions import ConfigurationError

optuna = pytest.importorskip("optuna")
from ictonyx.tuning import HyperparameterTuner  # noqa: E402


class _SpyRF(ScikitLearnModelWrapper):
    """Records what fit() receives and what the builder received; counts cleanup() calls."""

    seen: list = []
    seen_seeds: list = []
    cleanups: int = 0

    def fit(self, train_data, validation_data=None, **kw):
        _SpyRF.seen.append((kw.get("run_seed"), kw.get("batch_size")))
        return super().fit(train_data, validation_data)

    def cleanup(self):
        _SpyRF.cleanups += 1
        super().cleanup()


def _spy_builder(cfg):
    _SpyRF.seen_seeds.append(cfg.get("run_seed"))
    return _SpyRF(RandomForestClassifier(n_estimators=5, random_state=cfg.get("run_seed")))


SPACE = {
    "max_depth": optuna.distributions.IntDistribution(1, 5),
    "batch_size": optuna.distributions.IntDistribution(8, 64),
}


def _tune(X, y, seed, metric="val_accuracy", **init):
    _SpyRF.seen.clear()
    _SpyRF.seen_seeds.clear()
    _SpyRF.cleanups = 0
    t = HyperparameterTuner(
        _spy_builder,
        ArraysDataHandler(X, y),
        ModelConfig({"epochs": 1}),
        metric=metric,
        n_evals_per_trial=3,
        seed=seed,
        **init,
    )
    t.tune(SPACE, max_evals=3, direction="maximize")
    return t


def test_tuner_seeds_reach_the_builder(X, y):
    """sklearn wrappers are seeded via the constructor, not fit() (build_fit_kwargs rule)."""
    _tune(X, y, seed=1)
    assert len(_SpyRF.seen_seeds) == 9
    assert None not in _SpyRF.seen_seeds
    assert len(set(_SpyRF.seen_seeds)) == 9
    # fit() correctly received NO run_seed / batch_size for an sklearn wrapper
    assert all(s is None and b is None for s, b in _SpyRF.seen)


def test_tuner_batch_size_reaches_non_sklearn_fit(X, y):
    """sklearn wrappers don't take batch_size (contract); a generic wrapper does."""
    seen = []

    class _Spy(RecordingSpy):
        def fit(self, train_data, validation_data=None, **kw):
            seen.append(kw.get("batch_size"))
            return super().fit(train_data, validation_data, **kw)

        def evaluate(self, data, **kw):  # the tuner calls evaluate(data=...) by keyword
            return {"accuracy": 0.5}

    t = HyperparameterTuner(
        lambda cfg: _Spy(None),
        ArraysDataHandler(X, y),
        ModelConfig({}),
        n_evals_per_trial=1,
        seed=0,
    )
    t.tune(
        {"batch_size": optuna.distributions.IntDistribution(8, 64)},
        max_evals=2,
        direction="maximize",
    )
    assert all(b is not None and 8 <= b <= 64 for b in seen)


def test_tuner_reproducible(X, y):
    a = _tune(X, y, seed=7)
    seeds_a = list(_SpyRF.seen_seeds)
    b = _tune(X, y, seed=7)
    assert seeds_a == _SpyRF.seen_seeds
    assert (
        a.get_trials_dataframe()["params_max_depth"].tolist()
        == b.get_trials_dataframe()["params_max_depth"].tolist()
    )


def test_tuner_bad_metric_fails_on_first_trial(X, y):
    t = HyperparameterTuner(
        _spy_builder,
        ArraysDataHandler(X, y),
        ModelConfig({}),
        metric="val_loss",
        n_evals_per_trial=1,
        seed=0,
    )
    with pytest.raises(ConfigurationError, match="could not resolve metric"):
        t.tune({"max_depth": optuna.distributions.IntDistribution(1, 5)}, max_evals=5)
    # It raised on trial 1, not after 5 pruned trials.
    assert t._optuna_study is None


def test_tuner_default_metric_resolves_for_sklearn(X, y):
    t = HyperparameterTuner(
        _spy_builder, ArraysDataHandler(X, y), ModelConfig({}), n_evals_per_trial=1, seed=0
    )
    t.tune(
        {"max_depth": optuna.distributions.IntDistribution(1, 5)}, max_evals=2, direction="maximize"
    )
    assert t.metric == "val_accuracy"


def test_tuner_requires_direction_when_metric_unknown(X, y):
    t = HyperparameterTuner(_spy_builder, ArraysDataHandler(X, y), ModelConfig({}), seed=0)
    with pytest.raises(ValueError, match="direction"):
        t.tune({"max_depth": optuna.distributions.IntDistribution(1, 5)}, max_evals=1)


def test_tuner_calls_cleanup_every_evaluation(X, y):
    _tune(X, y, seed=1)
    assert _SpyRF.cleanups >= 9  # 9 explicit from the tuner's finally; __del__ may add more


def test_tuner_seeds_through_set_run_seeds():
    src = inspect.getsource(ictonyx.tuning)
    assert "set_run_seeds(" in src
    assert "simplefilter" not in src
    assert "trial._suggest" not in src
