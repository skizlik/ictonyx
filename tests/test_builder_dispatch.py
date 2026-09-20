"""v0.4.10 C15: builder dispatch and construction errors (v12 1.32, 2.53)."""

import functools

import pytest
from _spy_wrappers import RecordingSpy, build_recording_spy
from sklearn.linear_model import LogisticRegression

import ictonyx as ix
from ictonyx.api import _get_model_builder
from ictonyx.config import ModelConfig
from ictonyx.core import PYTORCH_AVAILABLE, TENSORFLOW_AVAILABLE
from ictonyx.exceptions import ConfigurationError


def test_wrapper_instance_gets_builder_hint():
    with pytest.raises(ValueError, match="builder function"):
        _get_model_builder(RecordingSpy(None))


def test_partial_builder_still_works(X, y):
    builder = functools.partial(build_recording_spy)
    r = ix.variability_study(builder, data=(X, y), runs=2, seed=0, verbose=False)
    assert r.n_runs == 2


def test_bad_config_key_raises_configuration_error(X, y):
    class Strict:
        """Rejects unknown kwargs and has no random_state -> the TypeError is not a seed problem."""

        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

        def score(self, X, y):
            return 0.0

    class StrictVarKw(Strict):
        def __init__(self, alpha=1.0, **kw):
            if kw:
                raise TypeError(f"unexpected keyword argument(s) {sorted(kw)}")
            super().__init__(alpha)

    with pytest.raises(ConfigurationError, match="Failed to construct StrictVarKw") as exc:
        ix.variability_study(
            StrictVarKw, data=(X, y.astype(float)), runs=2, seed=0, verbose=False, bogus_kw=1
        )
    assert "bogus_kw" in str(exc.value.__cause__)


def test_dropped_config_key_warns_once(X, y):
    with pytest.warns(UserWarning, match="does not accept \\['bogus_kw'\\]") as rec:
        ix.variability_study(
            LogisticRegression, data=(X, y), runs=3, seed=0, verbose=False, bogus_kw=1
        )
    assert sum("does not accept" in str(w.message) for w in rec) == 1


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
def test_nn_module_instance_raises_builder_hint():
    import torch.nn as nn

    with pytest.raises(ValueError, match="builder function|pass a class"):
        _get_model_builder(nn.Sequential(nn.Linear(4, 2)))


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_keras_model_instance_raises_builder_hint():
    import tensorflow as tf

    m = tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    with pytest.raises(ValueError, match="builder function|pass a class"):
        _get_model_builder(m)
