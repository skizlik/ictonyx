"""v0.4.10: Keras initial weights follow run_seed (v12 0.7).

Skipped without TensorFlow; exercised on the CI Linux job. Depends on C1's
set_run_seeds() calling keras.utils.set_random_seed before the builder runs.
"""

import numpy as np
import pytest

import ictonyx as ix
from ictonyx.config import ModelConfig
from ictonyx.core import TENSORFLOW_AVAILABLE
from ictonyx.runners import set_run_seeds

pytestmark = pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")


def _keras_builder(cfg):
    import tensorflow as tf

    m = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    m.compile(optimizer="adam", loss="mse")
    return ix.KerasModelWrapper(m, task="regression")


def _weights(cfg):
    return _keras_builder(cfg).model.get_weights()


def test_keras_initial_weights_follow_run_seed():
    """Same seed -> identical initial weights; different seed -> different."""
    cfg = ModelConfig({"run_seed": 123})
    set_run_seeds(123)
    w1 = _weights(cfg)
    set_run_seeds(123)
    w2 = _weights(cfg)
    set_run_seeds(456)
    w3 = _weights(cfg)
    assert all(np.array_equal(p, q) for p, q in zip(w1, w2))
    assert not all(np.array_equal(p, q) for p, q in zip(w1, w3))


def test_keras_same_study_seed_same_history(X, y):
    """Two studies with the same seed must produce identical per-run val_loss.
    If this fails while the test above passes, clear_session() inside fit() is
    resetting Keras's seed generator after the runner seeded it."""
    y_reg = X[:, 0].astype("float32")
    kw = dict(data=(X, y_reg), runs=2, epochs=2, seed=9, verbose=False)
    r1 = ix.variability_study(_keras_builder, **kw)
    r2 = ix.variability_study(_keras_builder, **kw)
    assert r1.get_metric_values("val_loss") == pytest.approx(r2.get_metric_values("val_loss"))
