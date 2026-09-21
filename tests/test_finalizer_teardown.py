"""v0.4.10 C22: garbage-collecting a wrapper never tears down process-global framework state."""

import gc
import types

import pytest
from sklearn.linear_model import LogisticRegression

import ictonyx as ix
import ictonyx.memory as memory
from ictonyx.core import ScikitLearnModelWrapper
from ictonyx.memory import MemoryManager


class _FakeTF:
    """Stands in for the tensorflow module inside ictonyx.memory."""

    def __init__(self):
        self.clear_calls = 0
        self.reset_calls = 0
        me = self
        self.keras = types.SimpleNamespace(
            backend=types.SimpleNamespace(
                clear_session=lambda: setattr(me, "clear_calls", me.clear_calls + 1)
            )
        )
        self.compat = types.SimpleNamespace(
            v1=types.SimpleNamespace(
                reset_default_graph=lambda: setattr(me, "reset_calls", me.reset_calls + 1)
            )
        )
        self.config = types.SimpleNamespace(list_physical_devices=lambda kind: [])


@pytest.fixture
def fake_tf(monkeypatch):
    tf = _FakeTF()
    monkeypatch.setattr(memory, "HAS_TENSORFLOW", True)
    monkeypatch.setattr(memory, "tf", tf, raising=False)
    return tf


def test_cleanup_is_local_and_release_is_global(fake_tf):
    w = ScikitLearnModelWrapper(LogisticRegression())
    w.cleanup()
    assert fake_tf.clear_calls == 0
    w.release()
    assert fake_tf.clear_calls == 1 and fake_tf.reset_calls == 1


def test_garbage_collecting_a_wrapper_does_not_clear_the_session(fake_tf):
    w = ScikitLearnModelWrapper(LogisticRegression())
    del w
    gc.collect()
    assert fake_tf.clear_calls == 0


def test_memory_manager_default_cleanup_is_local(fake_tf):
    m = MemoryManager(use_process_isolation=False, verbose=False)
    res = m.cleanup()
    assert fake_tf.clear_calls == 0
    assert not any(a.startswith("tf_") for a in res.actions)
    res = m.cleanup(global_teardown=True)
    assert fake_tf.clear_calls == 1
    assert "tf_clear_session" in res.actions


def test_runner_uses_global_teardown_between_runs(X, y, monkeypatch):
    seen = []
    real = MemoryManager.cleanup

    def spy(self, *a, **kw):
        seen.append(kw.get("global_teardown", False))
        return real(self, *a, **kw)

    monkeypatch.setattr(MemoryManager, "cleanup", spy)
    ix.variability_study(LogisticRegression, data=(X, y), runs=2, seed=0, verbose=False)
    assert True in seen  # the runner's explicit between-run teardown
    assert seen.count(True) >= 2  # once per run, at least


@pytest.mark.skipif(not ix.TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_keras_model_outside_ictonyx_survives_a_wrapper_gc(X):
    """The reported hazard: a temporary sklearn wrapper going out of scope wiped
    the user's unrelated Keras models."""
    import numpy as np
    import tensorflow as tf

    outside = tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(1)])
    outside.compile(optimizer="adam", loss="mse")
    before = outside.predict(X[:5].astype("float32"), verbose=0)

    w = ScikitLearnModelWrapper(LogisticRegression())
    del w
    gc.collect()

    after = outside.predict(X[:5].astype("float32"), verbose=0)
    assert np.allclose(before, after)
