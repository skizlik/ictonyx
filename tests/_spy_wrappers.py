"""Module-level spy wrappers for runner tests.

Module-level (not defined inside a test) so that spawn-based process
isolation can import them by name in the child process.
"""

import numpy as np

from ictonyx import BaseModelWrapper, TrainingResult


class _Base(BaseModelWrapper):
    """Minimal concrete wrapper: every abstract method stubbed."""

    def predict(self, d, **k):
        return np.zeros(len(d))

    def predict_proba(self, d, **k):
        return np.zeros((len(d), 2))

    def evaluate(self, d, **k):
        return {"accuracy": 0.5}

    def assess(self, t):
        return {}

    def save_model(self, p):
        pass

    @classmethod
    def load_model(cls, p):
        pass

    def _cleanup_implementation(self):
        pass


class RecordingSpy(_Base):
    """Records run_seed and the number of fit() kwargs into the history."""

    def fit(self, train_data, validation_data=None, **kw):
        self.training_result = TrainingResult(
            history={
                "val_seed_seen": [float(kw.get("run_seed", -1))],
                "val_n_kwargs": [float(len(kw))],
                "val_accuracy": [0.5],
            },
            params={},
        )


class EvalRaisesSpy(RecordingSpy):
    """Trains fine; evaluate() always raises."""

    def evaluate(self, d, **k):
        raise RuntimeError("boom")


def build_recording_spy(cfg):
    return RecordingSpy(None)


def build_eval_raises_spy(cfg):
    return EvalRaisesSpy(None)


FAIL_SEED = None  # set by tests; FailsOnRunSpy raises when run_seed == FAIL_SEED


class FailsOnRunSpy(RecordingSpy):
    """Raises in fit() for one specific run, to exercise failed_runs bookkeeping."""

    def fit(self, train_data, validation_data=None, **kw):
        import _spy_wrappers as m

        if kw.get("run_seed") == m.FAIL_SEED:
            raise RuntimeError("planned failure")
        super().fit(train_data, validation_data, **kw)


def build_fails_on_run(cfg):
    return FailsOnRunSpy(None)


class NamedLRSpy(_Base):
    """fit() names learning_rate explicitly, so build_fit_kwargs forwards it."""

    def fit(self, train_data, validation_data=None, learning_rate=None, **kw):
        self.training_result = TrainingResult(
            history={
                "val_lr_seen": [float(learning_rate if learning_rate is not None else -1)],
                "val_accuracy": [0.5],
            },
            params={},
        )


def build_named_lr_spy(cfg):
    return NamedLRSpy(None)
