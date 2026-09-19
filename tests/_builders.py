"""Module-level builders and estimators for tests.

Everything here must be importable by name so process-isolation (spawn) and
joblib workers can pickle it. Test files import from this module; nothing here
imports from test files.
"""

import numpy as np


class _AlwaysFails:
    """Every fit raises. Exercises the first-run-fatal path."""

    def __init__(self, **kw): ...

    def fit(self, X, y):
        raise RuntimeError("boom")

    def predict(self, X):
        return np.zeros(len(X))

    def score(self, X, y):
        return 0.0


class _FailsAfterFirst:
    """Run 1 succeeds; every later run raises. Exercises the failure-rate guard."""

    _first_seed = None

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        if _FailsAfterFirst._first_seed is None:
            _FailsAfterFirst._first_seed = self.rs
        if self.rs != _FailsAfterFirst._first_seed:
            raise RuntimeError("boom")
        self.classes_ = np.unique(y)
        self.c_ = 0

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.0


class _FailsSometimes:
    """~30% of child seeds fail; deterministic per base seed."""

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        if self.rs is not None and self.rs % 10 < 3:
            raise RuntimeError("boom")
        self.classes_ = np.unique(y)
        self.c_ = 0

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.0


class _FailsOnSecondRun:
    """Run 1 succeeds; run 2 fails; runs 3+ succeed. One hole, deterministic."""

    _calls = 0

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        _FailsOnSecondRun._calls += 1
        if _FailsOnSecondRun._calls == 2:
            raise RuntimeError("boom")
        self.classes_ = np.unique(y)
        self.c_ = 0

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.0


class _InterruptsAfter:
    """Trains a stochastic tree; raises KeyboardInterrupt on call number ``stop_after + 1``.

    Set ``_InterruptsAfter.stop_after = k`` (or None to never interrupt) and
    reset ``_InterruptsAfter.calls = 0`` before each study. Standard mode only
    (class-level state does not survive a spawned child).
    """

    stop_after = None
    calls = 0

    def __init__(self, random_state=None, **kw):
        from sklearn.tree import DecisionTreeClassifier

        self.rs = random_state
        self._tree = DecisionTreeClassifier(max_features=1, random_state=random_state)

    def fit(self, X, y):
        _InterruptsAfter.calls += 1
        if (
            _InterruptsAfter.stop_after is not None
            and _InterruptsAfter.calls > _InterruptsAfter.stop_after
        ):
            raise KeyboardInterrupt
        self._tree.fit(X, y)
        self.classes_ = self._tree.classes_
        return self

    def predict(self, X):
        return self._tree.predict(X)

    def score(self, X, y):
        return self._tree.score(X, y)
