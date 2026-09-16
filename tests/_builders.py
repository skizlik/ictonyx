"""Module-level builders and estimators for tests.

Everything here must be importable by name so process-isolation (spawn) and
joblib workers can pickle it. Test files import from this module; nothing here
imports from test files.
"""

import numpy as np  # noqa: F401  (used by builders added in later commits)


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
        self.c_ = 0

    def predict(self, X):
        return np.zeros(len(X))

    def score(self, X, y):
        return 0.0


class _FailsSometimes:
    """~30% of child seeds fail; deterministic per base seed."""

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        if self.rs is not None and self.rs % 10 < 3:
            raise RuntimeError("boom")
        self.c_ = 0

    def predict(self, X):
        return np.zeros(len(X))

    def score(self, X, y):
        return 0.0
