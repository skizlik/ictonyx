"""
Shared pytest fixtures for the ictonyx test suite.

Import automatically by pytest — no explicit import needed in test files.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_classification_arrays():
    """80-sample binary classification dataset, reproducible."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def small_multiclass_arrays():
    """60-sample 3-class dataset, reproducible."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 4))
    y = np.array([i % 3 for i in range(60)])
    return X, y


@pytest.fixture
def small_regression_arrays():
    """60-sample regression dataset, reproducible."""
    rng = np.random.default_rng(42)
    X = rng.random((60, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.05, 60)
    return X, y


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    """Minimal ModelConfig suitable for quick training runs."""
    from ictonyx.config import ModelConfig

    return ModelConfig({"epochs": 2, "batch_size": 16, "learning_rate": 0.01, "verbose": 0})


# ---------------------------------------------------------------------------
# DataHandler fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_classification_handler(small_classification_arrays):
    """ArraysDataHandler wrapping small_classification_arrays."""
    from ictonyx.data import ArraysDataHandler

    X, y = small_classification_arrays
    return ArraysDataHandler(X, y, val_split=0.2, test_split=0.1)


@pytest.fixture
def tabular_regression_handler(small_regression_arrays):
    """ArraysDataHandler wrapping small_regression_arrays."""
    from ictonyx.data import ArraysDataHandler

    X, y = small_regression_arrays
    return ArraysDataHandler(X, y, val_split=0.2, test_split=0.1)
