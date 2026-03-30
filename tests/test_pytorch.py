"""
tests/test_pytorch.py

Integration tests for PyTorchModelWrapper.predict().

BUG-CORE-02: the classification branch of predict() had no return statement
and silently returned None to any caller that captured the result. The
regression branch was unaffected.

Every test here trains the wrapper for 2 epochs before calling predict(),
so that model.eval(), device placement, and all internal state accumulated
during fit() are live. This catches state-dependent bugs that pure unit
tests — which skip fit() entirely — cannot reach.

Fixtures are module-scoped so each wrapper is trained once per session.
At 2 epochs on small CPU tensors this takes well under a second per fixture.
"""

import numpy as np
import pytest

from ictonyx.core import PYTORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures — data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def binary_data(rng):
    """80-sample binary classification dataset, 4 features."""
    X = rng.standard_normal((80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture(scope="module")
def multiclass_data(rng):
    """90-sample 3-class dataset, 6 features."""
    X = rng.standard_normal((90, 6)).astype(np.float32)
    y = np.array([i % 3 for i in range(90)], dtype=np.int64)
    return X[:70], y[:70], X[70:], y[70:]


@pytest.fixture(scope="module")
def regression_data(rng):
    """80-sample regression dataset, 4 features."""
    X = rng.standard_normal((80, 4)).astype(np.float32)
    y = (
        X @ np.array([1.5, -2.0, 0.5, 0.3], dtype=np.float32)
        + rng.standard_normal(80).astype(np.float32) * 0.1
    )
    return X[:60], y[:60], X[60:], y[60:]


# ---------------------------------------------------------------------------
# Fixtures — trained wrappers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_binary_classifier(binary_data):
    """Binary classifier trained for 2 epochs."""
    import torch
    import torch.nn as nn

    from ictonyx.core import PyTorchModelWrapper

    X_train, y_train, X_val, y_val = binary_data
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    wrapper = PyTorchModelWrapper(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.01},
        task="classification",
        device="cpu",
    )
    wrapper.fit(
        (X_train, y_train),
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
    )
    return wrapper, X_val, y_val


@pytest.fixture(scope="module")
def trained_multiclass_classifier(multiclass_data):
    """3-class classifier trained for 2 epochs."""
    import torch
    import torch.nn as nn

    from ictonyx.core import PyTorchModelWrapper

    X_train, y_train, X_val, y_val = multiclass_data
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))
    wrapper = PyTorchModelWrapper(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.01},
        task="classification",
        device="cpu",
    )
    wrapper.fit(
        (X_train, y_train),
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
    )
    return wrapper, X_val, y_val


@pytest.fixture(scope="module")
def trained_regressor(regression_data):
    """Regression wrapper trained for 2 epochs."""
    import torch
    import torch.nn as nn

    from ictonyx.core import PyTorchModelWrapper

    X_train, y_train, X_val, y_val = regression_data
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    wrapper = PyTorchModelWrapper(
        model=model,
        criterion=nn.MSELoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.01},
        task="regression",
        device="cpu",
    )
    wrapper.fit(
        (X_train, y_train),
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
    )
    return wrapper, X_val, y_val


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


class TestBinaryClassifierPredict:
    """predict() on a trained binary classifier."""

    def test_returns_not_none(self, trained_binary_classifier):
        """BUG-CORE-02 regression: predict() must not return None."""
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        assert result is not None, (
            "predict() returned None for a binary classification task. "
            "The classification branch is missing 'return self.predictions'."
        )

    def test_return_equals_attribute(self, trained_binary_classifier):
        """Return value must be the same array as wrapper.predictions."""
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        np.testing.assert_array_equal(
            result,
            wrapper.predictions,
            err_msg="predict() return value differs from wrapper.predictions.",
        )

    def test_return_is_ndarray(self, trained_binary_classifier):
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        assert isinstance(
            result, np.ndarray
        ), f"predict() returned {type(result).__name__}, expected np.ndarray."

    def test_shape_matches_input(self, trained_binary_classifier):
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        assert result.shape == (len(X_val),), f"Expected shape ({len(X_val)},), got {result.shape}."

    def test_dtype_is_integer(self, trained_binary_classifier):
        """Classification must return integer class indices, not floats."""
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        assert np.issubdtype(
            result.dtype, np.integer
        ), f"Expected integer dtype, got {result.dtype}."

    def test_labels_are_binary(self, trained_binary_classifier):
        """All predicted labels must be in {0, 1}."""
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val)
        unexpected = set(result.tolist()) - {0, 1}
        assert not unexpected, f"predict() returned labels outside {{0, 1}}: {unexpected}."

    def test_is_deterministic(self, trained_binary_classifier):
        """Two consecutive predict() calls on the same input must agree."""
        wrapper, X_val, _ = trained_binary_classifier
        r1 = wrapper.predict(X_val)
        r2 = wrapper.predict(X_val)
        np.testing.assert_array_equal(
            r1, r2, err_msg="predict() is non-deterministic on the same input."
        )

    def test_single_sample(self, trained_binary_classifier):
        """predict() must handle a batch of exactly one sample."""
        wrapper, X_val, _ = trained_binary_classifier
        result = wrapper.predict(X_val[:1])
        assert result is not None
        assert result.shape == (
            1,
        ), f"Single-sample predict() returned shape {result.shape}, expected (1,)."
        assert result[0] in {0, 1}, f"Single-sample prediction {result[0]} is not in {{0, 1}}."


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------


class TestMulticlassClassifierPredict:
    """predict() on a trained 3-class classifier."""

    def test_returns_not_none(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val)
        assert result is not None, "predict() returned None for 3-class task."

    def test_return_equals_attribute(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_shape_matches_input(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val)
        assert result.shape == (len(X_val),), f"Expected shape ({len(X_val)},), got {result.shape}."

    def test_dtype_is_integer(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val)
        assert np.issubdtype(
            result.dtype, np.integer
        ), f"Expected integer dtype, got {result.dtype}."

    def test_labels_in_valid_range(self, trained_multiclass_classifier):
        """All predicted labels must be in {0, 1, 2}."""
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val)
        unexpected = set(result.tolist()) - {0, 1, 2}
        assert not unexpected, f"predict() returned labels outside {{0, 1, 2}}: {unexpected}."

    def test_is_deterministic(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        r1 = wrapper.predict(X_val)
        r2 = wrapper.predict(X_val)
        np.testing.assert_array_equal(r1, r2)

    def test_single_sample(self, trained_multiclass_classifier):
        wrapper, X_val, _ = trained_multiclass_classifier
        result = wrapper.predict(X_val[:1])
        assert result is not None
        assert result.shape == (1,)
        assert result[0] in {0, 1, 2}


# ---------------------------------------------------------------------------
# Regression (symmetric — must not regress after the fix)
# ---------------------------------------------------------------------------


class TestRegressorPredict:
    """
    predict() on a trained regression wrapper. The regression branch
    returned correctly before BUG-CORE-02; these tests confirm the fix
    did not disturb it.
    """

    def test_returns_not_none(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val)
        assert result is not None, (
            "predict() returned None for regression — the fix broke the " "regression path."
        )

    def test_return_equals_attribute(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val)
        np.testing.assert_array_equal(result, wrapper.predictions)

    def test_return_is_ndarray(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val)
        assert isinstance(
            result, np.ndarray
        ), f"predict() returned {type(result).__name__}, expected np.ndarray."

    def test_shape_matches_input(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val)
        assert result.shape == (len(X_val),), f"Expected shape ({len(X_val)},), got {result.shape}."

    def test_dtype_is_float(self, trained_regressor):
        """Regression must return float values, not integers."""
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val)
        assert np.issubdtype(
            result.dtype, np.floating
        ), f"Expected float dtype, got {result.dtype}."

    def test_is_deterministic(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        r1 = wrapper.predict(X_val)
        r2 = wrapper.predict(X_val)
        np.testing.assert_array_equal(r1, r2)

    def test_single_sample(self, trained_regressor):
        wrapper, X_val, _ = trained_regressor
        result = wrapper.predict(X_val[:1])
        assert result is not None
        assert result.shape == (1,)
        assert np.issubdtype(result.dtype, np.floating)
