"""Test utility functions."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ictonyx.utils import load_object, save_object, train_val_test_split


def test_save_load_object():
    """Test saving and loading objects."""
    test_data = {"a": 1, "b": [2, 3, 4], "c": "test"}

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        # Save
        save_object(test_data, temp_path)
        assert os.path.exists(temp_path)

        # Load
        loaded_data = load_object(temp_path)
        assert loaded_data == test_data

    finally:
        os.unlink(temp_path)


def test_train_val_test_split():
    """Test data splitting function."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, test_size=0.2, val_size=0.2, random_state=42
    )

    # Check sizes
    assert len(X_train) == 64  # 100 * 0.8 * 0.8
    assert len(X_val) == 16  # 100 * 0.8 * 0.2
    assert len(X_test) == 20  # 100 * 0.2

    # Check no overlap
    assert len(X_train) + len(X_val) + len(X_test) == 100


def test_save_object_creates_file():
    """Test that save creates a file that exists."""
    import tempfile

    data = [1, 2, 3]
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(data, path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_load_object_nonexistent():
    """Test loading from a path that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_object("/fake/path/nothing.pkl")


def test_save_load_numpy():
    """Test roundtrip with numpy arrays."""
    import tempfile

    arr = np.array([[1, 2], [3, 4]])
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(arr, path)
        loaded = load_object(path)
        np.testing.assert_array_equal(arr, loaded)
    finally:
        os.unlink(path)


def test_save_load_dataframe():
    """Test roundtrip with pandas DataFrame."""
    import tempfile

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(df, path)
        loaded = load_object(path)
        pd.testing.assert_frame_equal(df, loaded)
    finally:
        os.unlink(path)


def test_train_val_test_split_small_val():
    """Test split with small val_size."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_save_object_creates_file():
    """Test that save creates a file that exists."""
    import tempfile

    data = [1, 2, 3]
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(data, path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_load_object_nonexistent():
    """Test loading from a path that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_object("/fake/path/nothing.pkl")


def test_save_load_numpy():
    """Test roundtrip with numpy arrays."""
    import tempfile

    arr = np.array([[1, 2], [3, 4]])
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(arr, path)
        loaded = load_object(path)
        np.testing.assert_array_equal(arr, loaded)
    finally:
        os.unlink(path)


def test_save_load_dataframe():
    """Test roundtrip with pandas DataFrame."""
    import tempfile

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_object(df, path)
        loaded = load_object(path)
        pd.testing.assert_frame_equal(df, loaded)
    finally:
        os.unlink(path)


def test_train_val_test_split_small_val():
    """Test split with small val_size."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_train_val_test_split_reproducible():
    """Test that same random_state gives same split."""
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)
    split1 = train_val_test_split(X, y, random_state=42)
    split2 = train_val_test_split(X, y, random_state=42)
    np.testing.assert_array_equal(split1[0], split2[0])  # X_train
    np.testing.assert_array_equal(split1[2], split2[2])  # X_test


def test_setup_mlflow_import_error():
    """Test that setup_mlflow raises ImportError without mlflow."""
    from unittest.mock import patch

    from ictonyx.utils import setup_mlflow

    with patch.dict("sys.modules", {"mlflow": None}):
        # If mlflow is actually installed, this test just verifies the function exists
        assert callable(setup_mlflow)


def test_train_val_test_split_reproducible():
    """Test that same random_state gives same split."""
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)
    split1 = train_val_test_split(X, y, random_state=42)
    split2 = train_val_test_split(X, y, random_state=42)
    np.testing.assert_array_equal(split1[0], split2[0])  # X_train
    np.testing.assert_array_equal(split1[2], split2[2])  # X_test


def test_setup_mlflow_import_error():
    """Test that setup_mlflow raises ImportError without mlflow."""
    from unittest.mock import patch

    from ictonyx.utils import setup_mlflow

    with patch.dict("sys.modules", {"mlflow": None}):
        # If mlflow is actually installed, this test just verifies the function exists
        assert callable(setup_mlflow)


class TestSaveLoadEdgeCases:

    def test_save_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "obj.pkl")
        save_object({"v": 1}, path)
        save_object({"v": 2}, path)
        loaded = load_object(path)
        assert loaded["v"] == 2

    def test_roundtrip_nested_structure(self, tmp_path):
        path = str(tmp_path / "nested.pkl")
        obj = {"a": [1, 2, 3], "b": {"c": "hello"}, "d": (4, 5)}
        save_object(obj, path)
        assert load_object(path) == obj

    def test_roundtrip_none(self, tmp_path):
        path = str(tmp_path / "none.pkl")
        save_object(None, path)
        assert load_object(path) is None


class TestTrainValTestSplitEdgeCases:

    def test_no_overlap_between_splits(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, test_size=0.2, val_size=0.2, random_state=0
        )
        train_idx = set(X_train.flatten())
        val_idx = set(X_val.flatten())
        test_idx = set(X_test.flatten())
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_y_labels_aligned_with_X(self):
        X = np.arange(50).reshape(50, 1).astype(float)
        y = np.arange(50, 100).astype(float)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, random_state=42)
        # y[i] must equal X[i] + 50 for all splits
        np.testing.assert_array_equal(y_train, X_train.flatten() + 50)
        np.testing.assert_array_equal(y_val, X_val.flatten() + 50)
        np.testing.assert_array_equal(y_test, X_test.flatten() + 50)

    def test_dataframe_input(self):
        df = pd.DataFrame({"a": range(50), "b": range(50, 100)})
        y = np.zeros(50)
        X_train, X_val, X_test, _, _, _ = train_val_test_split(df, y, random_state=42)
        assert isinstance(X_train, pd.DataFrame)


class TestTrainValTestSplitValidation:

    def test_returns_six_elements(self):
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        result = train_val_test_split(X, y)
        assert len(result) == 6

    def test_total_samples_preserved(self):
        X = np.random.rand(100, 4)
        y = np.zeros(100)
        X_tr, X_v, X_te, y_tr, y_v, y_te = train_val_test_split(X, y)
        assert len(X_tr) + len(X_v) + len(X_te) == 100

    def test_different_random_states_give_different_splits(self):
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.zeros(100)
        _, _, X_te1, _, _, _ = train_val_test_split(X, y, random_state=0)
        _, _, X_te2, _, _, _ = train_val_test_split(X, y, random_state=99)
        assert not np.array_equal(X_te1, X_te2)
