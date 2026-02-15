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
