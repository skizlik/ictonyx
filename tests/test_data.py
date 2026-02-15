"""Test data handlers."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ictonyx.core import TENSORFLOW_AVAILABLE
from ictonyx.data import (
    ArraysDataHandler,
    ImageDataHandler,
    TabularDataHandler,
    TextDataHandler,
    TimeSeriesDataHandler,
    auto_resolve_handler,
)


class TestTabularDataHandler:
    """Test TabularDataHandler."""

    def test_tabular_handler_creation(self):
        """Test creating tabular data handler."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {"feature1": range(10), "feature2": range(10, 20), "target": [0, 1] * 5}
            )
            df.to_csv(f.name, index=False)
            path = f.name

        handler = TabularDataHandler(path, target_column="target")

        assert handler.data_type == "tabular"
        assert handler.return_format == "split_arrays"
        assert handler.target_column == "target"

        # Clean up
        import os

        os.unlink(path)

    def test_tabular_load(self):
        """Test loading tabular data."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "x1": np.random.rand(100),
                    "x2": np.random.rand(100),
                    "y": np.random.randint(0, 2, 100),
                }
            )
            df.to_csv(f.name, index=False)
            path = f.name

        handler = TabularDataHandler(path, target_column="y")
        data = handler.load(test_split=0.2, val_split=0.1)

        assert "train_data" in data
        assert "val_data" in data
        assert "test_data" in data

        X_train, y_train = data["train_data"]
        assert len(X_train) > 0
        assert len(y_train) == len(X_train)

        # Clean up
        import os

        os.unlink(path)


class TestArraysDataHandler:
    """Tests for the new ArraysDataHandler class."""

    def test_initialization_mismatch(self):
        """Test error when X and y have different lengths."""
        X = [1, 2, 3]
        y = [1, 2]  # Short
        with pytest.raises(ValueError, match="Length mismatch"):
            ArraysDataHandler(X, y)

    def test_load_splitting(self):
        """Test that load() correctly splits the data."""
        X = np.arange(100).reshape(-1, 1)
        y = np.zeros(100)

        handler = ArraysDataHandler(X, y)
        splits = handler.load(test_split=0.2, val_split=0.2)

        X_train, y_train = splits["train_data"]
        X_val, y_val = splits["val_data"]
        X_test, y_test = splits["test_data"]

        # Check sizes (approximate due to int rounding)
        assert 55 <= len(X_train) <= 65
        assert 15 <= len(X_val) <= 25
        assert 15 <= len(X_test) <= 25


class TestAutoResolveHandler:
    """Tests for the auto_resolve_handler factory function."""

    def test_resolve_dataframe(self):
        """Test that pandas DataFrames resolve to TabularDataHandler."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        handler = auto_resolve_handler(df, target_column="target")

        assert isinstance(handler, TabularDataHandler)
        assert handler.input_df is not None
        assert len(handler.input_df) == 2

    def test_resolve_dataframe_missing_target(self):
        """Test error when target is missing for DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="target_column is required"):
            auto_resolve_handler(df)

    def test_resolve_arrays(self):
        """Test that tuple of arrays resolves to ArraysDataHandler."""
        X = np.array([[1], [2]])
        y = np.array([0, 1])

        # Case 1: Numpy arrays
        handler = auto_resolve_handler((X, y))
        assert isinstance(handler, ArraysDataHandler)

        # Case 2: Lists
        handler_lists = auto_resolve_handler(([1, 2], [0, 1]))
        assert isinstance(handler_lists, ArraysDataHandler)

    def test_resolve_csv_tabular(self):
        """Test resolving CSV file to TabularDataHandler."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,target\n1,0\n2,1")
            path = f.name

        try:
            # If target_column is provided, assume Tabular
            handler = auto_resolve_handler(path, target_column="target")
            assert isinstance(handler, TabularDataHandler)
        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_resolve_image_dir(self):
        """Test resolving directory to ImageDataHandler."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy class folders
            os.makedirs(os.path.join(tmp_dir, "class_a"))
            os.makedirs(os.path.join(tmp_dir, "class_b"))

            # Should detect as Image handler
            handler = auto_resolve_handler(tmp_dir, image_size=(64, 64))
            assert isinstance(handler, ImageDataHandler)

    def test_resolve_ambiguous_file(self):
        """Test error when file type is ambiguous."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("data")
            path = f.name

        try:
            # No target_column, text_column, or sequence_length provided
            with pytest.raises(ValueError, match="Ambiguous file input"):
                auto_resolve_handler(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestArraysDataHandlerEdgeCases:
    """Edge cases for ArraysDataHandler."""

    def test_no_test_split(self):
        """Test with test_split=0."""
        X = np.arange(50).reshape(-1, 1)
        y = np.zeros(50)
        handler = ArraysDataHandler(X, y)
        splits = handler.load(test_split=0, val_split=0.2)

        assert splits["test_data"] is None
        X_train, _ = splits["train_data"]
        X_val, _ = splits["val_data"]
        assert len(X_train) + len(X_val) == 50

    def test_no_val_split(self):
        """Test with val_split=0."""
        X = np.arange(50).reshape(-1, 1)
        y = np.zeros(50)
        handler = ArraysDataHandler(X, y)
        splits = handler.load(test_split=0.2, val_split=0)

        assert splits["val_data"] is None
        X_train, _ = splits["train_data"]
        X_test, _ = splits["test_data"]
        assert len(X_train) + len(X_test) == 50

    def test_no_splits_at_all(self):
        """Test with both splits at 0 — all data goes to train."""
        X = np.arange(20).reshape(-1, 1)
        y = np.ones(20)
        handler = ArraysDataHandler(X, y)
        splits = handler.load(test_split=0, val_split=0)

        assert splits["test_data"] is None
        assert splits["val_data"] is None
        X_train, _ = splits["train_data"]
        assert len(X_train) == 20

    def test_splits_sum_too_large(self):
        """Test error when splits sum to >= 1.0."""
        handler = ArraysDataHandler(np.zeros((10, 2)), np.zeros(10))
        with pytest.raises(ValueError, match="must be < 1.0"):
            handler.load(test_split=0.5, val_split=0.5)

    def test_data_type_property(self):
        """Test data_type and return_format."""
        handler = ArraysDataHandler(np.zeros((5, 2)), np.zeros(5))
        assert handler.data_type == "arrays"
        assert handler.return_format == "split_arrays"

    def test_accepts_lists(self):
        """Test that plain Python lists work."""
        handler = ArraysDataHandler([1, 2, 3, 4, 5], [0, 1, 0, 1, 0])
        assert handler.X.shape == (5,)
        assert handler.y.shape == (5,)

    def test_small_dataset(self):
        """Test with very small dataset (2 samples)."""
        handler = ArraysDataHandler(np.array([[1], [2]]), np.array([0, 1]))
        # Should not crash — splits may be empty but shouldn't error
        splits = handler.load(test_split=0.5, val_split=0)
        assert splits["train_data"] is not None


class TestTabularDataHandlerFromDataFrame:
    """Test TabularDataHandler initialized with a DataFrame."""

    def test_from_dataframe(self):
        """Test creating handler from DataFrame directly."""
        df = pd.DataFrame(
            {
                "a": np.random.rand(50),
                "b": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        handler = TabularDataHandler(df, target_column="target")
        splits = handler.load(test_split=0.2, val_split=0.1)

        X_train, y_train = splits["train_data"]
        assert X_train.shape[1] == 2  # two feature columns
        assert len(y_train) == len(X_train)

    def test_missing_target_column(self):
        """Test error when target column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        handler = TabularDataHandler(df, target_column="nonexistent")
        with pytest.raises((KeyError, ValueError)):
            handler.load()


class TestAutoResolveHandlerPassthrough:
    """Test that passing an existing DataHandler returns it unchanged."""

    def test_passthrough(self):
        """Test DataHandler passthrough."""
        original = ArraysDataHandler(np.zeros((5, 2)), np.zeros(5))
        result = auto_resolve_handler(original)
        assert result is original

    def test_nonexistent_path(self):
        """Test error for nonexistent file path."""
        with pytest.raises(FileNotFoundError):
            auto_resolve_handler("/fake/path/data.csv")

    def test_unsupported_type(self):
        """Test error for unsupported data type."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            auto_resolve_handler(12345)
