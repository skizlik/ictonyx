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


# =============================================================================
# ADD TO: tests/test_data.py  (paste at the bottom)
# =============================================================================
# No new imports needed — existing file has os, tempfile, np, pd, pytest,
# ArraysDataHandler, TabularDataHandler, auto_resolve_handler, etc.


class TestTabularDataHandlerCoverage:
    """Target uncovered TabularDataHandler lines."""

    def test_legacy_data_path_kwarg(self):
        """Test backward-compat data_path keyword (line ~400)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(f.name, index=False)
            path = f.name
        try:
            handler = TabularDataHandler(data_path=path, target_column="y")
            splits = handler.load()
            assert "train_data" in splits
        finally:
            os.unlink(path)

    def test_no_data_raises(self):
        """Test error when neither data nor data_path provided."""
        with pytest.raises(ValueError, match="Must provide"):
            TabularDataHandler(target_column="y")

    def test_no_target_raises(self):
        """Test error when target_column not provided."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="target_column"):
            TabularDataHandler(data=df, target_column=None)

    def test_invalid_data_type_raises(self):
        """Test error when data is neither str nor DataFrame."""
        with pytest.raises(TypeError, match="str path or DataFrame"):
            TabularDataHandler(data=12345, target_column="y")

    def test_features_parameter(self):
        """Test that features param selects specific columns."""
        df = pd.DataFrame(
            {
                "a": np.random.rand(50),
                "b": np.random.rand(50),
                "c": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        handler = TabularDataHandler(data=df, target_column="target", features=["a", "b"])
        splits = handler.load()
        X_train, y_train = splits["train_data"]
        # Should only have 2 feature columns (a, b), not 3
        assert X_train.shape[1] == 2

    def test_features_missing_column_raises(self):
        """Test error when a requested feature doesn't exist."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        handler = TabularDataHandler(data=df, target_column="target", features=["a", "nonexistent"])
        with pytest.raises(ValueError, match="not found in data"):
            handler.load()

    def test_target_column_not_in_data(self):
        """Test error when target column missing from DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        handler = TabularDataHandler(data=df, target_column="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            handler.load()

    def test_empty_csv_raises(self):
        """Test error when CSV file is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")  # empty file
            path = f.name
        try:
            handler = TabularDataHandler(data=path, target_column="y")
            with pytest.raises((ValueError, RuntimeError)):
                handler.load()
        finally:
            os.unlink(path)

    def test_csv_with_custom_sep(self):
        """Test loading a TSV (tab-separated) file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df = pd.DataFrame({"x1": range(20), "x2": range(20, 40), "target": [0, 1] * 10})
            df.to_csv(f.name, sep="\t", index=False)
            path = f.name
        try:
            handler = TabularDataHandler(data=path, target_column="target", sep="\t")
            splits = handler.load()
            X_train, y_train = splits["train_data"]
            assert X_train.shape[1] == 2
        finally:
            os.unlink(path)

    def test_load_no_val_split(self):
        """Test with val_split=0."""
        df = pd.DataFrame(
            {
                "a": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")
        splits = handler.load(test_split=0.2, val_split=0)
        assert splits["val_data"] == (None, None) or splits["val_data"] is None

    def test_load_no_test_split(self):
        """Test with test_split=0."""
        df = pd.DataFrame(
            {
                "a": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")
        splits = handler.load(test_split=0, val_split=0.2)
        assert splits["test_data"] is None or splits["test_data"] == (None, None)

    def test_load_splits_sum_too_large(self):
        """Test error when splits sum >= 1.0."""
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        handler = TabularDataHandler(data=df, target_column="target")
        with pytest.raises(ValueError, match="< 1.0"):
            handler.load(test_split=0.6, val_split=0.5)


class TestDataHandlerGetInfo:
    """Test get_data_info method."""

    def test_arrays_handler_info(self):
        handler = ArraysDataHandler(np.zeros((10, 3)), np.zeros(10))
        info = handler.get_data_info()
        assert info["data_type"] == "arrays"
        assert info["return_format"] == "split_arrays"

    def test_tabular_handler_info_from_df(self):
        df = pd.DataFrame({"a": [1], "target": [0]})
        handler = TabularDataHandler(data=df, target_column="target")
        info = handler.get_data_info()
        assert info["data_type"] == "tabular"


class TestAutoResolveHandlerExtended:
    """Additional auto_resolve_handler edge cases."""

    def test_resolve_csv_with_text_column(self):
        """Test that text_column kwarg routes to TextDataHandler (or ImportError)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\nhello,0\nworld,1\n")
            path = f.name
        try:
            try:
                handler = auto_resolve_handler(path, text_column="text", label_column="label")
                assert isinstance(handler, TextDataHandler)
            except ImportError:
                # TF not available — that's fine, the routing was correct
                pass
        finally:
            os.unlink(path)

    def test_resolve_csv_with_value_column(self):
        """Test that value_column kwarg routes to TimeSeriesDataHandler (or ImportError)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,value\n2024-01-01,100\n2024-01-02,101\n")
            path = f.name
        try:
            try:
                handler = auto_resolve_handler(path, value_column="value", sequence_length=5)
                assert isinstance(handler, TimeSeriesDataHandler)
            except ImportError:
                # TF not available — routing was correct
                pass
        finally:
            os.unlink(path)

    def test_resolve_directory_without_image_size(self):
        """Test error when directory given but image_size missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "class_a"))
            with pytest.raises(ValueError, match="image_size"):
                auto_resolve_handler(tmp_dir)
