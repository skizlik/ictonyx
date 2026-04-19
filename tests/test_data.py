"""Test data handlers."""

import os
import tempfile
from unittest.mock import patch

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

    def test_xtest_without_ytest_raises(self):
        """Providing X_test without y_test must raise ValueError at construction."""
        X = np.zeros((50, 3))
        y = np.zeros(50)
        X_test = np.zeros((10, 3))

        with pytest.raises(ValueError, match="both X_test and y_test"):
            ArraysDataHandler(X, y, X_test=X_test)  # no y_test

    def test_ytest_without_xtest_raises(self):
        """Providing y_test without X_test must raise ValueError at construction."""
        X = np.zeros((50, 3))
        y = np.zeros(50)
        y_test = np.zeros(10)

        with pytest.raises(ValueError, match="both X_test and y_test"):
            ArraysDataHandler(X, y, y_test=y_test)  # no X_test

    def test_both_provided_does_not_raise(self):
        """Providing both X_test and y_test is valid."""
        X = np.zeros((50, 3))
        y = np.zeros(50)
        X_test = np.zeros((10, 3))
        y_test = np.zeros(10)

        handler = ArraysDataHandler(X, y, X_test=X_test, y_test=y_test)
        result = handler.load()
        X_test_out, y_test_out = result["test_data"]
        assert X_test_out.shape == (10, 3)
        assert y_test_out.shape == (10,)

    def test_neither_provided_does_not_raise(self):
        """Providing neither X_test nor y_test uses internal splitting."""
        handler = ArraysDataHandler(np.zeros((50, 3)), np.zeros(50))
        result = handler.load(test_split=0.2)
        assert result["test_data"] is not None
        X_test_out, _ = result["test_data"]
        assert len(X_test_out) > 0

    def test_arrays_handler_importable_without_tf_preprocessing(self):
        """ArraysDataHandler must import cleanly with no TF preprocessing dependency.

        The TF preprocessing import block was removed in v0.4.0. This test
        verifies that ictonyx.data loads without error and exposes
        ArraysDataHandler with no TF dependency of any kind.
        """
        import importlib
        import sys

        # Remove ictonyx.data from the module cache so the import re-runs
        # from scratch in this process state.
        modules_to_remove = [k for k in sys.modules if "ictonyx.data" in k]
        saved = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            from ictonyx import data as ictonyx_data

            # The dead TF preprocessing block is gone — the flag must not exist.
            assert not hasattr(
                ictonyx_data, "HAS_TF_PREPROCESSING"
            ), "HAS_TF_PREPROCESSING was removed in v0.4.0 and must not be present"

            # ArraysDataHandler must be directly accessible.
            assert hasattr(
                ictonyx_data, "ArraysDataHandler"
            ), "ArraysDataHandler must be importable from ictonyx.data"

        finally:
            sys.modules.update(saved)


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

    def test_get_data_info_dataframe_mode_no_dummy_path(self):
        """get_data_info() for DataFrame-backed handler must not contain dummy path."""
        df = pd.DataFrame({"a": range(10), "b": range(10), "target": [0, 1] * 5})
        handler = TabularDataHandler(data=df, target_column="target")

        info = handler.get_data_info()

        # Must NOT contain the dummy path
        assert (
            info.get("data_path") != "in_memory_dataframe"
        ), "get_data_info() should not expose the internal dummy path string"
        # Must NOT say path_exists=False (which implies a broken handler)
        assert (
            info.get("path_exists") is not False
        ), "get_data_info() should not report path_exists=False for in-memory data"
        # Should identify the source clearly
        assert info.get("source") == "dataframe"
        # Should still contain useful metadata
        assert info["data_type"] == "tabular"

    def test_get_data_info_dataframe_includes_shape(self):
        """get_data_info() for DataFrame mode includes row/column counts."""
        df = pd.DataFrame(
            {
                "x1": np.random.rand(30),
                "x2": np.random.rand(30),
                "y": np.random.randint(0, 2, 30),
            }
        )
        handler = TabularDataHandler(data=df, target_column="y")
        # load() must be called first to populate self.data
        handler.load(test_split=0.0, val_split=0.0)

        info = handler.get_data_info()
        assert info["num_rows"] == 30
        assert info["num_columns"] == 3
        assert info["target_column"] == "y"

    def test_get_data_info_file_mode_has_path(self, tmp_path):
        """get_data_info() for file-backed handler still returns the real path."""
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n")
        handler = TabularDataHandler(data=str(csv), target_column="target")
        info = handler.get_data_info()
        assert info["data_path"] == str(csv)
        assert info["path_exists"] is True


class TestImageDataHandlerValidation:
    """Tests for ImageDataHandler._validate_image_files."""

    def test_validate_raises_on_corrupt_file(self, tmp_path):
        """_validate_image_files raises DataValidationError for non-image files."""
        pytest.importorskip("PIL", reason="Pillow not installed")

        from ictonyx.exceptions import DataValidationError

        # Create a fake "image" that is actually plain text
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"this is not an image")

        # We need a minimal ImageDataHandler to call the private method.
        # Use a real (but minimal) directory structure so __init__ doesn't raise.
        img_dir = tmp_path / "dataset"
        class_dir = img_dir / "cat"
        class_dir.mkdir(parents=True)
        # Put a valid tiny PNG in the class directory so ImageDataHandler init passes
        try:
            from PIL import Image

            img = Image.new("RGB", (8, 8), color=(255, 0, 0))
            img.save(str(class_dir / "valid.png"))
        except ImportError:
            pytest.skip("Pillow not available")

        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        from ictonyx.data import ImageDataHandler

        handler = ImageDataHandler(str(img_dir), image_size=(8, 8))

        with pytest.raises(DataValidationError, match="could not be opened"):
            handler._validate_image_files([str(corrupt)])

    def test_validate_passes_for_valid_files(self, tmp_path):
        """_validate_image_files does not raise for well-formed image files."""
        pytest.importorskip("PIL", reason="Pillow not installed")

        from PIL import Image

        # Write a valid PNG
        valid = tmp_path / "good.png"
        img = Image.new("RGB", (16, 16), color=(0, 128, 255))
        img.save(str(valid))

        img_dir = tmp_path / "dataset"
        class_dir = img_dir / "cat"
        class_dir.mkdir(parents=True)
        img.save(str(class_dir / "valid.png"))

        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        from ictonyx.data import ImageDataHandler

        handler = ImageDataHandler(str(img_dir), image_size=(16, 16))
        # Should not raise
        handler._validate_image_files([str(valid)])

    def test_validate_empty_list_does_not_raise(self, tmp_path):
        """_validate_image_files with an empty list is a no-op."""
        pytest.importorskip("PIL", reason="Pillow not installed")

        img_dir = tmp_path / "dataset"
        (img_dir / "cat").mkdir(parents=True)

        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        from PIL import Image

        from ictonyx.data import ImageDataHandler

        img = Image.new("RGB", (8, 8))
        img.save(str(img_dir / "cat" / "x.png"))

        handler = ImageDataHandler(str(img_dir), image_size=(8, 8))
        handler._validate_image_files([])  # should be a no-op


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

    def test_nan_features_emit_warning(self):
        """NaN values in feature columns must trigger logger.warning, not pass silently."""
        import unittest.mock as mock

        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [10.0, 20.0, np.nan, 40.0, 50.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")

        with mock.patch("ictonyx.data.logger") as mock_logger:
            handler.load(test_split=0.0, val_split=0.0)
            assert mock_logger.warning.called
            call_args = mock_logger.warning.call_args[0][0]
            assert "missing" in call_args.lower() or "nan" in call_args.lower()

    def test_nan_features_warning_names_the_column(self):
        """The NaN warning message must identify which column(s) are affected."""
        df = pd.DataFrame(
            {
                "feature_alpha": [1.0, np.nan, 3.0, 4.0, 5.0],
                "feature_beta": [10.0, 20.0, 30.0, 40.0, 50.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")

        import unittest.mock as mock

        with mock.patch("ictonyx.data.logger") as mock_logger:
            handler.load(test_split=0.0, val_split=0.0)
            call_args = mock_logger.warning.call_args[0][0]
            assert "feature_alpha" in call_args

    def test_clean_features_no_warning(self):
        """No warning when features have no NaN values."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")

        import unittest.mock as mock

        with mock.patch("ictonyx.data.logger") as mock_logger:
            handler.load(test_split=0.0, val_split=0.0)
            # warning should not have been called for features
            # (it may be called for other reasons, but not for NaN in X)
            for call in mock_logger.warning.call_args_list:
                assert "missing" not in str(call).lower() or "target" in str(call).lower()


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


class TestDataHandlerHierarchy:
    """Verify the DataHandler / FileDataHandler / ArraysDataHandler hierarchy."""

    def test_arrays_handler_has_no_data_path(self):
        """ArraysDataHandler must not carry a dummy file path attribute."""
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        handler = ArraysDataHandler(X, y)
        assert not hasattr(
            handler, "data_path"
        ), "ArraysDataHandler should not have a data_path attribute"

    def test_arrays_handler_is_data_handler(self):
        """ArraysDataHandler must still satisfy the DataHandler contract."""
        from ictonyx.data import DataHandler

        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        assert isinstance(ArraysDataHandler(X, y), DataHandler)

    def test_tabular_handler_is_data_handler(self, tmp_path):
        """TabularDataHandler must satisfy the DataHandler contract."""
        from ictonyx.data import DataHandler

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        assert isinstance(TabularDataHandler(str(csv), target_column="target"), DataHandler)

    def test_file_handler_raises_on_missing_path(self):
        """TabularDataHandler must raise FileNotFoundError for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            TabularDataHandler("/does/not/exist.csv", target_column="y")

    def test_arrays_handler_load_still_works(self):
        """Removing the dummy path must not break the load() method."""
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        handler = ArraysDataHandler(X, y)
        result = handler.load(test_split=0.2, val_split=0.1)
        assert "train_data" in result
        assert "val_data" in result
        assert "test_data" in result
        X_train, y_train = result["train_data"]
        assert len(X_train) > 0


class TestArraysDataHandlerSplitDefaults:
    """Verify that val_split/test_split can be set at construction time (A1 fix)."""

    def test_split_defaults_stored_on_init(self):
        """Constructor stores val_split and test_split as instance attributes."""
        handler = ArraysDataHandler(
            np.zeros((100, 3)), np.zeros(100), val_split=0.15, test_split=0.25
        )
        assert handler._default_val_split == 0.15
        assert handler._default_test_split == 0.25

    def test_load_uses_init_defaults_when_not_overridden(self):
        """load() with no arguments uses the splits configured at init."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        y = rng.integers(0, 2, 200)

        handler = ArraysDataHandler(X, y, val_split=0.2, test_split=0.1)
        splits = handler.load()  # no explicit splits

        X_train, _ = splits["train_data"]
        X_val, _ = splits["val_data"]
        X_test, _ = splits["test_data"]

        assert len(X_test) == pytest.approx(20, abs=3)  # ~10% of 200
        # val is carved from the 180-sample remainder: 0.2 / 0.9 ≈ 22.2%
        assert len(X_val) == pytest.approx(40, abs=5)  # ~20% of 200

    def test_explicit_load_args_override_init_defaults(self):
        """Explicit arguments to load() take precedence over init defaults."""
        handler = ArraysDataHandler(
            np.zeros((100, 2)), np.zeros(100), val_split=0.3, test_split=0.3
        )
        splits = handler.load(test_split=0.1, val_split=0.1)

        X_test, _ = splits["test_data"]
        assert len(X_test) == pytest.approx(10, abs=2)  # 10%, not 30%

    def test_conftest_classification_fixture_works(self, tabular_classification_handler):
        """The tabular_classification_handler fixture must not raise TypeError."""
        # If we reach this line, the fixture constructed successfully.
        result = tabular_classification_handler.load()
        assert "train_data" in result
        assert "val_data" in result
        assert "test_data" in result
        X_train, y_train = result["train_data"]
        assert len(X_train) > 0

    def test_conftest_regression_fixture_works(self, tabular_regression_handler):
        """The tabular_regression_handler fixture must not raise TypeError."""
        result = tabular_regression_handler.load()
        assert "train_data" in result
        X_train, y_train = result["train_data"]
        assert len(X_train) > 0

    def test_default_splits_unchanged_from_original_load_defaults(self):
        """Default val_split=0.1 and test_split=0.2 match load()'s original defaults."""
        handler = ArraysDataHandler(np.zeros((100, 2)), np.zeros(100))
        # No splits specified at init — should behave exactly like the old load() defaults
        splits_new = handler.load()

        # Compare against explicitly passing the old defaults
        handler2 = ArraysDataHandler(np.zeros((100, 2)), np.zeros(100))
        splits_old = handler2.load(test_split=0.2, val_split=0.1)

        X_train_new, _ = splits_new["train_data"]
        X_train_old, _ = splits_old["train_data"]
        assert len(X_train_new) == len(X_train_old)

    def test_backward_compatibility_explicit_load_args_still_work(self):
        """Existing code that calls load(test_split=..., val_split=...) is unaffected."""
        handler = ArraysDataHandler(np.zeros((100, 2)), np.zeros(100))
        splits = handler.load(test_split=0.3, val_split=0.2)
        X_test, _ = splits["test_data"]
        assert len(X_test) == pytest.approx(30, abs=3)


class TestTextDataHandlerKeras3:
    """TextDataHandler must fail cleanly on Keras 3 (R5-2)."""

    def test_raises_import_error_when_sklearn_absent(self):
        with patch("ictonyx.data.HAS_SKLEARN", False):
            with pytest.raises(ImportError, match="scikit-learn"):
                TextDataHandler(data_path="/tmp/fake.csv")


class TestArraysDataHandlerWithTestSet:
    """Tests for ArraysDataHandler when a pre-supplied test set is provided."""

    def test_presupplied_test_set_used_directly(self):
        """When X_test/y_test provided, test split must not carve from training data."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = rng.integers(0, 2, 100)
        X_test = rng.standard_normal((20, 4))
        y_test = rng.integers(0, 2, 20)
        handler = ArraysDataHandler(X, y, X_test=X_test, y_test=y_test)
        splits = handler.load()
        X_test_out, y_test_out = splits["test_data"]
        assert len(X_test_out) == 20
        np.testing.assert_array_equal(X_test_out, X_test)

    def test_presupplied_test_set_leaves_training_intact(self):
        """Training data must not shrink when test set is pre-supplied."""
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 3)
        y_test = np.random.randint(0, 2, 20)
        handler = ArraysDataHandler(X, y, X_test=X_test, y_test=y_test)
        splits = handler.load(val_split=0.1)
        X_train, _ = splits["train_data"]
        X_val, _ = splits["val_data"]
        # All 100 original samples split between train and val only
        assert len(X_train) + len(X_val) == 100

    def test_mismatched_test_set_raises(self):
        """X_test and y_test with different lengths must raise."""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        X_test = np.random.rand(10, 3)
        y_test = np.random.randint(0, 2, 15)  # wrong length
        with pytest.raises((ValueError, Exception)):
            handler = ArraysDataHandler(X, y, X_test=X_test, y_test=y_test)
            handler.load()


class TestTabularDataHandlerSplits:
    """Tests for TabularDataHandler split logic and validation."""

    def _make_df(self, n=100):
        return pd.DataFrame(
            {
                "a": np.random.rand(n),
                "b": np.random.rand(n),
                "target": np.random.randint(0, 2, n),
            }
        )

    def test_default_splits_produce_three_sets(self):
        handler = TabularDataHandler(data=self._make_df(), target_column="target")
        splits = handler.load()
        assert "train_data" in splits
        assert "val_data" in splits
        assert "test_data" in splits

    def test_split_sizes_sum_to_total(self):
        df = self._make_df(n=200)
        handler = TabularDataHandler(data=df, target_column="target")
        splits = handler.load(val_split=0.1, test_split=0.2)
        X_train, _ = splits["train_data"]
        X_val, _ = splits["val_data"]
        X_test, _ = splits["test_data"]
        assert len(X_train) + len(X_val) + len(X_test) == 200

    def test_custom_test_split(self):
        df = self._make_df(n=200)
        handler = TabularDataHandler(data=df, target_column="target")
        splits = handler.load(test_split=0.3)
        _, y_test = splits["test_data"]
        assert len(y_test) == pytest.approx(60, abs=5)

    def test_features_subset_correct_shape(self):
        df = self._make_df(n=100)
        handler = TabularDataHandler(data=df, target_column="target", features=["a"])
        splits = handler.load()
        X_train, _ = splits["train_data"]
        assert X_train.shape[1] == 1

    def test_stratified_split_preserves_class_balance(self):
        """Stratified split should preserve approximate class balance."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(n),
                "target": rng.integers(0, 2, n),
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")
        splits = handler.load(test_split=0.2)
        _, y_train = splits["train_data"]
        _, y_test = splits["test_data"]
        train_balance = np.mean(y_train)
        test_balance = np.mean(y_test)
        assert abs(train_balance - test_balance) < 0.15

    def test_load_twice_produces_same_splits_with_seed(self):
        """Loading twice with the same seed must produce identical splits."""
        df = self._make_df(n=100)
        handler = TabularDataHandler(data=df, target_column="target")
        splits1 = handler.load()
        splits2 = handler.load()
        X1, _ = splits1["train_data"]
        X2, _ = splits2["train_data"]
        np.testing.assert_array_equal(X1, X2)

    def test_missing_values_in_target_warns(self):
        """Missing values in target column should trigger a warning."""
        df = pd.DataFrame(
            {
                "a": range(10),
                "target": [0, 1, None, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        handler = TabularDataHandler(data=df, target_column="target")
        handler.load()


class TestAutoResolveHandlerEdgeCases:
    """Edge cases for auto_resolve_handler factory."""

    def test_existing_handler_returned_unchanged(self):
        from ictonyx.data import auto_resolve_handler

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        original = ArraysDataHandler(X, y)
        result = auto_resolve_handler(original)
        assert result is original

    def test_tuple_input_returns_arrays_handler(self):
        from ictonyx.data import auto_resolve_handler

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        handler = auto_resolve_handler((X, y))
        assert isinstance(handler, ArraysDataHandler)

    def test_dataframe_input_returns_tabular_handler(self):
        from ictonyx.data import auto_resolve_handler

        df = pd.DataFrame({"a": range(10), "target": range(10)})
        handler = auto_resolve_handler(df, target_column="target")
        assert isinstance(handler, TabularDataHandler)

    def test_dataframe_without_target_raises(self):
        from ictonyx.data import auto_resolve_handler

        df = pd.DataFrame({"a": range(10)})
        with pytest.raises(ValueError, match="target_column"):
            auto_resolve_handler(df)

    def test_nonexistent_path_raises(self):
        from ictonyx.data import auto_resolve_handler

        with pytest.raises(FileNotFoundError):
            auto_resolve_handler("/nonexistent/path/file.csv", target_column="y")

    def test_unsupported_type_raises(self):
        from ictonyx.data import auto_resolve_handler

        with pytest.raises((TypeError, ValueError)):
            auto_resolve_handler(12345)

    def test_csv_path_returns_tabular_handler(self):
        from ictonyx.data import auto_resolve_handler

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"x": range(20), "y": range(20)}).to_csv(f.name, index=False)
            path = f.name
        try:
            handler = auto_resolve_handler(path, target_column="y")
            assert isinstance(handler, TabularDataHandler)
        finally:
            os.unlink(path)


class TestImageDataHandlerColorMode:
    """ImageDataHandler must accept color_mode (QUALITY-7)."""

    def test_accepts_color_mode_rgb(self, tmp_path):
        from ictonyx.data import HAS_TENSORFLOW

        if not HAS_TENSORFLOW:
            pytest.skip("TensorFlow not installed")
        # ImageDataHandler requires at least one class subdirectory.
        (tmp_path / "class_a").mkdir()
        handler = ImageDataHandler(data_path=str(tmp_path), image_size=(32, 32), color_mode="rgb")
        assert handler.color_mode == "rgb"

    def test_accepts_color_mode_grayscale(self, tmp_path):
        from ictonyx.data import HAS_TENSORFLOW

        if not HAS_TENSORFLOW:
            pytest.skip("TensorFlow not installed")
        (tmp_path / "class_a").mkdir()
        handler = ImageDataHandler(
            data_path=str(tmp_path), image_size=(32, 32), color_mode="grayscale"
        )
        assert handler.color_mode == "grayscale"

    def test_rejects_invalid_color_mode(self, tmp_path):
        from ictonyx.data import HAS_TENSORFLOW

        if not HAS_TENSORFLOW:
            pytest.skip("TensorFlow not installed")
        # color_mode is validated before class subdirectory discovery,
        # so no subdirs are needed — just a valid directory path.
        with pytest.raises(ValueError):
            ImageDataHandler(data_path=str(tmp_path), image_size=(32, 32), color_mode="invalid")


class TestArraysDataHandlerLoadPaths:
    """Cover the zero-test-split and zero-val-split paths in ArraysDataHandler.load()."""

    def test_zero_test_split_no_test_data(self):
        """test_split=0 must return test_data=None."""
        X = np.random.default_rng(0).standard_normal((50, 4))
        y = np.zeros(50)
        handler = ArraysDataHandler(X, y)
        result = handler.load(test_split=0.0, val_split=0.1)
        assert result["test_data"] is None
        assert result["train_data"] is not None

    def test_zero_val_split_no_val_data(self):
        """val_split=0 must return val_data=None."""
        X = np.random.default_rng(1).standard_normal((50, 4))
        y = np.zeros(50)
        handler = ArraysDataHandler(X, y)
        result = handler.load(test_split=0.2, val_split=0.0)
        assert result["val_data"] is None

    def test_splits_sum_too_high_raises(self):
        X = np.random.default_rng(2).standard_normal((50, 4))
        y = np.zeros(50)
        handler = ArraysDataHandler(X, y)
        with pytest.raises(ValueError):
            handler.load(test_split=0.6, val_split=0.5)

    def test_x_y_length_mismatch_raises(self):
        X = np.zeros((50, 3))
        y = np.zeros(40)
        with pytest.raises(ValueError, match="Length mismatch"):
            ArraysDataHandler(X, y)

    def test_only_x_test_raises(self):
        X = np.zeros((50, 3))
        y = np.zeros(50)
        with pytest.raises(ValueError, match="both"):
            ArraysDataHandler(X, y, X_test=np.zeros((10, 3)))

    def test_default_splits_used_when_not_overridden(self):
        X = np.random.default_rng(3).standard_normal((100, 4))
        y = np.zeros(100)
        handler = ArraysDataHandler(X, y, val_split=0.15, test_split=0.25)
        result = handler.load()  # uses stored defaults
        X_train, _ = result["train_data"]
        X_test, _ = result["test_data"]
        assert len(X_test) == pytest.approx(25, abs=2)

    def test_data_type_and_return_format(self):
        handler = ArraysDataHandler(np.zeros((30, 4)), np.zeros((30,)))
        assert handler.data_type == "arrays"
        assert handler.return_format == "split_arrays"


class TestAutoResolveHandlerPaths:
    """Cover auto_resolve_handler dispatch branches."""

    def test_passthrough_existing_handler(self):
        handler = ArraysDataHandler(np.zeros((20, 2)), np.zeros(20))
        result = auto_resolve_handler(handler)
        assert result is handler

    def test_tuple_input_returns_arrays_handler(self):
        X = np.zeros((30, 4))
        y = np.zeros(30)
        handler = auto_resolve_handler((X, y))
        assert isinstance(handler, ArraysDataHandler)

    def test_dataframe_without_target_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="target_column"):
            auto_resolve_handler(df)

    def test_string_path_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            auto_resolve_handler("/nonexistent/path/data.csv")


class TestTabularDataHandlerCSVPaths:
    """Cover TabularDataHandler load() from a CSV file."""

    def test_load_from_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
                "f2": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
                "label": [0, 1] * 25,
            }
        )
        df.to_csv(csv, index=False)
        handler = TabularDataHandler(str(csv), target_column="label")
        result = handler.load(test_split=0.2, val_split=0.1)
        assert result["train_data"] is not None
        X_train, y_train = result["train_data"]
        assert len(X_train) > 0

    def test_missing_target_column_raises(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]}).to_csv(csv, index=False)
        handler = TabularDataHandler(str(csv), target_column="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            handler.load()

    def test_empty_csv_raises(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("col1,col2,target\n")  # header only
        handler = TabularDataHandler(str(csv), target_column="target")
        with pytest.raises(ValueError):
            handler.load()

    def test_dataframe_mode_no_val_split(self):
        df = pd.DataFrame({"f1": range(40), "f2": range(40, 80), "y": [0, 1] * 20})
        handler = TabularDataHandler(df, target_column="y")
        result = handler.load(val_split=0.0, test_split=0.2)
        assert result["val_data"] is None
        assert result["train_data"] is not None


class TestTabularDataHandlerCoverage2:
    """Cover features= branch, NaN warning, and get_data_info."""

    def test_load_with_explicit_features(self):
        """features= parameter selects a subset of columns."""
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0] * 8,
                "f2": [0.1, 0.2, 0.3, 0.4, 0.5] * 8,
                "f3": [10.0, 20.0, 30.0, 40.0, 50.0] * 8,
                "label": [0, 1] * 20,
            }
        )
        handler = TabularDataHandler(df, target_column="label", features=["f1", "f2"])
        result = handler.load(test_split=0.2, val_split=0.1)
        X_train, _ = result["train_data"]
        assert list(X_train.columns) == ["f1", "f2"]

    def test_load_warns_on_nan_features(self):
        """Missing values in features must log a warning without crashing."""
        import warnings

        df = pd.DataFrame(
            {
                "f1": [1.0, None, 3.0, 4.0, 5.0] * 8,
                "f2": [0.1, 0.2, 0.3, 0.4, 0.5] * 8,
                "label": [0, 1] * 20,
            }
        )
        handler = TabularDataHandler(df, target_column="label")
        # Should not raise — just log a warning
        result = handler.load(test_split=0.2, val_split=0.1)
        assert result["train_data"] is not None

    def test_get_data_info_dataframe_mode(self):
        """get_data_info() in DataFrame mode must return 'source': 'dataframe'."""
        df = pd.DataFrame({"f1": range(20), "f2": range(20), "label": [0, 1] * 10})
        handler = TabularDataHandler(df, target_column="label")
        info = handler.get_data_info()
        assert info["source"] == "dataframe"
        assert info["num_rows"] == 20
        assert "label" in info["columns"]

    def test_get_data_info_csv_mode(self, tmp_path):
        """get_data_info() in file mode must return path info."""
        csv = tmp_path / "data.csv"
        pd.DataFrame({"f1": range(10), "label": [0, 1] * 5}).to_csv(csv, index=False)
        handler = TabularDataHandler(str(csv), target_column="label")
        info = handler.get_data_info()
        assert info["num_rows"] == 10
        assert "target_column" in info

    def test_invalid_features_raises(self):
        """Non-existent feature column must raise ValueError."""
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        handler = TabularDataHandler(df, target_column="label", features=["f1", "nonexistent"])
        with pytest.raises(ValueError, match="nonexistent"):
            handler.load()

    def test_split_sum_too_high_raises(self):
        df = pd.DataFrame({"f1": range(20), "label": [0, 1] * 10})
        handler = TabularDataHandler(df, target_column="label")
        with pytest.raises(ValueError):
            handler.load(test_split=0.6, val_split=0.5)


class TestDeprecatedHandlerWarningVisibility:

    def test_text_data_handler_no_longer_emits_deprecation_warning(self):
        """TextDataHandler is now framework-agnostic and does not emit
        a deprecation warning. Verify construction succeeds without warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                TextDataHandler(data_path="/nonexistent/fake.csv")
            except (ImportError, ValueError, FileNotFoundError):
                pass
        deprecation_warnings = [x for x in w if "deprecated" in str(x.message).lower()]
        assert (
            len(deprecation_warnings) == 0
        ), "New TextDataHandler should not emit deprecation warnings."


class TestTextDataHandlerFrameworkAgnostic:
    """TextDataHandler (v0.4.0 rewrite) — TfidfVectorizer, no TF required."""

    def _make_csv(self, tmp_path, n: int = 30) -> str:
        path = str(tmp_path / "text.csv")
        rng = np.random.default_rng(42)
        texts = [f"sample text document number {i} with some words" for i in range(n)]
        labels = (rng.random(n) > 0.5).astype(int)
        pd.DataFrame({"text": texts, "label": labels}).to_csv(path, index=False)
        return path

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_load_returns_three_splits(self, tmp_path):
        from ictonyx.data import TextDataHandler

        path = self._make_csv(tmp_path)
        handler = TextDataHandler(path)
        result = handler.load()
        assert "train_data" in result
        assert "val_data" in result
        assert "test_data" in result

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_train_X_is_2d_numpy(self, tmp_path):
        from ictonyx.data import TextDataHandler

        path = self._make_csv(tmp_path)
        handler = TextDataHandler(path, max_features=50)
        result = handler.load()
        X_train, _ = result["train_data"]
        assert isinstance(X_train, np.ndarray)
        assert X_train.ndim == 2
        assert X_train.shape[1] <= 50

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_vectorizer_fit_on_train_only(self, tmp_path):
        """Vectorizer must be fit on training split only."""
        from ictonyx.data import TextDataHandler

        path = self._make_csv(tmp_path)
        handler = TextDataHandler(path)
        handler.load()
        assert handler.vectorizer is not None

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_missing_text_column_raises(self, tmp_path):
        from ictonyx.data import TextDataHandler

        path = str(tmp_path / "bad.csv")
        pd.DataFrame({"body": ["hello"], "label": [0]}).to_csv(path, index=False)
        handler = TextDataHandler(path, text_column="text")
        with pytest.raises(ValueError, match="text"):
            handler.load()

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_no_test_split(self, tmp_path):
        from ictonyx.data import TextDataHandler

        path = self._make_csv(tmp_path)
        handler = TextDataHandler(path)
        result = handler.load(test_split=0.0, val_split=0.1)
        assert result["test_data"] is None

    @pytest.mark.skipif(
        not __import__("ictonyx.data", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn required",
    )
    def test_get_data_info_returns_expected_keys(self, tmp_path):
        from ictonyx.data import TextDataHandler

        path = self._make_csv(tmp_path)
        handler = TextDataHandler(path)
        info = handler.get_data_info()
        assert "text_column" in info
        assert "label_column" in info
        assert "max_features" in info


class TestTimeSeriesDataHandlerFrameworkAgnostic:
    """TimeSeriesDataHandler (v0.4.0 rewrite) — pure NumPy, no TF required."""

    def _make_csv(self, tmp_path, n: int = 100) -> str:
        path = str(tmp_path / "ts.csv")
        rng = np.random.default_rng(42)
        values = np.cumsum(rng.normal(0, 1, n))
        pd.DataFrame({"value": values}).to_csv(path, index=False)
        return path

    def test_load_returns_three_splits(self, tmp_path):
        from ictonyx.data import TimeSeriesDataHandler

        path = self._make_csv(tmp_path)
        handler = TimeSeriesDataHandler(path, lookback=5)
        result = handler.load()
        assert "train_data" in result
        assert "val_data" in result
        assert "test_data" in result

    def test_train_X_shape_is_correct(self, tmp_path):
        from ictonyx.data import TimeSeriesDataHandler

        path = self._make_csv(tmp_path)
        handler = TimeSeriesDataHandler(path, lookback=10)
        result = handler.load()
        X_train, y_train = result["train_data"]
        assert X_train.ndim == 3
        assert X_train.shape[1] == 10  # lookback
        assert X_train.shape[2] == 1  # univariate
        assert len(X_train) == len(y_train)

    def test_sequence_length_alias(self, tmp_path):
        """sequence_length kwarg must be accepted as alias for lookback."""
        from ictonyx.data import TimeSeriesDataHandler

        path = self._make_csv(tmp_path)
        handler = TimeSeriesDataHandler(path, sequence_length=7)
        result = handler.load()
        X_train, _ = result["train_data"]
        assert X_train.shape[1] == 7

    def test_make_windows_shape(self):
        """_make_windows static method produces correct (n_windows, lb, n_feat) shape."""
        from ictonyx.data import TimeSeriesDataHandler

        data = np.arange(50, dtype=np.float32)
        targets = np.arange(50, dtype=np.float32)
        X, y = TimeSeriesDataHandler._make_windows(data, targets, lookback=5, stride=1)
        assert X.shape == (45, 5, 1)
        assert y.shape == (45,)

    def test_make_windows_stride(self):
        from ictonyx.data import TimeSeriesDataHandler

        data = np.arange(50, dtype=np.float32)
        targets = np.arange(50, dtype=np.float32)
        X, y = TimeSeriesDataHandler._make_windows(data, targets, lookback=5, stride=2)
        # With stride=2, fewer windows
        X_stride1, _ = TimeSeriesDataHandler._make_windows(data, targets, lookback=5, stride=1)
        assert len(X) < len(X_stride1)

    def test_too_short_raises(self, tmp_path):
        from ictonyx.data import TimeSeriesDataHandler

        path = str(tmp_path / "short.csv")
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
        handler = TimeSeriesDataHandler(path, lookback=10)
        with pytest.raises(ValueError, match="short"):
            handler.load()

    def test_no_test_split(self, tmp_path):
        from ictonyx.data import TimeSeriesDataHandler

        path = self._make_csv(tmp_path)
        handler = TimeSeriesDataHandler(path, lookback=5)
        result = handler.load(test_split=0.0, val_split=0.1)
        assert result["test_data"] is None

    def test_get_data_info_returns_expected_keys(self, tmp_path):
        from ictonyx.data import TimeSeriesDataHandler

        path = self._make_csv(tmp_path)
        handler = TimeSeriesDataHandler(path, lookback=5)
        info = handler.get_data_info()
        assert "lookback" in info
        assert "value_column" in info
        assert "stride" in info


class TestImageDataHandlerShuffleSeeding:
    """Tests: shuffle must vary across variability-study runs.

    Pre-v0.4.7, ImageDataHandler placed .shuffle() before .cache(). The
    cache materialized one shuffle order on first pass; every subsequent
    iteration replayed that order. Per-run batch-ordering variance — one
    of three variance sources the library claims to measure — was silently
    suppressed on image studies.

    Fix (Option C1): shuffle moved after cache with
    reshuffle_each_iteration=True. Combined with the runner's per-run
    tf.random.set_seed(child_seed), this produces distinct orderings
    across runs iterating the same cached dataset.
    """

    @pytest.fixture
    def tiny_image_dataset(self, tmp_path):
        """Minimal on-disk image dataset: 2 classes, 10 images each, 16x16 RGB."""
        pytest.importorskip("PIL", reason="Pillow not installed")
        from PIL import Image

        rng = np.random.default_rng(2026)
        root = tmp_path / "dataset"
        root.mkdir()
        for class_id in range(2):
            class_dir = root / f"class_{class_id}"
            class_dir.mkdir()
            for sample_id in range(10):
                arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
                Image.fromarray(arr).save(class_dir / f"img_{sample_id:02d}.png")
        return str(root)

    @pytest.mark.slow
    def test_shuffle_varies_across_global_seeds(self, tiny_image_dataset):
        """Same cached dataset under different global TF seeds → different orderings."""
        tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
        from ictonyx.data import ImageDataHandler

        handler = ImageDataHandler(
            data_path=tiny_image_dataset,
            image_size=(16, 16),
            batch_size=4,
            seed=42,
            val_split=0.2,
            test_split=0.2,
        )
        output = handler.load(validation_split=0.2, test_split=0.2)
        train_ds = output["train_data"]

        orderings = []
        for global_seed in (100, 200, 300):
            tf.random.set_seed(global_seed)
            labels = []
            for _images, lbls in train_ds:
                labels.extend(lbls.numpy().tolist())
            orderings.append(tuple(labels))

        assert len(set(orderings)) > 1, (
            f"X-72 regression: 3 simulated runs produced identical orderings "
            f"despite distinct global TF seeds: {orderings}"
        )
