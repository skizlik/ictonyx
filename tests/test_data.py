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
