"""Test data handlers."""
import pytest
import tempfile
import pandas as pd
import numpy as np
import os
from ictonyx.core import TENSORFLOW_AVAILABLE
from ictonyx.data import (
    TabularDataHandler,
    ImageDataHandler,
    TextDataHandler,
    TimeSeriesDataHandler,
    ArraysDataHandler,
    auto_resolve_handler
)

class TestTabularDataHandler:
    """Test TabularDataHandler."""
    
    def test_tabular_handler_creation(self):
        """Test creating tabular data handler."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'feature1': range(10),
                'feature2': range(10, 20),
                'target': [0, 1] * 5
            })
            df.to_csv(f.name, index=False)
            path = f.name
        
        handler = TabularDataHandler(path, target_column='target')
        
        assert handler.data_type == 'tabular'
        assert handler.return_format == 'split_arrays'
        assert handler.target_column == 'target'
        
        # Clean up
        import os
        os.unlink(path)
    
    def test_tabular_load(self):
        """Test loading tabular data."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'x1': np.random.rand(100),
                'x2': np.random.rand(100),
                'y': np.random.randint(0, 2, 100)
            })
            df.to_csv(f.name, index=False)
            path = f.name
        
        handler = TabularDataHandler(path, target_column='y')
        data = handler.load(test_split=0.2, val_split=0.1)
        
        assert 'train_data' in data
        assert 'val_data' in data
        assert 'test_data' in data
        
        X_train, y_train = data['train_data']
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

        X_train, y_train = splits['train_data']
        X_val, y_val = splits['val_data']
        X_test, y_test = splits['test_data']

        # Check sizes (approximate due to int rounding)
        assert 55 <= len(X_train) <= 65
        assert 15 <= len(X_val) <= 25
        assert 15 <= len(X_test) <= 25


class TestAutoResolveHandler:
    """Tests for the auto_resolve_handler factory function."""

    def test_resolve_dataframe(self):
        """Test that pandas DataFrames resolve to TabularDataHandler."""
        df = pd.DataFrame({'a': [1, 2], 'target': [0, 1]})
        handler = auto_resolve_handler(df, target_column='target')

        assert isinstance(handler, TabularDataHandler)
        assert handler.input_df is not None
        assert len(handler.input_df) == 2

    def test_resolve_dataframe_missing_target(self):
        """Test error when target is missing for DataFrame."""
        df = pd.DataFrame({'a': [1, 2]})
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,target\n1,0\n2,1")
            path = f.name

        try:
            # If target_column is provided, assume Tabular
            handler = auto_resolve_handler(path, target_column='target')
            assert isinstance(handler, TabularDataHandler)
        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_resolve_image_dir(self):
        """Test resolving directory to ImageDataHandler."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy class folders
            os.makedirs(os.path.join(tmp_dir, 'class_a'))
            os.makedirs(os.path.join(tmp_dir, 'class_b'))

            # Should detect as Image handler
            handler = auto_resolve_handler(tmp_dir, image_size=(64, 64))
            assert isinstance(handler, ImageDataHandler)

    def test_resolve_ambiguous_file(self):
        """Test error when file type is ambiguous."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("data")
            path = f.name

        try:
            # No target_column, text_column, or sequence_length provided
            with pytest.raises(ValueError, match="Ambiguous file input"):
                auto_resolve_handler(path)
        finally:
            if os.path.exists(path):
                os.remove(path)