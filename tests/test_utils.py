"""Test utility functions."""
import pytest
import tempfile
import os
from ictonyx.utils import save_object, load_object, train_val_test_split
import numpy as np
import pandas as pd


def test_save_load_object():
    """Test saving and loading objects."""
    test_data = {'a': 1, 'b': [2, 3, 4], 'c': 'test'}
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
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
    assert len(X_val) == 16    # 100 * 0.8 * 0.2
    assert len(X_test) == 20   # 100 * 0.2
    
    # Check no overlap
    assert len(X_train) + len(X_val) + len(X_test) == 100
