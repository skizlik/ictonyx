# ictonyx/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Union, Tuple, List, Any, Optional

def save_object(obj: Any, path: str):
    """Saves a Python object to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {path}")

def load_object(path: str) -> Any:
    """Loads a Python object from a file using pickle."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {path}")
    return obj

def train_val_test_split(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                         test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Splits data into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def setup_mlflow(experiment_name: str = "ictonyx_experiment",
                 tracking_uri: Optional[str] = None) -> str:
    """
    Sets up MLflow tracking with sensible defaults.

    Args:
        experiment_name: Name for the MLflow experiment
        tracking_uri: MLflow tracking server URI (None uses local)

    Returns:
        str: The experiment ID
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for setup_mlflow. Install with: pip install mlflow")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new MLflow experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing MLflow experiment: {experiment_name}")

    return experiment_id