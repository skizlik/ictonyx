# ictonyx/utils.py

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import settings

logger = settings.logger


def save_object(obj: Any, path: str):
    """Serialize a Python object to disk using pickle.

    Warning:
        The resulting file is a pickle file. Pickle files can execute
        arbitrary code when loaded. Only share with trusted recipients
        and only load files from trusted sources.

    Args:
        obj: Any picklable Python object.
        path: Destination file path (typically ``.pkl``).
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {path}")


def load_object(path: str) -> Any:
    """Deserialize a Python object from a pickle file.

    Warning:
        Pickle deserialization can execute arbitrary code. Only load
        files created by :func:`save_object` or received from a fully
        trusted source.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Object file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Object loaded from {path}")
    return obj


def train_val_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    split_basis: str = "auto",
) -> Tuple:
    """Split arrays into training, validation, and test sets.

    Performs two sequential splits: first separating the test set, then
    splitting the remainder into train and validation.

    Args:
        X: Feature array or DataFrame.
        y: Label array or Series.
        test_size: Fraction of the original dataset reserved for test.
            Default 0.2.
        val_size: Fraction reserved for validation. See ``split_basis``
            for interpretation. Default 0.2.
        random_state: Seed for reproducibility. Default 42.
        split_basis: Controls how ``val_size`` is interpreted.

            - ``"original"``: ``val_size`` is a fraction of the original
              dataset. ``test_size=0.2, val_size=0.2`` produces a
              60/20/20 train/val/test split. Matches the behavior of
              :class:`ArraysDataHandler` and :class:`ImageDataHandler`.

            - ``"remainder"``: ``val_size`` is a fraction of the
              post-test remainder (legacy behavior through v0.4.6).
              ``test_size=0.2, val_size=0.2`` produces a 64/16/20 split.

            - ``"auto"`` (default in v0.4.7): emit a ``DeprecationWarning``
              and behave as ``"remainder"`` for backward compatibility.
              The default changes to ``"original"`` in v0.5.0.

    Returns:
        Tuple of ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError(
            "scikit-learn is required for train_val_test_split. "
            "Install with: pip install ictonyx[sklearn]"
        )

    if split_basis not in ("original", "remainder", "auto"):
        raise ValueError(
            f"split_basis must be 'original', 'remainder', or 'auto'; " f"got {split_basis!r}."
        )

    if split_basis == "auto":
        warnings.warn(
            "train_val_test_split: val_size is currently interpreted as "
            "a fraction of the post-test remainder, producing different "
            "split sizes than ArraysDataHandler and ImageDataHandler for "
            "the same numerical arguments. In v0.5.0, the default will "
            "change to fraction-of-original to match the data handlers. "
            "Pass split_basis='original' to opt in now, or "
            "split_basis='remainder' to preserve legacy behavior silently.",
            DeprecationWarning,
            stacklevel=2,
        )
        split_basis = "remainder"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if split_basis == "original":
        adj_val_size = val_size / (1 - test_size) if test_size < 1 else val_size
    else:  # split_basis == "remainder"
        adj_val_size = val_size

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=adj_val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def setup_mlflow(
    experiment_name: str = "ictonyx_experiment",
    tracking_uri: Optional[str] = None,
    autolog: bool = False,
    tags: Optional[Dict[str, str]] = None,
) -> str:
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
        logger.info(f"Created new MLflow experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        _exp = mlflow.get_experiment_by_name(experiment_name)
        if _exp is None:
            raise RuntimeError(
                f"MLflow experiment '{experiment_name}' not found after creation attempt."
            )
        experiment_id = _exp.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")

    if autolog:
        try:
            import mlflow.sklearn

            mlflow.sklearn.autolog()
        except Exception:
            pass
        try:
            import mlflow.tensorflow

            mlflow.tensorflow.autolog()
        except Exception:
            pass

    if tags:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    return experiment_id
