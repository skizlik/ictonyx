# ictonyx/data.py

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .settings import logger

# Define public API
__all__ = [
    "DataHandler",
    "ImageDataHandler",
    "TabularDataHandler",
    "TextDataHandler",
    "TimeSeriesDataHandler",
    "ArraysDataHandler",  # NEW
    "auto_resolve_handler",  # NEW
]

# Optional TensorFlow imports
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

# Optional TensorFlow preprocessing imports
try:
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    HAS_TF_PREPROCESSING = True
except (ImportError, AttributeError):
    TimeseriesGenerator = None
    Tokenizer = None
    pad_sequences = None
    HAS_TF_PREPROCESSING = False

# Optional matplotlib for data visualization
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

# Optional sklearn for data splitting and text vectorisation
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    train_test_split = None
    TfidfVectorizer = None


class DataHandler(ABC):
    """Abstract base class for all data handlers.

    Provides a consistent interface regardless of data source.
    Handlers that load from the file system should inherit from
    :class:`FileDataHandler` instead of this class directly.

    All subclasses must implement :meth:`load`, :attr:`data_type`,
    and :attr:`return_format`. The returned dict from :meth:`load`
    must have keys ``'train_data'``, ``'val_data'``, and
    ``'test_data'``, each a ``(X, y)`` tuple or ``None``.
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> Dict[str, Any]:
        """Load and split the dataset.

        Returns:
            Dict with keys ``'train_data'``, ``'val_data'``,
            ``'test_data'``.
        """
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """String identifier for the data type this handler processes."""
        pass

    @property
    @abstractmethod
    def return_format(self) -> str:
        """String describing the format returned by :meth:`load`."""
        pass

    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "data_type": self.data_type,
            "return_format": self.return_format,
        }


class FileDataHandler(DataHandler, ABC):
    """Abstract base for file-system-backed data handlers.

    Owns path existence validation. ``ImageDataHandler``,
    ``TabularDataHandler``, ``TextDataHandler``, and
    ``TimeSeriesDataHandler`` inherit from this class.

    Args:
        data_path: Path to the data source (file or directory).

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._validate_data_path()

    def _validate_data_path(self):
        """Validate that the data path exists."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def get_data_info(self) -> Dict[str, Any]:
        info = super().get_data_info()
        info["data_path"] = self.data_path
        info["path_exists"] = os.path.exists(self.data_path)
        return info


class ImageDataHandler(FileDataHandler):
    """Data handler for image classification datasets in directory format.

    Expects a directory structure where each subdirectory is a class
    label containing image files. Images are loaded, resized, normalized
    to [0, 1], and split into train/val/test sets.

    Requires TensorFlow for image preprocessing. Install with
    ``pip install tensorflow``.

    Args:
        data_path: Path to the root directory containing class subdirectories.
        image_size: Target ``(height, width)`` for resizing, e.g. ``(64, 64)``.
        color_mode: ``'rgb'`` (3 channels) or ``'grayscale'`` (1 channel).
            Default ``'rgb'``.

    Raises:
        ValueError: If the path is not a directory, contains no class
            subdirectories, or no valid image files are found.
    """

    @property
    def data_type(self) -> str:
        return "image"

    @property
    def return_format(self) -> str:
        return "tf_datasets"

    def __init__(
        self,
        data_path: str,
        image_size: Tuple[int, int],
        batch_size: int = 32,
        seed: int = 42,
        color_mode: str = "rgb",
        val_split: float = 0.2,
        test_split: float = 0.1,
    ):
        """
        Initialize the image data handler.

        Args:
            data_path: Path to directory containing class subdirectories.
            image_size: ``(height, width)`` tuple for resizing images.
            batch_size: Batch size for the datasets. Default 32.
            seed: Random seed for reproducibility. Default 42.
            color_mode: ``'rgb'`` (default) or ``'grayscale'``.
            val_split: Fraction of data for validation. Default 0.2.
            test_split: Fraction of data for test. Default 0.1.
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for ImageDataHandler. "
                "Install with: pip install tensorflow"
            )

        super().__init__(data_path)

        if not os.path.isdir(self.data_path):
            raise ValueError("ImageDataHandler requires a directory path with class subdirectories")

        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        if color_mode not in ("rgb", "grayscale"):
            raise ValueError(f"color_mode must be 'rgb' or 'grayscale', got '{color_mode}'.")
        self.color_mode = color_mode
        self.val_split = val_split
        self.test_split = test_split

        # Discover class names
        try:
            self.class_names = sorted(
                [
                    d
                    for d in os.listdir(self.data_path)
                    if os.path.isdir(os.path.join(self.data_path, d)) and not d.startswith(".")
                ]
            )
        except PermissionError:
            raise PermissionError(f"Cannot read directory: {self.data_path}")

        if not self.class_names:
            raise ValueError(f"No class subdirectories found in {self.data_path}")

        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")

    def _get_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """Helper to collect all file paths and their corresponding integer labels."""
        all_image_paths = []
        all_labels = []
        class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Common image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)

            try:
                files = os.listdir(class_path)
            except PermissionError:
                raise PermissionError(f"Cannot read class directory: {class_path}")

            # Filter for valid image files
            image_files = [f for f in files if os.path.splitext(f.lower())[1] in valid_extensions]

            if not image_files:
                logger.warning(f"No valid image files found in {class_path}")
                continue

            image_paths = [os.path.join(class_path, fname) for fname in image_files]
            all_image_paths.extend(image_paths)
            all_labels.extend([class_to_idx[class_name]] * len(image_paths))

        if not all_image_paths:
            raise ValueError(f"No valid image files found in any class directories")

        return all_image_paths, all_labels

    def _validate_image_files(self, file_paths: List[str]) -> None:
        """Scan all image paths before building the dataset.

        Checks that every file can be opened with PIL. Raises
        ``DataValidationError`` listing all unreadable files if any are found.
        This prevents corrupt images from being silently replaced with
        zero-filled tensors during training.

        Requires ``Pillow`` (``pip install Pillow``). If Pillow is not
        installed, skips validation with a warning.

        Args:
            file_paths: List of absolute paths to image files.

        Raises:
            DataValidationError: If any file cannot be opened as an image.
        """
        try:
            from PIL import Image as _PILImage
        except ImportError:
            logger.warning(
                "Pillow not installed — skipping image file validation. "
                "Install with: pip install Pillow"
            )
            return

        failed = []
        for path in file_paths:
            try:
                with _PILImage.open(path) as im:
                    im.verify()
            except Exception:
                failed.append(path)

        if failed:
            from .exceptions import DataValidationError

            sample = failed[:5]
            tail = f" ... and {len(failed) - 5} more" if len(failed) > 5 else ""
            raise DataValidationError(
                f"{len(failed)} image file(s) could not be opened:{tail}\n"
                + "\n".join(f"  {p}" for p in sample)
                + "\nFix or remove these files before training."
            )

    def _preprocess_image(self, file_path, label):
        """Load, decode, resize, and normalise a single image.

        Uses ``tf.image.decode_image`` which handles JPEG, PNG, BMP, and GIF
        automatically. This is the only pattern that works correctly inside
        ``tf.data.Dataset.map()`` — Python try/except blocks cannot intercept
        TF op errors in graph execution mode.
        """
        img = tf.io.read_file(file_path)
        # decode_image handles JPEG, PNG, BMP, GIF.
        # expand_animations=False collapses GIF frames to the first frame only.
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        # decode_image does not set static shape; set it explicitly so
        # downstream ops (resize) know the rank.
        img.set_shape([None, None, 3])
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, self.image_size)
        img = img / 255.0
        return img, label

    def load(
        self, validation_split: float = 0.2, test_split: float = 0.1, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Loads and splits the dataset into training, validation, and test sets,
        ensuring stratified sampling.

        Args:
            validation_split: Proportion of data for validation
            test_split: Proportion of data for testing

        Returns:
            Dict with 'train_data', 'val_data', 'test_data' as tf.data.Dataset objects
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for data splitting. "
                "Install with: pip install scikit-learn"
            )

        if validation_split + test_split >= 1.0:
            raise ValueError("Sum of validation_split and test_split must be < 1.0")

        if validation_split < 0 or test_split < 0:
            raise ValueError("Split values must be non-negative")

        all_image_paths, all_labels = self._get_image_paths_and_labels()
        self._validate_image_files(all_image_paths)
        total_files = len(all_image_paths)

        logger.info(f"Total images found: {total_files}")

        # Check class distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for class_idx, count in zip(unique_labels, counts):
            class_name = self.class_names[class_idx]
            logger.info(f"  {class_name}: {count} images")

        # Split into training and test sets first, with stratification
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                all_image_paths,
                all_labels,
                test_size=test_split,
                random_state=self.seed,
                stratify=all_labels,
            )
        else:
            X_train, X_test, y_train, y_test = all_image_paths, [], all_labels, []

        # Split training into train/validation
        X_val, y_val = [], []
        if validation_split > 0 and len(X_train) > 1:
            # Recalculate validation split based on remaining training data
            adj_val_split = (
                validation_split / (1 - test_split) if test_split > 0 else validation_split
            )

            if adj_val_split < 1.0:  # Only split if we won't take everything
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=adj_val_split,
                    random_state=self.seed,
                    stratify=y_train,
                )

        # Convert the lists of paths and labels into TensorFlow datasets
        def create_tf_dataset(paths, labels, shuffle=True):
            if not paths:
                return None

            path_ds = tf.data.Dataset.from_tensor_slices(paths)
            label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

            ds = tf.data.Dataset.zip((path_ds, label_ds))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(paths), seed=self.seed)

            ds = ds.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(self.batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            return ds

        train_ds = create_tf_dataset(X_train, y_train, shuffle=True)
        val_ds = create_tf_dataset(X_val, y_val, shuffle=False) if X_val else None
        test_ds = create_tf_dataset(X_test, y_test, shuffle=False) if X_test else None

        logger.info(
            f"Data splits created - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return {"train_data": train_ds, "val_data": val_ds, "test_data": test_ds}

    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the image dataset."""
        info = super().get_data_info()

        try:
            all_paths, all_labels = self._get_image_paths_and_labels()
            unique_labels, counts = np.unique(all_labels, return_counts=True)

            class_distribution = {
                self.class_names[label]: count for label, count in zip(unique_labels, counts)
            }

            info.update(
                {
                    "num_classes": len(self.class_names),
                    "class_names": self.class_names,
                    "total_images": len(all_paths),
                    "class_distribution": class_distribution,
                    "image_size": self.image_size,
                    "batch_size": self.batch_size,
                }
            )
        except Exception as e:
            info["error"] = f"Could not analyze dataset: {e}"

        return info


class TabularDataHandler(FileDataHandler):
    """Data handler for structured tabular data from CSV files or DataFrames.

    Loads data, validates columns, and splits into train/val/test sets.
    Supports both file-path and in-memory DataFrame initialization.

    Args:
        data: Path to a CSV file, or a ``pd.DataFrame``.
        target_column: Name of the column containing labels.
        features: Optional list of feature column names. If ``None``,
            all columns except ``target_column`` are used.
        sep: CSV delimiter (only used for file paths). Default ``','``.
        header: Header row number (only used for file paths). Default 0.

    Raises:
        ValueError: If ``data`` or ``target_column`` is missing, or if
            the target column is not found in the data.
        TypeError: If ``data`` is neither a string nor a DataFrame.
    """

    @property
    def data_type(self) -> str:
        return "tabular"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(
        self,
        data: Union[str, pd.DataFrame] = None,
        target_column: Optional[str] = None,
        features: Optional[List[str]] = None,
        sep: str = ",",
        header: int = 0,
        **kwargs,
    ):
        """
        Initialize the tabular data handler.

        Args:
            data: Path to CSV file OR a pandas DataFrame object.
            target_column: Name of the target column.
            features: List of feature column names (None = all except target).
            sep: CSV separator (only used if data is a path).
            header: Header row number (only used if data is a path).
            **kwargs: Captures legacy 'data_path' argument.
        """
        # ------------------------------------------------------------------
        # FIX: Backward Compatibility for data_path keyword argument
        # ------------------------------------------------------------------
        if data is None and "data_path" in kwargs:
            data = kwargs["data_path"]

        if data is None:
            raise ValueError("Must provide 'data' (or legacy 'data_path') argument.")

        if target_column is None:
            raise ValueError("Must provide 'target_column' argument.")

        # Handle path string vs DataFrame
        if isinstance(data, str):
            self.input_df = None
            super().__init__(data)
        elif isinstance(data, pd.DataFrame):
            self.input_df = data.copy()
            # Bypass file check by passing a dummy string, but relying on _validate_data_path override
            super().__init__("in_memory_dataframe")
        else:
            raise TypeError(f"data must be str path or DataFrame, got {type(data)}")

        self.target_column = target_column
        self.features = features
        self.sep = sep
        self.header = header
        self.data: Optional[pd.DataFrame] = None

    def _validate_data_path(self):
        """Override validation to skip file check if using DataFrame."""
        # Check if we are in DataFrame mode (input_df set in __init__)
        if hasattr(self, "input_df") and self.input_df is not None:
            return  # Skip check, data is already in memory

        # Otherwise, perform standard file check
        super()._validate_data_path()

    def _load_and_validate_data(self):
        """Load (if needed) and validate the tabular data."""

        # 1. Load Data
        if self.input_df is not None:
            self.data = self.input_df
        else:
            try:
                self.data = pd.read_csv(self.data_path, sep=self.sep, header=self.header)
            except pd.errors.EmptyDataError:
                raise ValueError(f"CSV file is empty: {self.data_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load CSV file: {e}")

        if self.data.empty:
            raise ValueError("Loaded dataset is empty")

        # 2. Validate Columns
        if self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available: {list(self.data.columns)}"
            )

        # 3. Validate Features
        if self.features:
            missing = set(self.features) - set(self.data.columns)
            if missing:
                raise ValueError(f"Features not found in data: {missing}")

        logger.info(f"Loaded tabular data: {len(self.data)} rows, {len(self.data.columns)} columns")

    # ... (Keep existing load() and get_data_info() methods identical to previous version) ...
    def load(
        self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42, **kwargs: Any
    ) -> Dict[str, Any]:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for data splitting.")

        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        self._load_and_validate_data()
        assert self.data is not None

        # Prepare features and target
        if self.features:
            X = self.data[self.features]
        else:
            X = self.data.drop(columns=[self.target_column])

        y = self.data[self.target_column]

        # Check for missing values in features
        if X.isnull().any().any():
            null_cols = X.isnull().sum()
            null_cols = null_cols[null_cols > 0]
            total_missing = int(X.isnull().sum().sum())
            logger.warning(
                f"Features contain {total_missing} missing value(s) across "
                f"{len(null_cols)} column(s): {null_cols.to_dict()}. "
                f"Consider imputing before training — many models will error "
                f"or produce incorrect results on NaN inputs."
            )

        if y.isnull().any():
            logger.warning(f"{y.isnull().sum()} missing values in target column")

        # Split data
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, None, y, None

        X_val, y_val = None, None
        if val_split > 0 and len(X_train) > 1:
            adj_val_split = val_split / (1 - test_split) if test_split > 0 else val_split
            if adj_val_split < 1.0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=adj_val_split, random_state=random_state
                )

        return {
            "train_data": (X_train, y_train),
            "val_data": (X_val, y_val) if X_val is not None else None,
            "test_data": (X_test, y_test) if X_test is not None else None,
        }

    def get_data_info(self) -> Dict[str, Any]:
        """Return metadata about the tabular dataset.

        When operating from an in-memory DataFrame, returns a ``source`` key
        with ``'dataframe'`` instead of a ``data_path`` / ``path_exists`` pair,
        which would otherwise show a nonsensical dummy path.
        """
        # Build base info without the FileDataHandler dummy-path fields.
        if self.input_df is not None:
            # In-memory DataFrame mode — FileDataHandler.get_data_info() would
            # return data_path="in_memory_dataframe", path_exists=False, which
            # is confusing. Build our own base dict instead.
            info: Dict[str, Any] = {
                "data_type": self.data_type,
                "return_format": self.return_format,
                "source": "dataframe",
            }
        else:
            # File-backed mode — use FileDataHandler's info normally.
            info = super().get_data_info()

        try:
            if self.data is None:
                self._load_and_validate_data()
            assert self.data is not None

            info.update(
                {
                    "num_rows": len(self.data),
                    "num_columns": len(self.data.columns),
                    "target_column": self.target_column,
                    "columns": list(self.data.columns),
                }
            )
        except Exception as e:
            info["error"] = str(e)

        return info


class TextDataHandler(FileDataHandler):
    """Framework-agnostic text classification data handler.

    Converts raw text to TF-IDF feature vectors compatible with all three
    model wrappers (Keras, PyTorch, sklearn). Does not require TensorFlow.

    Requires scikit-learn: ``pip install ictonyx[sklearn]``

    Args:
        data_path: Path to a CSV file containing text and label columns.
        text_column: Name of the column containing raw text. Default
            ``'text'``.
        label_column: Name of the column containing labels. Default
            ``'label'``.
        max_features: Maximum number of TF-IDF features. Default 10000.
        val_split: Fraction of training data to use for validation.
            Default 0.1.
        test_split: Fraction of all data to hold out for testing.
            Default 0.2.
    """

    @property
    def data_type(self) -> str:
        return "text"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(
        self,
        data_path: str,
        text_column: str = "text",
        label_column: str = "label",
        max_features: int = 10000,
        val_split: float = 0.1,
        test_split: float = 0.2,
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "TextDataHandler requires scikit-learn. "
                "Install with: pip install ictonyx[sklearn]"
            )

        super().__init__(data_path)

        if not os.path.isfile(self.data_path):
            raise ValueError("TextDataHandler requires a path to an existing CSV file.")

        self.text_column = text_column
        self.label_column = label_column
        self.max_features = max_features
        self.val_split = val_split
        self.test_split = test_split
        self.vectorizer: Optional[Any] = None

    def load(
        self,
        test_split: Optional[float] = None,
        val_split: Optional[float] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load text data, vectorise with TF-IDF, and split.

        Returns:
            Dict with ``'train_data'``, ``'val_data'``, ``'test_data'`` as
            ``(X, y)`` tuples of numpy arrays. ``X`` has shape
            ``(n_samples, max_features)``.
        """
        _test_split = test_split if test_split is not None else self.test_split
        _val_split = val_split if val_split is not None else self.val_split

        if _test_split + _val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load text data from {self.data_path}: {e}")

        for col, name in [(self.text_column, "text"), (self.label_column, "label")]:
            if col not in df.columns:
                raise ValueError(
                    f"{name} column '{col}' not found. " f"Available columns: {list(df.columns)}"
                )

        df[self.text_column] = df[self.text_column].fillna("")
        if df[self.label_column].isnull().any():
            raise ValueError(
                f"{df[self.label_column].isnull().sum()} missing values in "
                f"label column '{self.label_column}'."
            )

        texts = df[self.text_column].tolist()
        labels = df[self.label_column].values

        # First split: carve out test set
        if _test_split > 0:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                texts, labels, test_size=_test_split, random_state=random_state
            )
        else:
            X_trainval, y_trainval = texts, labels
            X_test, y_test = None, None

        # Second split: carve out val set from train
        if _val_split > 0:
            adj_val = _val_split / (1.0 - _test_split) if _test_split > 0 else _val_split
            if adj_val < 1.0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, test_size=adj_val, random_state=random_state
                )
            else:
                X_train, y_train = X_trainval, y_trainval
                X_val, y_val = None, None
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val, y_val = None, None

        # Fit TF-IDF on training text only; transform val and test
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray().astype(np.float32)
        X_val_vec = (
            self.vectorizer.transform(X_val).toarray().astype(np.float32)
            if X_val is not None
            else None
        )
        X_test_vec = (
            self.vectorizer.transform(X_test).toarray().astype(np.float32)
            if X_test is not None
            else None
        )

        logger.info(
            f"TextDataHandler: {X_train_vec.shape[0]} train, "
            f"{X_val_vec.shape[0] if X_val_vec is not None else 0} val, "
            f"{X_test_vec.shape[0] if X_test_vec is not None else 0} test samples. "
            f"Features: {X_train_vec.shape[1]}."
        )

        return {
            "train_data": (X_train_vec, y_train),
            "val_data": (X_val_vec, y_val) if X_val_vec is not None else None,
            "test_data": (X_test_vec, y_test) if X_test_vec is not None else None,
        }

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the text dataset."""
        info = super().get_data_info()
        info.update(
            {
                "text_column": self.text_column,
                "label_column": self.label_column,
                "max_features": self.max_features,
                "vectorizer": (
                    "TfidfVectorizer" if self.vectorizer is None else str(self.vectorizer)
                ),
            }
        )
        return info


class TimeSeriesDataHandler(FileDataHandler):
    """Framework-agnostic time series data handler using NumPy sliding windows.

    Creates overlapping input windows from a univariate or multivariate
    time series. Compatible with all three model wrappers (Keras, PyTorch,
    sklearn). Does not require TensorFlow.

    Args:
        data_path: Path to a CSV file containing time series data.
        lookback: Number of time steps in each input window.
        value_column: Name of the column to use as the target series.
            Default ``'value'``.
        target_column: Name of the column to predict. If ``None``, uses
            ``value_column`` shifted by one step (next-step prediction).
            Default ``None``.
        stride: Step between consecutive windows. Default 1.
        val_split: Fraction of data to use for validation (chronological).
            Default 0.1.
        test_split: Fraction of data to hold out for testing (chronological,
            taken from the end). Default 0.2.
    """

    @property
    def data_type(self) -> str:
        return "timeseries"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(
        self,
        data_path: str,
        lookback: int = 10,
        sequence_length: Optional[int] = None,
        value_column: str = "value",
        target_column: Optional[str] = None,
        stride: int = 1,
        val_split: float = 0.1,
        test_split: float = 0.2,
    ):
        # Accept sequence_length as an alias for lookback (backward compat)
        if sequence_length is not None:
            lookback = sequence_length
        super().__init__(data_path)

        if not os.path.isfile(self.data_path):
            raise ValueError("TimeSeriesDataHandler requires a path to an existing CSV file.")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}.")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")

        self.lookback = lookback
        self.value_column = value_column
        self.target_column = target_column
        self.stride = stride
        self.val_split = val_split
        self.test_split = test_split

    @staticmethod
    def _make_windows(
        data: np.ndarray,
        targets: np.ndarray,
        lookback: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create (X, y) pairs from a time series array.

        Args:
            data: Feature array of shape ``(n_timesteps, n_features)`` or
                ``(n_timesteps,)`` for univariate series.
            targets: Target array of shape ``(n_timesteps,)``.
            lookback: Input window length.
            stride: Step between consecutive windows.

        Returns:
            Tuple of ``(X, y)`` where X has shape
            ``(n_windows, lookback, n_features)`` and y has shape
            ``(n_windows,)``.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = range(lookback, len(data), stride)
        X = np.stack([data[i - lookback : i] for i in indices]).astype(np.float32)
        y = targets[list(indices)].astype(np.float32)
        return X, y

    def load(
        self,
        test_split: Optional[float] = None,
        val_split: Optional[float] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load CSV, build sliding windows, and split chronologically.

        Splits are chronological (not shuffled) to avoid data leakage.
        ``random_state`` is accepted for API compatibility but not used.

        Returns:
            Dict with ``'train_data'``, ``'val_data'``, ``'test_data'`` as
            ``(X, y)`` tuples. X has shape
            ``(n_windows, lookback, n_features)``.
        """
        _test_split = test_split if test_split is not None else self.test_split
        _val_split = val_split if val_split is not None else self.val_split

        if _test_split + _val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load time series data from {self.data_path}: {e}")

        if self.value_column not in df.columns:
            raise ValueError(
                f"value_column '{self.value_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Build feature array and target array
        feature_data = df[self.value_column].values.astype(np.float32)

        target_col = self.target_column or self.value_column
        if target_col not in df.columns:
            raise ValueError(
                f"target_column '{target_col}' not found. " f"Available columns: {list(df.columns)}"
            )
        target_data = df[target_col].values.astype(np.float32)

        n = len(feature_data)
        if n < self.lookback + 1:
            raise ValueError(
                f"Dataset too short ({n} rows) for lookback={self.lookback}. "
                f"Need at least {self.lookback + 1} rows."
            )

        # Chronological split indices
        test_size = int(n * _test_split)
        val_size = int(n * _val_split)
        train_size = n - test_size - val_size

        if train_size < self.lookback + 1:
            raise ValueError(
                f"Training split too small ({train_size} rows) for " f"lookback={self.lookback}."
            )

        feat_train = feature_data[:train_size]
        tgt_train = target_data[:train_size]

        feat_val = feature_data[train_size : train_size + val_size] if val_size > 0 else None
        tgt_val = target_data[train_size : train_size + val_size] if val_size > 0 else None

        feat_test = feature_data[train_size + val_size :] if test_size > 0 else None
        tgt_test = target_data[train_size + val_size :] if test_size > 0 else None

        X_train, y_train = self._make_windows(feat_train, tgt_train, self.lookback, self.stride)

        val_data = None
        if feat_val is not None and len(feat_val) > self.lookback:
            X_val, y_val = self._make_windows(feat_val, tgt_val, self.lookback, self.stride)
            val_data = (X_val, y_val)

        test_data = None
        if feat_test is not None and len(feat_test) > self.lookback:
            X_test, y_test = self._make_windows(feat_test, tgt_test, self.lookback, self.stride)
            test_data = (X_test, y_test)

        logger.info(
            f"TimeSeriesDataHandler: {len(X_train)} train windows, "
            f"{len(val_data[0]) if val_data else 0} val, "
            f"{len(test_data[0]) if test_data else 0} test. "
            f"Window shape: {X_train.shape[1:]}."
        )

        return {
            "train_data": (X_train, y_train),
            "val_data": val_data,
            "test_data": test_data,
        }

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the time series dataset."""
        info = super().get_data_info()
        info.update(
            {
                "lookback": self.lookback,
                "value_column": self.value_column,
                "target_column": self.target_column,
                "stride": self.stride,
            }
        )
        return info


class ArraysDataHandler(DataHandler):
    """Data handler for pre-loaded in-memory arrays.

    Use this when data is already available as numpy arrays or Python
    lists, bypassing file I/O entirely. This is the handler used when
    passing ``(X, y)`` tuples to :func:`~ictonyx.api.variability_study`.

    Args:
        X: Feature array or list. Converted to ``np.ndarray`` on init.
        y: Label array or list. Must have the same length as ``X``.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths.
    """

    @property
    def data_type(self) -> str:
        return "arrays"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(
        self,
        X: Union[np.ndarray, List, Any],
        y: Union[np.ndarray, List, Any],
        X_test: Optional[Union[np.ndarray, List, Any]] = None,
        y_test: Optional[Union[np.ndarray, List, Any]] = None,
        val_split: float = 0.1,
        test_split: float = 0.2,
    ):
        """Data handler for pre-loaded in-memory arrays.

        Use this when data is already available as numpy arrays or Python
        lists, bypassing file I/O entirely. Optionally accepts a pre-held-out
        test set; if provided, it is passed through unchanged and only a
        train/val split is performed on the training arrays.

        Args:
            X: Feature array or list. Converted to ``np.ndarray`` on init.
            y: Label array or list. Must have the same length as ``X``.
            X_test: Optional pre-held-out test features. If provided,
                ``load()`` will not carve a test set from ``X``/``y``.
            y_test: Optional pre-held-out test labels. Required if
                ``X_test`` is provided.
            val_split: Default validation fraction passed to ``load()``
            when not overridden there. Default ``0.1``.
            test_split: Default test fraction passed to ``load()``
            when not overridden there. Default ``0.2``.

        Raises:
            ValueError: If ``X`` and ``y`` have different lengths.
            ValueError: If exactly one of ``X_test`` / ``y_test`` is provided.
            ValueError: If ``X_test`` and ``y_test`` have different lengths.
        """

        self.X = np.array(X)
        self.y = np.array(y)
        self.X_test_provided = np.array(X_test) if X_test is not None else None
        self.y_test_provided = np.array(y_test) if y_test is not None else None
        self._default_val_split = val_split
        self._default_test_split = test_split

        if len(self.X) != len(self.y):
            raise ValueError(f"Length mismatch: X has {len(self.X)}, y has {len(self.y)}")

        # Validate test set symmetry: both or neither must be provided.
        if (X_test is None) != (y_test is None):
            provided = "X_test" if X_test is not None else "y_test"
            missing = "y_test" if X_test is not None else "X_test"
            raise ValueError(
                f"ArraysDataHandler requires both X_test and y_test, or neither. "
                f"Got {provided} but not {missing}."
            )

        if self.X_test_provided is not None and self.y_test_provided is not None:
            if len(self.X_test_provided) != len(self.y_test_provided):
                raise ValueError(
                    f"Test set length mismatch: X_test has {len(self.X_test_provided)}, "
                    f"y_test has {len(self.y_test_provided)}"
                )

    def load(
        self,
        test_split: Optional[float] = None,
        val_split: Optional[float] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Split the pre-loaded arrays.

        If X_test and y_test were provided at construction, they are used
        as the test set directly and only a train/val split is performed.
        Otherwise, test_split is carved out of the provided arrays first.

        Args:
            test_split: Fraction to reserve for test. Defaults to the
                value passed to ``__init__`` (default ``0.2``). Pass
                ``0`` to skip the test split.
            val_split: Fraction to reserve for validation. Defaults to
                the value passed to ``__init__`` (default ``0.1``).
            random_state: Seed for reproducibility. Default ``42``.
        """

        # Resolve defaults: explicit argument wins over stored default.
        _test_split = self._default_test_split if test_split is None else test_split
        _val_split = self._default_val_split if val_split is None else val_split

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for splitting.")

        if _test_split + _val_split >= 1.0:
            raise ValueError("Sum of splits must be < 1.0")

            # --- Path 1: Pre-held-out test set provided ---
        if self.X_test_provided is not None:
            if _val_split > 0 and len(self.X) > 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=_val_split, random_state=random_state
                )
            else:
                X_train, y_train = self.X, self.y
                X_val, y_val = None, None

            logger.info(
                f"Array splits - Train: {len(X_train)}, "
                f"Val: {len(X_val) if X_val is not None else 0}, "
                f"Test: {len(self.X_test_provided)} (pre-provided)"
            )
            return {
                "train_data": (X_train, y_train),
                "val_data": (X_val, y_val) if X_val is not None else None,
                "test_data": (self.X_test_provided, self.y_test_provided),
            }

            # --- Path 2: Internal split ---
        if _test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=_test_split, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = self.X, None, self.y, None

        X_val, y_val = None, None
        if _val_split > 0 and len(X_train) > 1:
            adj_val_split = _val_split / (1 - _test_split) if _test_split > 0 else _val_split
            if adj_val_split < 1.0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=adj_val_split, random_state=random_state
                )

        logger.info(
            f"Array splits - Train: {len(X_train)}, "
            f"Val: {len(X_val) if X_val is not None else 0}, "
            f"Test: {len(X_test) if X_test is not None else 0}"
        )
        return {
            "train_data": (X_train, y_train),
            "val_data": (X_val, y_val) if X_val is not None else None,
            "test_data": (X_test, y_test) if X_test is not None else None,
        }


def auto_resolve_handler(
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray], Any],
    target_column: Optional[str] = None,
    **kwargs,
) -> DataHandler:
    """
    Factory function to automatically detect and initialize the appropriate DataHandler.

    Args:
        data: The input data. Supported formats:
            - str: Path to a directory (Images)
            - str: Path to a CSV file (Tabular, Text, or TimeSeries)
            - pd.DataFrame: Data object (Tabular)
            - tuple: (X, y) arrays (Arrays)
        target_column: Name of target column (Required for DataFrame/Tabular).
        **kwargs: Additional arguments passed to the specific handler constructor
            (e.g., 'image_size', 'text_column', 'sequence_length').

    Returns:
        DataHandler: An initialized instance of the appropriate handler.

    Raises:
        ValueError: If handler cannot be determined or required args missing.
        FileNotFoundError: If string path does not exist.
    """
    # 0. If data is already a DataHandler, just return it.
    if isinstance(data, DataHandler):
        return data

    # 1. Handle Tuple (X, y) -> Arrays
    if isinstance(data, tuple) and len(data) == 2:
        # Simple heuristic: check if elements have shape or length
        if hasattr(data[0], "shape") or hasattr(data[0], "__len__"):
            return ArraysDataHandler(data[0], data[1])

    # 2. Handle Pandas DataFrame -> Tabular
    if isinstance(data, pd.DataFrame):
        if not target_column:
            raise ValueError("target_column is required when passing a DataFrame.")
        return TabularDataHandler(data, target_column=target_column, **kwargs)

    # 3. Handle String Paths
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"Data path not found: {data}")

        # Directory -> Images
        if os.path.isdir(data):
            if "image_size" not in kwargs:
                raise ValueError("image_size=(H, W) is required for image data directories.")
            return ImageDataHandler(data, **kwargs)

        # File -> Tabular, Text, or TimeSeries
        if os.path.isfile(data):
            # Heuristic: Check kwargs for specific handler signals

            # Text
            if "text_column" in kwargs:
                return TextDataHandler(data, **kwargs)

            # TimeSeries
            if "value_column" in kwargs or "sequence_length" in kwargs:
                return TimeSeriesDataHandler(data, **kwargs)

            # Default to Tabular if target provided
            if target_column:
                return TabularDataHandler(data, target_column=target_column, **kwargs)

            # If we get here, we have a file but don't know how to handle it
            raise ValueError(
                "Ambiguous file input. Please provide:\n"
                " - 'target_column' for Tabular data\n"
                " - 'text_column' for Text data\n"
                " - 'sequence_length' for TimeSeries data"
            )

    # 4. Fallback
    raise TypeError(
        f"Unsupported data type: {type(data)}. Expected str, DataFrame, or (X, y) tuple."
    )
