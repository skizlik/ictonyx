# ictonyx/data.py

import os
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
except ImportError:
    TimeseriesGenerator = None
    Tokenizer = None
    pad_sequences = None
    HAS_TF_PREPROCESSING = False

# Optional matplotlib for data visualization
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# Optional sklearn for data splitting
try:
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    train_test_split = None


class DataHandler(ABC):
    """Abstract base class for loading and splitting datasets.

    Provides a consistent interface for all data formats. Concrete
    subclasses handle specific formats:

    * :class:`TabularDataHandler` — CSV files and DataFrames.
    * :class:`ImageDataHandler` — Image directories with class subfolders.
    * :class:`TextDataHandler` — Text datasets with a text column.
    * :class:`TimeSeriesDataHandler` — Sequential data with windowing.
    * :class:`ArraysDataHandler` — Pre-loaded ``(X, y)`` numpy arrays.

    All subclasses return a dict from :meth:`load` with keys
    ``'train_data'``, ``'val_data'``, and ``'test_data'``, each
    containing a ``(X, y)`` tuple or ``None`` if that split was not
    requested.

    For automatic format detection, use
    :func:`auto_resolve_handler` or pass data directly to
    :func:`~ictonyx.api.variability_study`.

    Args:
        data_path: Path to the data source (file or directory).

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """

    def __init__(self, data_path: str):
        """
        Initialize the data handler with a path to the data source.

        Args:
            data_path (str): Path to the data source (file or directory)
        """
        self.data_path = data_path
        self._validate_data_path()

    def _validate_data_path(self):
        """Validate that the data path exists."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    @abstractmethod
    def load(self, **kwargs) -> Dict[str, Any]:
        """
        Loads and prepares the dataset, and splits it into training, validation,
        and test sets.

        Returns:
            Dict[str, Any]: A dictionary containing the data splits, with keys
                            'train_data', 'val_data', and 'test_data'.
        """
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return a string identifier for the data type this handler processes."""
        pass

    @property
    @abstractmethod
    def return_format(self) -> str:
        """
        Return a string describing the format of data returned by load().

        Possible values:
        - "tf_datasets": Returns tf.data.Dataset objects
        - "split_arrays": Returns (X, y) tuple arrays
        - "generators": Returns generator objects
        """
        pass

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dict with dataset statistics and properties
        """
        return {
            "data_path": self.data_path,
            "data_type": self.data_type,
            "return_format": self.return_format,
            "path_exists": os.path.exists(self.data_path),
        }


class ImageDataHandler(DataHandler):
    """
    A concrete data handler for image datasets.

    This class is specialized for datasets with a directory structure where
    folder names are class labels.
    """

    @property
    def data_type(self) -> str:
        return "image"

    @property
    def return_format(self) -> str:
        return "tf_datasets"

    def __init__(
        self, data_path: str, image_size: Tuple[int, int], batch_size: int = 32, seed: int = 42
    ):
        """
        Initialize the image data handler.

        Args:
            data_path: Path to directory containing class subdirectories
            image_size: (height, width) tuple for resizing images
            batch_size: Batch size for the datasets
            seed: Random seed for reproducibility
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

    def _preprocess_image(self, file_path, label):
        """Helper to load and preprocess a single image from its path."""
        try:
            # Load the raw data from the file as a string
            img = tf.io.read_file(file_path)

            # Try to decode as different formats
            try:
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:
                try:
                    img = tf.image.decode_png(img, channels=3)
                except tf.errors.InvalidArgumentError:
                    # If both fail, try generic decode_image
                    img = tf.image.decode_image(img, channels=3)
                    img.set_shape([None, None, 3])  # Set shape for decode_image

            # Convert to float and resize
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, self.image_size)

            # Normalize the pixel values to [0, 1]
            img = img / 255.0

            return img, label

        except Exception as e:
            # Create a placeholder image if loading fails
            logger.warning(f"Failed to load image {file_path}: {e}")
            placeholder = tf.zeros((*self.image_size, 3), dtype=tf.float32)
            return placeholder, label

    def load(self, validation_split: float = 0.2, test_split: float = 0.1) -> Dict[str, Any]:
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


class TabularDataHandler(DataHandler):
    """
    A concrete data handler for structured, tabular data.
    Accepts either a file path (CSV) or a pandas DataFrame.
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
        target_column: str = None,
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
        self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42
    ) -> Dict[str, Any]:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for data splitting.")

        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        self._load_and_validate_data()

        # Prepare features and target
        if self.features:
            X = self.data[self.features]
        else:
            X = self.data.drop(columns=[self.target_column])

        y = self.data[self.target_column]

        # Check for missing values
        if X.isnull().any().any():
            null_counts = X.isnull().sum()
            # Only print warning, don't crash
            # print(f"Warning: Missing values in features")

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
        info = super().get_data_info()
        try:
            if self.data is None:
                self._load_and_validate_data()

            info.update(
                {
                    "num_rows": len(self.data),
                    "num_columns": len(self.data.columns),
                    "target_column": self.target_column,
                }
            )
        except Exception as e:
            info["error"] = str(e)
        return info


class TextDataHandler(DataHandler):
    """
    A concrete data handler for text data.
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
        max_words: int = 10000,
        max_len: int = 100,
    ):
        """
        Initialize the text data handler.

        Args:
            data_path: Path to CSV file with text data
            text_column: Name of column containing text
            label_column: Name of column containing labels
            max_words: Maximum vocabulary size
            max_len: Maximum sequence length
        """
        if not HAS_TF_PREPROCESSING:
            raise ImportError(
                "TensorFlow preprocessing is required for TextDataHandler. "
                "Install with: pip install tensorflow"
            )

        super().__init__(data_path)

        if not os.path.isfile(self.data_path):
            raise ValueError("TextDataHandler requires a CSV file path")

        self.text_column = text_column
        self.label_column = label_column
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.data = None

    def _load_and_validate_data(self):
        """Load and validate the text data."""
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load text data: {e}")

        if self.text_column not in self.data.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found. "
                f"Available columns: {list(self.data.columns)}"
            )

        if self.label_column not in self.data.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found. "
                f"Available columns: {list(self.data.columns)}"
            )

        # Check for missing values
        if self.data[self.text_column].isnull().any():
            null_count = self.data[self.text_column].isnull().sum()
            logger.warning(f"{null_count} missing values in text column")
            # Fill with empty strings
            self.data[self.text_column] = self.data[self.text_column].fillna("")

        if self.data[self.label_column].isnull().any():
            null_count = self.data[self.label_column].isnull().sum()
            raise ValueError(f"{null_count} missing values in label column - cannot proceed")

        logger.info(f"Loaded text data: {len(self.data)} samples")

    def load(
        self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Load and split the text data.

        Returns:
            Dict with 'train_data', 'val_data', 'test_data' as (X, y) tuples
            where X contains tokenized and padded sequences
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for data splitting. "
                "Install with: pip install scikit-learn"
            )

        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        self._load_and_validate_data()

        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")

        try:
            self.tokenizer.fit_on_texts(self.data[self.text_column])
            sequences = self.tokenizer.texts_to_sequences(self.data[self.text_column])
            padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        except Exception as e:
            raise RuntimeError(f"Text preprocessing failed: {e}")

        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        logger.info(f"Sequence length: {self.max_len}")

        # Split data
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                padded_sequences,
                self.data[self.label_column],
                test_size=test_split,
                random_state=random_state,
            )
        else:
            X_train, X_test = padded_sequences, None
            y_train, y_test = self.data[self.label_column], None

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
        """Get information about the text dataset."""
        info = super().get_data_info()

        try:
            if self.data is None:
                self._load_and_validate_data()

            info.update(
                {
                    "num_samples": len(self.data),
                    "text_column": self.text_column,
                    "label_column": self.label_column,
                    "max_words": self.max_words,
                    "max_len": self.max_len,
                    "avg_text_length": self.data[self.text_column].str.len().mean(),
                    "label_distribution": self.data[self.label_column].value_counts().to_dict(),
                }
            )

            if self.tokenizer:
                info["vocab_size"] = len(self.tokenizer.word_index)

        except Exception as e:
            info["error"] = f"Could not analyze dataset: {e}"

        return info


class TimeSeriesDataHandler(DataHandler):
    """
    A concrete data handler for time series data.
    """

    @property
    def data_type(self) -> str:
        return "timeseries"

    @property
    def return_format(self) -> str:
        return "generators"

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 10,
        value_column: str = "value",
        batch_size: int = 32,
    ):
        """
        Initialize the time series data handler.

        Args:
            data_path: Path to CSV file with time series data
            sequence_length: Length of input sequences
            value_column: Name of column containing values
            batch_size: Batch size for generators
        """
        if not HAS_TF_PREPROCESSING:
            raise ImportError(
                "TensorFlow preprocessing is required for TimeSeriesDataHandler. "
                "Install with: pip install tensorflow"
            )

        super().__init__(data_path)

        if not os.path.isfile(self.data_path):
            raise ValueError("TimeSeriesDataHandler requires a CSV file path")

        self.sequence_length = sequence_length
        self.value_column = value_column
        self.batch_size = batch_size
        self.data = None

    def _load_and_validate_data(self):
        """Load and validate the time series data."""
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load time series data: {e}")

        if self.value_column not in self.data.columns:
            raise ValueError(
                f"Value column '{self.value_column}' not found. "
                f"Available columns: {list(self.data.columns)}"
            )

        # Validate data types
        try:
            self.data[self.value_column] = pd.to_numeric(
                self.data[self.value_column], errors="coerce"
            )
        except Exception:
            raise ValueError(f"Cannot convert '{self.value_column}' to numeric values")

        if self.data[self.value_column].isnull().any():
            null_count = self.data[self.value_column].isnull().sum()
            raise ValueError(f"{null_count} non-numeric or missing values in '{self.value_column}'")

        if len(self.data) < self.sequence_length + 1:
            raise ValueError(
                f"Dataset too short ({len(self.data)} samples) for sequence_length={self.sequence_length}"
            )

        logger.info(f"Loaded time series: {len(self.data)} time points")

    def load(
        self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Load and split the time series data.

        Returns:
            Dict with 'train_data', 'val_data', 'test_data' as TimeseriesGenerator objects
        """
        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        self._load_and_validate_data()

        # Convert to numpy array
        data = self.data[self.value_column].values.reshape(-1, 1)
        total_len = len(data)

        # Calculate split indices (chronological splits for time series)
        train_size = int(total_len * (1 - test_split - val_split))
        val_size = int(total_len * val_split) if val_split > 0 else 0

        # Split data chronologically
        train_data = data[:train_size]

        val_data = None
        test_data = None

        if val_size > 0:
            val_data = data[train_size : train_size + val_size]
            if test_split > 0:
                test_data = data[train_size + val_size :]
        elif test_split > 0:
            test_data = data[train_size:]

        # Create generators
        def create_generator(data_array):
            if data_array is None or len(data_array) < self.sequence_length + 1:
                return None
            return TimeseriesGenerator(
                data_array, data_array, length=self.sequence_length, batch_size=self.batch_size
            )

        train_gen = create_generator(train_data)
        val_gen = create_generator(val_data)
        test_gen = create_generator(test_data)

        logger.info(
            f"Data splits - Train: {len(train_data) if train_data is not None else 0}, "
            f"Val: {len(val_data) if val_data is not None else 0}, "
            f"Test: {len(test_data) if test_data is not None else 0}"
        )

        return {"train_data": train_gen, "val_data": val_gen, "test_data": test_gen}


class ArraysDataHandler(DataHandler):
    """
    A concrete data handler for raw Numpy arrays or Lists.
    Useful when data is already loaded into memory as (X, y) tuples.
    """

    @property
    def data_type(self) -> str:
        return "arrays"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(self, X: Union[np.ndarray, List, Any], y: Union[np.ndarray, List, Any]):
        """
        Initialize with existing arrays.

        Args:
            X: Feature array/list (n_samples, n_features)
            y: Label array/list (n_samples,)
        """
        # Dummy path since data is in memory
        super().__init__("in_memory_arrays")

        self.X = np.array(X)
        self.y = np.array(y)

        if len(self.X) != len(self.y):
            raise ValueError(f"Length mismatch: X has {len(self.X)}, y has {len(self.y)}")

    def _validate_data_path(self):
        """Skip path validation for in-memory arrays."""
        pass

    def load(
        self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42
    ) -> Dict[str, Any]:
        """Split the pre-loaded arrays."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for splitting.")

        if test_split + val_split >= 1.0:
            raise ValueError("Sum of splits must be < 1.0")

        # Split 1: Separate Test
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_split, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = self.X, None, self.y, None

        # Split 2: Separate Val from Train
        X_val, y_val = None, None
        if val_split > 0 and len(X_train) > 1:
            adj_val_split = val_split / (1 - test_split) if test_split > 0 else val_split

            if adj_val_split < 1.0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=adj_val_split, random_state=random_state
                )

        logger.info(
            f"Array splits - Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test) if X_test is not None else 0}"
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
