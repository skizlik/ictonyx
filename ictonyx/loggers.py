import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from . import settings

logger = settings.logger

# Optional MLflow dependency
try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.sklearn

    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

# Optional system info dependencies
try:
    import platform
    import psutil

    HAS_SYSTEM_INFO = True
except ImportError:
    platform = None
    psutil = None
    HAS_SYSTEM_INFO = False


class BaseLogger:
    """
    A base class for logging experiment data.

    The logger stores all logged data in a history object and can optionally
    print it to the console.
    """

    def __init__(self, verbose: bool = True, print_params: bool = False,
                 print_metrics: bool = False):
        self.verbose = verbose
        self.print_params = print_params
        self.print_metrics = print_metrics
        self.history: Dict[str, Any] = {'params': {}, 'metrics': []}

    def log_params(self, params: Dict[str, Any]):
        """Logs a dictionary of parameters and stores it in history."""
        self.history['params'].update(params)

        if self.verbose and self.print_params:
            print(f"Parameters: {params}")

    def log_metric(self, key: str, value: float, step: int = 0):
        """Logs a single metric and stores it in history."""
        metric_entry = {'key': key, 'value': value, 'step': step}
        self.history['metrics'].append(metric_entry)

        if self.verbose and self.print_metrics:
            print(f" - Step {step}: {key} = {value:.4f}")

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Logs multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Logs an artifact (file) - base implementation does nothing."""
        if self.verbose:
            print(f"Would log artifact: {artifact_path}")

    def log_model(self, model, artifact_path: str = "model"):
        """Logs a model - base implementation does nothing."""
        if self.verbose:
            print(f"Would log model to: {artifact_path}")

    def log_figure(self, figure, artifact_name: str):
        """Logs a matplotlib figure - base implementation does nothing."""
        if self.verbose:
            print(f"Would log figure: {artifact_name}")

    def set_tags(self, tags: Dict[str, str]):
        """Sets tags for the run - base implementation does nothing."""
        if self.verbose:
            print(f"Would set tags: {tags}")

    def end_run(self):
        """Called at the end of a run to finalize logging."""
        if self.verbose:
            print("-" * 50)

    def get_history(self) -> Dict[str, Any]:
        """Returns the logged history."""
        return self.history


class MLflowLogger(BaseLogger):
    """
    An implementation of the BaseLogger for MLflow with comprehensive tracking capabilities.
    """

    def __init__(self,
                 run_name: Optional[str] = None,
                 experiment_name: str = "ictonyx_experiment",
                 tracking_uri: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize MLflow logger.

        Args:
            run_name: Name for this specific run
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None uses local mlruns/)
            verbose: Whether to print logging messages
        """
        if not HAS_MLFLOW:
            raise ImportError(
                "MLflow is required for MLflowLogger. Install with: pip install mlflow\n"
                "Alternative: Use BaseLogger for basic logging without MLflow integration."
            )

        super().__init__(verbose)

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

        # Start the run
        self._run = mlflow.start_run(run_name=run_name)
        self._run_id = self._run.info.run_id

        if self.verbose:
            print(f"Started MLflow run: {self._run_id}")
            print(f"Experiment: {experiment_name}")
            if tracking_uri:
                print(f"Tracking URI: {tracking_uri}")

    @property
    def run_id(self) -> str:
        """Get the current run ID."""
        return self._run_id

    @property
    def experiment_name(self) -> str:
        """Get the current experiment name."""
        return mlflow.get_experiment(self._run.info.experiment_id).name

    def log_params(self, params: Dict[str, Any]):
        """Logs parameters to MLflow and stores in history."""
        super().log_params(params)

        # MLflow has limitations on parameter types, so convert appropriately
        mlflow_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                # Convert complex types to strings
                mlflow_params[key] = str(value)
            elif value is None:
                mlflow_params[key] = "None"
            else:
                mlflow_params[key] = value

        mlflow.log_params(mlflow_params)

    def log_metric(self, key: str, value: float, step: int = 0):
        """Logs a metric to MLflow and stores in history."""
        super().log_metric(key, value, step)

        # Handle NaN values
        if np.isnan(value) or np.isinf(value):
            if self.verbose:
                print(f"Warning: Skipping invalid metric value for {key}: {value}")
            return

        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Logs multiple metrics efficiently."""
        # Filter out invalid values
        valid_metrics = {}
        for key, value in metrics.items():
            if not (np.isnan(value) or np.isinf(value)):
                valid_metrics[key] = value
            elif self.verbose:
                print(f"Warning: Skipping invalid metric value for {key}: {value}")

        if valid_metrics:
            mlflow.log_metrics(valid_metrics, step=step)

        # Update history
        for key, value in valid_metrics.items():
            super().log_metric(key, value, step)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Logs an artifact file to MLflow."""
        super().log_artifact(artifact_path, artifact_name)

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        if artifact_name:
            # Create a temporary directory with the desired name structure
            with tempfile.TemporaryDirectory() as tmp_dir:
                dest_path = os.path.join(tmp_dir, artifact_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copy file to temporary location with desired name
                import shutil
                shutil.copy2(artifact_path, dest_path)
                mlflow.log_artifacts(tmp_dir)
        else:
            mlflow.log_artifact(artifact_path)

    def log_model(self, model, artifact_path: str = "model", **kwargs):
        """
        Logs a model to MLflow using the appropriate flavor.

        Args:
            model: The model object to log
            artifact_path: Path within the run's artifact directory
            **kwargs: Additional arguments passed to the specific model logging function
        """
        super().log_model(model, artifact_path)

        try:
            if hasattr(model, 'save') and 'keras' in str(type(model)).lower():
                # TensorFlow/Keras model
                mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
            elif hasattr(model, 'fit') and hasattr(model, 'predict'):
                # Scikit-learn model
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            else:
                # Generic Python model
                mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)

            if self.verbose:
                print(f"Logged model of type {type(model).__name__} to {artifact_path}")

        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to log model: {e}")
                print("Attempting to save as pickle...")

            # Fallback: save as pickle artifact
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                import pickle
                pickle.dump(model, tmp_file)
                tmp_file.flush()
                self.log_artifact(tmp_file.name, f"{artifact_path}/model.pkl")
                os.unlink(tmp_file.name)

    def log_figure(self, figure, artifact_name: str):
        """Logs a matplotlib figure as an artifact."""
        super().log_figure(figure, artifact_name)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            figure.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
            tmp_file.flush()
            self.log_artifact(tmp_file.name, artifact_name)
            os.unlink(tmp_file.name)

    def log_dataframe(self, df: pd.DataFrame, artifact_name: str):
        """Logs a pandas DataFrame as a CSV artifact."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file.flush()
            self.log_artifact(tmp_file.name, artifact_name)
            os.unlink(tmp_file.name)

    def log_confusion_matrix(self, cm_df: pd.DataFrame, artifact_name: str = "confusion_matrix.csv"):
        """Logs a confusion matrix DataFrame."""
        self.log_dataframe(cm_df, artifact_name)

    def log_training_history(self, history: Union[Dict, pd.DataFrame],
                             artifact_name: str = "training_history.csv"):
        """Logs training history as both metrics and artifact."""
        if isinstance(history, dict):
            history_df = pd.DataFrame(history)
        else:
            history_df = history.copy()

        # Log each epoch's metrics
        for epoch in history_df.index:
            epoch_metrics = {}
            for column in history_df.columns:
                value = history_df.loc[epoch, column]
                if pd.notna(value):
                    epoch_metrics[column] = value

            if epoch_metrics:
                self.log_metrics(epoch_metrics, step=epoch)

        # Also save as artifact
        self.log_dataframe(history_df, artifact_name)

    def set_tags(self, tags: Dict[str, str]):
        """Sets tags for the current run."""
        super().set_tags(tags)

        # Convert all values to strings
        string_tags = {key: str(value) for key, value in tags.items()}
        mlflow.set_tags(string_tags)

    def log_system_info(self):
        """Logs system information as tags."""
        system_tags = {}

        # Always available Python info
        import sys
        system_tags.update({
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_platform": sys.platform
        })

        # Optional detailed system info
        if HAS_SYSTEM_INFO:
            system_tags.update({
                "system": platform.system(),
                "cpu_count": str(psutil.cpu_count()),
                "memory_gb": str(round(psutil.virtual_memory().total / (1024 ** 3), 2))
            })
        elif self.verbose:
            print("Warning: platform/psutil not available, logging basic system info only")

        # Try to get GPU info if TensorFlow available
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            system_tags["gpu_count"] = str(len(gpus))
            if gpus:
                system_tags["gpu_names"] = str([gpu.name for gpu in gpus])
        except Exception:
            system_tags["gpu_count"] = "0"

        self.set_tags(system_tags)

    def end_run(self):
        """Ends the MLflow run."""
        super().end_run()
        mlflow.end_run()

        if self.verbose:
            print(f"Ended MLflow run: {self._run_id}")
            print(f"View results at: {mlflow.get_tracking_uri()}")

    def get_run_url(self) -> str:
        """Gets the URL to view this run in MLflow UI."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.startswith('file://'):
            return f"http://localhost:5000/#/experiments/{self._run.info.experiment_id}/runs/{self._run_id}"
        else:
            return f"{tracking_uri}/#/experiments/{self._run.info.experiment_id}/runs/{self._run_id}"

