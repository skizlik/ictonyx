import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import settings

logger = settings.logger

# Optional MLflow dependency
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow

    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    HAS_MLFLOW = False

# Optional system info dependencies
try:
    import platform

    import psutil

    HAS_SYSTEM_INFO = True
except ImportError:
    platform = None  # type: ignore[assignment]
    psutil = None  # type: ignore[assignment]
    HAS_SYSTEM_INFO = False


class BaseLogger:
    """Base experiment logger with in-memory history tracking.

    Stores parameters and per-step metrics in a history dict. All
    ``log_*`` methods for artifacts, models, and figures are no-ops in
    the base class — subclasses (e.g. :class:`MLflowLogger`) override
    them to persist data to external systems.

    This class is used as the default logger when no tracker is passed
    to :class:`~ictonyx.runners.ExperimentRunner`.

    Args:
        verbose: If ``True``, print activity to stdout. Default ``True``.
        print_params: If ``True`` and ``verbose``, print logged parameters.
            Default ``False``.
        print_metrics: If ``True`` and ``verbose``, print logged metrics.
            Default ``False``.

    Attributes:
        history: Dict with keys ``'params'`` (Dict) and ``'metrics'``
            (List of ``{'key', 'value', 'step'}`` dicts).
    """

    def __init__(
        self, verbose: bool = True, print_params: bool = False, print_metrics: bool = False
    ):
        self.verbose = verbose
        self.print_params = print_params
        self.print_metrics = print_metrics
        self.history: Dict[str, Any] = {"params": {}, "metrics": []}

    def log_params(self, params: Dict[str, Any]):
        """Logs a dictionary of parameters and stores it in history."""
        self.history["params"].update(params)

        if self.verbose and self.print_params:
            logger.info(f"Parameters: {params}")

    def log_metric(self, key: str, value: float, step: int = 0):
        """Logs a single metric and stores it in history."""
        metric_entry = {"key": key, "value": value, "step": step}
        self.history["metrics"].append(metric_entry)

        if self.verbose and self.print_metrics:
            logger.info(f" - Step {step}: {key} = {value:.4f}")

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Logs multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Logs an artifact (file) - base implementation does nothing."""
        logger.debug(f"Would log artifact: {artifact_path}")

    def log_model(self, model, artifact_path: str = "model"):
        """Logs a model - base implementation does nothing."""
        logger.debug(f"Would log model to: {artifact_path}")

    def log_figure(self, figure, artifact_name: str):
        """Logs a matplotlib figure - base implementation does nothing."""
        logger.debug(f"Would log figure: {artifact_name}")

    def set_tags(self, tags: Dict[str, str]):
        """Sets tags for the run - base implementation does nothing."""
        if self.verbose:
            logger.info(f"Would set tags: {tags}")

    def end_run(self):
        """Called at the end of a run to finalize logging."""
        if self.verbose:
            logger.info("-" * 50)

    def start_child_run(self, run_name: Optional[str] = None) -> str:
        """No-op for base logger. Returns empty string."""
        return ""

    def end_child_run(self) -> None:
        """No-op for base logger."""
        pass

    def get_history(self) -> Dict[str, Any]:
        """Returns the logged history."""
        return self.history


class MLflowLogger(BaseLogger):
    """Experiment logger backed by MLflow.

    Extends :class:`BaseLogger` to persist parameters, metrics, artifacts,
    models, and figures to an MLflow tracking server. Creates a new MLflow
    run on initialization and ends it on :meth:`end_run`.

    Requires ``mlflow``. Install with ``pip install mlflow``.

    Args:
        run_name: Optional display name for this MLflow run.
        experiment_name: MLflow experiment name. Created if it does not
            exist. Default ``'ictonyx_experiment'``.
        tracking_uri: MLflow tracking server URI. If ``None``, uses the
            local ``mlruns/`` directory. Default ``None``.
        verbose: Print logging activity to stdout. Default ``True``.

    Raises:
        ImportError: If ``mlflow`` is not installed.
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: str = "ictonyx_experiment",
        tracking_uri: Optional[str] = None,
        verbose: bool = True,
    ):
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
            _exp = mlflow.get_experiment_by_name(experiment_name)
            if _exp is None:
                raise RuntimeError(
                    f"MLflow experiment '{experiment_name}' not found after creation attempt."
                )
            experiment_id = _exp.experiment_id

        mlflow.set_experiment(experiment_name)

        # Start the run
        self._run = mlflow.start_run(run_name=run_name)
        self._run_id = self._run.info.run_id

        self._current_child_run_id: Optional[str] = None
        if self.verbose:
            logger.info(f"Started MLflow run: {self._run_id}")
            logger.info(f"Experiment: {experiment_name}")
            if tracking_uri:
                logger.info(f"Tracking URI: {tracking_uri}")

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
                logger.warning(f"Skipping invalid metric value for {key}: {value}")
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
                logger.warning(f"Skipping invalid metric value for {key}: {value}")

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
        """Log a model to MLflow using the framework-appropriate flavor."""
        super().log_model(model, artifact_path)
        try:
            _logged = False

            if not _logged:
                try:
                    import tensorflow as tf

                    if isinstance(model, tf.keras.Model):
                        mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
                        _logged = True
                except (ImportError, AttributeError):
                    pass

            if not _logged:
                try:
                    import torch

                    if isinstance(model, torch.nn.Module):
                        mlflow.pytorch.log_model(model, artifact_path, **kwargs)
                        _logged = True
                except (ImportError, AttributeError):
                    pass

            if not _logged:
                try:
                    from sklearn.base import BaseEstimator

                    if isinstance(model, BaseEstimator):
                        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
                        _logged = True
                except (ImportError, AttributeError):
                    pass

            if not _logged:
                mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)

            if self.verbose:
                logger.info(f"Logged model of type {type(model).__name__} to {artifact_path}")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not log model to MLflow: {e}")

    def log_figure(self, figure, artifact_name: str):
        """Logs a matplotlib figure as an artifact."""
        super().log_figure(figure, artifact_name)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            figure.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
            tmp_file.flush()
            self.log_artifact(tmp_file.name, artifact_name)
            os.unlink(tmp_file.name)

    def log_dataframe(self, df: pd.DataFrame, artifact_name: str):
        """Logs a pandas DataFrame as a CSV artifact."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file.flush()
            self.log_artifact(tmp_file.name, artifact_name)
            os.unlink(tmp_file.name)

    def log_confusion_matrix(
        self, cm_df: pd.DataFrame, artifact_name: str = "confusion_matrix.csv"
    ):
        """Logs a confusion matrix DataFrame."""
        self.log_dataframe(cm_df, artifact_name)

    def log_training_history(
        self, history: Union[Dict, pd.DataFrame], artifact_name: str = "training_history.csv"
    ):
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

        system_tags.update(
            {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "python_platform": sys.platform,
            }
        )

        # Optional detailed system info
        if HAS_SYSTEM_INFO:
            system_tags.update(
                {
                    "system": platform.system(),
                    "cpu_count": str(psutil.cpu_count()),
                    "memory_gb": str(round(psutil.virtual_memory().total / (1024**3), 2)),
                }
            )
        elif self.verbose:
            logger.warning("platform/psutil not available, logging basic system info only")

        # Try to get GPU info if TensorFlow available
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            system_tags["gpu_count"] = str(len(gpus))
            if gpus:
                system_tags["gpu_names"] = str([gpu.name for gpu in gpus])
        except Exception:
            system_tags["gpu_count"] = "0"

        self.set_tags(system_tags)

    def start_child_run(self, run_name: Optional[str] = None) -> str:
        """Start a nested child run under the current parent run."""
        if not HAS_MLFLOW:
            return ""
        child_run = mlflow.start_run(run_name=run_name, nested=True)
        self._current_child_run_id = child_run.info.run_id
        return child_run.info.run_id

    def end_child_run(self) -> None:
        """End the current nested child run."""
        if not HAS_MLFLOW:
            return
        mlflow.end_run()
        self._current_child_run_id = None

    def end_run(self):
        """Ends the MLflow run."""
        super().end_run()
        mlflow.end_run()

        if self.verbose:
            logger.info(f"Ended MLflow run: {self._run_id}")
            logger.info(f"View results at: {mlflow.get_tracking_uri()}")

    def log_study_summary(self, results: Any) -> None:
        """Log statistical summary of a completed study to the parent run.

        Logs mean and SD for all tracked validation metrics, and mean
        and SD for all tracked test metrics when test data is present.

        Args:
            results: A VariabilityStudyResults object from run_study()
                or variability_study().
        """
        if not HAS_MLFLOW:
            return

        for metric_name, values in results.final_metrics.items():
            if values:
                self.log_metric(f"{metric_name}_mean", float(np.mean(values)))
                self.log_metric(f"{metric_name}_sd", float(np.std(values, ddof=1)))
                self.log_metric(f"{metric_name}_min", float(np.min(values)))
                self.log_metric(f"{metric_name}_max", float(np.max(values)))

        if results.has_test_data:
            test_keys = sorted({k for m in results.final_test_metrics for k in m if k != "run_id"})
            for key in test_keys:
                vals = [m[key] for m in results.final_test_metrics if key in m]
                if vals:
                    self.log_metric(f"{key}_mean", float(np.mean(vals)))
                    self.log_metric(f"{key}_sd", float(np.std(vals, ddof=1)))

    def get_run_url(self) -> str:
        """Return the MLflow UI URL for this run.

        For remote tracking servers, uses the configured tracking URI directly.
        For local file-based tracking (``file://``), defaults to
        ``http://localhost:5000`` — the standard ``mlflow ui`` address.
        Override with the ``MLFLOW_UI_URL`` environment variable when serving
        the UI on a non-default host or port.
        """
        import os

        tracking_uri = mlflow.get_tracking_uri()
        exp_id = self._run.info.experiment_id
        run_id = self._run_id

        if tracking_uri.startswith("file://"):
            ui_base = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000").rstrip("/")
            return f"{ui_base}/#/experiments/{exp_id}/runs/{run_id}"

        return f"{tracking_uri.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
