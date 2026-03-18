# ictonyx/runners.py
"""
Experiment runners for variability studies with memory management.
Supports both standard and process-isolated execution modes.
"""

import gc
import itertools
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import ModelConfig
from .core import BaseModelWrapper
from .data import DataHandler
from .loggers import BaseLogger
from .memory import get_memory_info, get_memory_manager

# The SYSTEM Logger (Standard Python Convention)
from .settings import logger

# Optional TensorFlow for cleanup
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

# Optional progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False


class ExperimentRunner:
    """Core engine for running variability studies with memory management.

    Trains a model multiple times with different random seeds, collecting
    per-epoch metrics from each run. Supports standard in-process execution
    and subprocess isolation for GPU memory cleanup.

    This class is the low-level interface. Most users should prefer
    :func:`~ictonyx.api.variability_study`, which handles data resolution
    and model wrapping automatically.

    Args:
        model_builder: A callable ``f(ModelConfig) -> BaseModelWrapper``
            that constructs a fresh model for each run.
        data_handler: A :class:`~ictonyx.data.DataHandler` that provides
            train/val/test splits via its ``load()`` method.
        model_config: A :class:`~ictonyx.config.ModelConfig` containing
            training hyperparameters (epochs, batch_size, learning_rate, etc.).
        tracker: An optional :class:`~ictonyx.loggers.BaseLogger` for
            experiment tracking. Defaults to a basic in-memory logger.
        use_process_isolation: If ``True``, each run executes in a child
            process, ensuring GPU memory is fully released between runs.
            Required for long Keras/TF studies that would otherwise OOM.
            Default ``False``.
        gpu_memory_limit: Optional GPU memory cap in MB, applied per run
            when using TensorFlow. Default ``None`` (no limit).
        seed: Base random seed for reproducibility. Run *i* uses
            ``seed + i``. If ``None``, a random seed is generated and stored.
        verbose: If ``True``, log study progress to stdout. If ``tqdm`` is
            installed, a progress bar replaces per-run log messages.
            Default ``True``.
    """

    def __init__(
        self,
        model_builder: Callable[[ModelConfig], BaseModelWrapper],
        data_handler: DataHandler,
        model_config: ModelConfig,
        tracker: Optional[BaseLogger] = None,
        use_process_isolation: bool = False,
        gpu_memory_limit: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the experiment runner.

        See class docstring for full parameter descriptions.
        """
        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        self._child_seeds: List[int] = []

        # The Experiment Tracker (Records results/metrics)
        self.tracker = tracker or BaseLogger()  # <--- RENAMED ATTR

        self.use_process_isolation = use_process_isolation
        self.gpu_memory_limit = gpu_memory_limit
        self.verbose = verbose
        self._progress_bar = HAS_TQDM and verbose

        # Initialize memory manager
        self.memory_manager = get_memory_manager(
            use_process_isolation=use_process_isolation, gpu_memory_limit=gpu_memory_limit
        )

        # Setup memory constraints for standard mode
        if not use_process_isolation:
            setup_success = self.memory_manager.setup()
            if not setup_success and verbose:
                warnings.warn(
                    "Memory setup incomplete. Consider using process isolation "
                    "for better memory control: use_process_isolation=True"
                )

        # Load and prepare data
        if verbose:
            logger.info("Loading and preparing data...")  # <--- System Logger

        try:
            data_dict = self.data_handler.load()
            self.train_data = data_dict["train_data"]
            self.val_data = data_dict.get("val_data")
            self.test_data = data_dict.get("test_data")

            if verbose:
                logger.info("Data loaded successfully")
                if self.val_data is None:
                    logger.warning("No validation data provided")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        # Initialize result storage
        self.all_runs_metrics: List[pd.DataFrame] = []
        self.final_metrics: Dict[str, List[float]] = {}
        self.final_test_metrics: List[Dict[str, Any]] = []
        self.failed_runs: List[int] = []

        # Validate process isolation if enabled
        if use_process_isolation:
            self._validate_process_isolation()

    def _validate_process_isolation(self):
        """Validate that process isolation can work with current setup."""
        import pickle

        # Check if model builder is picklable
        try:
            pickle.dumps(self.model_builder)
        except Exception as e:
            raise ValueError(f"model_builder must be picklable for process isolation: {e}")

        # Check if data is picklable (warn if large)
        try:
            import sys

            data_size = sys.getsizeof(pickle.dumps(self.train_data))
            if data_size > 500_000_000:  # 500MB
                warnings.warn(
                    f"Training data is large ({data_size / 1e6:.0f}MB). "
                    "Process isolation may use significant memory for serialization."
                )
        except Exception:
            warnings.warn(
                "Could not determine data size. Process isolation may have issues "
                "with non-picklable data types like tf.data.Dataset"
            )

    @staticmethod
    def _set_seeds(seed: int):
        """Set all relevant RNGs for reproducibility.

        Called before each run with a unique per-run seed (base_seed + run_id).
        This controls the environment's randomness; model-internal RNGs
        (e.g. sklearn's random_state) remain the user's responsibility.
        """
        random.seed(seed)
        np.random.seed(seed)

        # TensorFlow
        if HAS_TENSORFLOW:
            tf.random.set_seed(seed)

        # PyTorch (will require revision when full PyTorch support is added)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _run_log(self, msg: str, level: str = "info"):
        """Log a per-run message, respecting tqdm when active.

        Info messages are suppressed when the progress bar is active (the bar
        replaces them). Warnings and errors always print, routed through
        tqdm.write() to avoid colliding with the progress bar.
        """
        if not self.verbose:
            return
        if self._progress_bar:
            if level == "info":
                return  # progress bar replaces routine info
            tqdm.write(f"{level.upper()}: {msg}")
        else:
            getattr(logger, level)(msg)

    def _update_progress_postfix(self, pbar, metrics_df):
        """Show the most informative metric on the progress bar."""
        exclude = {"run_num", "epoch"}
        columns = [c for c in metrics_df.columns if c not in exclude]

        # Prefer: val metric that isn't loss > any val metric > first metric
        val_metrics = [c for c in columns if "val" in c and "loss" not in c]
        if val_metrics:
            col = val_metrics[0]
        elif columns:
            col = columns[0]
        else:
            return

        pbar.set_postfix({col: f"{metrics_df[col].iloc[-1]:.4f}"})

    def _run_single_fit(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Run a single training iteration.

        Dispatches to :meth:`_run_single_fit_isolated` or
        :meth:`_run_single_fit_standard` depending on the
        ``use_process_isolation`` setting.

        Args:
            run_id: 1-based index of this run within the study.
            epochs: Number of training epochs for this run.

        Returns:
            A ``pd.DataFrame`` of per-epoch metrics if the run succeeded,
            or ``None`` if it failed. Failed runs are recorded in
            :attr:`failed_runs`.
        """
        if self.use_process_isolation:
            return self._run_single_fit_isolated(run_id, epochs)
        else:
            return self._run_single_fit_standard(run_id, epochs)

    def _run_single_fit_isolated(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Execute a training run in an isolated subprocess.

        Serializes the model builder, data handler, and config, then trains
        in a child process. This guarantees that GPU memory (especially
        TensorFlow session state) is fully released between runs.

        Args:
            run_id: 1-based run index.
            epochs: Training epochs.

        Returns:
            ``pd.DataFrame`` of per-epoch metrics, or ``None`` on failure.
        """
        self._run_log(f" - Run {run_id}: Starting in isolated process...")

        # Log run start (Metric Tracker)
        self.tracker.log_params({"run_id": run_id, "mode": "isolated"})

        # Execute in subprocess
        result = self.memory_manager.run_isolated(
            _isolated_training_function,
            args=(
                self.model_builder,
                self.model_config,
                self.train_data,
                self.val_data,
                self.test_data,
                epochs,
                run_id,
                self._child_seeds[run_id - 1],
            ),
        )

        # Process results
        if result["success"]:
            try:
                # Extract history
                history_data = result["result"]["history"]
                if not history_data:
                    self._run_log(f" - Run {run_id}: No training history returned", level="warning")
                    self.failed_runs.append(run_id)
                    return None

                # Create DataFrame
                history_df = pd.DataFrame(history_data)
                history_df["run_num"] = run_id
                history_df["epoch"] = range(1, len(history_df) + 1)

                # Standardize column names
                history_df.rename(
                    columns={"accuracy": "train_accuracy", "loss": "train_loss"}, inplace=True
                )

                # Store final values for ALL tracked metrics
                for col in history_df.columns:
                    if col not in ("run_num", "epoch"):
                        final_value = float(history_df[col].iloc[-1])
                        if col not in self.final_metrics:
                            self.final_metrics[col] = []
                        self.final_metrics[col].append(final_value)
                        self.tracker.log_metric(f"final_{col}", final_value, step=run_id)

                # Store test metrics
                test_metrics = result["result"].get("test_metrics")
                if test_metrics:
                    self.final_test_metrics.append(test_metrics)
                    for key, value in test_metrics.items():
                        self.tracker.log_metric(f"final_test_{key}", value, step=run_id)

                if self.verbose:
                    self._run_log(f" - Run {run_id}: Completed successfully (isolated)")

                return history_df

            except Exception as e:
                self._run_log(f" - Run {run_id}: Failed to process results: {e}", level="error")
                self.failed_runs.append(run_id)
                return None
        else:
            # Training failed
            error_msg = result.get("error", "Unknown error")
            self._run_log(f" - Run {run_id}: Failed - {error_msg}", level="error")
            if "traceback" in result:
                self._run_log(f"   Traceback: {result['traceback'][:500]}...", level="error")

            self.failed_runs.append(run_id)
            return None

    def _run_single_fit_standard(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Execute a training run in the current process.

        Builds a fresh model, loads data, trains, evaluates on the test set
        (if available), and collects metrics into a DataFrame. Cleans up
        the model after each run via :meth:`BaseModelWrapper.cleanup`.

        Args:
            run_id: 1-based run index.
            epochs: Training epochs.

        Returns:
            ``pd.DataFrame`` of per-epoch metrics, or ``None`` on failure.
        """
        self._run_log(f" - Run {run_id}: Training...")

        # Set deterministic seeds for this run
        if self._child_seeds:
            self._set_seeds(self._child_seeds[run_id - 1])

        # Log run start (Metric Tracker)
        self.tracker.log_params({"run_id": run_id, "mode": "standard"})

        wrapped_model = None
        try:
            # Build model
            wrapped_model = self.model_builder(self.model_config)

            # Train
            wrapped_model.fit(
                train_data=self.train_data,
                validation_data=self.val_data,
                epochs=epochs,
                batch_size=self.model_config.get("batch_size", 32),
                verbose=self.model_config.get("verbose", 0),
            )

            # Extract training result
            if wrapped_model.training_result is None:
                self._run_log(f" - Run {run_id}: No training result produced", level="warning")
                self.failed_runs.append(run_id)
                return None

            history_dict = wrapped_model.training_result.history

            history_df = pd.DataFrame(history_dict)
            history_df["run_num"] = run_id
            history_df["epoch"] = range(1, len(history_df) + 1)

            # Standardize column names
            history_df.rename(
                columns={"accuracy": "train_accuracy", "loss": "train_loss"}, inplace=True
            )

            # Store final values for ALL tracked metrics
            for col in history_df.columns:
                if col not in ("run_num", "epoch"):
                    final_value = float(history_df[col].iloc[-1])
                    if col not in self.final_metrics:
                        self.final_metrics[col] = []
                    self.final_metrics[col].append(final_value)
                    self.tracker.log_metric(f"final_{col}", final_value, step=run_id)

            # Evaluate on test data
            if self.test_data is not None:
                try:
                    test_metrics = wrapped_model.evaluate(data=self.test_data)
                    self.final_test_metrics.append(test_metrics)
                    for key, value in test_metrics.items():
                        self.tracker.log_metric(f"final_test_{key}", value, step=run_id)
                except Exception as e:
                    self._run_log(f"   Warning: Test evaluation failed: {e}", level="warning")

            self._run_log(f" - Run {run_id}: Completed successfully")

            return history_df

        except Exception as e:
            self._run_log(f" - Run {run_id}: Failed with error: {e}", level="error")
            self.failed_runs.append(run_id)
            return None

        finally:
            # Cleanup
            if wrapped_model is not None:
                try:
                    if hasattr(wrapped_model, "cleanup"):
                        wrapped_model.cleanup()
                    del wrapped_model
                except Exception:
                    pass

            # Perform memory cleanup
            cleanup_result = self.memory_manager.cleanup()
            if cleanup_result.memory_freed_mb and cleanup_result.memory_freed_mb > 10:
                self._run_log(f"   Freed {cleanup_result.memory_freed_mb:.1f}MB")

    def run_study(
        self,
        num_runs: int = 5,
        epochs_per_run: Optional[int] = None,
        stop_on_failure_rate: float = 0.8,
    ) -> "VariabilityStudyResults":
        """Execute the complete variability study.

        Trains the model ``num_runs`` times, collecting per-epoch training
        metrics from each run. Resets all internal accumulators before
        starting, so calling ``run_study()`` twice on the same runner is safe.

        If ``tqdm`` is installed and ``verbose=True``, a progress bar is
        displayed with a live postfix showing the most informative validation
        metric from the latest completed run.

        Args:
            num_runs: Number of independent training runs. Default 5.
            epochs_per_run: Epochs per run. If ``None``, uses the ``epochs``
                value from :attr:`model_config`. Default ``None``.
            stop_on_failure_rate: If the fraction of failed runs exceeds this
                threshold, the study halts early. Set to ``1.0`` to never
                stop early. Default ``0.8``.

        Returns:
            :class:`VariabilityStudyResults` with per-run DataFrames, final
            metric distributions, and optional test-set metrics.
        """

        # Reset state from any previous run
        self.all_runs_metrics = []
        self.final_metrics = {}
        self.final_test_metrics = []
        self.failed_runs = []

        # Generate independent child seeds up front.
        # SeedSequence guarantees uncorrelated children regardless of proximity.
        _ss = np.random.SeedSequence(self.seed)
        self._child_seeds: list[int] = [
            int(child.generate_state(1)[0]) for child in _ss.spawn(num_runs)
        ]

        if epochs_per_run is None:
            epochs_per_run = self.model_config.get("epochs", 10)

        # Log study parameters (Metric Tracker)
        self.tracker.log_params(
            {
                "num_runs": num_runs,
                "epochs_per_run": epochs_per_run,
                "use_process_isolation": self.use_process_isolation,
                "gpu_memory_limit": self.gpu_memory_limit,
                "seed": self.seed,
            }
        )
        self.tracker.log_params(self.model_config.params)

        # Print study configuration (System Logger)
        if self.verbose:
            mode = "with process isolation" if self.use_process_isolation else "in standard mode"
            logger.info("\nStarting Variability Study")
            logger.info(f"  Runs: {num_runs}")
            logger.info(f"  Epochs per run: {epochs_per_run}")
            logger.info(f"  Execution mode: {mode}")
            logger.info(f"  Seed: {self.seed}")
            if self.gpu_memory_limit:
                logger.info(f"  GPU memory limit: {self.gpu_memory_limit}MB")
            logger.info("")

        # Execute runs
        run_iter = range(num_runs)
        if self._progress_bar:
            run_iter = tqdm(run_iter, desc="Variability Study", unit="run")

        try:
            for i in run_iter:
                # Check failure rate
                if i > 0:
                    completed = i  # runs 0..i-1 have completed
                    failure_rate = len(self.failed_runs) / completed
                    if failure_rate >= stop_on_failure_rate:
                        self._run_log(
                            f"Stopping due to high failure rate: {failure_rate:.1%}",
                            level="error",
                        )
                        break

                # Run single training
                metrics_df = self._run_single_fit(run_id=i + 1, epochs=epochs_per_run)
                if metrics_df is not None:
                    self.all_runs_metrics.append(metrics_df)
                    if self._progress_bar:
                        self._update_progress_postfix(run_iter, metrics_df)

                # Log memory info periodically (suppressed when progress bar is active)
                if (i + 1) % 10 == 0 and self.verbose and not self._progress_bar:
                    memory_info = get_memory_info()
                    if "process_rss_mb" in memory_info:
                        logger.info(f"  Memory check: {memory_info['process_rss_mb']:.1f}MB")

        except KeyboardInterrupt:
            if self.verbose:
                logger.warning(f"\n\nStudy interrupted after {len(self.all_runs_metrics)} runs")

        finally:
            # Final cleanup for standard mode
            if not self.use_process_isolation:
                final_cleanup = self.memory_manager.cleanup()
                if self.verbose and final_cleanup.memory_freed_mb:
                    logger.info(f"\nFinal cleanup freed {final_cleanup.memory_freed_mb:.1f}MB")

            self.tracker.end_run()

        # Print summary
        if self.verbose:
            successful = len(self.all_runs_metrics)
            logger.info("\nStudy Summary:")
            logger.info(f"  Successful runs: {successful}/{num_runs}")
            if self.failed_runs:
                logger.warning(f"  Failed runs: {self.failed_runs}")
            for metric_name, values in self.final_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    logger.info(f"  {metric_name}: {mean_val:.4f} (SD = {std_val:.4f})")

        return VariabilityStudyResults(
            all_runs_metrics=self.all_runs_metrics,
            final_metrics=self.final_metrics,
            final_test_metrics=self.final_test_metrics,
            seed=self.seed,
        )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the completed study.

        Returns:
            Dict with keys:

            * ``total_runs``, ``successful_runs``, ``failed_runs``,
              ``failure_rate`` — run accounting.
            * ``<metric>_mean``, ``<metric>_std``, ``<metric>_min``,
              ``<metric>_max`` — descriptive statistics for each tracked
              metric (e.g. ``val_accuracy_mean``).
        """
        stats = {
            "total_runs": len(self.all_runs_metrics) + len(self.failed_runs),
            "successful_runs": len(self.all_runs_metrics),
            "failed_runs": len(self.failed_runs),
            "failure_rate": len(self.failed_runs)
            / max(1, len(self.all_runs_metrics) + len(self.failed_runs)),
        }

        for metric_name, values in self.final_metrics.items():
            if values:
                stats[f"{metric_name}_mean"] = np.mean(values)
                stats[f"{metric_name}_std"] = np.std(values)
                stats[f"{metric_name}_min"] = np.min(values)
                stats[f"{metric_name}_max"] = np.max(values)

        return stats


# Module-level function for subprocess execution (must be picklable)
def _isolated_training_function(
    model_builder, config, train_data, val_data, test_data, epochs, run_id, run_seed=None
):
    """
    Training function executed in isolated subprocess.
    """
    import gc

    try:
        # Set seeds in subprocess
        if run_seed is not None:
            import random

            random.seed(run_seed)
            np.random.seed(run_seed)
            try:
                import tensorflow as tf

                tf.random.set_seed(run_seed)
            except ImportError:
                pass
            try:
                import torch

                torch.manual_seed(run_seed)
            except ImportError:
                pass

        # Build model in subprocess
        model = model_builder(config)

        # Train model
        model.fit(
            train_data=train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=config.get("batch_size", 32),
            verbose=config.get("verbose", 0),
        )

        # Extract history
        history = {}
        if model.training_result is not None:
            history = dict(model.training_result.history)

        # Evaluate on test data if available
        test_metrics = {}
        if test_data is not None:
            try:
                test_metrics = model.evaluate(data=test_data)
                # Ensure all values are serializable
                if isinstance(test_metrics, dict):
                    test_metrics = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in test_metrics.items()
                    }
            except Exception as e:
                test_metrics = {"error": str(e)}

        # Cleanup before returning
        if hasattr(model, "cleanup"):
            model.cleanup()
        del model
        gc.collect()

        return {"history": history, "test_metrics": test_metrics, "run_id": run_id}

    except Exception as e:
        # Return error information
        import traceback

        return {
            "history": {},
            "test_metrics": {},
            "run_id": run_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@dataclass
class VariabilityStudyResults:
    """Container for variability study results with analysis methods.

    Returned by :func:`~ictonyx.api.variability_study` and
    :meth:`ExperimentRunner.run_study`. Holds per-run training history
    DataFrames, aggregated final-epoch metrics, and optional test-set
    evaluation results.

    Attributes:
        all_runs_metrics: List of DataFrames, one per successful run. Each
            DataFrame has one row per epoch and columns for every tracked
            metric (e.g. ``train_accuracy``, ``val_loss``).
        final_metrics: Dict mapping metric names to lists of final-epoch
            values across runs. Example:
            ``{'val_accuracy': [0.82, 0.85, 0.83]}``.
        final_test_metrics: List of dicts, one per run, containing test-set
            evaluation results (empty if no test set was used).
        seed: The base random seed used for the study, for reproducibility.
    """

    all_runs_metrics: List[pd.DataFrame]
    final_metrics: Dict[str, List[float]]
    final_test_metrics: List[Dict[str, Any]]
    seed: Optional[int] = None

    @property
    def n_runs(self) -> int:
        """Number of successful runs."""
        return len(self.all_runs_metrics)

    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get collected final values for a specific metric.

        Args:
            metric_name: Metric key, e.g. 'val_accuracy', 'train_loss', 'val_f1'

        Returns:
            List of final-epoch values, one per run.

        Raises:
            KeyError: If metric was not tracked.
        """
        if metric_name not in self.final_metrics:
            available = sorted(self.final_metrics.keys())
            raise KeyError(f"Metric '{metric_name}' not found. Available: {available}")
        return self.final_metrics[metric_name]

    def get_final_metrics(self, metric_name: str = "val_accuracy") -> Dict[str, float]:
        """Extract final metric values for each run (labeled run_1, run_2, ...).

        For backward compatibility with plotting and statistical comparison code.
        """
        metrics = {}
        for i, df in enumerate(self.all_runs_metrics):
            if metric_name in df.columns:
                metrics[f"run_{i + 1}"] = float(df[metric_name].iloc[-1])
        return metrics

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics across all runs."""
        return sorted(self.final_metrics.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame with one row per run."""
        if not self.all_runs_metrics:
            return pd.DataFrame()

        rows = []
        for i, df in enumerate(self.all_runs_metrics):
            row = {"run_id": i + 1}

            for col in df.columns:
                if col not in {"run_num", "epoch", "run_id"}:
                    row[f"final_{col}"] = float(df[col].iloc[-1])

            if i < len(self.final_test_metrics):
                for key, value in self.final_test_metrics[i].items():
                    row[f"test_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def summarize(self) -> str:
        """Generate text summary of results."""
        lines = [
            "Variability Study Results",
            "=" * 30,
            f"Successful runs: {self.n_runs}",
            f"Seed: {self.seed}",
        ]

        for metric_name, values in sorted(self.final_metrics.items()):
            if values:
                lines.extend(
                    [
                        f"{metric_name}:",
                        f"  Mean: {np.mean(values):.4f}",
                        f"  Std:  {np.std(values):.4f}",
                        f"  Min:  {np.min(values):.4f}",
                        f"  Max:  {np.max(values):.4f}",
                    ]
                )

        return "\n".join(lines)

    def compare_models_statistically(
        self,
        metric_name: str = "val_accuracy",
        alpha: float = 0.05,
        correction_method: str = "holm",
    ) -> Dict[str, Any]:
        """Perform statistical comparison of runs for the specified metric.

        Each run contributes exactly one observation (its final-epoch metric
        value), satisfying the independence assumption required by Mann-Whitney U
        and Kruskal-Wallis tests.
        """
        from .analysis import compare_multiple_models

        if not self.all_runs_metrics:
            raise ValueError("No run metrics available for statistical comparison")

        if metric_name not in self.final_metrics:
            available = self.get_available_metrics()
            raise ValueError(
                f"Metric '{metric_name}' not found in results. " f"Available metrics: {available}"
            )

        values = self.final_metrics[metric_name]
        if len(values) < 2:
            raise ValueError(
                f"At least 2 runs are required for statistical comparison, "
                f"got {len(values)} for metric '{metric_name}'."
            )

        # Each run IS one observation — a single-element Series.
        # Using final_metrics directly guarantees scalars, not epoch series.
        metrics_dict: Dict[str, pd.Series] = {
            f"run_{i + 1}": pd.Series([v], name=f"run_{i + 1}") for i, v in enumerate(values)
        }

        return compare_multiple_models(
            model_results=metrics_dict,
            alpha=alpha,
            correction_method=correction_method,
        )


@dataclass
class GridStudyResults:
    """Container for grid study results across multiple parameter configurations.

    Returned by :func:`~ictonyx.api.run_grid_study`. Holds one
    :class:`VariabilityStudyResults` per parameter configuration, the
    param grid that generated them, and the base config used as the
    template for all runs.

    Attributes:
        results:     Dict mapping frozen parameter tuples to
                     VariabilityStudyResults. Keys are sorted tuples of
                     (param_name, value) pairs for hashability and
                     deterministic ordering.
        param_grid:  The original param_grid dict passed by the caller.
        base_config: The ModelConfig used as the template for all runs.
        metric:      The primary metric used for summary statistics.
    """

    results: Dict[Tuple, VariabilityStudyResults]
    param_grid: Dict[str, List[Any]]
    base_config: "ModelConfig"
    metric: str = "val_accuracy"

    @staticmethod
    def _config_key(param_combo: Dict[str, Any]) -> Tuple:
        """Convert a parameter combination dict to a hashable, sorted tuple key."""
        return tuple(sorted(param_combo.items()))

    @property
    def n_configurations(self) -> int:
        """Total number of parameter configurations run."""
        return len(self.results)

    @property
    def n_runs_per_config(self) -> int:
        """Number of runs per configuration (from first result)."""
        if not self.results:
            return 0
        return next(iter(self.results.values())).n_runs

    def get_results_for_config(self, param_combo: Dict[str, Any]) -> VariabilityStudyResults:
        """Retrieve VariabilityStudyResults for a specific parameter combination.

        Args:
            param_combo: Dict of parameter names to values, e.g.
                         {'learning_rate': 0.001, 'batch_size': 8}

        Returns:
            :class:`VariabilityStudyResults` for that configuration.

        Raises:
            KeyError: If the configuration was not found in results.
        """
        key = self._config_key(param_combo)
        if key not in self.results:
            raise KeyError(
                f"Configuration {param_combo} not found in results. "
                f"Available configurations: {self.list_configurations()}"
            )
        return self.results[key]

    def list_configurations(self) -> List[Dict[str, Any]]:
        """Return list of all parameter combinations as dicts."""
        return [dict(key) for key in self.results.keys()]

    def to_dataframe(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Summary DataFrame with one row per configuration.

        Columns: all parameter names, mean, sd, se, min, max, n.

        Args:
            metric: Metric to summarize. Defaults to self.metric.

        Returns:
            DataFrame sorted by first parameter in param_grid.
        """
        metric = metric or self.metric
        rows = []

        for key, result in self.results.items():
            param_combo = dict(key)
            try:
                values = pd.Series(result.get_metric_values(metric))
                n = len(values)
                row = {
                    **param_combo,
                    "mean": round(values.mean(), 4),
                    "sd": round(values.std(), 4),
                    "se": round(values.std() / np.sqrt(n), 4),
                    "min": round(values.min(), 4),
                    "max": round(values.max(), 4),
                    "n": n,
                }
            except KeyError:
                row = {
                    **param_combo,
                    "mean": np.nan,
                    "sd": np.nan,
                    "se": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n": 0,
                }
            rows.append(row)

        df = pd.DataFrame(rows)
        sort_cols = list(self.param_grid.keys())
        return df.sort_values(sort_cols).reset_index(drop=True)

    def summarize(self, metric: Optional[str] = None) -> str:
        """Generate text summary of grid study results.

        Args:
            metric: Metric to summarize. Defaults to self.metric.
        """
        metric = metric or self.metric
        df = self.to_dataframe(metric)
        lines = [
            "Grid Study Results",
            "=" * 40,
            f"Configurations: {self.n_configurations}",
            f"Runs per configuration: {self.n_runs_per_config}",
            f"Total runs: {self.n_configurations * self.n_runs_per_config}",
            f"Metric: {metric}",
            "",
            df.to_string(index=False),
        ]
        return "\n".join(lines)


def run_variability_study(
    model_builder: Callable[[ModelConfig], BaseModelWrapper],
    data_handler: DataHandler,
    model_config: ModelConfig,
    num_runs: int = 5,
    epochs_per_run: Optional[int] = None,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    gpu_memory_limit: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> VariabilityStudyResults:
    """Run a complete variability study (convenience function).

    Creates an :class:`ExperimentRunner` and calls
    :meth:`~ExperimentRunner.run_study`. This is the lower-level entry
    point; most users should prefer :func:`~ictonyx.api.variability_study`.

    Args:
        model_builder: Callable ``f(ModelConfig) -> BaseModelWrapper``.
        data_handler: A :class:`~ictonyx.data.DataHandler` instance.
        model_config: A :class:`~ictonyx.config.ModelConfig` instance.
        num_runs: Number of training runs. Default 5.
        epochs_per_run: Epochs per run. Default ``None`` (uses config).
        tracker: Optional experiment logger. Default ``None``.
        use_process_isolation: Run each iteration in a subprocess.
            Default ``False``.
        gpu_memory_limit: Optional GPU memory cap in MB.
        seed: Base random seed. If ``None``, one is generated.
        verbose: Print progress to stdout. Default ``True``.

    Returns:
        :class:`VariabilityStudyResults`.
    """
    runner = ExperimentRunner(
        model_builder=model_builder,
        data_handler=data_handler,
        model_config=model_config,
        tracker=tracker,
        use_process_isolation=use_process_isolation,
        gpu_memory_limit=gpu_memory_limit,
        seed=seed,
        verbose=verbose,
    )

    return runner.run_study(num_runs=num_runs, epochs_per_run=epochs_per_run)


def run_grid_study(
    model_builder: Callable,
    data_handler: "DataHandler",
    base_config: "ModelConfig",
    param_grid: Dict[str, List[Any]],
    num_runs: int = 10,
    metric: str = "val_accuracy",
    use_process_isolation: bool = True,
    dry_run: bool = False,
) -> GridStudyResults:
    """Run a variability study across a grid of parameter configurations.

    For each combination in the Cartesian product of ``param_grid``,
    creates a modified copy of ``base_config``, runs a full variability
    study, and collects results. All runs use process isolation by default
    to prevent CUDA memory accumulation across configurations.

    Args:
        model_builder:         Callable accepting a ModelConfig, returning
                               a compiled model.
        data_handler:          DataHandler instance providing train/val data.
        base_config:           ModelConfig used as template. Values are
                               overridden per configuration.
        param_grid:            Dict mapping parameter names to lists of
                               values to sweep. Example::

                                   {
                                       'learning_rate': [0.001, 0.0001],
                                       'batch_size':    [8, 16]
                                   }

                               Generates the full Cartesian product:
                               4 configurations × num_runs each.
        num_runs:              Number of training runs per configuration.
        metric:                Primary metric for summary statistics.
        use_process_isolation: If True, each run executes in a subprocess.
                               Strongly recommended to prevent GPU memory
                               accumulation across configurations.
        dry_run:               If True, print execution plan and return
                               without training. Use to verify configuration
                               before committing to a long run.

    Returns:
        :class:`GridStudyResults`

    Raises:
        ValueError: If param_grid is empty or contains no values.

    Example::

        results = run_grid_study(
            model_builder=build_cnn,
            data_handler=data_handler,
            base_config=config,
            param_grid={'learning_rate': [0.001, 0.0001, 0.00001]},
            num_runs=12,
            use_process_isolation=True
        )
        print(results.summarize())
        df = results.to_dataframe()
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter with values.")

    for param_name, values in param_grid.items():
        if not values:
            raise ValueError(
                f"param_grid['{param_name}'] is empty. "
                f"Each parameter must have at least one value."
            )

    # Build Cartesian product of all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    n_configs = len(combinations)
    total_runs = n_configs * num_runs

    if dry_run:
        print("Grid Study — Dry Run")
        print("=" * 40)
        print(f"Parameter grid:")
        for name, values in param_grid.items():
            print(f"  {name}: {values}")
        print(f"\nConfigurations:         {n_configs}")
        print(f"Runs per configuration: {num_runs}")
        print(f"Total training runs:    {total_runs}")
        print(f"Metric:                 {metric}")
        print(f"Process isolation:      {use_process_isolation}")
        print("\nConfigurations to run:")
        for i, combo in enumerate(combinations, 1):
            print(f"  {i:>3}. {combo}")
        return GridStudyResults(
            results={}, param_grid=param_grid, base_config=base_config, metric=metric
        )

    results_dict: Dict[Tuple, VariabilityStudyResults] = {}

    for i, param_combo in enumerate(combinations, 1):
        print(f"\n{'='*50}")
        print(f"Configuration {i}/{n_configs}: {param_combo}")
        print(f"{'='*50}")

        config = base_config.copy().update(param_combo)

        result = run_variability_study(
            model_builder=model_builder,
            data_handler=data_handler,
            model_config=config,
            num_runs=num_runs,
            use_process_isolation=use_process_isolation,
        )

        key = GridStudyResults._config_key(param_combo)
        results_dict[key] = result

        try:
            values = pd.Series(result.get_metric_values(metric))
            print(
                f"  Mean: {values.mean():.4f}  SD: {values.std():.4f}  "
                f"SE: {values.std()/np.sqrt(len(values)):.4f}"
            )
        except KeyError:
            print(f"  Warning: metric '{metric}' not found in results.")

    return GridStudyResults(
        results=results_dict, param_grid=param_grid, base_config=base_config, metric=metric
    )
