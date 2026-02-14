# ictonyx/runners.py
"""
Experiment runners for variability studies with memory management.
Supports both standard and process-isolated execution modes.
"""

import gc
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
    """
    Engine for running variability studies with memory management.
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
        """
        Initialize experiment runner.

        Args:
            tracker: Object to track experiment metrics (e.g. MLflowLogger).
                     If None, uses basic BaseLogger.
            seed: Base random seed for reproducibility. Each run uses
                  seed + run_id. If None, a random seed is generated and
                  stored so the study can still be reproduced after the fact.
        """
        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))

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
                logger.info(f"Data loaded successfully")
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
        """Run a single training iteration."""
        if self.use_process_isolation:
            return self._run_single_fit_isolated(run_id, epochs)
        else:
            return self._run_single_fit_standard(run_id, epochs)

    def _run_single_fit_isolated(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Execute training in isolated subprocess."""
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
                self.seed + run_id,
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
        """Execute training in standard mode (in-process)."""
        self._run_log(f" - Run {run_id}: Training...")

        # Set deterministic seeds for this run
        self._set_seeds(self.seed + run_id)

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
        stop_on_failure_rate: float = 0.5,
    ) -> "VariabilityStudyResults":
        """Execute the complete variability study."""

        # Reset state from any previous run
        self.all_runs_metrics = []
        self.final_metrics = {}
        self.final_test_metrics = []
        self.failed_runs = []

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
            logger.info(f"\nStarting Variability Study")
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
            logger.info(f"\nStudy Summary:")
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
        """Get summary statistics for the completed study."""
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
    """Container for variability study results with analysis methods."""

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
        """Perform statistical comparison of runs for the specified metric."""
        from .analysis import compare_multiple_models

        if not self.all_runs_metrics:
            raise ValueError("No run metrics available for statistical comparison")

        final_metrics = self.get_final_metrics(metric_name)

        if not final_metrics:
            available = self.get_available_metrics()
            raise ValueError(
                f"Metric '{metric_name}' not found in results. " f"Available metrics: {available}"
            )

        metrics_dict = {}
        for run_name, value in final_metrics.items():
            run_idx = int(run_name.split("_")[-1]) - 1
            if run_idx < len(self.all_runs_metrics):
                df = self.all_runs_metrics[run_idx]
                if metric_name in df.columns:
                    metrics_dict[run_name] = df[metric_name]
                else:
                    metrics_dict[run_name] = pd.Series([value], name=run_name)
            else:
                metrics_dict[run_name] = pd.Series([value], name=run_name)

        return compare_multiple_models(
            model_results=metrics_dict, alpha=alpha, correction_method=correction_method
        )


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
    """Run a complete variability study.

    Args:
        seed: Base random seed. Each run uses seed + run_id. If None,
              a random seed is generated automatically.
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
