# ictonyx/runners.py
"""
Experiment runners for variability studies with memory management.
Supports both standard and process-isolated execution modes.
"""

import gc
import itertools
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from .analysis import StatisticalTestResult

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from .config import ModelConfig
from .core import BaseModelWrapper, ScikitLearnModelWrapper
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

# version for _schema validation
from ._version import __version__ as _ICTONYX_VERSION


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
        seed: Base random seed for reproducibility. Each run receives an
            independent child seed derived via ``np.random.SeedSequence.spawn()``,
            which guarantees statistically uncorrelated RNG streams.
            If ``None``, a random seed is generated and stored.
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
        self.seed: int = (
            seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        )
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

    @staticmethod
    @contextmanager
    def _deterministic_cudnn():
        try:
            import torch

            if torch.cuda.is_available():
                prev_det = torch.backends.cudnn.deterministic
                prev_bench = torch.backends.cudnn.benchmark
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                try:
                    yield
                finally:
                    torch.backends.cudnn.deterministic = prev_det
                    torch.backends.cudnn.benchmark = prev_bench
            else:
                yield
        except ImportError:
            yield

    def _validate_process_isolation(self):
        """Validate that process isolation can work with current setup.

        Uses cloudpickle when available (matching actual execution), falling
        back to standard pickle. This correctly validates lambdas and
        notebook-defined functions that cloudpickle supports but pickle does not.
        """
        import pickle as _serializer

        try:
            import cloudpickle as _serializer
        except ImportError:
            pass

        try:
            _serializer.dumps(self.model_builder)
        except Exception as e:
            serializer_name = getattr(_serializer, "__name__", "pickle")
            raise ValueError(
                f"model_builder could not be serialised with {serializer_name} "
                f"for process isolation: {e}\n"
                "Ensure your model builder is a picklable function, class, or lambda. "
                "Notebook cells that reference closed-over variables may fail "
                "with standard pickle; install cloudpickle for broader support."
            )

        # Check data size and serialisability
        import sys

        try:
            data_size = sys.getsizeof(_serializer.dumps(self.train_data))
            if data_size > 500_000_000:
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

        Called before each run with a unique per-run seed derived via
        SeedSequence.spawn(). This controls global RNGs for numpy, Python,
        TensorFlow, and PyTorch. sklearn estimators that accept random_state
        are seeded at wrapper construction time when possible.
        """
        random.seed(seed)
        np.random.seed(seed)

        # TensorFlow
        if HAS_TENSORFLOW:
            tf.random.set_seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    @staticmethod
    def _standardize_history_df(df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw framework history columns to ictonyx standard names.

        Only renames columns that are present; absent columns are silently
        skipped. Returns a new DataFrame; does not modify in place.
        """
        rename_map = {
            "accuracy": "train_accuracy",
            "loss": "train_loss",
            "r2": "train_r2",
            "mse": "train_mse",
            "rmse": "train_rmse",
            "mae": "train_mae",
        }
        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        return df.rename(columns=existing, inplace=False)

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
                history_df = self._standardize_history_df(history_df)

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
                    self.final_test_metrics.append({"run_id": run_id, **test_metrics})
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
            child_seed = self._child_seeds[run_id - 1]
            self._set_seeds(child_seed)
            # Inject run_seed into config so class-based builders can pass it
            # as random_state. Mirrors the behaviour of isolated mode.
            self.model_config.set("run_seed", child_seed)

        # Log run start (Metric Tracker)
        self.tracker.log_params({"run_id": run_id, "mode": "standard"})

        wrapped_model = None
        try:
            # Build model
            wrapped_model = self.model_builder(self.model_config)

            # Train — use deterministic cudnn for reproducibility, restore after
            # Build kwargs conditionally: sklearn wrappers ignore training-loop
            # kwargs and emit DeprecationWarning when they're passed. Only forward
            # epochs/batch_size/verbose to wrappers that use them.
            fit_kwargs = {
                "train_data": self.train_data,
                "validation_data": self.val_data,
            }
            if not isinstance(wrapped_model, ScikitLearnModelWrapper):
                fit_kwargs["epochs"] = epochs
                fit_kwargs["batch_size"] = self.model_config.get("batch_size", 32)
                fit_kwargs["verbose"] = self.model_config.get("verbose", 0)
                fit_kwargs["run_seed"] = self.model_config.get("run_seed")

            with self._deterministic_cudnn():
                wrapped_model.fit(**fit_kwargs)

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
            history_df = self._standardize_history_df(history_df)

            # Store final values for ALL tracked metrics
            self._extract_and_store_final_metrics(history_df, run_id=run_id)

            # Evaluate on test data
            if self.test_data is not None:
                try:
                    with self._deterministic_cudnn():
                        test_metrics = wrapped_model.evaluate(data=self.test_data)
                        self.final_test_metrics.append({"run_id": run_id, **test_metrics})
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

    def _extract_and_store_final_metrics(self, history_df: pd.DataFrame, run_id: int = 0) -> None:
        """Store final-epoch metric values from a completed run's history DataFrame.

        Shared by both the sequential and parallel execution paths. The sequential
        path (via _run_single_fit_standard) passes run_id for tracker logging.
        The parallel collection loop passes no run_id, so tracker logging is skipped.

        Args:
            history_df: Per-epoch metrics DataFrame produced by a completed run.
            run_id: 1-based run index for tracker logging. 0 means skip logging.
        """
        for col in history_df.columns:
            if col not in ("run_num", "epoch"):
                final_value = float(history_df[col].iloc[-1])
                if col not in self.final_metrics:
                    self.final_metrics[col] = []
                self.final_metrics[col].append(final_value)
                if run_id:
                    self.tracker.log_metric(f"final_{col}", final_value, step=run_id)

    def run_study(
        self,
        num_runs: int = 20,
        epochs_per_run: Optional[int] = None,
        stop_on_failure_rate: float = 0.8,
        checkpoint_dir: Optional[str] = None,
        use_parallel: bool = False,
        n_jobs: int = -1,
    ) -> "VariabilityStudyResults":
        """Execute the complete variability study.

        Trains the model ``num_runs`` times, collecting per-epoch training
        metrics from each run. Resets all internal accumulators before
        starting, so calling ``run_study()`` twice on the same runner is safe.

        If ``tqdm`` is installed and ``verbose=True``, a progress bar is
        displayed with a live postfix showing the most informative validation
        metric from the latest completed run.

        Args:
            num_runs: Number of independent training runs. Default 20.
            epochs_per_run: Epochs per run. If ``None``, uses the ``epochs``
                value from :attr:`model_config`. Default ``None``.
            stop_on_failure_rate: If the fraction of failed runs exceeds this
                threshold, the study halts early. Set to ``1.0`` to never
                stop early. Default ``0.8``.
            checkpoint_dir: Optional path to a directory for saving progress
                after each completed run. If the directory contains a
                ``checkpoint.pkl`` file from a previous interrupted run,
                execution resumes from where it left off. Default ``None``
                (no checkpointing).
            use_parallel: If ``True``, fan training runs across multiple
                processes using ``joblib.Parallel``. Safe for sklearn
                models. Not recommended for Keras/TF. Mutually exclusive
                with ``use_process_isolation``. Default ``False``.
            n_jobs: Number of parallel workers. ``-1`` uses all available
                CPUs. Ignored when ``use_parallel=False``. Default ``-1``.

        Returns:
            :class:`VariabilityStudyResults` with per-run DataFrames, final
            metric distributions, and optional test-set metrics.
        """

        if num_runs < 2:
            raise ValueError(
                f"num_runs must be at least 2 to measure variability, got {num_runs}. "
                "A single run produces no distribution to analyse. "
                "Call wrapper.fit() and wrapper.evaluate() directly for single-run use."
            )

        if use_parallel and self.use_process_isolation:
            raise ValueError(
                "use_parallel=True and use_process_isolation=True are mutually exclusive. "
                "Use one or the other, not both."
            )

        if use_parallel and HAS_TENSORFLOW:
            warnings.warn(
                "use_parallel=True with Keras/TF models may cause GPU memory conflicts "
                "or session state corruption. Consider use_process_isolation=True instead.",
                UserWarning,
                stacklevel=2,
            )

        # Reset state from any previous run — must happen BEFORE checkpoint load
        self.all_runs_metrics.clear()
        self.final_metrics.clear()
        self.final_test_metrics.clear()
        self.failed_runs.clear()

        # Resume from checkpoint if available
        completed_run_ids: set = set()
        if checkpoint_dir is not None:
            import os
            import pickle as _pickle

            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "rb") as _f:
                        _prior_data = _pickle.load(_f)
                    _schema = _prior_data.get("_schema_version")
                    if _schema != _ICTONYX_VERSION:
                        warnings.warn(
                            f"Checkpoint schema version '{_schema}' does not match "
                            f"current '{_ICTONYX_VERSION}'. The checkpoint was written by a "
                            "different version of Ictonyx and may be incompatible. "
                            "Delete the checkpoint directory to start fresh.",
                            UserWarning,
                            stacklevel=2,
                        )
                    self.all_runs_metrics = list(_prior_data["all_runs_metrics"])
                    self.final_metrics = dict(_prior_data["final_metrics"])
                    self.final_test_metrics = list(_prior_data["final_test_metrics"])
                    self.failed_runs = list(
                        _prior_data.get("failed_runs", [])
                    )  # added v0.4.4; absent in older checkpoints
                    completed_run_ids = {
                        int(df["run_num"].iloc[0]) for df in self.all_runs_metrics if not df.empty
                    }
                    logger.info(
                        f"Resuming from checkpoint: "
                        f"{len(completed_run_ids)} of {num_runs} runs already complete."
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not load checkpoint from {checkpoint_path}: {e}. "
                        "Starting from scratch."
                    )

        if use_parallel and self.test_data is not None:
            warnings.warn(
                "use_parallel=True: test set evaluation is not supported in parallel "
                "mode. Each training run executes in a joblib worker process; test "
                "metrics cannot be returned to the parent. has_test_data will be "
                "False and final_test_metrics will be empty. Use "
                "use_process_isolation=True if you need per-run test evaluation.",
                UserWarning,
                stacklevel=2,
            )

        # Generate independent child seeds up front.
        # SeedSequence guarantees uncorrelated children regardless of proximity.
        _ss = np.random.SeedSequence(self.seed)
        self._child_seeds = [int(child.generate_state(1)[0]) for child in _ss.spawn(num_runs)]

        if epochs_per_run is None:
            epochs_per_run = self.model_config.get("epochs_per_run") or self.model_config.get(
                "epochs", 10
            )

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

        if use_parallel and not self.use_process_isolation:
            # --- Parallel execution path ---
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "joblib is required for parallel execution. " "Install with: pip install joblib"
                )

            logger.info(f"Running {num_runs} runs in parallel (n_jobs={n_jobs})...")

            # Build the list of run_ids being dispatched, preserving order.
            # parallel_results[j] corresponds to dispatched_run_ids[j].
            dispatched_run_ids = [
                i + 1 for i in range(num_runs) if (i + 1) not in completed_run_ids
            ]

            parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._run_single_fit)(run_id=run_id, epochs=epochs_per_run)
                for run_id in dispatched_run_ids
            )

            for run_id, metrics_df in zip(dispatched_run_ids, parallel_results):
                if metrics_df is not None:
                    self.all_runs_metrics.append(metrics_df)
                    self._extract_and_store_final_metrics(metrics_df)
                else:
                    self.failed_runs.append(run_id)

        else:
            # --- Sequential execution path ---
            try:
                for i in run_iter:
                    # Skip runs already completed in a prior checkpoint
                    if (i + 1) in completed_run_ids:
                        continue

                    # Check failure rate
                    if i > 0:
                        completed = i
                        failure_rate = len(self.failed_runs) / max(
                            1, completed + len(self.failed_runs)
                        )
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

                        # Save checkpoint after each successful run
                        if checkpoint_dir is not None:
                            import os
                            import pickle as _pickle

                            os.makedirs(checkpoint_dir, exist_ok=True)
                            _checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
                            _checkpoint_data = {
                                "_schema_version": _ICTONYX_VERSION,
                                "all_runs_metrics": list(self.all_runs_metrics),
                                "final_metrics": dict(self.final_metrics),
                                "final_test_metrics": list(self.final_test_metrics),
                                "seed": self.seed,
                                "failed_runs": list(self.failed_runs),
                            }
                            _tmp_path = _checkpoint_path + ".tmp"
                            with open(_tmp_path, "wb") as _f:
                                _pickle.dump(_checkpoint_data, _f)
                            os.replace(_tmp_path, _checkpoint_path)

                    # Log memory info periodically
                    if (i + 1) % 10 == 0 and self.verbose and not self._progress_bar:
                        memory_info = get_memory_info()
                        if "process_rss_mb" in memory_info:
                            logger.info(f"  Memory check: {memory_info['process_rss_mb']:.1f}MB")

            except KeyboardInterrupt:
                if self.verbose:
                    logger.warning(f"\n\nStudy interrupted after {len(self.all_runs_metrics)} runs")

            # --- Cleanup and return (runs regardless of which path was taken) ---
        if not self.use_process_isolation:
            final_cleanup = self.memory_manager.cleanup()
            if self.verbose and final_cleanup.memory_freed_mb:
                logger.info(f"\nFinal cleanup freed {final_cleanup.memory_freed_mb:.1f}MB")

        self.tracker.end_run()

        if self.verbose:
            successful = len(self.all_runs_metrics)
            logger.info("\nStudy Summary:")
            logger.info(f"  Successful runs: {successful}/{num_runs}")
            if self.failed_runs:
                logger.warning(f"  Failed runs: {self.failed_runs}")
            for metric_name, values in self.final_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    logger.info(f"  {metric_name}: {mean_val:.4f} (SD = {std_val:.4f})")

        results = VariabilityStudyResults(
            all_runs_metrics=self.all_runs_metrics,
            final_metrics=self.final_metrics,
            final_test_metrics=self.final_test_metrics,
            seed=self.seed,
            run_seeds=list(self._child_seeds),
        )

        if hasattr(self.tracker, "log_study_summary"):
            self.tracker.log_study_summary(results)

        return results

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
                stats[f"{metric_name}_mean"] = float(np.mean(values))
                stats[f"{metric_name}_std"] = float(np.std(values, ddof=1))
                stats[f"{metric_name}_min"] = float(np.min(values))
                stats[f"{metric_name}_max"] = float(np.max(values))

        return stats


# Module-level function for subprocess execution (must be picklable)
def _isolated_training_function(
    model_builder, config, train_data, val_data, test_data, epochs, run_id, run_seed=None
):
    """Training function executed in isolated subprocess.

    Exceptions are intentionally not caught here. They propagate to the
    subprocess worker (_cloudpickle_subprocess_worker or
    _standard_subprocess_worker), which puts {"success": False} on the
    result queue. This ensures training failures surface as proper failed
    runs rather than as empty history.
    """
    import gc

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

    # Build model in subprocess — inject run_seed so class-based builders
    # can pass it as random_state to sklearn estimators.
    if run_seed is not None:
        config = config.copy()
        config.set("run_seed", run_seed)
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
        run_seeds: List of per-run child seeds generated from the base
            seed via ``SeedSequence.spawn()``. Empty when results are
            reconstructed from MLflow or loaded from JSON.
    """

    all_runs_metrics: List[pd.DataFrame]
    final_metrics: Dict[str, List[float]]
    final_test_metrics: List[Dict[str, Any]]
    seed: Optional[int] = None
    run_seeds: List[int] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        """Number of successful runs."""
        return len(self.all_runs_metrics)

    def __repr__(self) -> str:
        metrics = list(self.final_metrics.keys())
        test_part = ""
        if self.has_test_data:
            test_keys = list({k for m in self.final_test_metrics for k in m if k != "run_id"})
            test_part = f", test_metrics={test_keys}"
        return (
            f"VariabilityStudyResults("
            f"n_runs={self.n_runs}, "
            f"seed={self.seed}, "
            f"metrics={metrics}"
            f"{test_part})"
        )

    @property
    def has_test_data(self) -> bool:
        """True if at least one run produced test-set metrics."""
        return bool(self.final_test_metrics)

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

        .. deprecated::
            Emits :class:`UserWarning`. Use :meth:`get_metric_values` instead.
            ``get_final_metrics()`` will be removed in v0.5.0.
        """
        warnings.warn(
            "get_final_metrics() is deprecated and will be removed in v0.5.0. "
            "Use get_metric_values() instead.",
            UserWarning,
            stacklevel=2,
        )
        metrics = {}
        for i, df in enumerate(self.all_runs_metrics):
            if metric_name in df.columns:
                metrics[f"run_{i + 1}"] = float(df[metric_name].iloc[-1])
        return metrics

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics across all runs."""
        return sorted(self.final_metrics.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame with one row per run.

        Test metrics are aligned by ``run_id``, not list position, so a
        failed test evaluation for one run does not corrupt the alignment
        of subsequent runs.
        """
        if not self.all_runs_metrics:
            return pd.DataFrame()

        # Index test metrics by run_id for reliable O(1) lookup.
        test_by_run: Dict[int, Dict] = {
            m["run_id"]: {k: v for k, v in m.items() if k != "run_id"}
            for m in self.final_test_metrics
            if "run_id" in m
        }

        rows = []
        for i, df in enumerate(self.all_runs_metrics):
            run_id = int(df["run_num"].iloc[0]) if "run_num" in df.columns else i + 1
            row: Dict[str, Any] = {"run_id": run_id}

            for col in df.columns:
                if col not in {"run_num", "epoch", "run_id"}:
                    row[f"final_{col}"] = float(df[col].iloc[-1])

            for key, value in test_by_run.get(run_id, {}).items():
                row[f"test_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def preferred_metric(
        self,
        base: str = "accuracy",
        context: Literal["scalar", "epoch"] = "scalar",
    ) -> str:
        """Return the preferred metric name for this study.

        The preferred metric depends on the *context* in which it will be
        used:

        - **scalar** (default): Terminal metric reporting. Returns
          ``f"test_{base}"`` when test data is present and the base metric
          was tracked, ``f"val_{base}"`` otherwise.
        - **epoch**: Per-epoch plotting and analysis. Always returns
          ``f"val_{base}"`` because test metrics are scalar (single
          final value) and per-epoch plotters cannot resolve them.

        Before returning, verifies the resolved key exists in the study's
        tracked metrics. If neither the primary nor alternate prefix
        resolves, raises a helpful ``KeyError`` naming the attempted keys,
        what metrics *are* tracked, and how to pass ``base`` explicitly.

        Note:
            ``final_test_metrics`` stores metrics under bare keys (e.g.
            ``"accuracy"``, not ``"test_accuracy"``). This method compares
            against the bare key and returns the prefixed form.

        Args:
            base: Base metric name without prefix. Default ``"accuracy"``.
            context: ``"scalar"`` for terminal reporting (prefers
                ``test_*`` when available), ``"epoch"`` for per-epoch
                plotters (always ``val_*``). Default ``"scalar"`` for
                backward compatibility.

        Returns:
            The most appropriate metric key for this study's results.

        Raises:
            KeyError: If neither the primary candidate nor the alternate
                prefix resolves to a tracked metric on this study. The
                error message names the attempted keys, the available
                val/test metrics, and suggests how to fix the call.
        """
        # Build the list of candidate prefixed keys to try, in priority order.
        # Scalar context: prefer test_* when test data present, fall back to val_*.
        # Epoch context: only val_* makes sense (test metrics are scalar).
        candidates: list[str] = []
        if context == "epoch":
            candidates.append(f"val_{base}")
        else:
            # scalar
            if self.has_test_data:
                tracked_test_set = {k for m in self.final_test_metrics for k in m if k != "run_id"}
                if base in tracked_test_set:
                    # Primary candidate: the bare-key test metric, returned with prefix.
                    candidates.append(f"test_{base}")
            candidates.append(f"val_{base}")

        # Try each candidate against self.final_metrics (for val_*) or the
        # test_metrics tracking (for test_*). A test_* candidate only reaches
        # here if it was already verified present above, so it's safe to return
        # directly. For val_*, verify against self.final_metrics.
        for candidate in candidates:
            if candidate.startswith("test_"):
                # Already verified present when the candidate was added.
                return candidate
            if candidate in self.final_metrics:
                return candidate

        # Neither primary nor alternate candidate resolved. Build a helpful
        # error message naming what was attempted and what's available.
        tracked_val = sorted(self.final_metrics.keys())
        tracked_test = (
            sorted({k for m in self.final_test_metrics for k in m if k != "run_id"})
            if self.has_test_data
            else []
        )

        message_parts = [
            f"preferred_metric(base={base!r}, context={context!r}) could not "
            f"resolve a tracked metric.",
            f"Tried: {candidates}.",
            f"Available val metrics: {tracked_val}.",
        ]
        if self.has_test_data:
            message_parts.append(f"Available test metrics: {tracked_test}.")

        # Try to suggest a better base: if the user asked for 'accuracy' but
        # we see something like 'mse' or 'r2', infer they're on a regression
        # study and offer the first available val metric as a hint.
        if tracked_val:
            suggested_base = tracked_val[0].replace("val_", "").replace("test_", "")
            message_parts.append(
                f"Did you mean base={suggested_base!r}? "
                f"Pass base explicitly to preferred_metric(), or pass "
                f"metric=<name> directly to the plotting or analysis function."
            )
        else:
            message_parts.append(
                "No val metrics tracked at all — this study may have failed "
                "to record metrics. Check the runner's metric extraction."
            )

        raise KeyError(" ".join(message_parts))

    def summarize(self) -> str:
        lines = [
            "Variability Study Results",
            "=" * 30,
            f"Successful runs: {self.n_runs}",
            f"Seed: {self.seed}",
        ]

        def _format_metric_block(metric_name: str, values: list) -> list:
            n = len(values)
            mean = np.mean(values)
            sd = np.std(values, ddof=1) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 else float("nan")
            block = [
                f"{metric_name}:",
                f"  N:                {n}",
                f"  Mean:             {mean:.4f}",
                f"  SD (sample, N-1): {sd:.4f}",
                f"  SE:               {se:.4f}",
                f"  Min:              {np.min(values):.4f}",
                f"  Max:              {np.max(values):.4f}",
            ]
            return block

        if self.has_test_data:
            lines += ["", "Test Set Metrics:", "-" * 20]
            test_keys = sorted({k for m in self.final_test_metrics for k in m if k != "run_id"})
            for key in test_keys:
                values = [m[key] for m in self.final_test_metrics if key in m]
                if values:
                    lines.extend(_format_metric_block(key, values))
            lines += ["", "Validation Metrics:", "-" * 20]
        else:
            lines += [
                "",
                "Note: no held-out test data — validation metrics shown. "
                "Provide test data via DataHandler for unbiased evaluation.",
            ]

        for metric_name, values in sorted(self.final_metrics.items()):
            if values:
                lines.extend(_format_metric_block(metric_name, values))

        return "\n".join(lines)

    def get_epoch_statistics(
        self,
        metric: str,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """Compute per-epoch mean, SD, SE, and confidence band across all runs.

        Aligns all runs at each epoch position. Runs with fewer epochs
        than the maximum are excluded from epochs beyond their last.

        Args:
            metric: Column name in per-run DataFrames, e.g.
                ``'val_accuracy'``, ``'train_loss'``.
            confidence: Confidence level for the t-based confidence interval
                on the per-epoch mean across runs. The interval uses a
                Student t-distribution with ``n_runs - 1`` degrees of freedom.
                Default 0.95.

        Returns:
            DataFrame with columns: ``epoch``, ``mean``, ``sd``,
            ``se``, ``ci_lower``, ``ci_upper``, ``n_runs``.

        Raises:
            ValueError: If no runs are available.
            KeyError: If *metric* is not found in any run's DataFrame.

        Example::

            stats = results.get_epoch_statistics("val_accuracy")
            import matplotlib.pyplot as plt
            plt.fill_between(
                stats["epoch"], stats["ci_lower"], stats["ci_upper"],
                alpha=0.2, label="95% CI"
            )
            plt.plot(stats["epoch"], stats["mean"], label="Mean")
            plt.legend()
        """

        if not self.all_runs_metrics:
            raise ValueError("No runs available.")

        if not (0 < confidence < 1):
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence!r}. " "Use 0.95 for 95%, not 95."
            )

        available_runs = [df for df in self.all_runs_metrics if metric in df.columns]
        if not available_runs:
            all_cols = sorted({col for df in self.all_runs_metrics for col in df.columns})
            raise KeyError(
                f"Metric '{metric}' not found in any run. " f"Available metrics: {all_cols}"
            )

        _min_len = min(len(df) for df in available_runs)
        _max_len = max(len(df) for df in available_runs)
        if _min_len != _max_len:
            warnings.warn(
                f"get_epoch_statistics('{metric}'): runs have unequal epoch counts "
                f"(min {_min_len}, max {_max_len}). Late-epoch statistics are "
                "computed from fewer runs. The 'n_runs' column reflects this.",
                UserWarning,
                stacklevel=2,
            )

        max_epochs = max(len(df) for df in available_runs)
        alpha = 1 - confidence
        rows = []

        for epoch_idx in range(max_epochs):
            epoch_vals = [
                float(df[metric].iloc[epoch_idx]) for df in available_runs if len(df) > epoch_idx
            ]
            if len(epoch_vals) < 2:
                continue

            arr = np.array(epoch_vals)
            mean = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1))
            se = sd / np.sqrt(len(arr))

            if len(arr) > 1:
                t_crit = _scipy_stats.t.ppf(1 - alpha / 2, df=len(arr) - 1)
                ci_half = t_crit * se
                ci_lower = float(mean - ci_half)
                ci_upper = float(mean + ci_half)
            else:
                ci_lower = float("nan")
                ci_upper = float("nan")

            rows.append(
                {
                    "epoch": epoch_idx + 1,
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_runs": len(epoch_vals),
                }
            )

        return pd.DataFrame(rows)

    def get_test_metric_values(self, metric: str) -> List[float]:
        """Get per-run final values for a test-set metric.

        Accepts either bare key (``'accuracy'``) or prefixed form
        (``'test_accuracy'``) — both are resolved against the bare key,
        which is how ``final_test_metrics`` actually stores values.

        Args:
            metric: Metric key, e.g. ``'accuracy'`` or ``'test_accuracy'``.

        Returns:
            List of values, one per run, in run order.

        Raises:
            KeyError: If the metric was not tracked or no test data was provided.
        """
        bare = metric[len("test_") :] if metric.startswith("test_") else metric
        values = [m[bare] for m in self.final_test_metrics if bare in m]
        if not values:
            available = sorted({k for m in self.final_test_metrics for k in m if k != "run_id"})
            raise KeyError(
                f"Test metric '{metric}' not found. "
                f"Available (stored without 'test_' prefix): {available}. "
                "If no test data was provided, use get_metric_values() instead."
            )
        return values

    def test_against_null(
        self,
        null_value: float = 0.5,
        metric: Optional[str] = None,
        alpha: float = 0.05,
    ) -> "StatisticalTestResult":
        """Test whether a metric's distribution differs from a null value.

        Applies a one-sample Wilcoxon signed-rank test to the per-run final
        values of ``metric``.

        Args:
            null_value: The null hypothesis value. Use 0.5 for chance-level
                binary classification accuracy.
            metric: Metric name. Must be a key in ``final_metrics``.
            alpha: Significance threshold (default 0.05).

        Returns:
            A StatisticalTestResult with ``is_significant``, ``p_value``,
            ``statistic``, and ``conclusion``.

        Note:
            The Wilcoxon signed-rank test requires at least 6 non-zero
            differences for reliable results. Use ``num_runs >= 20`` for
            reliable inference — consistent with the library-wide minimum
            for Mann-Whitney U tests.
        """
        from .analysis import _wilcoxon_signed_rank_impl

        if metric is None:
            metric = self.preferred_metric("accuracy")

            # Route to the correct metric store based on prefix
        if metric.startswith("test_"):
            try:
                values = pd.Series(self.get_test_metric_values(metric))
            except KeyError:
                available_test = sorted(
                    {k for m in self.final_test_metrics for k in m if k != "run_id"}
                )
                raise ValueError(
                    f"Metric '{metric}' not found in test metrics. "
                    f"Available test metrics: {available_test}. "
                    f"Available val metrics: {list(self.final_metrics.keys())}"
                )
        else:
            if metric not in self.final_metrics:
                available = list(self.final_metrics.keys())
                raise ValueError(f"Metric '{metric}' not found. Available: {available}")
            values = pd.Series(self.get_metric_values(metric))

        return _wilcoxon_signed_rank_impl(values, null_value=null_value, alpha=alpha)

    def compare_models_statistically(self, *args, **kwargs):
        """Removed in v0.3.10.

        This method applied Kruskal-Wallis to groups of one observation each
        (one per run), which is statistically incoherent — within-group variance
        is undefined with a single observation.

        To compare a single model's run distribution against a null value, use::

            results.test_against_null(null_value=0.5, metric="val_accuracy")

        To compare multiple distinct models against each other, use::

            ix.compare_models(models=[model_a, model_b], data=..., runs=20)
        """
        raise AttributeError(
            "compare_models_statistically() was removed in v0.3.10. "
            "It applied Kruskal-Wallis to single-observation groups (one per run), "
            "producing statistically incoherent results.\n\n"
            "For single-model null testing: results.test_against_null(null_value=0.5)\n"
            "For cross-model comparison:    ix.compare_models([model_a, model_b], data=...)"
        )

    def save(self, path: str) -> None:
        """Persist results to disk as a plain dict via pickle.

        Preserves all_runs_metrics, final_metrics, final_test_metrics,
        seed, and run_seeds. Restore with :meth:`load`.

        Args:
            path: File path. Recommended extension: ``.pkl``.
        """
        import pickle

        data = {
            "_schema_version": _ICTONYX_VERSION,
            "all_runs_metrics": self.all_runs_metrics,
            "final_metrics": self.final_metrics,
            "final_test_metrics": self.final_test_metrics,
            "seed": self.seed,
            "run_seeds": list(self.run_seeds),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "VariabilityStudyResults":
        """Restore results previously saved with :meth:`save`.

        Args:
            path: File path written by :meth:`save`.

        Returns:
            Reconstructed VariabilityStudyResults.
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        _schema = data.get("_schema_version")
        if _schema is None:
            warnings.warn(
                "Loading results saved without a schema version. "
                "This file predates ictonyx v0.4.5; compatibility is "
                "best-effort.",
                UserWarning,
                stacklevel=2,
            )
        elif _schema != _ICTONYX_VERSION:
            warnings.warn(
                f"Results schema version '{_schema}' does not match "
                f"current '{_ICTONYX_VERSION}'. The file was written by a "
                "different version of Ictonyx and may be incompatible.",
                UserWarning,
                stacklevel=2,
            )
        return cls(
            all_runs_metrics=data["all_runs_metrics"],
            final_metrics=data["final_metrics"],
            final_test_metrics=data["final_test_metrics"],
            seed=data.get("seed"),
            run_seeds=list(data.get("run_seeds", [])),
        )

    def to_json(self) -> str:
        """Serialise final metrics to a compact JSON string.

        Does not include ``all_runs_metrics`` (per-epoch DataFrames).
        Use :meth:`save` / :meth:`load` for full round-trip fidelity.

        Returns:
            JSON string containing ``final_metrics``, ``final_test_metrics``,
            ``seed``, and ``n_runs``.
        """
        import json

        return json.dumps(
            {
                "n_runs": self.n_runs,
                "seed": self.seed,
                "final_metrics": self.final_metrics,
                "final_test_metrics": self.final_test_metrics,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "VariabilityStudyResults":
        """Reconstruct a VariabilityStudyResults from a JSON string.

        Symmetric with :meth:`to_json`. Note that ``to_json`` does not
        serialize ``all_runs_metrics`` (per-epoch DataFrames), so the
        reconstructed object has an empty ``all_runs_metrics`` and
        empty ``run_seeds``. Use :meth:`save` / :meth:`load` for full
        round-trip fidelity including per-epoch history.

        Args:
            json_str: JSON string produced by :meth:`to_json`.

        Returns:
            VariabilityStudyResults with ``final_metrics``,
            ``final_test_metrics``, and ``seed`` populated from the
            JSON. ``all_runs_metrics`` is empty; ``run_seeds`` is empty.

        Raises:
            ValueError: If the JSON is malformed or missing required keys.

        Example:
            >>> serialized = results.to_json()
            >>> restored = VariabilityStudyResults.from_json(serialized)
            >>> assert restored.final_metrics == results.final_metrics
        """
        import json

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON passed to from_json: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"from_json expects a JSON object, got {type(data).__name__}.")

        required = ("final_metrics", "final_test_metrics")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"from_json JSON missing required keys: {missing}. "
                f"Keys present: {sorted(data.keys())}."
            )

        return cls(
            all_runs_metrics=[],
            final_metrics=data["final_metrics"],
            final_test_metrics=data["final_test_metrics"],
            seed=data.get("seed"),
            run_seeds=[],
        )

    @classmethod
    def from_mlflow_experiment(
        cls,
        experiment_name: str,
        metric: str,
        run_filter: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> "VariabilityStudyResults":
        """Reconstruct results from a completed MLflow experiment.

        Allows running ictonyx statistical analysis on existing MLflow
        experiments without re-running training.

        Args:
            experiment_name: Name of the MLflow experiment.
            metric: Metric to extract per run (e.g. 'val_accuracy').
            run_filter: Optional MLflow filter string
                (e.g. "params.seed = '42'").
            tracking_uri: MLflow tracking URI. Uses the environment's
                MLFLOW_TRACKING_URI if not set.

        Returns:
            VariabilityStudyResults with final_metrics populated from
            the experiment's runs. all_runs_metrics is empty —
            per-epoch history is not reconstructed from MLflow.

        Raises:
            ImportError: If mlflow is not installed.
            ValueError: If no matching runs are found, or if the
                requested metric is not present in the runs.
        """
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for from_mlflow_experiment(). "
                "Install with: pip install ictonyx[mlflow]"
            )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        runs_df = cast(
            pd.DataFrame,
            mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=run_filter or "",
                order_by=["start_time ASC"],
            ),
        )

        if runs_df.empty:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}' " f"with filter: {run_filter!r}"
            )

        col = f"metrics.{metric}"
        if col not in runs_df.columns:
            available = [
                c.replace("metrics.", "") for c in runs_df.columns if c.startswith("metrics.")
            ]
            raise ValueError(
                f"Metric '{metric}' not found in experiment '{experiment_name}'. "
                f"Available metrics: {available}"
            )

        values = runs_df[col].dropna().tolist()
        if not values:
            raise ValueError(
                f"Metric '{metric}' exists in experiment '{experiment_name}' "
                "but contains no non-null values."
            )

        return cls(
            all_runs_metrics=[],
            final_metrics={metric: values},
            final_test_metrics=[],
            seed=None,
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
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "se": float(values.std(ddof=1) / np.sqrt(n)),
                    "min": float(values.min()),
                    "max": float(values.max()),
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

        # Empty results (from dry_run=True) produces a DataFrame with no
        # columns. Sorting by param-grid columns would raise KeyError.
        # Return a typed empty DataFrame with the expected schema so
        # downstream consumers (summarize, external code) get a
        # well-formed result.
        if df.empty:
            schema_cols = list(self.param_grid.keys()) + [
                "mean",
                "sd",
                "se",
                "min",
                "max",
                "n",
            ]
            return pd.DataFrame(columns=schema_cols)

        sort_cols = list(self.param_grid.keys())
        return df.sort_values(sort_cols).reset_index(drop=True)

    def summarize(self, metric: Optional[str] = None) -> str:
        """Generate text summary of grid study results.

        For executed studies, returns a summary of configurations and
        their metric statistics. For dry-run results (empty
        ``self.results``), returns a preview showing the parameter grid
        and the planned configurations that would be executed.

        Args:
            metric: Metric to summarize. Defaults to self.metric.
        """
        metric = metric or self.metric

        # Dry-run path: no results to summarize, but self.param_grid
        # lets us preview what would have been run. Runs-per-config
        # is not preserved on a dry-run result in v0.4.7, so this
        # preview reports the planned configurations without a run
        # count. The architectural fix to carry num_runs through to
        # the dry-run result object is scheduled for v0.5.0.
        if not self.results:
            import itertools

            param_names = list(self.param_grid.keys())
            param_value_lists = [self.param_grid[k] for k in param_names]
            combinations = list(itertools.product(*param_value_lists))
            n_planned_configs = len(combinations)

            lines = [
                "Grid Study Results (Dry Run — not yet executed)",
                "=" * 40,
                "Parameter grid:",
            ]
            for name, param_values in self.param_grid.items():
                lines.append(f"  {name}: {param_values}")
            lines.extend(
                [
                    "",
                    f"Planned configurations: {n_planned_configs}",
                    f"Metric:                 {metric}",
                    "",
                    "Configurations to run:",
                ]
            )
            for i, combo in enumerate(combinations, 1):
                combo_dict = dict(zip(param_names, combo))
                lines.append(f"  {i:>3}. {combo_dict}")
            return "\n".join(lines)

        # Normal path: study was executed, report statistics.
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
    num_runs: int = 20,
    epochs_per_run: Optional[int] = None,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    gpu_memory_limit: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
) -> VariabilityStudyResults:
    """Run a complete variability study (convenience function).

    Creates an :class:`ExperimentRunner` and calls
    :meth:`~ExperimentRunner.run_study`. This is the lower-level entry
    point; most users should prefer :func:`~ictonyx.api.variability_study`.

    Args:
        model_builder: Callable ``f(ModelConfig) -> BaseModelWrapper``.
        data_handler: A :class:`~ictonyx.data.DataHandler` instance.
        model_config: A :class:`~ictonyx.config.ModelConfig` instance.
        num_runs: Number of training runs. Default 20.
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

    return runner.run_study(
        num_runs=num_runs,
        epochs_per_run=epochs_per_run,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def run_grid_study(
    model_builder: Callable,
    data_handler: "DataHandler",
    base_config: "ModelConfig",
    param_grid: Dict[str, List[Any]],
    num_runs: int = 20,
    metric: str = "val_accuracy",
    use_process_isolation: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    seed: Optional[int] = None,
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
        num_runs:              Number of training runs per configuration. Default 20.
        metric:                Primary metric for summary statistics.
        use_process_isolation: If True, each run executes in a subprocess.
                               Strongly recommended to prevent GPU memory
                               accumulation across configurations.
        dry_run:               If True, print execution plan and return
                               without training. Use to verify configuration
                               before committing to a long run.
        verbose:               If ``True``, log study progress.
                               Default ``True``.
        seed:                  Base random seed for reproducibility. Each
                               configuration receives an independent child seed
                               derived via ``SeedSequence.spawn()``. If ``None``,
                               each configuration uses a random seed.

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
            num_runs=20,
            use_process_isolation=True
        )
        print(results.summarize())
        df = results.to_dataframe()
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter with values.")

    for param_name, param_vals in param_grid.items():
        if not param_vals:
            raise ValueError(
                f"param_grid['{param_name}'] is empty. "
                f"Each parameter must have at least one value."
            )

    if num_runs < 20:
        warnings.warn(
            f"run_grid_study(): num_runs={num_runs} may be insufficient for "
            "reliable statistical inference across configurations. "
            "Consider num_runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # Build Cartesian product of all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    n_configs = len(combinations)
    total_runs = n_configs * num_runs

    if dry_run:
        if verbose:
            logger.info("Grid Study — Dry Run")
            logger.info("=" * 40)
            logger.info("Parameter grid:")
            for name, param_values in param_grid.items():
                logger.info(f"  {name}: {param_values}")
            logger.info(f"\nConfigurations:         {n_configs}")
            logger.info(f"Runs per configuration: {num_runs}")
            logger.info(f"Total training runs:    {total_runs}")
            logger.info(f"Metric:                 {metric}")
            logger.info(f"Process isolation:      {use_process_isolation}")
            logger.info("\nConfigurations to run:")
            for i, combo in enumerate(combinations, 1):
                logger.info(f"  {i:>3}. {combo}")
        return GridStudyResults(
            results={}, param_grid=param_grid, base_config=base_config, metric=metric
        )

    results_dict: Dict[Tuple, VariabilityStudyResults] = {}

    # Pre-generate independent child seeds via SeedSequence rather than
    # seed+i, which can produce correlated RNG states for nearby seeds.
    if seed is not None:
        _ss = np.random.SeedSequence(seed)
        config_seeds: List[Optional[int]] = [
            int(s.generate_state(1)[0]) for s in _ss.spawn(n_configs)
        ]
    else:
        config_seeds = [None] * n_configs

    for i, param_combo in enumerate(combinations, 1):
        if verbose:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Configuration {i}/{n_configs}: {param_combo}")
            logger.info(f"{'=' * 50}")

        config = base_config.copy().update(param_combo)

        result = run_variability_study(
            model_builder=model_builder,
            data_handler=data_handler,
            model_config=config,
            num_runs=num_runs,
            use_process_isolation=use_process_isolation,
            seed=config_seeds[i - 1],
        )

        key = GridStudyResults._config_key(param_combo)
        results_dict[key] = result

        try:
            values: pd.Series = pd.Series(result.get_metric_values(metric))
            if verbose:
                logger.info(
                    f"  Mean: {values.mean():.4f}  SD: {values.std(ddof=1):.4f}  "
                    f"SE: {values.std(ddof=1) / np.sqrt(len(values)):.4f}"
                )
        except KeyError:
            if verbose:
                logger.warning(f"  Warning: metric '{metric}' not found in results.")

    return GridStudyResults(
        results=results_dict, param_grid=param_grid, base_config=base_config, metric=metric
    )
