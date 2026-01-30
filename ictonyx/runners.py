# ictonyx/runners.py
"""
Experiment runners for variability studies with memory management.
Supports both standard and process-isolated execution modes.
"""

import gc
import warnings
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from .core import BaseModelWrapper
from .config import ModelConfig
from .data import DataHandler
from .memory import get_memory_manager, get_memory_info
from .loggers import BaseLogger

# The SYSTEM Logger (Standard Python Convention)
from .settings import logger

# Optional TensorFlow for cleanup
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False


class ExperimentRunner:
    """
    Engine for running variability studies with memory management.
    """

    def __init__(self,
                 model_builder: Callable[[ModelConfig], BaseModelWrapper],
                 data_handler: DataHandler,
                 model_config: ModelConfig,
                 tracker: Optional[BaseLogger] = None,  # <--- RENAMED
                 use_process_isolation: bool = False,
                 gpu_memory_limit: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize experiment runner.

        Args:
            tracker: Object to track experiment metrics (e.g. MLflowLogger).
                     If None, uses basic BaseLogger.
        """
        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config

        # The Experiment Tracker (Records results/metrics)
        self.tracker = tracker or BaseLogger()  # <--- RENAMED ATTR

        self.use_process_isolation = use_process_isolation
        self.gpu_memory_limit = gpu_memory_limit
        self.verbose = verbose

        # Initialize memory manager
        self.memory_manager = get_memory_manager(
            use_process_isolation=use_process_isolation,
            gpu_memory_limit=gpu_memory_limit
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
            self.train_data = data_dict['train_data']
            self.val_data = data_dict.get('val_data')
            self.test_data = data_dict.get('test_data')

            if verbose:
                logger.info(f"Data loaded successfully")
                if self.val_data is None:
                    logger.warning("No validation data provided")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        # Initialize result storage
        self.all_runs_metrics: List[pd.DataFrame] = []
        self.final_val_accuracies: List[float] = []
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
            raise ValueError(
                f"model_builder must be picklable for process isolation: {e}"
            )

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

    def _run_single_fit(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Run a single training iteration."""
        if self.use_process_isolation:
            return self._run_single_fit_isolated(run_id, epochs)
        else:
            return self._run_single_fit_standard(run_id, epochs)

    def _run_single_fit_isolated(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Execute training in isolated subprocess."""
        if self.verbose:
            logger.info(f" - Run {run_id}: Starting in isolated process...")

        # Log run start (Metric Tracker)
        self.tracker.log_params({'run_id': run_id, 'mode': 'isolated'})

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
                run_id
            )
        )

        # Process results
        if result['success']:
            try:
                # Extract history
                history_data = result['result']['history']
                if not history_data:
                    if self.verbose:
                        logger.warning(f" - Run {run_id}: No training history returned")
                    self.failed_runs.append(run_id)
                    return None

                # Create DataFrame
                history_df = pd.DataFrame(history_data)
                history_df['run_num'] = run_id
                history_df['epoch'] = range(1, len(history_df) + 1)

                # Standardize column names
                history_df.rename(columns={
                    'accuracy': 'train_accuracy',
                    'loss': 'train_loss'
                }, inplace=True)

                # Store validation accuracy
                if 'val_accuracy' in history_df.columns:
                    final_val_acc = float(history_df['val_accuracy'].iloc[-1])
                    self.final_val_accuracies.append(final_val_acc)
                    self.tracker.log_metric('final_val_accuracy', final_val_acc, step=run_id)

                # Store test metrics
                test_metrics = result['result'].get('test_metrics')
                if test_metrics:
                    self.final_test_metrics.append(test_metrics)
                    for key, value in test_metrics.items():
                        self.tracker.log_metric(f'final_test_{key}', value, step=run_id)

                if self.verbose:
                    logger.info(f" - Run {run_id}: Completed successfully (isolated)")

                return history_df

            except Exception as e:
                if self.verbose:
                    logger.error(f" - Run {run_id}: Failed to process results: {e}")
                self.failed_runs.append(run_id)
                return None
        else:
            # Training failed
            error_msg = result.get('error', 'Unknown error')
            if self.verbose:
                logger.error(f" - Run {run_id}: Failed - {error_msg}")
                if 'traceback' in result and self.verbose:
                    logger.info(f"   Traceback: {result['traceback'][:500]}...")

            self.failed_runs.append(run_id)
            return None

    def _run_single_fit_standard(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """Execute training in standard mode (in-process)."""
        if self.verbose:
            logger.info(f" - Run {run_id}: Training...")

        # Log run start (Metric Tracker)
        self.tracker.log_params({'run_id': run_id, 'mode': 'standard'})

        wrapped_model = None
        try:
            # Build model
            wrapped_model = self.model_builder(self.model_config)

            # Train
            wrapped_model.fit(
                train_data=self.train_data,
                validation_data=self.val_data,
                epochs=epochs,
                batch_size=self.model_config.get('batch_size', 32),
                verbose=self.model_config.get('verbose', 0)
            )

            # Extract history
            if not hasattr(wrapped_model, 'history') or wrapped_model.history is None:
                if self.verbose:
                    logger.warning(f" - Run {run_id}: No training history produced")
                self.failed_runs.append(run_id)
                return None

            # Create DataFrame from history
            if hasattr(wrapped_model.history, 'history'):
                history_dict = wrapped_model.history.history
            else:
                history_dict = wrapped_model.history

            history_df = pd.DataFrame(history_dict)
            history_df['run_num'] = run_id
            history_df['epoch'] = range(1, len(history_df) + 1)

            # Standardize column names
            history_df.rename(columns={
                'accuracy': 'train_accuracy',
                'loss': 'train_loss'
            }, inplace=True)

            # Store validation accuracy
            if self.val_data is not None and 'val_accuracy' in history_df.columns:
                final_val_acc = float(history_df['val_accuracy'].iloc[-1])
                self.final_val_accuracies.append(final_val_acc)
                self.tracker.log_metric('final_val_accuracy', final_val_acc, step=run_id)

            # Evaluate on test data
            if self.test_data is not None:
                try:
                    test_metrics = wrapped_model.evaluate(data=self.test_data)
                    self.final_test_metrics.append(test_metrics)
                    for key, value in test_metrics.items():
                        self.tracker.log_metric(f'final_test_{key}', value, step=run_id)
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"   Warning: Test evaluation failed: {e}")

            if self.verbose:
                logger.info(f" - Run {run_id}: Completed successfully")

            return history_df

        except Exception as e:
            if self.verbose:
                logger.error(f" - Run {run_id}: Failed with error: {e}")
            self.failed_runs.append(run_id)
            return None

        finally:
            # Cleanup
            if wrapped_model is not None:
                try:
                    if hasattr(wrapped_model, 'cleanup'):
                        wrapped_model.cleanup()
                    del wrapped_model
                except:
                    pass

            # Perform memory cleanup
            cleanup_result = self.memory_manager.cleanup()
            if self.verbose and cleanup_result.memory_freed_mb:
                if cleanup_result.memory_freed_mb > 10:  # Only report significant cleanup
                    logger.info(f"   Freed {cleanup_result.memory_freed_mb:.1f}MB")

    def run_study(self, num_runs: int = 5,
                  epochs_per_run: Optional[int] = None,
                  stop_on_failure_rate: float = 0.5) -> Tuple[
        List[pd.DataFrame], List[float], List[Dict[str, Any]]]:
        """Execute the complete variability study."""
        if epochs_per_run is None:
            epochs_per_run = self.model_config.get('epochs', 10)

        # Log study parameters (Metric Tracker)
        self.tracker.log_params({
            'num_runs': num_runs,
            'epochs_per_run': epochs_per_run,
            'use_process_isolation': self.use_process_isolation,
            'gpu_memory_limit': self.gpu_memory_limit
        })
        self.tracker.log_params(self.model_config.params)

        # Print study configuration (System Logger)
        if self.verbose:
            mode = "with process isolation" if self.use_process_isolation else "in standard mode"
            logger.info(f"\nStarting Variability Study")
            logger.info(f"  Runs: {num_runs}")
            logger.info(f"  Epochs per run: {epochs_per_run}")
            logger.info(f"  Execution mode: {mode}")
            if self.gpu_memory_limit:
                logger.info(f"  GPU memory limit: {self.gpu_memory_limit}MB")
            logger.info("")

        # Execute runs
        try:
            for i in range(num_runs):
                # Check failure rate
                if i > 0:
                    failure_rate = len(self.failed_runs) / (i + 1)
                    if failure_rate > stop_on_failure_rate:
                        if self.verbose:
                            logger.error(f"\nStopping due to high failure rate: {failure_rate:.1%}")
                        break

                # Run single training
                metrics_df = self._run_single_fit(run_id=i + 1, epochs=epochs_per_run)
                if metrics_df is not None:
                    self.all_runs_metrics.append(metrics_df)

                # Log memory info periodically
                if (i + 1) % 10 == 0 and self.verbose:
                    memory_info = get_memory_info()
                    if 'process_rss_mb' in memory_info:
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
            if self.final_val_accuracies:
                mean_acc = np.mean(self.final_val_accuracies)
                std_acc = np.std(self.final_val_accuracies)
                logger.info(f"  Val accuracy: {mean_acc:.4f} (SD = {std_acc:.4f})")

        return self.all_runs_metrics, self.final_val_accuracies, self.final_test_metrics

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the completed study."""
        stats = {
            'total_runs': len(self.all_runs_metrics) + len(self.failed_runs),
            'successful_runs': len(self.all_runs_metrics),
            'failed_runs': len(self.failed_runs),
            'failure_rate': len(self.failed_runs) / max(1, len(self.all_runs_metrics) + len(self.failed_runs))
        }

        if self.final_val_accuracies:
            stats.update({
                'val_accuracy_mean': np.mean(self.final_val_accuracies),
                'val_accuracy_std': np.std(self.final_val_accuracies),
                'val_accuracy_min': np.min(self.final_val_accuracies),
                'val_accuracy_max': np.max(self.final_val_accuracies)
            })

        return stats


# Module-level function for subprocess execution (must be picklable)
def _isolated_training_function(model_builder, config, train_data,
                                val_data, test_data, epochs, run_id):
    """
    Training function executed in isolated subprocess.
    """
    import gc

    try:
        # Build model in subprocess
        model = model_builder(config)

        # Train model
        model.fit(
            train_data=train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=config.get('batch_size', 32),
            verbose=config.get('verbose', 0)
        )

        # Extract history
        history = {}
        if hasattr(model, 'history'):
            if hasattr(model.history, 'history'):
                # Keras History object
                history = dict(model.history.history)
            else:
                # Already a dict
                history = dict(model.history)

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
                test_metrics = {'error': str(e)}

        # Cleanup before returning
        if hasattr(model, 'cleanup'):
            model.cleanup()
        del model
        gc.collect()

        return {
            'history': history,
            'test_metrics': test_metrics,
            'run_id': run_id
        }

    except Exception as e:
        # Return error information
        import traceback
        return {
            'history': {},
            'test_metrics': {},
            'run_id': run_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@dataclass
class VariabilityStudyResults:
    """
    Container for variability study results with analysis methods.
    """
    all_runs_metrics: List[pd.DataFrame]
    final_val_accuracies: List[float]
    final_test_metrics: List[Dict[str, Any]]

    @property
    def n_runs(self) -> int:
        """Number of successful runs."""
        return len(self.all_runs_metrics)

    def get_final_metrics(self, metric_name: str = 'val_accuracy') -> Dict[str, float]:
        """Extract final metric values for each run."""
        metrics = {}
        for i, df in enumerate(self.all_runs_metrics):
            if metric_name in df.columns:
                metrics[f'run_{i + 1}'] = float(df[metric_name].iloc[-1])
        return metrics

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics across all runs."""
        if not self.all_runs_metrics:
            return []

        # Get all unique column names from all DataFrames
        all_columns = set()
        for df in self.all_runs_metrics:
            all_columns.update(df.columns)

        # Exclude non-metric columns
        exclude = {'run_num', 'epoch', 'run_id'}
        metrics = sorted([col for col in all_columns if col not in exclude])

        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame with one row per run."""
        if not self.all_runs_metrics:
            return pd.DataFrame()

        rows = []
        for i, df in enumerate(self.all_runs_metrics):
            row = {'run_id': i + 1}

            # Get final values for all metrics
            for col in df.columns:
                if col not in {'run_num', 'epoch', 'run_id'}:
                    row[f'final_{col}'] = float(df[col].iloc[-1])

            # Add test metrics if available
            if i < len(self.final_test_metrics):
                for key, value in self.final_test_metrics[i].items():
                    row[f'test_{key}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def summarize(self) -> str:
        """Generate text summary of results."""
        lines = [
            "Variability Study Results",
            "=" * 30,
            f"Successful runs: {self.n_runs}",
        ]

        if self.final_val_accuracies:
            mean_acc = np.mean(self.final_val_accuracies)
            std_acc = np.std(self.final_val_accuracies)
            lines.extend([
                f"Final validation accuracy:",
                f"  Mean: {mean_acc:.4f}",
                f"  Std:  {std_acc:.4f}",
                f"  Min:  {np.min(self.final_val_accuracies):.4f}",
                f"  Max:  {np.max(self.final_val_accuracies):.4f}"
            ])

        return "\n".join(lines)

    def compare_models_statistically(self,
                                     metric_name: str = 'val_accuracy',
                                     alpha: float = 0.05,
                                     correction_method: str = 'holm') -> Dict[str, Any]:
        """
        Perform statistical comparison of runs for the specified metric.
        """
        from .analysis import compare_multiple_models

        if not self.all_runs_metrics:
            raise ValueError("No run metrics available for statistical comparison")

        # Extract the specified metric for each run
        final_metrics = self.get_final_metrics(metric_name)

        if not final_metrics:
            available = self.get_available_metrics()
            raise ValueError(
                f"Metric '{metric_name}' not found in results. "
                f"Available metrics: {available}"
            )

        # Convert to the format expected by compare_multiple_models
        metrics_dict = {}
        for run_name, value in final_metrics.items():
            run_idx = int(run_name.split('_')[-1]) - 1

            if run_idx < len(self.all_runs_metrics):
                df = self.all_runs_metrics[run_idx]
                if metric_name in df.columns:
                    metrics_dict[run_name] = df[metric_name]
                else:
                    metrics_dict[run_name] = pd.Series([value], name=run_name)
            else:
                metrics_dict[run_name] = pd.Series([value], name=run_name)

        # Perform statistical comparison
        comparison_results = compare_multiple_models(
            model_results=metrics_dict,
            alpha=alpha,
            correction_method=correction_method
        )

        return comparison_results


# Convenience function
def run_variability_study(
        model_builder: Callable[[ModelConfig], BaseModelWrapper],
        data_handler: DataHandler,
        model_config: ModelConfig,
        num_runs: int = 5,
        epochs_per_run: Optional[int] = None,
        tracker: Optional[BaseLogger] = None,  # <--- RENAMED
        use_process_isolation: bool = False,
        gpu_memory_limit: Optional[int] = None,
        verbose: bool = True) -> VariabilityStudyResults:
    """
    Run a complete variability study.
    """
    runner = ExperimentRunner(
        model_builder=model_builder,
        data_handler=data_handler,
        model_config=model_config,
        tracker=tracker,  # <--- RENAMED
        use_process_isolation=use_process_isolation,
        gpu_memory_limit=gpu_memory_limit,
        verbose=verbose
    )

    all_metrics, final_accs, test_metrics = runner.run_study(
        num_runs=num_runs,
        epochs_per_run=epochs_per_run
    )

    return VariabilityStudyResults(all_metrics, final_accs, test_metrics)