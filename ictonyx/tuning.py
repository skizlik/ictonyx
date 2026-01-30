import numpy as np
from typing import Dict, Any, Callable
import warnings

# Optional hyperopt dependency
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from hyperopt.pyll.base import scope

    HAS_HYPEROPT = True
except ImportError:
    fmin = tpe = hp = Trials = STATUS_OK = scope = None
    HAS_HYPEROPT = False

# Optional TensorFlow for memory management
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

from .core import BaseModelWrapper
from .config import ModelConfig
from .data import DataHandler


class HyperparameterTuner:
    """
    A class to automate hyperparameter tuning for a given model using Hyperopt.

    Requires hyperopt to be installed: pip install hyperopt
    """

    def __init__(self, model_builder: Callable[[ModelConfig], BaseModelWrapper],
                 data_handler: DataHandler,
                 model_config: ModelConfig,
                 metric: str = 'val_loss'):
        """
        Initialize the hyperparameter tuner.

        Args:
            model_builder: Function that takes ModelConfig and returns BaseModelWrapper
            data_handler: DataHandler instance for loading data
            model_config: Base ModelConfig that will be updated with hyperparameters
            metric: Metric to optimize (e.g., 'val_loss', 'val_accuracy')
        """
        if not HAS_HYPEROPT:
            raise ImportError("Hyperopt is required for hyperparameter tuning. "
                              "Install with: pip install hyperopt")

        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.metric = metric
        self.trials = Trials()

        # Load data once during initialization
        print("Loading data for hyperparameter tuning...")
        try:
            data_dict = self.data_handler.load()
            self.train_data = data_dict['train_data']
            self.val_data = data_dict.get('val_data')

            if self.val_data is None:
                raise ValueError("Validation data is required for hyperparameter tuning. "
                                 "Ensure your data handler provides 'val_data'.")

            print(f"Data loaded successfully. Training on: {type(self.train_data)}")

        except Exception as e:
            raise RuntimeError(f"Failed to load data for hyperparameter tuning: {e}")

    def tune(self, param_space: Dict[str, Any], max_evals: int = 100) -> Dict[str, Any]:
        """
        Runs hyperparameter tuning using hyperopt.

        Args:
            param_space: The hyperparameter search space using hyperopt distributions
            max_evals: The number of hyperparameter combinations to try

        Returns:
            Dict containing the best parameters found
        """
        if not isinstance(param_space, dict) or not param_space:
            raise ValueError("param_space must be a non-empty dictionary of hyperopt distributions")

        if max_evals <= 0:
            raise ValueError("max_evals must be positive")

        print(f"Starting hyperparameter optimization with {max_evals} evaluations...")
        print(f"Optimizing metric: {self.metric}")
        print(f"Search space: {list(param_space.keys())}")

        def objective(params: Dict[str, Any]) -> Dict[str, Any]:
            """
            Objective function for hyperopt optimization.

            Args:
                params: Dictionary of hyperparameters to evaluate

            Returns:
                Dictionary with 'loss' (value to minimize) and 'status'
            """
            trial_num = len(self.trials.trials) + 1
            print(f"\nTrial {trial_num}/{max_evals}: {params}")

            try:
                # Suppress warnings during optimization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Clear TensorFlow session if available to prevent memory leaks
                    if HAS_TENSORFLOW:
                        tf.keras.backend.clear_session()

                    # Create a copy of the base config to avoid modifying the original
                    trial_config = ModelConfig(self.model_config.params.copy())
                    trial_config.merge(params)

                    # Build a new model with the trial hyperparameters
                    wrapped_model = self.model_builder(trial_config)

                    # Train the model
                    history = wrapped_model.fit(
                        train_data=self.train_data,
                        validation_data=self.val_data,
                        epochs=getattr(trial_config, 'epochs', 10),
                        verbose=0  # Keep training quiet during tuning
                    )

                    # Extract the target metric
                    if not hasattr(history, 'history') or not history.history:
                        raise ValueError("Model training did not return a valid history object")

                    if self.metric not in history.history:
                        available_metrics = list(history.history.keys())
                        raise ValueError(f"Metric '{self.metric}' not found in training history. "
                                         f"Available metrics: {available_metrics}")

                    # Get the final value of the specified metric
                    metric_values = history.history[self.metric]
                    if not metric_values:
                        raise ValueError(f"No values found for metric '{self.metric}'")

                    final_metric_value = metric_values[-1]

                    # Validate the metric value
                    if not isinstance(final_metric_value, (int, float)) or np.isnan(final_metric_value):
                        raise ValueError(f"Invalid metric value: {final_metric_value}")

                    # Convert to loss (hyperopt minimizes)
                    if 'accuracy' in self.metric.lower() or 'precision' in self.metric.lower() or 'recall' in self.metric.lower():
                        # For metrics we want to maximize, return negative
                        loss = -float(final_metric_value)
                    else:
                        # For metrics we want to minimize (loss, error)
                        loss = float(final_metric_value)

                    print(f"  Result: {self.metric} = {final_metric_value:.4f}")

                    return {
                        'loss': loss,
                        'status': STATUS_OK,
                        'eval_time': getattr(history, 'params', {}).get('epochs', 0),
                        'final_metric': final_metric_value
                    }

            except Exception as e:
                print(f"  Trial failed: {str(e)}")
                # Return a high loss value for failed trials
                return {
                    'loss': float('inf'),
                    'status': STATUS_OK,  # Don't fail the entire optimization
                    'error': str(e)
                }

        # Run the optimization
        try:
            best_params = fmin(
                fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=self.trials,
                verbose=False  # We handle our own logging
            )

            # Get the best trial results
            best_trial = self.trials.best_trial
            best_loss = best_trial['result']['loss']

            # Convert loss back to metric value for reporting
            if 'accuracy' in self.metric.lower() or 'precision' in self.metric.lower() or 'recall' in self.metric.lower():
                best_metric_value = -best_loss
            else:
                best_metric_value = best_loss

            print(f"\nOptimization completed!")
            print(f"Best {self.metric}: {best_metric_value:.4f}")
            print(f"Best parameters: {best_params}")

            return best_params

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            if self.trials.trials:
                # Return the best parameters found so far
                best_params = self.trials.argmin
                print(f"Returning best parameters found in {len(self.trials.trials)} trials: {best_params}")
                return best_params
            else:
                raise RuntimeError("No trials completed before interruption")

        except Exception as e:
            raise RuntimeError(f"Hyperparameter optimization failed: {e}")

    def get_best_trial(self) -> Dict[str, Any]:
        """
        Get details about the best trial after optimization.

        Returns:
            Dictionary with best trial information
        """
        if not self.trials.trials:
            raise RuntimeError("No trials have been run yet. Call tune() first.")

        best_trial = self.trials.best_trial
        best_loss = best_trial['result']['loss']

        # Convert loss back to metric value
        if 'accuracy' in self.metric.lower() or 'precision' in self.metric.lower() or 'recall' in self.metric.lower():
            best_metric_value = -best_loss
        else:
            best_metric_value = best_loss

        return {
            'best_params': self.trials.argmin,
            'best_metric_value': best_metric_value,
            'total_trials': len(self.trials.trials),
            'successful_trials': len([t for t in self.trials.trials if t['result']['loss'] != float('inf')])
        }

    def get_trials_dataframe(self) -> 'pd.DataFrame':
        """
        Get a pandas DataFrame with all trial results.

        Returns:
            DataFrame with trial parameters and results
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for trials DataFrame. Install with: pip install pandas")

        if not self.trials.trials:
            raise RuntimeError("No trials have been run yet. Call tune() first.")

        # Extract data from trials
        trial_data = []
        for i, trial in enumerate(self.trials.trials):
            row = {'trial_id': i}

            # Add parameters
            if 'misc' in trial and 'vals' in trial['misc']:
                for param_name, param_values in trial['misc']['vals'].items():
                    if param_values:  # Check if list is not empty
                        row[param_name] = param_values[0]

            # Add results
            result = trial.get('result', {})
            row['loss'] = result.get('loss', float('inf'))
            row['final_metric'] = result.get('final_metric', None)
            row['status'] = result.get('status', 'UNKNOWN')

            if 'error' in result:
                row['error'] = result['error']

            trial_data.append(row)

        return pd.DataFrame(trial_data)


# Utility function for common search spaces
def create_search_space() -> Dict[str, Any]:
    """
    Creates common hyperparameter search spaces for different model types.

    Returns:
        Dictionary of example search space definitions
    """
    if not HAS_HYPEROPT:
        raise ImportError("Hyperopt required to create search spaces. Install with: pip install hyperopt")

    return {
        'neural_network': {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
            'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
            'epochs': hp.choice('epochs', [10, 20, 50, 100]),
            'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)
        },
        'xgboost': {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 500]),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0)
        },
        'random_forest': {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 500]),
            'max_depth': hp.choice('max_depth', [None, 5, 10, 15, 20]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4])
        }
    }