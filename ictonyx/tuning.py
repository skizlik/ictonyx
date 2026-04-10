import time
import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from .settings import logger

# Optional hyperopt dependency
try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
    from hyperopt.pyll.base import scope

    HAS_HYPEROPT = True
except ImportError:
    fmin = tpe = hp = Trials = STATUS_OK = scope = None
    HAS_HYPEROPT = False

# Optional Optuna dependency
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAS_OPTUNA = False

# Optional TensorFlow for memory management
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

from .config import ModelConfig
from .core import BaseModelWrapper
from .data import DataHandler


def _should_minimize(metric: str) -> bool:
    """Return True if lower values are better for this metric."""
    maximize_keywords = ("accuracy", "precision", "recall", "r2", "f1", "auc")
    return not any(kw in metric.lower() for kw in maximize_keywords)


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna as the primary backend.

    Trains each trial configuration ``n_evals_per_trial`` times and uses
    the mean metric as the objective, consistent with the library's thesis
    that a single training run is not a reliable measurement.

    Requires optuna: ``pip install ictonyx[tuning]``

    The legacy Hyperopt backend is still available but deprecated and will
    be removed in v0.5.0.
    """

    def __init__(
        self,
        model_builder: Callable[[ModelConfig], BaseModelWrapper],
        data_handler: DataHandler,
        model_config: ModelConfig,
        metric: str = "val_loss",
        n_evals_per_trial: int = 3,
        stability_weight: float = 0.0,
    ):
        """
        Args:
            model_builder: Function returning BaseModelWrapper given ModelConfig.
            data_handler: DataHandler for loading data. Data is loaded lazily
                at tune() time, not during construction.
            model_config: Base ModelConfig updated with trial parameters.
            metric: Metric to optimize. Default 'val_loss'.
            n_evals_per_trial: Independent training runs per trial. The trial
                objective is the mean metric across these runs. Default 3.
                Set to 1 to reproduce single-run (old) behavior.
            stability_weight: If > 0, penalizes run-to-run variance.
                Minimisation objectives (e.g. ``'val_loss'``):
                ``objective = mean + stability_weight * std``.
                Maximisation objectives (e.g. ``'val_accuracy'``):
                ``objective = mean - stability_weight * std``.
                In both cases, higher variance worsens the objective score.
                Default 0.0 (variance not penalized).
        """
        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.metric = metric
        self.n_evals_per_trial = n_evals_per_trial
        self.stability_weight = stability_weight
        self._data_dict: Optional[Dict[str, Any]] = None
        self.train_data = None
        self.val_data = None
        self.best_params: Optional[Dict[str, Any]] = None
        self._optuna_study = None
        # Legacy hyperopt trials object — populated only when using hyperopt backend
        self.trials = Trials() if HAS_HYPEROPT else None

    def _resolve_direction(self, direction: str) -> str:
        """Resolve 'auto' to 'minimize' or 'maximize' based on metric name."""
        if direction != "auto":
            return direction
        maximize_keywords = ("accuracy", "precision", "recall", "r2", "f1", "auc")
        return (
            "maximize" if any(kw in self.metric.lower() for kw in maximize_keywords) else "minimize"
        )

    def _ensure_data_loaded(self) -> None:
        """Load data lazily on first call. Validates val_data is present."""
        if self._data_dict is not None:
            return
        logger.info("Loading data for hyperparameter tuning...")
        self._data_dict = self.data_handler.load()
        self.train_data = self._data_dict["train_data"]
        self.val_data = self._data_dict.get("val_data")
        if self.val_data is None:
            raise ValueError(
                "Validation data is required for hyperparameter tuning. "
                "Ensure your DataHandler provides val_data."
            )
        logger.info(f"Data loaded. Training on: {type(self.train_data)}")

    def tune(
        self,
        param_space: Dict[str, Any],
        max_evals: int = 100,
        direction: str = "auto",
        timeout: Optional[float] = None,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimisation using Optuna.

        Args:
            param_space: Dict mapping parameter names to Optuna distributions.
                Example::

                    import optuna
                    {
                        "learning_rate": optuna.distributions.FloatDistribution(
                            1e-4, 1e-1, log=True),
                        "n_estimators": optuna.distributions.IntDistribution(50, 500),
                    }

            max_evals: Number of trials. Default 100.
            direction: ``'minimize'``, ``'maximize'``, or ``'auto'``. When
                ``'auto'``, infers from metric name: maximise for
                accuracy/f1/r2/auc; minimise for loss/mse/mae. Default
                ``'auto'``.
            timeout: Optional wall-clock time limit in seconds.
            n_jobs: Number of parallel Optuna workers. Default 1.

        Returns:
            Dict of best hyperparameters found.
        """
        if not HAS_OPTUNA:
            # Fall back to deprecated hyperopt backend
            return self._tune_hyperopt(param_space, max_evals)

        if not isinstance(param_space, dict) or not param_space:
            raise ValueError("param_space must be a non-empty dict of Optuna distributions.")

        self._ensure_data_loaded()
        resolved_direction = self._resolve_direction(direction)

        def objective(trial: "optuna.Trial") -> float:
            params = {}
            for name, dist in param_space.items():
                params[name] = trial._suggest(name, dist)

            config = self.model_config.copy()
            config.update(params)

            seed_seq = np.random.SeedSequence(trial.number)
            child_seeds = seed_seq.spawn(self.n_evals_per_trial)
            metric_values = []

            for i, child_seed in enumerate(child_seeds):
                run_seed = int(child_seed.generate_state(1)[0])
                run_config = config.copy()
                run_config["run_seed"] = run_seed
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrapper = self.model_builder(run_config)
                        wrapper.fit(
                            train_data=self.train_data,
                            validation_data=self.val_data,
                            epochs=run_config.get("epochs", 10),
                            verbose=0,
                        )
                    result = wrapper.evaluate(data=self.val_data)
                    if self.metric in result:
                        metric_values.append(float(result[self.metric]))
                    elif (
                        wrapper.training_result is not None
                        and self.metric in wrapper.training_result.history
                    ):
                        vals = wrapper.training_result.history[self.metric]
                        if vals:
                            metric_values.append(float(vals[-1]))
                except Exception as e:
                    logger.warning(f"Trial {trial.number} run {i + 1} failed: {e}")

            if not metric_values:
                raise optuna.TrialPruned()

            mean_val = float(np.mean(metric_values))

            if self.stability_weight > 0 and len(metric_values) > 1:
                std_val = float(np.std(metric_values, ddof=1))
                penalty = self.stability_weight * std_val
                return (
                    mean_val + penalty if resolved_direction == "minimize" else mean_val - penalty
                )

            return mean_val

        study = optuna.create_study(direction=resolved_direction)
        study.optimize(objective, n_trials=max_evals, timeout=timeout, n_jobs=n_jobs)

        self._optuna_study = study
        best = study.best_params
        self.best_params = best
        logger.info(f"Optimisation complete. Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best}")
        return best

    def _tune_hyperopt(self, param_space: Dict[str, Any], max_evals: int) -> Dict[str, Any]:
        """Legacy Hyperopt backend. Deprecated — will be removed in v0.5.0."""
        warnings.warn(
            "The Hyperopt backend is deprecated in v0.4.0 and will be removed in v0.5.0. "
            "Switch to Optuna by passing Optuna distributions to tune(). "
            "Install Optuna with: pip install ictonyx[tuning]",
            DeprecationWarning,
            stacklevel=3,
        )
        if not HAS_HYPEROPT:
            raise ImportError(
                "Hyperopt is required for the legacy backend. "
                "Install with: pip install hyperopt, or switch to Optuna."
            )

        if not isinstance(param_space, dict) or not param_space:
            raise ValueError("param_space must be a non-empty dictionary of hyperopt distributions")
        if max_evals <= 0:
            raise ValueError("max_evals must be positive")

        # Load data if not already loaded
        self._ensure_data_loaded()

        logger.info(f"Starting hyperopt optimization with {max_evals} evaluations...")

        def objective(params: Dict[str, Any]) -> Dict[str, Any]:
            assert self.trials is not None
            trial_num = len(self.trials.trials) + 1
            logger.info(f"\nTrial {trial_num}/{max_evals}: {params}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if HAS_TENSORFLOW:
                        tf.keras.backend.clear_session()
                    trial_config = ModelConfig(self.model_config.params.copy())
                    trial_config.update(params)
                    wrapped_model = self.model_builder(trial_config)
                    _start = time.time()
                    wrapped_model.fit(
                        train_data=self.train_data,
                        validation_data=self.val_data,
                        epochs=trial_config.get("epochs", 10),
                        verbose=0,
                    )
                    _elapsed = time.time() - _start
                    training_result = wrapped_model.training_result
                    if training_result is None or not training_result.history:
                        raise ValueError("Model training did not produce a TrainingResult.")
                    history_dict = training_result.history
                    if self.metric not in history_dict:
                        raise ValueError(
                            f"Metric '{self.metric}' not found. "
                            f"Available: {list(history_dict.keys())}"
                        )
                    metric_values = history_dict[self.metric]
                    final_value = metric_values[-1]
                    loss = (
                        -float(final_value)
                        if not _should_minimize(self.metric)
                        else float(final_value)
                    )
                    return {
                        "loss": loss,
                        "status": STATUS_OK,
                        "eval_time": _elapsed,
                        "final_metric": final_value,
                    }
            except Exception as e:
                logger.warning(f"  Trial failed: {e}")
                return {"loss": float("inf"), "status": STATUS_OK, "error": str(e)}

        try:
            best_params = fmin(
                fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=self.trials,
                verbose=False,
            )
            result = space_eval(param_space, best_params)
            self.best_params = result
            return result
        except Exception as e:
            raise RuntimeError(f"Hyperopt optimization failed: {e}")

    def get_best_trial(self) -> Dict[str, Any]:
        """Get details about the best trial after optimisation.

        Returns:
            Dict with best trial information.

        Raises:
            RuntimeError: If tune() has not been called yet.
        """
        if self._optuna_study is not None:
            best = self._optuna_study.best_trial
            return {
                "best_params": best.params,
                "best_metric_value": best.value,
                "total_trials": len(self._optuna_study.trials),
                "successful_trials": len(
                    [
                        t
                        for t in self._optuna_study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
            }
        elif self.trials is not None and self.trials.trials:
            # Legacy hyperopt path
            best_trial = self.trials.best_trial
            best_loss = best_trial["result"]["loss"]
            best_metric_value = -best_loss if not _should_minimize(self.metric) else best_loss
            return {
                "best_params": self.trials.argmin,
                "best_metric_value": best_metric_value,
                "total_trials": len(self.trials.trials),
                "successful_trials": len(
                    [t for t in self.trials.trials if t["result"]["loss"] != float("inf")]
                ),
            }
        else:
            raise RuntimeError("No trials have been run yet. Call tune() first.")

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with all trial results.

        Returns:
            DataFrame with trial parameters and results.

        Raises:
            RuntimeError: If tune() has not been called yet.
        """
        if self._optuna_study is not None:
            return self._optuna_study.trials_dataframe()
        elif self.trials is not None and self.trials.trials:
            # Legacy hyperopt path
            _trials = self.trials  # narrow type for mypy
            trial_data = []
            for i, trial in enumerate(_trials.trials):
                row: Dict[str, Any] = {"trial_id": i}
                if "misc" in trial and "vals" in trial["misc"]:
                    for param_name, param_values in trial["misc"]["vals"].items():
                        if param_values:
                            row[param_name] = param_values[0]
                result = trial.get("result", {})
                row["loss"] = result.get("loss", float("inf"))
                row["final_metric"] = result.get("final_metric", None)
                row["status"] = result.get("status", "UNKNOWN")
                if "error" in result:
                    row["error"] = result["error"]
                trial_data.append(row)
            return pd.DataFrame(trial_data)
        else:
            raise RuntimeError("No trials have been run yet. Call tune() first.")


# Utility function for common search spaces
def create_search_space() -> Dict[str, Any]:
    """
    Creates common hyperparameter search spaces for different model types.

    Returns:
        Dictionary of example search space definitions
    """
    if not HAS_HYPEROPT:
        raise ImportError(
            "Hyperopt required to create search spaces. Install with: pip install hyperopt"
        )

    return {
        "neural_network": {
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-1)),
            "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
            "epochs": hp.choice("epochs", [10, 20, 50, 100]),
            "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
        },
        "xgboost": {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500]),
            "max_depth": hp.choice("max_depth", [3, 5, 7, 9]),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
        },
        "random_forest": {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500]),
            "max_depth": hp.choice("max_depth", [None, 5, 10, 15, 20]),
            "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
            "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4]),
        },
    }
