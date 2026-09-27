import time
import warnings
from typing import Any, Callable, Dict, Literal, Optional, cast

import numpy as np
import pandas as pd

from .settings import logger

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
from .exceptions import ConfigurationError
from .runners import build_fit_kwargs, set_run_seeds

_METRIC_CANDIDATES = ("val_accuracy", "val_r2", "val_loss")


def _resolve_direction(direction: str, metric: Optional[str]) -> str:
    """Resolve 'auto' to 'minimize' or 'maximize' from the metric name.

    One implementation for both backends (v12 7.39: the previous two copies
    could drift). Raises if 'auto' is requested before the metric is known.
    """
    if direction != "auto":
        return direction
    if metric is None:
        raise ValueError(
            "direction='auto' cannot be resolved before the metric is known. "
            "Pass metric= to HyperparameterTuner, or direction='minimize'/'maximize'."
        )
    # 0.5.0 (register 3.51): the library's one direction table. The former
    # six-keyword list minimised iou/dice/map/ndcg/mcc/kappa/fbeta/... and
    # maximised accuracy_loss.
    from .analysis import metric_direction

    d = metric_direction(metric)
    if d == "higher":
        return "maximize"
    if d == "lower":
        return "minimize"
    raise ConfigurationError(
        f"direction='auto' cannot tell whether higher or lower {metric!r} is better. "
        "Pass direction='maximize' or direction='minimize'."
    )


def _resolve_metric(
    requested: Optional[str], wrapper: BaseModelWrapper, eval_result: Dict[str, Any]
) -> "tuple[str, float]":
    """Return ``(metric_name, value)`` for one evaluation, or raise ConfigurationError.

    Lookup for ``val_<x>``: ``eval_result[<x>]`` (a fresh evaluation on the
    validation set) then ``history[val_<x>][-1]``. For any other name:
    ``eval_result[name]`` then ``history[name][-1]``. With ``requested=None``
    the first candidate in ``_METRIC_CANDIDATES`` that resolves is used.
    Raising here surfaces a bad metric on the FIRST trial instead of after
    every trial has been pruned (v12 1.31).
    """
    hist = wrapper.training_result.history if wrapper.training_result is not None else {}
    names = [requested] if requested else list(_METRIC_CANDIDATES)
    for name in names:
        bare = name[4:] if name.startswith("val_") else None
        if bare is not None and bare in eval_result:
            return name, float(eval_result[bare])
        if name in eval_result:
            return name, float(eval_result[name])
        if name in hist and hist[name]:
            return name, float(hist[name][-1])
    raise ConfigurationError(
        f"HyperparameterTuner could not resolve metric {requested or _METRIC_CANDIDATES}. "
        f"evaluate() produced {sorted(eval_result)}; training history has {sorted(hist)}. "
        "Pass metric= as one of these names (val_<x> reads evaluate()'s <x>)."
    )


def _suggest_public(trial: "optuna.Trial", name: str, dist: Any) -> Any:
    """Dispatch an Optuna distribution to the public suggest_* API (v12 2.31)."""
    D = optuna.distributions
    if isinstance(dist, D.FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, D.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, D.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    raise TypeError(f"Unsupported Optuna distribution for {name!r}: {type(dist).__name__}")


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna as the primary backend.

    Trains each trial configuration ``n_evals_per_trial`` times and uses
    the mean metric as the objective, consistent with the library's thesis
    that a single training run is not a reliable measurement.

    Requires optuna: ``pip install ictonyx[tuning]``

    The Hyperopt backend was removed in v0.5.0 (promise ledger #10).
    """

    def __init__(
        self,
        model_builder: Callable[[ModelConfig], BaseModelWrapper],
        data_handler: DataHandler,
        model_config: ModelConfig,
        metric: Optional[str] = None,
        n_evals_per_trial: int = 3,
        stability_weight: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            model_builder: Function returning BaseModelWrapper given ModelConfig.
            data_handler: DataHandler for loading data. Data is loaded lazily
                at tune() time, not during construction.
            model_config: Base ModelConfig updated with trial parameters.
            metric: Metric to optimize. ``val_<x>`` reads ``evaluate()``'s
                ``<x>`` on the validation set, falling back to the last
                training-history value. ``None`` (default) resolves to the
                first of ``val_accuracy``, ``val_r2``, ``val_loss`` the model
                produces; with ``None`` you must pass ``direction=`` to
                ``tune()``. An unresolvable metric raises on the first trial.
            seed: Base seed for the Optuna sampler and for every evaluation's
                ``run_seed``. Same seed, same trial sequence, same runs.
                ``None`` draws a random seed (recorded in ``resolved_seed``).
            n_evals_per_trial: Independent training runs per trial. The trial
                objective is the mean metric across these runs. Default 3.
                Set to 1 to reproduce single-run (old) behavior.
            stability_weight: If > 0, penalizes run-to-run variance.
                Minimisation objectives (e.g. ``'val_loss'``):
                ``objective = mean + stability_weight * std``.
                Maximisation objectives (e.g. ``'val_accuracy'``):
                ``objective = mean - stability_weight * std``.
                In both cases, higher variance worsens the objective score.
                Default 0.0 (variance not penalized). The SD is estimated from
                ``n_evals_per_trial`` runs; at the default 3 its relative
                standard error is about 50%, so the penalty is mostly noise
                unless ``n_evals_per_trial`` is raised.
        """
        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.metric = metric
        self.n_evals_per_trial = n_evals_per_trial
        self.stability_weight = stability_weight
        self.seed = seed
        self.resolved_seed: Optional[int] = None
        if not 0.0 <= self.stability_weight <= 1.0:
            raise ValueError(
                f"stability_weight must be between 0.0 and 1.0, "
                f"got {stability_weight!r}. "
                "Values > 1 apply an unstable over-penalization of variance."
            )
        self._data_dict: Optional[Dict[str, Any]] = None
        self.train_data = None
        self.val_data = None
        self.best_params: Optional[Dict[str, Any]] = None
        self._optuna_study: Optional[Any] = None

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
            # The Hyperopt fallback was removed in v0.5.0 (promise ledger #10).
            raise ImportError(
                "HyperparameterTuner.tune() requires Optuna: pip install ictonyx[tuning]. "
                "The Hyperopt backend was removed in v0.5.0."
            )

        if not isinstance(param_space, dict) or not param_space:
            raise ValueError("param_space must be a non-empty dict of Optuna distributions.")
        if max_evals <= 0:
            raise ValueError(f"max_evals must be positive, got {max_evals}.")

        self._ensure_data_loaded()
        resolved_direction = _resolve_direction(direction, self.metric)

        # One base seed drives the sampler and every evaluation (v12 2.36, 0.8).
        base_seed = (
            self.seed if self.seed is not None else int(np.random.default_rng().integers(0, 2**31))
        )
        self.resolved_seed = base_seed
        trial_seeds = [
            int(ss.generate_state(1)[0])
            for ss in np.random.SeedSequence(base_seed).spawn(max_evals)
        ]
        resolved: Dict[str, str] = {}

        def objective(trial: "optuna.Trial") -> float:
            params = {
                name: _suggest_public(trial, name, dist) for name, dist in param_space.items()
            }
            config = self.model_config.copy(keep_frozen=False).update(params)
            child_seeds = np.random.SeedSequence(trial_seeds[trial.number]).spawn(
                self.n_evals_per_trial
            )
            metric_values = []

            for i, child_seed in enumerate(child_seeds):
                run_seed = int(child_seed.generate_state(1)[0])
                run_config = config.copy().set("run_seed", run_seed)
                wrapper = None
                try:
                    set_run_seeds(run_seed)  # global RNGs, same owner as the runner
                    wrapper = self.model_builder(run_config)
                    wrapper.fit(  # the fit() contract: run_seed, batch_size, etc. (0.8)
                        **build_fit_kwargs(
                            run_config,
                            run_config.get("epochs", 10),
                            run_seed,
                            wrapper,
                            self.train_data,
                            self.val_data,
                        )
                    )
                    result = wrapper.evaluate(data=self.val_data)
                    name, value = _resolve_metric(self.metric, wrapper, result)
                    resolved.setdefault("metric", name)
                    metric_values.append(value)
                except ConfigurationError:
                    raise  # a bad metric fails the first trial, not the last (1.31)
                except Exception as e:
                    logger.warning(f"Trial {trial.number} run {i + 1} failed: {e}")
                finally:
                    if wrapper is not None:  # release the model every time
                        try:
                            wrapper.release()
                        except Exception:
                            pass

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

        sampler = optuna.samplers.TPESampler(seed=base_seed)
        study = optuna.create_study(
            direction=cast(Literal["minimize", "maximize"], resolved_direction),
            sampler=sampler,
        )
        study.optimize(objective, n_trials=max_evals, timeout=timeout, n_jobs=n_jobs)
        self.metric = self.metric or resolved.get("metric")

        self._optuna_study = study
        best = study.best_params
        self.best_params = best
        logger.info(f"Optimisation complete. Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best}")
        return best

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
        else:
            raise RuntimeError("No trials have been run yet. Call tune() first.")


# Utility function for common search spaces
def create_search_space(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Removed in v0.5.0 with the Hyperopt backend (promise ledger #10).

    Raises:
        RemovedAPIError: Always. This tombstone is removed in v0.6.0.
    """
    from .exceptions import _removed

    raise _removed(
        "create_search_space() (Hyperopt search spaces)",
        "an Optuna param_space: {name: optuna.distributions.*}",
        since="0.4.0",
    )
