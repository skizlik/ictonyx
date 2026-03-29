# ictonyx/api.py
"""
The High-Level API for Ictonyx.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import settings
from .analysis import ModelComparisonResults
from .analysis import compare_multiple_models as _stat_compare
from .config import ModelConfig
from .core import PYTORCH_AVAILABLE, SKLEARN_AVAILABLE, TENSORFLOW_AVAILABLE, BaseModelWrapper
from .data import DataHandler, auto_resolve_handler
from .exceptions import ConfigurationError
from .loggers import BaseLogger
from .runners import VariabilityStudyResults
from .runners import run_variability_study as _run_study

# Resolve torch.nn once at import time so isinstance checks below are reliable
# and don't require repeated try/except blocks throughout this module.
if PYTORCH_AVAILABLE:
    import torch.nn as _torch_nn
else:
    _torch_nn = None  # type: ignore[assignment]


def variability_study(
    model: Any,
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 10,
    epochs: int = 10,
    batch_size: int = 32,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> VariabilityStudyResults:
    """Run a variability study: train a model N times and collect distributions.

    Trains the same model architecture on the same data ``runs`` times, each
    with a different random seed, and returns the full distribution of
    training metrics. This is the primary entry point for most users.

    The function handles data resolution, model wrapping, configuration,
    and execution automatically. For finer control, use
    :class:`~ictonyx.runners.ExperimentRunner` directly.

    Args:
        model: What to train. Accepted forms:

            * A **class** with ``fit``/``predict`` (e.g. ``RandomForestClassifier``) —
              a fresh instance is created for each run.
            * A **callable** ``f(config) -> BaseModelWrapper`` — called once per run.
            * A **Keras Model** or **PyTorch nn.Module** — auto-wrapped.
            * An **instance** with ``fit``/``predict`` — works, but a warning is
              emitted because fitted state persists between runs.

        data: The dataset. Accepted forms:

            * ``pd.DataFrame`` — requires ``target_column``.
            * ``(X, y)`` tuple of array-likes.
            * ``str`` path to a CSV file (requires ``target_column``) or an
              image directory (requires ``image_size`` in kwargs).
            * An existing :class:`~ictonyx.data.DataHandler` instance.

        target_column: Column name containing labels. Required when ``data``
            is a DataFrame or CSV path.
        runs: Number of independent training runs. Default 10.
        epochs: Training epochs per run. Ignored by scikit-learn models.
            Default 10.
        batch_size: Batch size per run. Ignored by scikit-learn models.
            Default 32.
        tracker: Optional :class:`~ictonyx.loggers.BaseLogger` (or subclass
            such as ``MLflowLogger``) for experiment tracking. If ``None``,
            a basic in-memory logger is used.
        use_process_isolation: If ``True``, each run executes in a subprocess
            to guarantee GPU memory cleanup. Useful for Keras/TF models that
            leak memory across runs. Default ``False``.
        seed: Base random seed. Each run receives an independent child seed
            derived via ``np.random.SeedSequence.spawn()``, guaranteeing
            statistically uncorrelated RNG streams. If ``None``, a random
            seed is generated and stored in the results.
        verbose: If ``False``, suppress all training output. Default ``True``.
        **kwargs: Additional arguments forwarded to both the
            :class:`~ictonyx.config.ModelConfig` and the data handler
            (e.g. ``image_size``, ``test_split``, ``val_split``).

    Returns:
        :class:`~ictonyx.runners.VariabilityStudyResults` containing
        per-run metric DataFrames, final metric distributions, and
        convenience methods for summarization and statistical analysis.

    Example::

        import ictonyx as ix
        from sklearn.ensemble import RandomForestClassifier

        results = ix.variability_study(
            model=RandomForestClassifier,
            data=df,
            target_column='target',
            runs=20,
        )
        print(results.summarize())
    """

    # Separate infrastructure kwargs (forwarded to the data handler) from
    # model kwargs (forwarded to ModelConfig). Add new DataHandler constructor
    # parameters to _INFRA_KWARGS to prevent them from appearing in ModelConfig.
    _INFRA_KWARGS = {
        "image_size",
        "test_split",
        "val_split",
        "gpu_memory_limit",
        "color_mode",
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k not in _INFRA_KWARGS}
    infra_kwargs = {k: v for k, v in kwargs.items() if k in _INFRA_KWARGS}

    # Apply verbose setting to global logger
    from .settings import set_verbose

    set_verbose(verbose)

    if runs < 20:
        warnings.warn(
            f"runs={runs} may be insufficient for reliable statistical inference. "
            "At n=5, Mann-Whitney U has very low power against small effects. "
            "Consider runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # 1. Prepare Data
    handler = auto_resolve_handler(data, target_column=target_column, **infra_kwargs)

    # 2. Prepare Model Builder
    # If the user passes a class (e.g. RandomForestClassifier), we instantiate it per run.
    # If they pass a function, we use it directly.
    builder = _get_model_builder(model)

    # 3. Configure
    # We explicitly pass kwargs into the config, trusting ModelConfig to ignore/store extras.
    config = ModelConfig(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": 1 if verbose else 0,
            **model_kwargs,
        }
    )

    # 4. Execute
    return _run_study(
        model_builder=builder,
        data_handler=handler,
        model_config=config,
        num_runs=runs,
        epochs_per_run=epochs,
        tracker=tracker,
        use_process_isolation=use_process_isolation,
        gpu_memory_limit=kwargs.get("gpu_memory_limit"),
        seed=seed,
        verbose=verbose,
    )


def compare_models(
    models: List[Any],
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 10,
    epochs: int = 10,
    metric: str = "val_accuracy",
    seed: Optional[int] = None,
    verbose: bool = True,
    paired: bool = False,
    **kwargs,
) -> ModelComparisonResults:
    """Run variability studies on multiple models and compare them statistically.

    Each model is trained ``runs`` times on the same data, producing a
    distribution of the chosen metric. The distributions are then compared
    using non-parametric statistical tests with automatic test selection:

    * **Two models**: Mann-Whitney U (or Wilcoxon signed-rank if paired).
    * **Three or more**: Kruskal-Wallis with post-hoc pairwise comparisons
      and multiple-comparison correction.

    Effect sizes (Cohen's d, rank-biserial) and bootstrap confidence
    intervals are included automatically.

    Args:
        models: List of models in any form accepted by
            :func:`variability_study` (classes, callables, instances, etc.).
        data: The dataset, in any form accepted by :func:`variability_study`.
        target_column: Column name containing labels, if applicable.
        runs: Number of independent training runs per model. Default 5.
        epochs: Training epochs per run. Default 10.
        metric: Metric name to compare across models. Must be a key in
            the training history (e.g. ``'val_accuracy'``, ``'val_loss'``,
            ``'val_f1'``). Default ``'val_accuracy'``.
        seed: Base random seed for reproducibility. All models use the same
            seed so the comparison can be reproduced exactly. If ``None``,
            a random seed is generated.
        verbose: If ``False``, suppress all output. Default ``True``.
        paired: If ``True``, use Wilcoxon signed-rank for two-model
            comparison. Correct when both models share the same seeds.
            Default ``False``.
        **kwargs: Forwarded to each :func:`variability_study` call.

    Returns:
        :class:`~ictonyx.analysis.ModelComparisonResults` containing the
        omnibus test, pairwise comparisons, raw metric distributions, and
        summary methods.

    Example::

        results = ix.compare_models(
            models=[RandomForestClassifier, GradientBoostingClassifier],
            data=df,
            target_column='target',
            runs=20,
            metric='val_accuracy',
            seed=42,
        )
        print(results.get_summary())
    """
    # Apply verbose setting to global logger
    from .settings import set_verbose

    set_verbose(verbose)

    if runs < 20:
        warnings.warn(
            f"runs={runs} may be insufficient for reliable statistical inference. "
            "At n=5, Mann-Whitney U has very low power against small effects. "
            "Consider runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve seed once — all models use the same base seed for reproducibility
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)
    results_store = {}

    settings.logger.info(f"--- Starting Comparison of {len(models)} Models (seed={seed}) ---")

    for model_input in models:
        base_name = _get_model_name(model_input)
        name = base_name
        counter = 1
        while name in results_store:
            name = f"{base_name}_{counter}"
            counter += 1
        settings.logger.info(f"Evaluating: {name}")

        study_results = variability_study(
            model=model_input,
            data=handler,
            runs=runs,
            epochs=epochs,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

        try:
            metric_values = study_results.get_metric_values(metric)
        except KeyError:
            available = study_results.get_available_metrics()
            hint = ""
            if metric == "val_accuracy" and "accuracy" in available:
                hint = (
                    " Tip: sklearn models only record 'val_accuracy' when "
                    "validation data is provided. Pass val_split=0.2 to your "
                    "DataHandler, or use metric='accuracy' instead."
                )
            elif metric.startswith("val_") and metric[4:] in available:
                hint = (
                    f" '{metric[4:]}' was recorded but '{metric}' was not. "
                    "Ensure your DataHandler provides a validation split."
                )
            raise ConfigurationError(
                f"compare_models() could not find metric '{metric}' for model "
                f"'{name}'. Available metrics: {available}.{hint}"
            ) from None

        if not metric_values:
            settings.logger.warning(
                f"Metric '{metric}' was tracked for '{name}' but contains no "
                "values — all training runs for this model may have failed."
            )
            continue
        results_store[name] = pd.Series(metric_values)

    if len(results_store) < 2:
        raise ValueError(
            f"Insufficient valid results for comparison: only {len(results_store)} model(s) "
            f"produced '{metric}' data. Check that your metric name is correct."
        )

    if paired and len(results_store) == 2:
        from .analysis import paired_wilcoxon_test

        names = list(results_store.keys())
        series_a = results_store[names[0]]
        series_b = results_store[names[1]]
        if len(series_a) != len(series_b):
            raise ValueError(
                f"compare_models(paired=True) requires equal run counts. "
                f"'{names[0]}' has {len(series_a)} runs but "
                f"'{names[1]}' has {len(series_b)} runs. "
                "Use paired=False for independent comparison."
            )
        paired_result = paired_wilcoxon_test(series_a, series_b)
        return ModelComparisonResults(
            overall_test=paired_result,
            raw_data=results_store,
            pairwise_comparisons={f"{names[0]}_vs_{names[1]}": paired_result},
            significant_comparisons=(
                [f"{names[0]}_vs_{names[1]}"] if paired_result.is_significant() else []
            ),
            correction_method="none",
            n_models=2,
            metric=metric,
        )

    stat_results = _stat_compare(results_store)

    return ModelComparisonResults(
        overall_test=stat_results["overall_test"],
        raw_data=results_store,
        pairwise_comparisons=stat_results.get("pairwise_comparisons", {}),
        significant_comparisons=stat_results.get("significant_comparisons", []),
        correction_method=stat_results.get("correction_method", "holm"),
        n_models=len(results_store),
        metric=metric,
    )


# --- Clean Helpers ---


def _get_model_builder(model: Any) -> Callable:
    """Normalize diverse model inputs into a consistent factory function.

    Accepts classes, callables, and instances and returns a callable
    with signature ``f(ModelConfig) -> BaseModelWrapper``. Instances
    are wrapped with a warning about state leakage between runs.

    Args:
        model: A model class, callable, or fitted/unfitted instance.

    Returns:
        A callable that takes a :class:`~ictonyx.config.ModelConfig` and
        returns a :class:`~ictonyx.core.BaseModelWrapper`.

    Raises:
        ValueError: If ``model`` is not a class, callable, or object with
            a ``fit`` method.
    """

    # If it's already a function, trust it.
    if callable(model) and not isinstance(model, type):
        return lambda conf: _ensure_wrapper(model(conf))

    # If it's a class (like RandomForestClassifier), instantiate it per run.
    if isinstance(model, type):

        def _build_from_class(conf, _model_class=model):
            import inspect

            if "random_state" in inspect.signature(_model_class).parameters:
                try:
                    return _ensure_wrapper(_model_class(random_state=conf.get("run_seed")))
                except TypeError:
                    return _ensure_wrapper(_model_class())
            return _ensure_wrapper(_model_class())

        return _build_from_class

    # If it's an instance, we need to clone it per run for independence.
    if hasattr(model, "fit"):
        return _build_instance_cloner(model)

    raise ValueError(f"Invalid model input: {model}")


def _build_instance_cloner(model: Any) -> Callable:
    """Creates a builder that produces independent copies of a model instance.

    For sklearn estimators, uses sklearn.base.clone() which creates an
    unfitted copy with the same hyperparameters. This preserves the user's
    configuration (e.g., n_estimators=100) while ensuring each run starts
    from scratch.

    For Keras and PyTorch instances, cloning is not reliably possible, so
    we raise an error guiding the user to pass a class or builder function.
    """
    # sklearn: clone() creates an unfitted copy with same hyperparameters
    if hasattr(model, "get_params"):
        try:
            from sklearn.base import clone

            # Test that clone works before committing to this path
            clone(model)
            settings.logger.info(
                f"Cloning {type(model).__name__} instance per run for independence."
            )
            return lambda conf: _ensure_wrapper(clone(model))
        except Exception as e:
            raise ValueError(
                f"Cannot clone sklearn model instance: {e}. "
                f"Pass the class instead: model={type(model).__name__}"
            )

    # Keras models: no clean clone path
    if "keras" in str(type(model)) or "tensorflow" in str(type(model)):
        raise ValueError(
            "Passing a Keras model instance risks weight leakage between runs. "
            "Pass a builder function instead:\n"
            "  def build_model(config):\n"
            "      model = Sequential([...])\n"
            "      model.compile(...)\n"
            "      return KerasModelWrapper(model)\n"
            "  ix.variability_study(model=build_model, ...)"
        )

    # PyTorch modules: no clean clone path
    if PYTORCH_AVAILABLE and isinstance(model, _torch_nn.Module):
        raise ValueError(
            "Passing a PyTorch model instance risks weight leakage between runs. "
            "Pass a builder function instead:\n"
            "  def build_model(config):\n"
            "      return PyTorchModelWrapper(MyNet(), ...)\n"
            "  ix.variability_study(model=build_model, ...)"
        )

    # Unknown instance with a fit method — cannot safely clone.
    # Reusing the same instance across runs would leak trained weights,
    # invalidating the independence assumption of the variability study.
    raise ValueError(
        f"Passed an instance of {type(model).__name__} that cannot be cloned. "
        "Reusing the same instance across runs would leak trained weights between "
        "runs, invalidating the variability study.\n\n"
        "Pass a class or builder function instead:\n"
        f"  ix.variability_study(model={type(model).__name__}, ...)\n"
        "  ix.variability_study(model=lambda config: MyModel(**config.params), ...)"
    )


def _ensure_wrapper(obj: Any) -> BaseModelWrapper:
    """Wrap a raw model object in the appropriate Ictonyx model wrapper.

    If ``obj`` is already a :class:`~ictonyx.core.BaseModelWrapper`, it is
    returned unchanged. Otherwise, the function inspects the object to
    determine the correct wrapper:

    * Keras / TensorFlow models → :class:`KerasModelWrapper`
    * PyTorch ``nn.Module`` → :class:`PyTorchModelWrapper`
    * Objects with ``fit``/``predict`` → :class:`ScikitLearnModelWrapper`

    Args:
        obj: A model object to wrap.

    Returns:
        A :class:`~ictonyx.core.BaseModelWrapper` subclass instance.

    Raises:
        ImportError: If the required framework (sklearn, TF, or PyTorch)
            is not installed.
        TypeError: If the object cannot be identified as a supported model.
    """
    if isinstance(obj, BaseModelWrapper):
        return obj

        # Framework-specific checks MUST come before the generic duck-typing check.
        # Keras models expose .fit() and .predict(), so they would be mis-wrapped as
        # ScikitLearnModelWrapper if the duck-typing branch ran first.
        # Keras / TensorFlow detection via string inspection — avoids a hard TF import.
    if "keras" in str(type(obj)).lower() or "tensorflow" in str(type(obj)).lower():
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to auto-wrap Keras models. "
                "Install with: pip install tensorflow"
            )
        from .core import KerasModelWrapper

        return KerasModelWrapper(obj)

        # PyTorch nn.Module detection via isinstance — reliable, no string heuristics.
    if PYTORCH_AVAILABLE and isinstance(obj, _torch_nn.Module):
        from .core import PyTorchModelWrapper

        return PyTorchModelWrapper(obj)
    # Generic duck-typing — only reached if the object is not Keras or PyTorch.

    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required to auto-wrap models with fit/predict. "
                "Install with: pip install scikit-learn"
            )
        from .core import ScikitLearnModelWrapper

        return ScikitLearnModelWrapper(obj)

    raise TypeError(f"Cannot wrap model of type: {type(obj)}")


def _get_model_name(obj: Any) -> str:
    """Extract a human-readable name from a model for logging and display.

    Checks ``__name__`` (functions/classes), then ``__class__.__name__``
    (instances), falling back to ``str(obj)``.

    Args:
        obj: A model class, function, or instance.

    Returns:
        A string suitable for use in log messages and result keys.
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)
