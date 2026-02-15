# ictonyx/api.py
"""
The High-Level API for Ictonyx.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import settings
from .analysis import compare_multiple_models as _stat_compare
from .config import ModelConfig
from .core import PYTORCH_AVAILABLE, SKLEARN_AVAILABLE, TENSORFLOW_AVAILABLE, BaseModelWrapper
from .data import DataHandler, auto_resolve_handler
from .loggers import BaseLogger
from .runners import VariabilityStudyResults
from .runners import run_variability_study as _run_study


def variability_study(
    model: Any,
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    seed: Optional[int] = None,
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
        runs: Number of independent training runs. Default 5.
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
        seed: Base random seed. Run *i* uses ``seed + i``, so every run is
            different but the full study is reproducible. If ``None``, a
            random seed is generated and stored in the results.
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

    # 1. Prepare Data
    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)

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
            "verbose": 1 if settings._VERBOSE else 0,
            **kwargs,
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
    )


def compare_models(
    models: List[Any],
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 5,
    epochs: int = 10,
    metric: str = "val_accuracy",
    **kwargs,
) -> Dict[str, Any]:
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
        **kwargs: Forwarded to each :func:`variability_study` call.

    Returns:
        Dict containing:

        * ``'overall_test'``: :class:`~ictonyx.analysis.StatisticalTestResult`
          for the omnibus test.
        * ``'pairwise_comparisons'``: Dict of pairwise
          :class:`~ictonyx.analysis.StatisticalTestResult` objects (if 3+
          models).
        * ``'raw_data'``: Dict mapping model names to ``pd.Series`` of
          metric values.
        * ``'error'``: Present only if fewer than 2 models produced valid
          results.

    Example::

        results = ix.compare_models(
            models=[RandomForestClassifier, GradientBoostingClassifier],
            data=df,
            target_column='target',
            runs=20,
            metric='val_accuracy',
        )
        print(results['overall_test'].get_summary())
    """

    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)
    results_store = {}

    settings.logger.info(f"--- Starting Comparison of {len(models)} Models ---")

    for model_input in models:
        name = _get_model_name(model_input)
        settings.logger.info(f"Evaluating: {name}")

        # Run study for this specific model
        # We recurse into variability_study to avoid duplicating logic
        # Pass the already-resolved handler so all models use identical data splits
        study_results = variability_study(
            model=model_input,
            data=handler,  # Pass the already resolved handler
            runs=runs,
            epochs=epochs,
            **kwargs,
        )

        # Extract metric
        metrics = study_results.get_final_metrics(metric)
        if not metrics:
            settings.logger.warning(f"No '{metric}' data found for {name}")
            continue

        results_store[name] = pd.Series(list(metrics.values()))

    # Statistical Analysis
    if len(results_store) < 2:
        return {"error": "Insufficient valid results for comparison"}

    return _stat_compare(results_store)


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
        return lambda conf: _ensure_wrapper(model())

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
    if "torch" in str(type(model).__mro__):
        raise ValueError(
            "Passing a PyTorch model instance risks weight leakage between runs. "
            "Pass a builder function instead:\n"
            "  def build_model(config):\n"
            "      return PyTorchModelWrapper(MyNet(), ...)\n"
            "  ix.variability_study(model=build_model, ...)"
        )

    # Unknown instance with a fit method — warn and allow, but this is risky
    settings.logger.warning(
        f"Passed an instance of {type(model).__name__}. Cannot clone — "
        f"weights may persist between runs. Consider passing a class or "
        f"builder function for independent runs."
    )
    return lambda conf: _ensure_wrapper(model)


def _ensure_wrapper(obj: Any) -> BaseModelWrapper:
    """Wrap a raw model object in the appropriate Ictonyx model wrapper.

    If ``obj`` is already a :class:`~ictonyx.core.BaseModelWrapper`, it is
    returned unchanged. Otherwise, the function inspects the object to
    determine the correct wrapper:

    * Objects with ``fit``/``predict`` → :class:`ScikitLearnModelWrapper`
    * Keras models → :class:`KerasModelWrapper`
    * PyTorch ``nn.Module`` → :class:`PyTorchModelWrapper`

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

    # Duck typing checks are Pythonic and readable
    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required to auto-wrap models with fit/predict. "
                "Install with: pip install scikit-learn"
            )
        from .core import ScikitLearnModelWrapper

        return ScikitLearnModelWrapper(obj)

    # String check avoids hard import of TensorFlow
    if "keras" in str(type(obj)) or "tensorflow" in str(type(obj)):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to auto-wrap Keras models. "
                "Install with: pip install tensorflow"
            )
        from .core import KerasModelWrapper

        return KerasModelWrapper(obj)

    # PyTorch nn.Module detection
    if "torch" in str(type(obj).__mro__):
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to auto-wrap torch.nn.Module models. "
                "Install with: pip install torch"
            )
        from .core import PyTorchModelWrapper

        return PyTorchModelWrapper(obj)

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
