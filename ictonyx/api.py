# ictonyx/api.py
"""
The High-Level API for Ictonyx.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""
import dataclasses
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
    runs: int = 20,
    epochs: int = 10,
    batch_size: int = 32,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
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
        runs: Number of independent training runs. Default 20.
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
        use_parallel: If ``True``, fan training runs across multiple
            processes using ``joblib``. Safe for sklearn models. Not
            recommended for Keras/TF models. Mutually exclusive with
            ``use_process_isolation``. Default ``False``.
        n_jobs: Number of parallel workers. ``-1`` uses all CPUs.
            Ignored when ``use_parallel=False``. Default ``-1``.
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
            "Mann-Whitney U has limited power against small effects below n=20. "
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
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def compare_models(
    models: List[Any],
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 20,
    epochs: int = 10,
    metric: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    paired: bool = True,
    **kwargs,
) -> ModelComparisonResults:
    """Run variability studies on multiple models and compare them statistically.

    Each model is trained ``runs`` times on the same data, producing a
    distribution of the chosen metric. The distributions are then compared
    using non-parametric statistical tests.

    **Seeding and pairing**

    All models receive the same base ``seed``. Internally,
    ``np.random.SeedSequence(seed).spawn(runs)`` generates per-run child seeds
    that are identical across models: model A's run *i* and model B's run *i*
    always share the same child seed. The runs are therefore **genuinely paired
    at the RNG level**, regardless of framework.

    For this reason ``paired`` defaults to ``True``:

    * **Two models (default, paired):** Paired Wilcoxon signed-rank test on the
      per-run differences. More powerful than the independent-samples alternative
      because it removes run-to-run noise that is common to both models.
    * **Two models (unpaired):** Kruskal-Wallis omnibus + Mann-Whitney U with
      Holm correction. Valid but does not exploit the RNG pairing.
    * **Three or more models:** Kruskal-Wallis omnibus + pairwise Mann-Whitney U
      with Holm correction, regardless of ``paired``. (Paired multi-group
      analysis requires a different design not yet implemented.)

    **Seeding guarantee by framework**

    * scikit-learn: exact (``random_state`` injected at wrapper construction).
    * PyTorch: approximately deterministic under ``cudnn.deterministic=True``;
      rare non-deterministic CUDA ops may introduce small deviations.
    * TF/Keras + GPU: not fully controllable; pairing is approximate.

    If exact pairing cannot be guaranteed (e.g. Keras with GPU), pass
    ``paired=False`` to fall back to the independent-samples test, which
    is always valid regardless of seeding.

    Args:
        models: List of models in any form accepted by
            :func:`variability_study` (classes, callables, instances, etc.).
        data: The dataset, in any form accepted by :func:`variability_study`.
        target_column: Column name containing labels, if applicable.
        runs: Number of independent training runs per model. Default 20.
        epochs: Training epochs per run. Default 10.
        metric: Metric name to compare across models. Must be a key in
            the training history (e.g. ``'val_accuracy'``, ``'val_loss'``,
            ``'val_f1'``). Default ``None`` (auto-resolved from results).
        seed: Base random seed for reproducibility. All models use the same
            seed so the comparison can be reproduced exactly. If ``None``,
            a random seed is generated and stored in the result.
        verbose: If ``False``, suppress all output. Default ``True``.
        paired: If ``True`` (default), use the paired Wilcoxon signed-rank
            test for two-model comparisons, exploiting the fact that all
            models receive identical per-run seeds by construction. Ignored
            when comparing three or more models (KW + MW is used regardless).
            Pass ``False`` to use the independent-samples test instead —
            appropriate when seeds are not shared or when comparing against
            externally produced results.
        **kwargs: Forwarded to each :func:`variability_study` call.

    Returns:
        :class:`~ictonyx.analysis.ModelComparisonResults` containing the
        omnibus test, pairwise comparisons, raw metric distributions, and
        summary methods.

    Example (two models, default paired analysis)::

        results = ix.compare_models(
            models=[RandomForestClassifier, GradientBoostingClassifier],
            data=df,
            target_column='target',
            runs=20,
            metric='val_accuracy',
            seed=42,
        )
        print(results.get_summary())

    Example (three models, or unpaired two-model comparison)::

        results = ix.compare_models(
            models=[ModelA, ModelB, ModelC],
            data=df,
            target_column='target',
            runs=20,
            seed=42,
        )
        # Unpaired two-model:
        results = ix.compare_models(
            models=[ModelA, ModelB],
            data=df,
            target_column='target',
            runs=20,
            seed=42,
            paired=False,   # fall back to KW + Mann-Whitney
        )
    """
    # Apply verbose setting to global logger
    from .settings import set_verbose

    set_verbose(verbose)

    if runs < 20:
        warnings.warn(
            f"runs={runs} may be insufficient for reliable statistical inference. "
            "Mann-Whitney U has limited power against small effects below n=20. "
            "Consider runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve seed once — all models use the same base seed for reproducibility
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    # Apply the same infra/model kwargs separation used in variability_study().
    # Without this split, model hyperparameters like learning_rate reach
    # auto_resolve_handler() and raise TypeError.
    _INFRA_KWARGS = {"image_size", "test_split", "val_split", "gpu_memory_limit", "color_mode"}
    infra_kwargs = {k: v for k, v in kwargs.items() if k in _INFRA_KWARGS}
    handler = auto_resolve_handler(data, target_column=target_column, **infra_kwargs)

    settings.logger.info(f"--- Starting Comparison of {len(models)} Models (seed={seed}) ---")

    # --- Loop 1: run all studies, store full VariabilityStudyResults objects ---
    studies: Dict[str, VariabilityStudyResults] = {}

    for model_input in models:
        base_name = _get_model_name(model_input)
        name = base_name
        counter = 1
        while name in studies:
            name = f"{base_name}_{counter}"
            counter += 1
        settings.logger.info(f"Evaluating: {name}")

        study_result = variability_study(
            model=model_input,
            data=handler,
            runs=runs,
            epochs=epochs,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
        studies[name] = study_result

    # --- Resolve metric now that all studies are complete ---
    if metric is None:
        metric = "val_accuracy"

    # --- Loop 2: extract metric values now that metric is a concrete string ---
    results_store = {}

    for name, study_result in studies.items():
        try:
            if metric.startswith("test_"):
                metric_values = study_result.get_test_metric_values(metric)
            else:
                metric_values = study_result.get_metric_values(metric)
        except KeyError:
            available = study_result.get_available_metrics()
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
        from .analysis import compare_two_models

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
        paired_result = compare_two_models(series_a, series_b, paired=True, random_state=seed)
        return ModelComparisonResults(
            overall_test=paired_result,
            raw_data=results_store,
            pairwise_comparisons={f"{names[0]}_vs_{names[1]}": dataclasses.replace(paired_result)},
            significant_comparisons=(
                [f"{names[0]}_vs_{names[1]}"] if paired_result.is_significant() else []
            ),
            correction_method="none",
            n_models=2,
            metric=metric,
        )

    if paired and len(results_store) > 2:
        warnings.warn(
            f"compare_models(): paired=True has no effect when comparing "
            f"{len(results_store)} models. Paired analysis is only available "
            "for exactly two models. Using Kruskal-Wallis + Mann-Whitney U "
            "(independent-samples) for this comparison.",
            UserWarning,
            stacklevel=2,
        )

    stat_results = _stat_compare(results_store, random_state=seed)
    stat_results.metric = metric
    stat_results.raw_data = results_store
    return stat_results


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

            sig = inspect.signature(_model_class)
            # Extract kwargs from the ModelConfig that the wrapper's constructor
            # actually accepts. Filter out infra keys the constructor doesn't know
            # about (e.g. run_seed), and `random_state` which we pass explicitly
            # below when the signature supports it.
            accepted = set(sig.parameters.keys())
            accepts_var_keyword = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            construction_kwargs = {
                k: v
                for k, v in conf.items()
                if k != "run_seed"
                and k != "random_state"
                and (accepts_var_keyword or k in accepted)
            }

            if "random_state" in accepted:
                try:
                    return _ensure_wrapper(
                        _model_class(
                            random_state=conf.get("run_seed"),
                            **construction_kwargs,
                        )
                    )
                except TypeError as e:
                    if "random_state" in str(e) or "unexpected keyword" in str(e):
                        warnings.warn(
                            f"Could not pass random_state to {_model_class.__name__}. "
                            f"Reproducibility not guaranteed. Original error: {e}",
                            UserWarning,
                            stacklevel=3,
                        )
                        return _ensure_wrapper(_model_class(**construction_kwargs))
                    raise ValueError(
                        f"Failed to construct {_model_class.__name__}: {e}. "
                        "Check that the class accepts the arguments in your ModelConfig."
                    ) from e
            return _ensure_wrapper(_model_class(**construction_kwargs))

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

    # Framework-specific checks MUST come before generic duck-typing.
    # Keras models expose .fit() and .predict() and would be mis-wrapped
    # as ScikitLearnModelWrapper if the duck-typing branch ran first.
    if TENSORFLOW_AVAILABLE:
        import tensorflow as tf

        if isinstance(obj, tf.keras.Model):
            from .core import KerasModelWrapper

            return KerasModelWrapper(obj)

    if PYTORCH_AVAILABLE and isinstance(obj, _torch_nn.Module):
        from .core import PyTorchModelWrapper

        return PyTorchModelWrapper(obj)

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


# AFTER (complete replacement)
def compare_results(
    results_a: "VariabilityStudyResults",
    results_b: "VariabilityStudyResults",
    metric: Optional[str] = None,
    paired: bool = True,
    seed: Optional[int] = None,
) -> "ModelComparisonResults":
    """Compare two pre-computed VariabilityStudyResults without re-running training.

    Extracts metric values from each results object and compares them
    statistically. Use this when you already have results from
    :func:`variability_study` and want to compare them without retraining.

    **Pairing:** If both results were produced with the same ``seed``, the runs
    are paired by construction. Pass ``paired=True`` (default) to exploit this
    with the more powerful paired Wilcoxon signed-rank test. Pass ``paired=False``
    to use the independent-samples Mann-Whitney U test instead.

    Args:
        results_a: First model's results.
        results_b: Second model's results.
        metric: Metric to compare. If ``None``, resolves via
            ``results_a.preferred_metric()``.
        paired: If ``True`` (default) and run counts are equal, use the
            paired Wilcoxon signed-rank test. Falls back to Mann-Whitney U
            with a ``UserWarning`` when run counts differ.
        seed: Random state for bootstrap CI computation. Defaults to ``None``
            (non-deterministic CIs).

    Returns:
        :class:`~ictonyx.analysis.ModelComparisonResults`.

    Raises:
        KeyError: If the resolved metric is not present in both results.
    """
    from .analysis import mann_whitney_test, paired_wilcoxon_test

    resolved = metric if metric is not None else results_a.preferred_metric("accuracy")

    if resolved.startswith("test_"):
        values_a = pd.Series(results_a.get_test_metric_values(resolved))
        values_b = pd.Series(results_b.get_test_metric_values(resolved))
    else:
        values_a = pd.Series(results_a.get_metric_values(resolved))
        values_b = pd.Series(results_b.get_metric_values(resolved))

    pair_key = "results_a_vs_results_b"

    from .analysis import compare_two_models

    if paired:
        if len(values_a) == len(values_b):
            test_result = compare_two_models(values_a, values_b, paired=True, random_state=seed)
        else:
            warnings.warn(
                f"compare_results(paired=True) requires equal run counts. "
                f"results_a has {len(values_a)} runs, results_b has {len(values_b)}. "
                "Falling back to unpaired comparison. "
                "Pass paired=False to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            test_result = compare_two_models(values_a, values_b, paired=False, random_state=seed)
    else:
        test_result = compare_two_models(values_a, values_b, paired=False, random_state=seed)

    return ModelComparisonResults(
        overall_test=test_result,
        raw_data={"results_a": values_a, "results_b": values_b},
        pairwise_comparisons={pair_key: test_result},
        significant_comparisons=([pair_key] if test_result.is_significant() else []),
        correction_method="none",
        n_models=2,
        metric=resolved,
    )
