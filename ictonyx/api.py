# ictonyx/api.py
"""
The High-Level API for Ictonyx.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""
import dataclasses
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
from .runners import FIT_KWARG_KEYS, VariabilityStudyResults
from .runners import run_variability_study as _run_study

# Resolve torch.nn once at import time so isinstance checks below are reliable
# and don't require repeated try/except blocks throughout this module.
if PYTORCH_AVAILABLE:
    import torch.nn as _torch_nn
else:
    _torch_nn = None  # type: ignore[assignment]


# Data-pipeline kwargs: routed to the DataHandler (or the runner), never into ModelConfig.
# Per handler type, so a key the handler would ignore is rejected instead of dropped (v12 0.6).
_SPLIT_KWARGS = frozenset({"test_split", "val_split", "split_seed", "stratify"})
_HANDLER_KWARGS: Dict[str, frozenset] = {
    "arrays": _SPLIT_KWARGS | {"validation_data", "X_val", "y_val", "X_test", "y_test"},
    "tabular": _SPLIT_KWARGS | {"features", "sep", "header", "return_frames"},
    "image": (_SPLIT_KWARGS - {"stratify"}) | {"image_size", "color_mode"},
    "text": _SPLIT_KWARGS | {"text_column", "label_column", "max_features"},
    "timeseries": (_SPLIT_KWARGS - {"stratify"})
    | {"value_column", "sequence_length", "lookback", "stride"},
}
_RUNNER_KWARGS = frozenset({"gpu_memory_limit", "process_timeout", "allow_isolation_fallback"})
_INFRA_KWARGS = frozenset().union(*_HANDLER_KWARGS.values()) | _RUNNER_KWARGS
_TS_TRIGGERS = ("value_column", "sequence_length", "lookback", "stride")


def _handler_kind(data: Any, kwargs: Dict[str, Any]) -> str:
    """Mirror auto_resolve_handler's routing so validation matches construction."""
    import os

    if isinstance(data, DataHandler):
        return data.data_type
    if isinstance(data, tuple) and len(data) == 2:
        return "arrays"
    if isinstance(data, pd.DataFrame):
        return "tabular"
    if isinstance(data, str):
        if os.path.isdir(data):
            return "image"
        if "text_column" in kwargs or "label_column" in kwargs:
            return "text"
        if any(k in kwargs for k in _TS_TRIGGERS):
            return "timeseries"
        return "tabular"
    raise TypeError(f"Unsupported data type: {type(data).__name__}")


_AMBIGUOUS_KWARGS = frozenset({"max_features"})
"""Names that are a data-handler argument for SOME kinds and a model hyperparameter
for others (max_features: TextDataHandler's TF-IDF vocabulary vs RandomForest).
When the detected handler does not accept them they route to the model; every
other cross-kind handler key raises rather than silently becoming a model param."""


def _split_kwargs(data: Any, kwargs: Dict[str, Any]) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """Split ``**kwargs`` into (infra, model) by the DETECTED handler kind."""
    kind = _handler_kind(data, kwargs)
    accepted = _HANDLER_KWARGS[kind] | _RUNNER_KWARGS
    stray = sorted(
        k for k in kwargs if k in _INFRA_KWARGS and k not in accepted and k not in _AMBIGUOUS_KWARGS
    )
    if stray:
        raise ConfigurationError(
            f"{stray} are not accepted for {kind!r} data and would have been ignored. "
            f"Accepted: {sorted(_HANDLER_KWARGS[kind])}."
        )
    infra = {k: v for k, v in kwargs.items() if k in accepted}
    model = {k: v for k, v in kwargs.items() if k not in accepted}
    return infra, model


def _resolve_handler(
    data: Any, target_column: Optional[str], infra_kwargs: Dict[str, Any]
) -> DataHandler:
    """Route infra kwargs to the right DataHandler; reject anything that would be dropped."""
    infra = {k: v for k, v in infra_kwargs.items() if k not in _RUNNER_KWARGS}
    kind = _handler_kind(data, infra)

    unknown = sorted(set(infra) - _HANDLER_KWARGS[kind])
    if unknown:
        raise ConfigurationError(
            f"{unknown} are not accepted for {kind!r} data and would have been ignored. "
            f"Accepted: {sorted(_HANDLER_KWARGS[kind])}."
        )

    if isinstance(data, DataHandler):
        if infra:
            raise ConfigurationError(
                f"{sorted(infra)} cannot be combined with a DataHandler instance; "
                "construct the handler with these arguments instead."
            )
        return data

    validation_data = infra.pop("validation_data", None)
    if validation_data is not None:
        if not (isinstance(validation_data, tuple) and len(validation_data) == 2):
            raise ConfigurationError("validation_data must be an (X_val, y_val) tuple.")
        infra["X_val"], infra["y_val"] = validation_data
    return auto_resolve_handler(data, target_column=target_column, **infra)


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
        stratify: Passed to ``ArraysDataHandler`` for tuple data.
        validation_data: ``(X_val, y_val)`` to use as the validation set
            instead of carving one from ``data``. Only with ``(X, y)`` tuple
            data. Raises ``ConfigurationError`` otherwise.
        verbose: If ``False``, suppress all training output. Default ``True``.
        use_parallel: If ``True``, fan training runs across multiple
            processes using ``joblib``. Safe for sklearn models. Not
            recommended for Keras/TF models. Mutually exclusive with
            ``use_process_isolation``. Default ``False``.
        n_jobs: Number of parallel workers. ``-1`` uses all CPUs.
            Ignored when ``use_parallel=False``. Default ``-1``.
        **kwargs: Data-handler keys are routed to the handler; everything else goes
            into ModelConfig. Accepted data-handler keys by input type:
              arrays (X, y):   test_split, val_split, split_seed, stratify,
                               validation_data, X_val, y_val, X_test, y_test
              DataFrame / CSV: test_split, val_split, split_seed, stratify,
                               features, sep, header, return_frames
              image directory: test_split, val_split, split_seed, image_size, color_mode
              text CSV:        test_split, val_split, split_seed, stratify,
                               text_column, label_column, max_features
              time-series CSV: test_split, val_split, split_seed,
                               value_column, sequence_length, lookback, stride
            A key not accepted for the detected input type raises ConfigurationError.
            split_seed (default 42) is shared across models in compare_models, so
            every model sees the same split.

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

    # Apply verbose setting to global logger
    from .settings import set_verbose

    set_verbose(verbose)

    # compare_models warns once for the whole comparison and asks each study not
    # to repeat it (v12 2.27). The key is popped so it never reaches ModelConfig.
    _suppress_runs_warning = bool(kwargs.pop("_suppress_runs_warning", False))
    if runs < 20 and not _suppress_runs_warning:
        warnings.warn(
            f"runs={runs} may be insufficient for reliable statistical inference. "
            "rank-based tests (paired Wilcoxon, Mann-Whitney U) have limited power against small effects below n=20. "
            "Consider runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # Separate infrastructure kwargs (forwarded to the data handler) from
    # model kwargs (forwarded to ModelConfig). Add new DataHandler constructor
    # parameters to _INFRA_KWARGS to prevent them from appearing in ModelConfig.

    infra_kwargs, model_kwargs = _split_kwargs(data, kwargs)

    # If model is a class and the user passed model_kwargs=dict (the pattern
    # documented in the HuggingFaceModelWrapper docstring), unpack it into
    # model_kwargs so individual constructor parameters flow through the
    # normal ModelConfig path. Without this unpacking, the entire dict lands
    # in ModelConfig as a single 'model_kwargs' key, which the wrapper's
    # __init__ then sees as one kwarg instead of unpacked arguments.
    if isinstance(model, type) and "model_kwargs" in model_kwargs:
        user_model_kwargs = model_kwargs.pop("model_kwargs")
        if isinstance(user_model_kwargs, dict):
            model_kwargs.update(user_model_kwargs)

    # 1. Prepare Data
    handler = _resolve_handler(data, target_column, infra_kwargs)

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
        process_timeout=kwargs.get("process_timeout"),
        allow_isolation_fallback=bool(kwargs.get("allow_isolation_fallback", False)),
    )


def _warn_incomplete_studies(studies: Dict[str, VariabilityStudyResults], metric: str) -> None:
    """Warn once per study whose runs failed or produced NaN (v12 2.67).

    Excluding runs that diverged understates variance and overstates the mean;
    the user should know the comparison is on fewer runs than requested.
    """
    for name, s in studies.items():
        vals = np.asarray(s.get_metric_values(metric), dtype=float)
        n_nan = int(np.isnan(vals).sum())
        if s.failed_runs or n_nan:
            warnings.warn(
                f"{name}: {len(s.failed_runs)} of {s.n_requested} runs failed and {n_nan} "
                f"produced a NaN {metric}; they are excluded from the comparison. "
                "Excluding runs that diverged understates variance and overstates the mean.",
                UserWarning,
                stacklevel=3,
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

    **What this establishes.** Inference is conditional on this fixed
    train/validation/test split and on seed-induced training randomness only.
    Test-set sampling error is shared by every run and is not propagated. A
    significant result therefore shows a seed-distribution shift *on this
    split*; it does not by itself show superiority on new splits or new data.

    **Seeding and pairing**

    All models receive the same base ``seed``. Internally,
    ``np.random.SeedSequence(seed).spawn(runs)`` generates per-run child seeds
    that are identical across models: model A's run *i* and model B's run *i*
    always share the same child seed. The runs are therefore **genuinely paired
    at the RNG level**, regardless of framework.

    For this reason ``paired`` defaults to ``True``:

    * **Two models (default, paired):** Paired Wilcoxon signed-rank test on the
      per-run differences. Pairing guarantees that run *i* of each model
      trained under the same child seed, so unequal or misaligned samples are
      impossible. It does **not** in general increase power: for different
      model families the shared seed drives unrelated random streams and the
      paired differences are approximately independent (observed Spearman
      correlation near 0 for, e.g., random forest vs. MLP). Plan ``runs`` as
      you would for an unpaired test.
    * **Two models (unpaired):** Mann-Whitney U (no omnibus test is run for
      two groups). Valid but does not use the RNG pairing.
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
            "rank-based tests (paired Wilcoxon, Mann-Whitney U) have limited power against small effects below n=20. "
            "Consider runs >= 20 for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve seed once — all models use the same base seed for reproducibility
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    # Split kwargs exactly as variability_study does. The handler is resolved
    # ONCE and shared, so every model sees the same split (data-level pairing).
    # Only model kwargs and runner kwargs are forwarded; handler kwargs must not
    # reach variability_study(data=handler), which rightly rejects them (v12 0.5).
    infra_kwargs, model_kwargs = _split_kwargs(data, kwargs)
    runner_kwargs = {k: v for k, v in infra_kwargs.items() if k in _RUNNER_KWARGS and v is not None}
    handler = _resolve_handler(data, target_column, infra_kwargs)

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
            _suppress_runs_warning=True,
            **runner_kwargs,
            **model_kwargs,
        )
        studies[name] = study_result

    # --- Resolve metric now that all studies are complete ---
    if metric is None:
        # Resolve from the metrics EVERY study produced, not the first one's
        # (v12 2.30: a regressor first and a classifier second raised for val_r2).
        per_model = {name: set(s.get_available_metrics()) for name, s in studies.items()}
        common = set.intersection(*per_model.values()) if per_model else set()
        for candidate in ("val_accuracy", "val_r2", "val_loss"):
            if candidate in common:
                metric = candidate
                break
        else:
            val_common = sorted(m for m in common if m.startswith("val_"))
            if val_common:
                metric = val_common[0]
            elif not any(per_model.values()):
                metric = "val_accuracy"  # nothing recorded anywhere: fall through to the
                #                            downstream "insufficient valid results" error
            else:
                raise ConfigurationError(
                    "compare_models() found no metric common to all models. Per model: "
                    + "; ".join(f"{n}: {sorted(m)}" for n, m in per_model.items())
                    + ". Pass metric= explicitly, or compare models of the same task type."
                )

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
        from .analysis import align_paired, compare_two_models

        names = list(results_store.keys())
        # All models ran under one seed, so pairing is valid; align on run id in
        # case either study lost a run.
        run_ids, va, vb = align_paired(studies[names[0]], studies[names[1]], metric)
        series_a = pd.Series(va, index=run_ids, name=names[0])
        series_b = pd.Series(vb, index=run_ids, name=names[1])
        _warn_incomplete_studies(studies, metric)
        paired_result = compare_two_models(series_a, series_b, paired=True, random_state=seed)
        return ModelComparisonResults(
            overall_test=paired_result,
            # raw_data is the aligned pairs the test actually used (v12 2.33)
            raw_data={names[0]: series_a, names[1]: series_b},
            run_counts={n: (s.n_requested, s.n_runs) for n, s in studies.items()},
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

    _warn_incomplete_studies(studies, metric)
    stat_results = _stat_compare(results_store, random_state=seed)
    stat_results.run_counts = {n: (s.n_requested, s.n_runs) for n, s in studies.items()}
    stat_results.metric = metric
    stat_results.raw_data = results_store
    return stat_results


# --- Clean Helpers ---


class _EnsureWrapperBuilder:
    """Picklable builder: call a user function, coerce the result to a BaseModelWrapper.

    Replaces the former ``lambda conf: _ensure_wrapper(model(conf))``, which
    stdlib pickle cannot serialise (1.14). Looks up ``_ensure_wrapper`` as a
    module global at call time so tests can monkeypatch it.
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, conf: ModelConfig) -> BaseModelWrapper:
        return _ensure_wrapper(self.fn(conf))

    def __repr__(self) -> str:
        return f"_EnsureWrapperBuilder({getattr(self.fn, '__name__', repr(self.fn))})"


class _CloneBuilder:
    """Picklable builder: ``sklearn.base.clone(model)`` per run, seeded per run.

    v12 0.9: clone() preserves the user's ``random_state``, so an instance
    with a fixed seed trained identical models on every run and one with
    ``random_state=None`` was seeded only through the global NumPy RNG. The
    clone now receives the run's child seed, exactly as the class path does.
    """

    _warned: bool = False  # once per process, like _WARNED_FIT_KWARGS

    def __init__(self, model: Any):
        self.model = model
        params = model.get_params() if hasattr(model, "get_params") else {}
        self.has_random_state = "random_state" in params
        self.user_random_state = params.get("random_state")

    def __call__(self, conf: ModelConfig) -> BaseModelWrapper:
        from sklearn.base import clone

        est = clone(self.model)
        run_seed = conf.get("run_seed")
        if self.has_random_state and run_seed is not None:
            est.set_params(random_state=run_seed)
            if self.user_random_state is not None and not _CloneBuilder._warned:
                _CloneBuilder._warned = True
                warnings.warn(
                    f"{type(self.model).__name__} instance has random_state="
                    f"{self.user_random_state}; it is overridden with the per-run child seed "
                    "so runs vary and pair across models. Pass the class instead to silence this.",
                    UserWarning,
                    stacklevel=4,
                )
        elif not self.has_random_state and not _CloneBuilder._warned:
            _CloneBuilder._warned = True
            warnings.warn(
                f"{type(self.model).__name__} has no random_state parameter; per-run variation "
                "depends on the global NumPy RNG only and results may be identical across runs.",
                UserWarning,
                stacklevel=4,
            )
        return _ensure_wrapper(est)


class _ClassBuilder:
    """Picklable builder: instantiate ``model_class`` per run from a ModelConfig."""

    def __init__(self, model_class: Type[Any]):
        self.model_class = model_class

    def __call__(self, conf: ModelConfig) -> BaseModelWrapper:
        return _build_from_class(conf, self.model_class)


def _build_from_class(conf: ModelConfig, _model_class: Type[Any]) -> BaseModelWrapper:
    """Instantiate ``_model_class`` per run from ``conf``. Module-level so it pickles."""
    sig = inspect.signature(_model_class)
    # Extract kwargs from the ModelConfig that the wrapper's constructor
    # actually accepts. Filter out infra keys the constructor doesn't know
    # about (e.g. run_seed), and `random_state` which we pass explicitly
    # below when the signature supports it.
    accepted = set(sig.parameters.keys())
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    # Runner concerns, never a constructor's. Training-loop keys and
    # validation_data are consumed by the runner; FIT_KWARG_KEYS are forwarded
    # to fit() by runners.build_fit_kwargs, the single owner of that contract.
    _RUNNER_ONLY_KWARGS = {"epochs", "batch_size", "verbose", "validation_data"} | FIT_KWARG_KEYS

    candidate = {
        k: v
        for k, v in conf.items()
        if k != "run_seed" and k != "random_state" and k not in _RUNNER_ONLY_KWARGS
    }
    construction_kwargs = {
        k: v for k, v in candidate.items() if accepts_var_keyword or k in accepted
    }
    dropped = sorted(set(candidate) - set(construction_kwargs))
    if dropped:
        _warn_dropped_construction_kwargs(_model_class, dropped)

    passes_random_state = ("random_state" in accepted or accepts_var_keyword) and not issubclass(
        _model_class, BaseModelWrapper
    )
    try:
        if passes_random_state:
            return _ensure_wrapper(
                _model_class(random_state=conf.get("run_seed"), **construction_kwargs)
            )
        return _ensure_wrapper(_model_class(**construction_kwargs))
    except TypeError as e:
        # v12 2.53: only a random_state complaint is a random_state problem. Any
        # other TypeError is a configuration error and is reported as one.
        if passes_random_state and "random_state" in str(e):
            try:
                built = _model_class(**construction_kwargs)  # retry without the seed
            except TypeError as e2:
                raise ConfigurationError(
                    f"Failed to construct {_model_class.__name__}: {e2}. "
                    "Check the keys in your ModelConfig against the constructor signature."
                ) from e2
            warnings.warn(
                f"Could not pass random_state to {_model_class.__name__}. "
                f"Reproducibility not guaranteed. Original error: {e}",
                UserWarning,
                stacklevel=3,
            )
            return _ensure_wrapper(built)
        raise ConfigurationError(
            f"Failed to construct {_model_class.__name__}: {e}. "
            "Check the keys in your ModelConfig against the constructor signature."
        ) from e


_WARNED_DROPPED_KWARGS: set = set()


def _warn_dropped_construction_kwargs(model_class: Type[Any], dropped: List[str]) -> None:
    """Warn once per (class, keys) that config keys the constructor cannot take were dropped."""
    key = (model_class.__name__, tuple(dropped))
    if key in _WARNED_DROPPED_KWARGS:
        return
    _WARNED_DROPPED_KWARGS.add(key)
    warnings.warn(
        f"{model_class.__name__} does not accept {dropped}; these ModelConfig keys were not "
        "passed to its constructor. Check for typos. (This becomes an error in v0.5.0.)",
        UserWarning,
        stacklevel=4,
    )


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

    # 1. A class (RandomForestClassifier, a wrapper subclass): instantiate per run.
    if isinstance(model, type):
        return _ClassBuilder(model)

    # 2. Framework model INSTANCES and wrapper instances, BEFORE the generic
    #    callable test: Keras models and nn.Modules are callable, so the old
    #    order treated them as builder functions and called forward(ModelConfig)
    #    (v12 1.32). _build_instance_cloner has the right message for them.
    if _is_framework_instance(model) or isinstance(model, BaseModelWrapper):
        return _build_instance_cloner(model)

    # 3. Any other callable is a builder function: trust it.
    if callable(model):
        return _EnsureWrapperBuilder(model)

    # 4. Anything else with fit() is an estimator instance to clone per run.
    if hasattr(model, "fit"):
        return _build_instance_cloner(model)

    raise ValueError(f"Invalid model input: {model}")


def _is_framework_instance(obj: Any) -> bool:
    """True for a Keras Model or torch nn.Module instance (both are callable)."""
    if PYTORCH_AVAILABLE and _torch_nn is not None and isinstance(obj, _torch_nn.Module):
        return True
    if TENSORFLOW_AVAILABLE:
        import tensorflow as tf

        if isinstance(obj, tf.keras.Model):
            return True
    return False


def _build_instance_cloner(model: Any) -> Callable:
    """Creates a builder that produces independent copies of a model instance.

    For sklearn estimators, uses sklearn.base.clone() which creates an
    unfitted copy with the same hyperparameters. This preserves the user's
    configuration (e.g., n_estimators=100) while ensuring each run starts
    from scratch.

    For Keras and PyTorch instances, cloning is not reliably possible, so
    we raise an error guiding the user to pass a class or builder function.
    """
    if isinstance(model, BaseModelWrapper):
        raise ValueError(
            f"{type(model).__name__} is a wrapper INSTANCE. Pass a builder function that "
            "returns a fresh wrapper per run (def build(config): return "
            f"{type(model).__name__}(...)), so every run starts from an untrained model."
        )

    # sklearn: clone() creates an unfitted copy with same hyperparameters
    if hasattr(model, "get_params"):
        try:
            from sklearn.base import clone

            # Test that clone works before committing to this path
            clone(model)
            settings.logger.info(
                f"Cloning {type(model).__name__} instance per run for independence."
            )
            return _CloneBuilder(model)
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


def compare_results(
    results_a: "VariabilityStudyResults",
    results_b: "VariabilityStudyResults",
    metric: Optional[str] = None,
    paired: bool = True,
    seed: Optional[int] = None,
    force_paired: bool = False,
) -> "ModelComparisonResults":
    """Compare two pre-computed VariabilityStudyResults without re-running training.

    Extracts metric values from each results object and compares them
    statistically. Use this when you already have results from
    :func:`variability_study` and want to compare them without retraining.

    **What this establishes.** Inference is conditional on this fixed
    train/validation/test split and on seed-induced training randomness only.
    Test-set sampling error is shared by every run and is not propagated. A
    significant result therefore shows a seed-distribution shift *on this
    split*; it does not by itself show superiority on new splits or new data.

    **Pairing:** If both results were produced with the same ``seed``, the runs
    are paired by construction. Pass ``paired=True`` (default) to exploit this
    with the more powerful paired Wilcoxon signed-rank test. Pass ``paired=False``
    to use the independent-samples Mann-Whitney U test instead.

    Args:
        results_a: First model's results.
        results_b: Second model's results.
        metric: Metric to compare. If ``None``, resolves via
            ``results_a.preferred_metric()``.
        paired: If ``True`` (default), align the two studies on run id via
            :func:`~ictonyx.analysis.align_paired` and use the paired Wilcoxon
            signed-rank test. Runs missing from either side are dropped with a
            ``UserWarning``. If the studies have different ``seed`` values (so
            their runs are not paired), falls back to Mann-Whitney U with a
            ``UserWarning``.
        force_paired: Skip the seed check in ``align_paired``. Use only when you
            know the runs are paired despite differing or missing seeds.
        seed: Random state for bootstrap CI computation. Defaults to ``None``
            (non-deterministic CIs).

    Returns:
        :class:`~ictonyx.analysis.ModelComparisonResults`.

    Raises:
        KeyError: If the resolved metric is not present in both results.
    """
    from .analysis import align_paired, compare_two_models

    resolved = metric if metric is not None else results_a.preferred_metric("accuracy")
    pair_key = "results_a_vs_results_b"

    if paired:
        try:
            _, va, vb = align_paired(results_a, results_b, resolved, force=force_paired)
        except ValueError as e:
            warnings.warn(
                f"compare_results(paired=True): {e} Falling back to an unpaired "
                "Mann-Whitney U test. Pass paired=False to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            paired = False
        else:
            values_a, values_b = pd.Series(va), pd.Series(vb)
            test_result = compare_two_models(values_a, values_b, paired=True, random_state=seed)

    if not paired:
        if resolved.startswith("test_"):
            values_a = pd.Series(results_a.get_test_metric_values(resolved))
            values_b = pd.Series(results_b.get_test_metric_values(resolved))
        else:
            values_a = pd.Series(results_a.get_metric_values(resolved))
            values_b = pd.Series(results_b.get_metric_values(resolved))
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
