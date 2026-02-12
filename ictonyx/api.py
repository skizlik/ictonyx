# ictonyx/api.py
"""
The High-Level API for Ictonyx.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""

from typing import Union, Callable, Any, Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from .core import BaseModelWrapper, TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
from .config import ModelConfig
from .data import auto_resolve_handler, DataHandler
from .runners import run_variability_study as _run_study
from .runners import VariabilityStudyResults
from .loggers import BaseLogger
from .analysis import compare_multiple_models as _stat_compare
from . import settings


def variability_study(
        model: Any,
        data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        target_column: Optional[str] = None,
        runs: int = 5,
        epochs: int = 10,
        batch_size: int = 32,
        tracker: Optional[BaseLogger] = None,
        use_process_isolation: bool = False,
        **kwargs
) -> VariabilityStudyResults:
    """
    Run a variability study with a single function call.
    """
    # 1. Prepare Data
    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)

    # 2. Prepare Model Builder
    # If the user passes a class (e.g. RandomForestClassifier), we instantiate it per run.
    # If they pass a function, we use it directly.
    builder = _get_model_builder(model)

    # 3. Configure
    # We explicitly pass kwargs into the config, trusting ModelConfig to ignore/store extras.
    config = ModelConfig({
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': 1 if settings._VERBOSE else 0,
        **kwargs
    })

    # 4. Execute
    return _run_study(
        model_builder=builder,
        data_handler=handler,
        model_config=config,
        num_runs=runs,
        epochs_per_run=epochs,
        tracker=tracker,
        use_process_isolation=use_process_isolation,
        gpu_memory_limit=kwargs.get('gpu_memory_limit')
    )


def compare_models(
        models: List[Any],
        data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        target_column: Optional[str] = None,
        runs: int = 5,
        epochs: int = 10,
        metric: str = 'val_accuracy',
        **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple models using rigorous statistical testing.
    """
    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)
    results_store = {}

    settings.logger.info(f"--- Starting Comparison of {len(models)} Models ---")

    for model_input in models:
        name = _get_model_name(model_input)
        settings.logger.info(f"Evaluating: {name}")

        # Run study for this specific model
        # We recurse into variability_study to avoid duplicating logic
        study_results = variability_study(
            model=model_input,
            data=handler,  # Pass the already resolved handler
            runs=runs,
            epochs=epochs,
            **kwargs
        )

        # Extract metric
        metrics = study_results.get_final_metrics(metric)
        if not metrics:
            settings.logger.warning(f"No '{metric}' data found for {name}")
            continue

        results_store[name] = pd.Series(list(metrics.values()))

    # Statistical Analysis
    if len(results_store) < 2:
        return {'error': 'Insufficient valid results for comparison'}

    return _stat_compare(results_store)


# --- Clean Helpers ---

def _get_model_builder(model: Any) -> Callable:
    """Standardizes model input into a factory function."""

    # If it's already a function, trust it.
    if callable(model) and not isinstance(model, type):
        return lambda conf: _ensure_wrapper(model(conf))

    # If it's a class (like RandomForestClassifier), instantiate it.
    if isinstance(model, type):
        return lambda conf: _ensure_wrapper(model())

    # If it's an instance, we must warn about state leakage.
    if hasattr(model, 'fit'):
        settings.logger.warning("Passed a model instance. Weights may persist between runs.")
        return lambda conf: _ensure_wrapper(model)

    raise ValueError(f"Invalid model input: {model}")


def _ensure_wrapper(obj: Any) -> BaseModelWrapper:
    """Ensures the object is wrapped in an Ictonyx wrapper."""
    if isinstance(obj, BaseModelWrapper):
        return obj

    # Duck typing checks are Pythonic and readable
    if hasattr(obj, 'fit') and hasattr(obj, 'predict'):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required to auto-wrap models with fit/predict. "
                "Install with: pip install scikit-learn"
            )
        from .core import ScikitLearnModelWrapper
        return ScikitLearnModelWrapper(obj)

    # String check avoids hard import of TensorFlow
    if 'keras' in str(type(obj)) or 'tensorflow' in str(type(obj)):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to auto-wrap Keras models. "
                "Install with: pip install tensorflow"
            )
        from .core import KerasModelWrapper
        return KerasModelWrapper(obj)

    raise TypeError(f"Cannot wrap model of type: {type(obj)}")

def _get_model_name(obj: Any) -> str:
    """Extracts a readable name from a model object/class/function."""
    if hasattr(obj, '__name__'): return obj.__name__
    if hasattr(obj, '__class__'): return obj.__class__.__name__
    return str(obj)