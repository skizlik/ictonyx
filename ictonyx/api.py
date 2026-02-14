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
    """Standardizes model input into a factory function.

    Handles three input types:
      - Callable (function): used directly as a builder. The user is
        responsible for returning a fresh model each call.
      - Class: instantiated fresh per run (e.g., RandomForestClassifier).
      - Instance: cloned per run for sklearn models. Keras and PyTorch
        instances are rejected because they cannot be cleanly cloned.
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
    """Ensures the object is wrapped in an Ictonyx wrapper."""
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
    """Extracts a readable name from a model object/class/function."""
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)
