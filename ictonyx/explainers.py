import numpy as np
from typing import Optional, List, Any, TYPE_CHECKING

# Optional SHAP dependency
try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

# Optional matplotlib dependency
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# Import BaseModelWrapper with TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .core import BaseModelWrapper


def _check_shap():
    """Check SHAP availability."""
    if not HAS_SHAP:
        raise ImportError("SHAP is required for explainability features. Install with: pip install shap")


def _check_matplotlib():
    """Check matplotlib availability."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for SHAP plotting. Install with: pip install matplotlib")


def plot_shap_summary(model_wrapper: 'BaseModelWrapper',
                      X_data: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      plot_type: str = "bar"):
    """
    Generates a SHAP summary plot for a model.

    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_data (np.ndarray): The data to explain.
        feature_names (Optional[List[str]]): List of feature names.
        plot_type (str): The type of plot to generate ("bar", "dot", "violin").
    """
    _check_shap()
    _check_matplotlib()

    model_type = type(model_wrapper.model).__name__

    if "Tree" in model_type:
        # For tree-based models, use TreeExplainer
        explainer = shap.TreeExplainer(model_wrapper.model)
        shap_values = explainer.shap_values(X_data)

    elif hasattr(model_wrapper.model, 'layers'):
        # For neural networks (Keras/TensorFlow models)
        print("Note: Using DeepExplainer for neural network. This may take some time...")
        try:
            # Use a subset of the data as background for efficiency
            background_size = min(100, len(X_data))
            background = X_data[:background_size]
            explainer = shap.DeepExplainer(model_wrapper.model, background)
            shap_values = explainer.shap_values(X_data)
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to KernelExplainer (this will be slower)...")
            # Fallback to KernelExplainer
            background_size = min(50, len(X_data))
            explainer = shap.KernelExplainer(model_wrapper.model.predict, X_data[:background_size])
            shap_values = explainer.shap_values(X_data)

    else:
        # For other models, use KernelExplainer
        print("Using KernelExplainer, which can be very slow for large datasets.")
        print("Consider using a smaller sample of your data for faster results.")

        # Use model's predict_proba if available, otherwise predict
        if hasattr(model_wrapper, 'predict_proba'):
            predict_fn = model_wrapper.predict_proba
        else:
            predict_fn = model_wrapper.predict

        # Use a smaller background dataset for efficiency
        background_size = min(50, len(X_data))
        explainer = shap.KernelExplainer(predict_fn, X_data[:background_size])
        shap_values = explainer.shap_values(X_data)

    # Handle the different formats SHAP might return
    if isinstance(shap_values, list):
        # Multi-class models return a list of arrays (one per class)
        # Plot the first class by default
        if len(shap_values) > 1:
            print(f"Multi-class model detected. Plotting SHAP values for class 0.")
            print(f"Model has {len(shap_values)} classes. You may want to plot other classes separately.")
        shap.summary_plot(shap_values[0], X_data, feature_names=feature_names, plot_type=plot_type)
    else:
        # Single output (binary classification or regression)
        shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type=plot_type)


def plot_shap_waterfall(model_wrapper: 'BaseModelWrapper',
                        X_data: np.ndarray,
                        sample_index: int = 0,
                        feature_names: Optional[List[str]] = None,
                        class_index: int = 0):
    """
    Generates a SHAP waterfall plot for a single prediction.

    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_data (np.ndarray): The data to explain.
        sample_index (int): Index of the sample to explain.
        feature_names (Optional[List[str]]): List of feature names.
        class_index (int): For multi-class models, which class to explain.
    """
    _check_shap()
    _check_matplotlib()

    if sample_index >= len(X_data):
        raise ValueError(f"sample_index {sample_index} is out of range for data with {len(X_data)} samples")

    model_type = type(model_wrapper.model).__name__

    # Get explainer (same logic as summary plot)
    if "Tree" in model_type:
        explainer = shap.TreeExplainer(model_wrapper.model)
    elif hasattr(model_wrapper.model, 'layers'):
        background_size = min(100, len(X_data))
        background = X_data[:background_size]
        explainer = shap.DeepExplainer(model_wrapper.model, background)
    else:
        predict_fn = getattr(model_wrapper, 'predict_proba', model_wrapper.predict)
        background_size = min(50, len(X_data))
        explainer = shap.KernelExplainer(predict_fn, X_data[:background_size])

    # Get SHAP values for the specific sample
    sample_data = X_data[sample_index:sample_index + 1]
    shap_values = explainer.shap_values(sample_data)

    # Handle multi-class vs single output
    if isinstance(shap_values, list):
        if class_index >= len(shap_values):
            raise ValueError(f"class_index {class_index} is out of range for model with {len(shap_values)} classes")
        shap_values_to_plot = shap_values[class_index][0]  # Get first (and only) sample
    else:
        shap_values_to_plot = shap_values[0]  # Get first (and only) sample

    # Create explanation object for waterfall plot
    if hasattr(shap, 'Explanation'):
        # Newer SHAP versions
        explanation = shap.Explanation(
            values=shap_values_to_plot,
            base_values=explainer.expected_value[class_index] if isinstance(explainer.expected_value,
                                                                            list) else explainer.expected_value,
            data=sample_data[0],
            feature_names=feature_names
        )
        shap.waterfall_plot(explanation)
    else:
        # Older SHAP versions - use force plot as fallback
        print("Waterfall plot not available in this SHAP version. Using force plot instead.")
        base_value = explainer.expected_value[class_index] if isinstance(explainer.expected_value,
                                                                         list) else explainer.expected_value
        shap.force_plot(base_value, shap_values_to_plot, sample_data[0], feature_names=feature_names)


def plot_shap_dependence(model_wrapper: 'BaseModelWrapper',
                         X_data: np.ndarray,
                         feature_name: str,
                         feature_names: Optional[List[str]] = None,
                         interaction_index: Optional[int] = None,
                         class_index: int = 0):
    """
    Generates a SHAP dependence plot showing how a feature affects predictions.

    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_data (np.ndarray): The data to explain.
        feature_name (str): Name of the feature to plot (or index if feature_names not provided).
        feature_names (Optional[List[str]]): List of feature names.
        interaction_index (Optional[int]): Feature index to use for coloring interaction effects.
        class_index (int): For multi-class models, which class to explain.
    """
    _check_shap()
    _check_matplotlib()

    # Convert feature name to index if necessary
    if feature_names and isinstance(feature_name, str):
        if feature_name not in feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in feature_names")
        feature_index = feature_names.index(feature_name)
    else:
        # Assume feature_name is already an index
        try:
            feature_index = int(feature_name)
        except (ValueError, TypeError):
            raise ValueError(f"feature_name must be either a string (if feature_names provided) or an integer index")

        if feature_index >= X_data.shape[1]:
            raise ValueError(f"Feature index {feature_index} is out of range for data with {X_data.shape[1]} features")

    model_type = type(model_wrapper.model).__name__

    try:
        # Get explainer
        if "Tree" in model_type or "Forest" in model_type or "XGB" in model_type or "LGB" in model_type:
            explainer = shap.TreeExplainer(model_wrapper.model)
        elif hasattr(model_wrapper.model, 'layers'):
            try:
                background_size = min(100, len(X_data))
                background = X_data[:background_size]
                explainer = shap.DeepExplainer(model_wrapper.model, background)
            except Exception:
                predict_fn = getattr(model_wrapper, 'predict_proba', model_wrapper.predict)
                background_size = min(50, len(X_data))
                explainer = shap.KernelExplainer(predict_fn, X_data[:background_size])
        else:
            predict_fn = getattr(model_wrapper, 'predict_proba', model_wrapper.predict)
            background_size = min(50, len(X_data))
            explainer = shap.KernelExplainer(predict_fn, X_data[:background_size])

        # Get SHAP values
        shap_values = explainer.shap_values(X_data)

        # Handle multi-class vs single output
        if isinstance(shap_values, list):
            if class_index >= len(shap_values):
                raise ValueError(f"class_index {class_index} is out of range for model with {len(shap_values)} classes")
            shap_values_to_plot = shap_values[class_index]
        else:
            shap_values_to_plot = shap_values

        # Create dependence plot
        shap.dependence_plot(
            feature_index,
            shap_values_to_plot,
            X_data,
            feature_names=feature_names,
            interaction_index=interaction_index,
            show=False
        )
        plt.show()

    except Exception as e:
        raise RuntimeError(f"SHAP dependence plot failed: {e}")


def get_shap_feature_importance(model_wrapper: 'BaseModelWrapper',
                                X_data: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                class_index: int = 0) -> np.ndarray:
    """
    Get feature importance scores based on mean absolute SHAP values.

    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_data (np.ndarray): The data to explain.
        feature_names (Optional[List[str]]): List of feature names.
        class_index (int): For multi-class models, which class to analyze.

    Returns:
        np.ndarray: Mean absolute SHAP values for each feature.
    """
    _check_shap()

    model_type = type(model_wrapper.model).__name__

    # Get explainer
    if "Tree" in model_type:
        explainer = shap.TreeExplainer(model_wrapper.model)
    elif hasattr(model_wrapper.model, 'layers'):
        background_size = min(100, len(X_data))
        background = X_data[:background_size]
        explainer = shap.DeepExplainer(model_wrapper.model, background)
    else:
        predict_fn = getattr(model_wrapper, 'predict_proba', model_wrapper.predict)
        background_size = min(50, len(X_data))
        explainer = shap.KernelExplainer(predict_fn, X_data[:background_size])

    # Get SHAP values
    shap_values = explainer.shap_values(X_data)

    # Handle multi-class vs single output
    if isinstance(shap_values, list):
        if class_index >= len(shap_values):
            raise ValueError(f"class_index {class_index} is out of range for model with {len(shap_values)} classes")
        shap_values_to_use = shap_values[class_index]
    else:
        shap_values_to_use = shap_values

    # Calculate mean absolute SHAP values
    feature_importance = np.mean(np.abs(shap_values_to_use), axis=0)

    return feature_importance