# ictonyx/__init__.py
# v0.4.6

"""
Ictonyx: A Machine Learning Framework for Variability and Reproducibility Analysis

Ictonyx provides tools for conducting systematic variability studies, statistical analysis
of model performance, experiment tracking, and comprehensive ML workflow management.
"""

import os

from ._version import __version__

# Core imports (always available)
from .config import ModelConfig
from .core import (
    HUGGINGFACE_AVAILABLE,
    PYTORCH_AVAILABLE,
    SKLEARN_AVAILABLE,
    TENSORFLOW_AVAILABLE,
    BaseModelWrapper,
    TrainingResult,
)

# Data handling (always available)
from .data import DataHandler

# Exception classes (always available)
from .exceptions import (
    ConfigurationError,
    DataValidationError,
    ExperimentError,
    IctonyxError,
    ModelError,
    StatisticalTestError,
)

# Loggers (always available - BaseLogger works without dependencies)
from .loggers import BaseLogger

# Memory management
from .memory import (
    MemoryManager,
    cleanup_gpu_memory,
    get_memory_info,
    get_memory_manager,
    managed_memory,
)

# Experiment runners (always available)
from .runners import (
    ExperimentRunner,
    GridStudyResults,
    VariabilityStudyResults,
    run_grid_study,
    run_variability_study,
)

# Global settings
from .settings import logger, set_display_plots, set_theme, set_verbose
from .utils import load_object, save_object, train_val_test_split

# Build __all__ list with core functionality
__all__ = [
    # Core classes
    "ModelConfig",
    "BaseModelWrapper",
    "TrainingResult",
    "BaseLogger",
    # Settings
    "set_verbose",
    "set_display_plots",
    "set_theme",
    # Data handling
    "DataHandler",
    # Experiment running
    "ExperimentRunner",
    "run_variability_study",
    "VariabilityStudyResults",
    "run_grid_study",
    "GridStudyResults",
    # Exception classes
    "IctonyxError",
    "DataValidationError",
    "ModelError",
    "ExperimentError",
    "StatisticalTestError",
    "ConfigurationError",
    # Utilities
    "load_object",
    "save_object",
    "train_val_test_split",
    # Memory cleanup
    "MemoryManager",
    "managed_memory",
    "get_memory_manager",
    "cleanup_gpu_memory",
    "get_memory_info",
    # Feature availability flags
    "TENSORFLOW_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "PYTORCH_AVAILABLE",
    "HUGGINGFACE_AVAILABLE",
    "__version__",
]

# Framework-specific model wrappers
if TENSORFLOW_AVAILABLE:
    from .core import KerasModelWrapper

    __all__.append("KerasModelWrapper")

if SKLEARN_AVAILABLE:
    from .core import ScikitLearnModelWrapper

    __all__.append("ScikitLearnModelWrapper")

if PYTORCH_AVAILABLE:
    from .core import PyTorchModelWrapper

    __all__.append("PyTorchModelWrapper")

if HUGGINGFACE_AVAILABLE:
    from .core import HuggingFaceModelWrapper

    __all__.append("HuggingFaceModelWrapper")

# Data handlers with their dependencies
_data_handlers_loaded = []

try:
    from .data import ImageDataHandler

    __all__.append("ImageDataHandler")
    _data_handlers_loaded.append("ImageDataHandler")
except ImportError:
    pass

try:
    from .data import TabularDataHandler

    __all__.append("TabularDataHandler")
    _data_handlers_loaded.append("TabularDataHandler")
except ImportError:
    pass

try:
    from .data import TextDataHandler

    __all__.append("TextDataHandler")
    _data_handlers_loaded.append("TextDataHandler")
except ImportError:
    pass

try:
    from .data import TimeSeriesDataHandler

    __all__.append("TimeSeriesDataHandler")
    _data_handlers_loaded.append("TimeSeriesDataHandler")
except ImportError:
    pass

# Arrays and Auto-Resolver
try:
    from .data import ArraysDataHandler, auto_resolve_handler

    __all__.extend(["ArraysDataHandler", "auto_resolve_handler"])
    _data_handlers_loaded.append("ArraysDataHandler")
except ImportError:
    pass

# Statistical analysis functions
try:
    from .analysis import (
        ModelComparisonResults,
        StatisticalTestResult,
        anova_test,
        apply_multiple_comparison_correction,
        assess_training_stability,
        calculate_autocorr,
        calculate_averaged_autocorr,
        check_convergence,
        check_equal_variances,
        check_independence,
        check_normality,
        cohens_d,
        compare_two_models,
        create_results_dataframe,
        eta_squared,
        generate_statistical_summary,
        get_confusion_matrix_df,
        kruskal_wallis_test,
        mann_whitney_test,
        minimum_detectable_effect,
        paired_wilcoxon_test,
        rank_biserial_correlation,
        required_runs,
        required_runs_paired,
        shapiro_wilk_test,
        validate_sample_sizes,
        wilcoxon_signed_rank_test,
    )

    __all__.extend(
        [
            "ModelComparisonResults",
            "StatisticalTestResult",
            "mann_whitney_test",
            "paired_wilcoxon_test",
            "anova_test",
            "kruskal_wallis_test",
            "shapiro_wilk_test",
            "compare_two_models",
            "generate_statistical_summary",
            "create_results_dataframe",
            "assess_training_stability",
            "calculate_autocorr",
            "calculate_averaged_autocorr",
            "check_convergence",
            "get_confusion_matrix_df",
            "validate_sample_sizes",
            "check_normality",
            "check_equal_variances",
            "check_independence",
            "cohens_d",
            "rank_biserial_correlation",
            "eta_squared",
            "apply_multiple_comparison_correction",
            "required_runs",
            "required_runs_paired",
            "minimum_detectable_effect",
        ]
    )
    _has_statistical_functions = True
except ImportError:
    _has_statistical_functions = False

# Bootstrap confidence intervals
try:
    from .bootstrap import (
        BootstrapCIResult,
        bootstrap_ci,
        bootstrap_effect_size_ci,
        bootstrap_hedges_g_ci,
        bootstrap_mean_difference_ci,
        bootstrap_paired_difference_ci,
    )

    __all__.extend(
        [
            "BootstrapCIResult",
            "bootstrap_ci",
            "bootstrap_mean_difference_ci",
            "bootstrap_effect_size_ci",
            "bootstrap_hedges_g_ci",
            "bootstrap_paired_difference_ci",
        ]
    )
    _has_bootstrap = True
except ImportError:
    _has_bootstrap = False

# Plotting functions
try:
    from .plotting import (
        plot_autocorr_vs_lag,
        plot_averaged_autocorr,
        plot_averaged_pacf,
        plot_comparison_boxplots,
        plot_comparison_forest,
        plot_confusion_matrix,
        plot_epoch_run_heatmap,
        plot_grid_study_heatmap,
        plot_pacf_vs_lag,
        plot_paired_deltas,
        plot_pairwise_comparison_matrix,
        plot_precision_recall_curve,
        plot_rank_correlation_over_epoch,
        plot_roc_curve,
        plot_run_distribution,
        plot_run_independence_diagnostics,
        plot_run_strip,
        plot_run_trajectories,
        plot_sequential_ci,
        plot_training_history,
        plot_training_stability,
        plot_variability_summary,
    )

    __all__.extend(
        [
            "plot_confusion_matrix",
            "plot_training_history",
            "plot_roc_curve",
            "plot_precision_recall_curve",
            "plot_variability_summary",
            "plot_run_trajectories",
            "plot_run_distribution",
            "plot_run_strip",
            "plot_autocorr_vs_lag",
            "plot_averaged_autocorr",
            "plot_pacf_vs_lag",
            "plot_averaged_pacf",
            "plot_run_independence_diagnostics",
            "plot_pairwise_comparison_matrix",
            "plot_training_stability",
            "plot_rank_correlation_over_epoch",
            "plot_comparison_boxplots",
            "plot_comparison_forest",
            "plot_paired_deltas",
            "plot_grid_study_heatmap",
            "plot_epoch_run_heatmap",
            "plot_sequential_ci",
        ]
    )
    _has_plotting_functions = True
except ImportError:
    _has_plotting_functions = False

# MLflow logger and utilities
try:
    from .loggers import MLflowLogger
    from .utils import setup_mlflow

    __all__.extend(["MLflowLogger", "setup_mlflow"])
    _has_mlflow_logger = True
except ImportError:
    _has_mlflow_logger = False

# Hyperparameter tuning
try:
    from .tuning import HyperparameterTuner, create_search_space

    __all__.extend(["HyperparameterTuner", "create_search_space"])
    _has_hyperparameter_tuning = True
except ImportError:
    _has_hyperparameter_tuning = False

# Explainability features
try:
    from .explainers import (
        get_shap_feature_importance,
        plot_shap_dependence,
        plot_shap_summary,
        plot_shap_waterfall,
    )

    __all__.extend(
        [
            "plot_shap_summary",
            "plot_shap_waterfall",
            "plot_shap_dependence",
            "get_shap_feature_importance",
        ]
    )
    _has_explainability = True
except ImportError:
    _has_explainability = False

# Ultra-Simple API
try:
    from .api import compare_models, compare_results, variability_study

    __all__.extend(["variability_study", "compare_models", "compare_results"])
except ImportError:
    pass


# Feature availability summary
def get_feature_availability() -> dict:
    """Get a comprehensive summary of which optional features are available."""
    try:
        import joblib

        has_process_isolation = True
    except ImportError:
        has_process_isolation = False

    return {
        "tensorflow_support": TENSORFLOW_AVAILABLE,
        "sklearn_support": SKLEARN_AVAILABLE,
        "pytorch_support": PYTORCH_AVAILABLE,
        "huggingface_support": HUGGINGFACE_AVAILABLE,
        "statistical_functions": _has_statistical_functions,
        "bootstrap_ci": _has_bootstrap,
        "plotting_functions": _has_plotting_functions,
        "mlflow_logger": _has_mlflow_logger,
        "hyperparameter_tuning": _has_hyperparameter_tuning,
        "explainability": _has_explainability,
        "data_handlers": _data_handlers_loaded,
        "memory_management": True,
        "process_isolation": has_process_isolation,
    }
