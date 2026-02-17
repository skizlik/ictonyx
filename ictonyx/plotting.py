# ictonyx/plotting.py
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import settings

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sn
    from matplotlib.figure import Figure

    HAS_PLOTTING = True
except ImportError:
    plt = None
    Figure = None
    sn = None
    HAS_PLOTTING = False

# Optional sklearn for metrics
try:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False

# Optional TensorFlow for utils
try:
    from tensorflow.keras.utils import to_categorical

    HAS_TENSORFLOW_UTILS = True
except ImportError:
    HAS_TENSORFLOW_UTILS = False

if TYPE_CHECKING:
    from .core import BaseModelWrapper


# --- Helpers ---


def _check_plotting():
    if not HAS_PLOTTING:
        raise ImportError(
            "matplotlib and seaborn are required. Install with: pip install matplotlib seaborn"
        )


# FIX THIS: eliminate backward compatibility, prefer to update functions


def _check_matplotlib():
    """Alias for _check_plotting for backward compatibility."""
    _check_plotting()


def _check_seaborn():
    """Alias for _check_plotting for backward compatibility."""
    _check_plotting()


def _check_sklearn_metrics():
    if not HAS_SKLEARN_METRICS:
        raise ImportError(
            "scikit-learn required for ROC/PR curves. Install with: pip install scikit-learn"
        )


def _check_tensorflow_utils():
    if not HAS_TENSORFLOW_UTILS:
        raise ImportError(
            "TensorFlow required for multi-class plotting. Install with: pip install tensorflow"
        )


def _finalize_plot(fig: "Figure", show_arg: Optional[bool]) -> Optional["Figure"]:
    """
    Centralized logic for showing/returning plots.
    If showing: returns None (to prevent Jupyter double-display).
    If not showing: returns the Figure object (for saving/modification).
    """
    should_show = show_arg if show_arg is not None else settings.should_display()

    if should_show:
        plt.show()
        return None  # Prevents Jupyter from rendering the return value

    return fig


def _find_metric_columns(df: pd.DataFrame, metric: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Smart column detection. Tries to find train/val columns for a metric
    even if prefixes don't match exactly.
    """
    base = metric.replace("train_", "").replace("val_", "")

    # Candidates for training
    train_candidates = [f"train_{base}", base, f"{base}_train", "loss" if base == "loss" else None]
    train_col = next((c for c in train_candidates if c and c in df.columns), None)

    # Candidates for validation
    val_candidates = [f"val_{base}", f"{base}_val", f"test_{base}"]
    val_col = next((c for c in val_candidates if c and c in df.columns), None)

    return train_col, val_col


# --- Standard Plots ---


def plot_confusion_matrix(
    cm_df: pd.DataFrame, title: str = "", show: Optional[bool] = None
) -> Optional["Figure"]:
    """Plot a confusion matrix as an annotated heatmap.

    Args:
        cm_df: A square ``pd.DataFrame`` where rows are true labels and
            columns are predicted labels. Values are integer counts.
            Typically produced by
            :func:`~ictonyx.analysis.get_confusion_matrix_df`.
        title: Plot title. Default: ``"Confusion Matrix"``.
        show: If ``True``, call ``plt.show()``. If ``False``, return the
            figure without displaying. If ``None`` (default), defer to
            the global :func:`~ictonyx.settings.set_display_plots` setting.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_plotting()

    fig = plt.figure(figsize=(10, 8))
    sn.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title if title else "Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    return _finalize_plot(fig, show)


def plot_training_history(
    history: Any, title: str = None, metrics: List[str] = None, show: Optional[bool] = None
) -> Optional["Figure"]:
    """Plot training and validation metrics over epochs.

    Creates one subplot per metric, each showing train and validation
    curves. Accepts DataFrames, dicts, lists of dicts, or Keras
    ``History`` objects.

    Args:
        history: Training history in any of the supported formats.
            Column/key naming convention: ``'train_accuracy'`` and
            ``'val_accuracy'``, or ``'accuracy'`` and ``'val_accuracy'``.
        title: Plot title. If ``None``, auto-generated from epoch count.
        metrics: List of metric base names to plot (e.g. ``['accuracy',
            'loss']``). If ``None``, auto-detected from available columns.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_plotting()

    # Convert various history formats to DataFrame
    if isinstance(history, pd.DataFrame):
        history_df = history
    elif isinstance(history, dict):
        history_df = pd.DataFrame(history)
    elif isinstance(history, list) and all(isinstance(i, dict) for i in history):
        history_df = pd.DataFrame(history)
    else:
        try:
            history_df = pd.DataFrame(history.history)
        except AttributeError:
            settings.logger.error("The provided history object format is not supported.")
            return None

    # Determine metrics
    if metrics is None:
        available = history_df.columns.tolist()
        default_metrics = []
        if "accuracy" in available or "train_accuracy" in available:
            default_metrics.append("accuracy")
        if "loss" in available or "train_loss" in available:
            default_metrics.append("loss")

        if default_metrics:
            metrics = default_metrics
        else:
            metrics = [col for col in available if not col.startswith("val_") and col != "epoch"]
            if not metrics:
                settings.logger.warning("No metrics found to plot.")
                return None

    if title is None:
        n_epochs = len(history_df)
        title = f"Training Progress: {n_epochs} Epochs"

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    colors = settings.THEME

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Use smart column finding
        train_col, val_col = _find_metric_columns(history_df, metric)

        if not train_col:
            continue

        metric_display = metric.replace("_", " ").title()
        if metric.lower() in ["mse", "mae", "rmse"]:
            metric_display = metric.upper()
        elif metric.lower() == "auc":
            metric_display = "AUC"

        epochs = range(1, len(history_df) + 1)

        ax.plot(
            epochs, history_df[train_col], label=f"Training", color=colors["train"], linewidth=2
        )

        if val_col:
            ax.plot(
                epochs, history_df[val_col], label=f"Validation", color=colors["val"], linewidth=2
            )
            final_train = history_df[train_col].iloc[-1]
            final_val = history_df[val_col].iloc[-1]
            ax.plot([], [], " ", label=f"Final: {final_train:.4f} / {final_val:.4f}")
        else:
            final_train = history_df[train_col].iloc[-1]
            ax.plot([], [], " ", label=f"Final: {final_train:.4f}")

        ax.set_title(metric_display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_display, fontsize=11)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Set y-axis limits for bounded metrics
        if any(x in metric.lower() for x in ["accuracy", "auc", "precision", "recall", "f1"]):
            ax.set_ylim([0, 1.05])

        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, color=colors.get("grid", "#e6e6e6"))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return _finalize_plot(fig, show)


def plot_roc_curve(
    model_wrapper: "BaseModelWrapper",
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "",
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot per-class ROC curves with AUC values.

    Binarizes labels using one-vs-rest and plots one ROC curve per class.
    Requires the model wrapper to have a ``predict_proba`` method.

    Args:
        model_wrapper: A trained :class:`~ictonyx.core.BaseModelWrapper`
            with ``predict_proba`` support.
        X_test: Test feature array.
        y_test: True class labels (integer-encoded).
        title: Plot title. Default: ``'Receiver Operating Characteristic'``.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if the model does
        not support probability predictions.
    """
    _check_plotting()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, "predict_proba"):
        settings.logger.warning(
            "Model does not have a predict_proba method. Cannot plot ROC curve."
        )
        return None

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    fig = plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC curve of class {i} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title if title else "Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    return _finalize_plot(fig, show)


def plot_precision_recall_curve(
    model_wrapper: "BaseModelWrapper",
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "",
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot per-class precision-recall curves with AUC values.

    Binarizes labels using one-vs-rest and plots one PR curve per class.
    Requires the model wrapper to have a ``predict_proba`` method.

    Args:
        model_wrapper: A trained :class:`~ictonyx.core.BaseModelWrapper`
            with ``predict_proba`` support.
        X_test: Test feature array.
        y_test: True class labels (integer-encoded).
        title: Plot title. Default: ``'Precision-Recall Curve'``.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if the model does
        not support probability predictions.
    """
    _check_plotting()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, "predict_proba"):
        settings.logger.warning(
            "Model does not have a predict_proba method. Cannot plot Precision-Recall curve."
        )
        return None

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    fig = plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(
            recall, precision, label=f"Precision-recall curve of class {i} (area = {pr_auc:.2f})"
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title if title else "Precision-Recall Curve")
    plt.legend(loc="lower left")

    return _finalize_plot(fig, show)


# --- Variability & Comparison Visualizations ---


def plot_variability_summary(
    all_runs_metrics_list: List[pd.DataFrame],
    final_metrics_series: Union[pd.Series, List],
    final_test_series: Optional[Union[pd.Series, List]] = None,
    metric: str = "accuracy",
    show_histogram: bool = True,
    show_boxplot: bool = False,
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot a multi-panel variability summary for a completed study.

    Generates a figure with:

    * **Overlaid training curves** — all runs plotted together with a
      mean line, showing epoch-by-epoch convergence and spread.
    * **Final metric distribution** — boxplot of final-epoch values,
      showing the spread of outcomes across runs.

    Args:
        all_runs_metrics: List of per-run DataFrames (one per successful
            run), as stored in
            :attr:`VariabilityStudyResults.all_runs_metrics`.
        final_val_series: ``pd.Series`` of final-epoch validation metric
            values across runs.
        final_test_series: Optional ``pd.Series`` of test-set metric values.
            If provided, a second boxplot is added.
        metric: Base metric name for labeling (default ``'accuracy'``).
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """

    _check_plotting()

    if not all_runs_metrics_list:
        settings.logger.warning("No run metrics provided to plot.")
        return None

    # 1. Resolve Columns using smart detection
    sample_run = all_runs_metrics_list[0]
    train_col, val_col = _find_metric_columns(sample_run, metric)

    if not train_col and not val_col:
        settings.logger.warning(
            f"Could not find columns for metric '{metric}'. Available: {list(sample_run.columns)}"
        )
        return None

    # Setup layout
    num_plots = 1 + int(show_histogram) + int(show_boxplot)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    # PANEL 1: Trajectories
    ax = axes[0]
    colors = settings.THEME

    # Clean metric name for display
    metric_display = metric.replace("val_", "").replace("train_", "").replace("_", " ").title()

    for df in all_runs_metrics_list:
        epochs = range(1, len(df) + 1)
        if train_col and train_col in df.columns:
            ax.plot(epochs, df[train_col], color=colors["train"], alpha=0.3)
        if val_col and val_col in df.columns:
            ax.plot(epochs, df[val_col], color=colors["val"], alpha=0.3)

    # Add Mean Lines
    if len(all_runs_metrics_list) > 1:
        if train_col:
            try:
                train_stack = np.array([run[train_col].values for run in all_runs_metrics_list])
                ax.plot(
                    range(1, train_stack.shape[1] + 1),
                    np.mean(train_stack, axis=0),
                    color=colors["train"],
                    linewidth=3,
                    label="Mean Train",
                )
            except ValueError:
                pass

        if val_col:
            try:
                val_stack = np.array([run[val_col].values for run in all_runs_metrics_list])
                ax.plot(
                    range(1, val_stack.shape[1] + 1),
                    np.mean(val_stack, axis=0),
                    color=colors["val"],
                    linewidth=3,
                    label="Mean Val",
                )
            except ValueError:
                pass

    ax.set_title(f"{metric_display} over {len(all_runs_metrics_list)} Runs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_display)
    ax.legend()
    ax.grid(True, alpha=0.3, color=colors.get("grid", "#e6e6e6"))

    plot_idx = 1

    # PANEL 2: Histogram
    if show_histogram and len(final_metrics_series) > 0:
        ax = axes[plot_idx]
        sn.histplot(final_metrics_series, kde=True, ax=ax, color=colors["val"], label="Validation")
        if final_test_series is not None:
            sn.histplot(final_test_series, kde=True, ax=ax, color=colors["test"], label="Test")
        ax.set_title(f"Final {metric_display} Distribution")
        ax.legend()
        plot_idx += 1

    # PANEL 3: Boxplot
    if show_boxplot and len(final_metrics_series) > 0:
        ax = axes[plot_idx]
        data = [final_metrics_series]
        labels = ["Val"]
        palette = [colors["val"]]

        if final_test_series is not None:
            data.append(final_test_series)
            labels.append("Test")
            palette.append(colors["test"])

        sn.boxplot(data=data, palette=palette, ax=ax)
        ax.set_xticklabels(labels)
        ax.set_title("Performance Spread")

    plt.tight_layout()
    return _finalize_plot(fig, show)


def plot_comparison_boxplots(
    comparison_results: Dict[str, Any], metric: str = "Accuracy", show: Optional[bool] = None
) -> Optional["Figure"]:
    """Side-by-side boxplots comparing metric distributions across models.

    Each model's runs are shown as a boxplot with individual data points
    overlaid. If pairwise statistical comparisons are present in the
    input, significance annotations are added to the title.

    Args:
        comparison_results: Either a dict returned by
            :func:`~ictonyx.api.compare_models` (must contain a
            ``'raw_data'`` key), or a plain dict mapping model names
            to lists/arrays of metric values.
        metric: Label for the y-axis (default ``'Accuracy'``).
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if the raw data
        cannot be extracted.
    """
    _check_plotting()

    if "raw_data" in comparison_results:
        data_dict = comparison_results["raw_data"]
    elif all(isinstance(v, (list, pd.Series, np.ndarray)) for v in comparison_results.values()):
        data_dict = comparison_results
    else:
        settings.logger.error("Could not find raw data for boxplots.")
        return None

    records = []
    for model_name, scores in data_dict.items():
        for score in scores:
            records.append({"Model": model_name, metric: score})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use themed palette if possible, else default
    sn.boxplot(data=df, x="Model", y=metric, hue="Model", palette="Blues", legend=False, ax=ax)
    sn.stripplot(data=df, x="Model", y=metric, color="black", alpha=0.3, ax=ax)

    # Statistical Annotations Title
    if "pairwise_comparisons" in comparison_results:
        pairs = comparison_results["pairwise_comparisons"]
        sig_pairs = [k for k, v in pairs.items() if v.is_significant()]
        if sig_pairs:
            subtitle = "Significant differences: " + ", ".join(sig_pairs)
            if len(subtitle) > 80:
                subtitle = subtitle[:77] + "..."
            ax.set_title(f"Model Comparison: {metric}\n{subtitle}", fontsize=10)
        else:
            ax.set_title(
                f"Model Comparison: {metric}\nNo significant differences found.", fontsize=10
            )

    return _finalize_plot(fig, show)


def plot_comparison_forest(
    comparison_results: Dict[str, Any],
    baseline_model: str,
    metric: str = "Accuracy",
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Forest plot of effect sizes relative to a baseline model.

    For each non-baseline model, shows the mean difference from the
    baseline with a 95% confidence interval, plotted as a horizontal
    error bar. A vertical dashed line at zero marks no difference.

    Args:
        comparison_results: Dict with a ``'raw_data'`` key mapping model
            names to arrays of metric values.
        baseline_model: Name of the model to use as the reference. Must
            be a key in ``raw_data``.
        metric: Label for the x-axis (default ``'Accuracy'``).
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if the baseline
        model is not found in the data.
    """
    _check_plotting()

    if "raw_data" in comparison_results:
        data_dict = comparison_results["raw_data"]
    elif all(isinstance(v, (list, pd.Series, np.ndarray)) for v in comparison_results.values()):
        data_dict = comparison_results
    else:
        settings.logger.error("Could not find raw data.")
        return None

    if baseline_model not in data_dict:
        settings.logger.error(f"Baseline model '{baseline_model}' not found in results.")
        return None

    baseline_scores = np.array(data_dict[baseline_model])
    baseline_mean = np.mean(baseline_scores)

    models = []
    diff_means = []
    cis = []
    colors = []

    for name, scores in data_dict.items():
        if name == baseline_model:
            continue

        scores = np.array(scores)
        diff = scores.mean() - baseline_mean

        # Welch's t-interval approximation
        se_diff = np.sqrt(
            np.var(scores, ddof=1) / len(scores)
            + np.var(baseline_scores, ddof=1) / len(baseline_scores)
        )
        ci = 1.96 * se_diff

        models.append(name)
        diff_means.append(diff)
        cis.append(ci)

        if diff - ci > 0:
            colors.append(settings.THEME["test"])  # Better
        elif diff + ci < 0:
            colors.append(settings.THEME["significant"])  # Worse
        else:
            colors.append("gray")  # Neutral

    fig, ax = plt.subplots(figsize=(8, len(models) * 0.8 + 2))

    y_pos = np.arange(len(models))
    for i, (dm, yp, ci, col) in enumerate(zip(diff_means, y_pos, cis, colors)):
        ax.errorbar(dm, yp, xerr=ci, fmt="o", color="black", ecolor=col, capsize=5)

    ax.axvline(0, color=settings.THEME["baseline"], linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel(f"Difference in {metric} (vs {baseline_model})")
    ax.set_title(f"Model Performance vs Baseline ({baseline_model})")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    return _finalize_plot(fig, show)


def plot_pairwise_comparison_matrix(
    comparison_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 10),
    show_effect_sizes: bool = True,
    annotate_significance: bool = True,
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Heatmap matrix of pairwise p-values, significance, and effect sizes.

    Creates 2–3 side-by-side heatmaps showing p-values, binary
    significance indicators, and (optionally) effect sizes for every
    pair of models.

    Args:
        comparison_results: Dict returned by
            :func:`~ictonyx.api.compare_models`, containing a
            ``'pairwise_comparisons'`` key.
        figsize: Figure dimensions. Default ``(12, 10)``.
        show_effect_sizes: If ``True``, include an effect-size heatmap.
            Default ``True``.
        annotate_significance: If ``True``, mark significant cells with
            asterisks. Default ``True``.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if no pairwise
        comparisons are present.
    """
    _check_plotting()
    _check_seaborn()

    if "pairwise_comparisons" not in comparison_results:
        settings.logger.warning("No pairwise comparisons to plot.")
        return None

    pairwise = comparison_results["pairwise_comparisons"]
    model_names = set()
    for comp_name in pairwise.keys():
        names = comp_name.split("_vs_")
        model_names.update(names)
    model_names = sorted(list(model_names))
    n_models = len(model_names)

    p_value_matrix = np.ones((n_models, n_models))
    significance_matrix = np.zeros((n_models, n_models))
    effect_size_matrix = np.zeros((n_models, n_models))

    for comp_name, result in pairwise.items():
        name1, name2 = comp_name.split("_vs_")
        i, j = model_names.index(name1), model_names.index(name2)
        p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value
        p_value_matrix[i, j] = p_val
        p_value_matrix[j, i] = p_val
        if result.is_significant():
            significance_matrix[i, j] = 1
            significance_matrix[j, i] = 1
        if result.effect_size is not None:
            effect_size_matrix[i, j] = result.effect_size
            effect_size_matrix[j, i] = result.effect_size

    n_plots = 2 if not show_effect_sizes or not np.any(effect_size_matrix) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    def format_annotation(data, i, j):
        val = data[i, j]
        if i == j:
            return ""
        text = f"{val:.3f}"
        if annotate_significance and significance_matrix[i, j] == 1:
            if val < 0.001:
                text += "\n***"
            elif val < 0.01:
                text += "\n**"
            elif val < 0.05:
                text += "\n*"
        return text

    annot_matrix = np.array(
        [
            [format_annotation(p_value_matrix, i, j) for j in range(n_models)]
            for i in range(n_models)
        ]
    )
    mask = np.eye(n_models, dtype=bool)

    sn.heatmap(
        p_value_matrix,
        xticklabels=model_names,
        yticklabels=model_names,
        annot=annot_matrix,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        ax=axes[0],
        mask=mask,
        cbar_kws={"label": "P-value"},
        linewidths=0.5,
        linecolor="gray",
    )
    axes[0].set_title(f"Corrected P-values", fontsize=12, fontweight="bold")

    sig_annot = np.array(
        [
            ["✓" if significance_matrix[i, j] and i != j else "" for j in range(n_models)]
            for i in range(n_models)
        ]
    )
    sn.heatmap(
        significance_matrix,
        xticklabels=model_names,
        yticklabels=model_names,
        annot=sig_annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[1],
        mask=mask,
        cbar_kws={"label": "Significant", "ticks": [0, 1]},
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Significant Differences (α = 0.05)", fontsize=12, fontweight="bold")

    if show_effect_sizes and np.any(effect_size_matrix):
        effect_annot = np.array(
            [
                [f"{effect_size_matrix[i, j]:.3f}" if i != j else "" for j in range(n_models)]
                for i in range(n_models)
            ]
        )
        sn.heatmap(
            np.abs(effect_size_matrix),
            xticklabels=model_names,
            yticklabels=model_names,
            annot=effect_annot,
            fmt="",
            cmap="YlOrRd",
            ax=axes[2],
            mask=mask,
            cbar_kws={"label": "Effect Size (absolute)"},
            linewidths=0.5,
            linecolor="gray",
        )
        axes[2].set_title("Effect Sizes", fontsize=12, fontweight="bold")

    plt.suptitle(f"Pairwise Comparison Matrix ({n_models} models)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return _finalize_plot(fig, show)


def plot_training_stability(
    stability_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot training stability diagnostics from a stability analysis.

    Creates a multi-panel figure showing convergence behavior, loss
    distributions, and coefficient of variation across runs.

    Args:
        stability_results: Dict produced by a stability analysis function,
            containing keys such as ``'n_runs'``, ``'common_length'``,
            ``'final_loss_mean'``, ``'final_loss_std'``, ``'final_loss_cv'``,
            ``'stability_assessment'``, ``'converged_runs'``, and
            ``'final_losses_list'``.
        figsize: Figure dimensions in inches. Default ``(12, 8)``.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_matplotlib()

    if "error" in stability_results:
        settings.logger.error(f"Cannot plot training stability: {stability_results['error']}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Final loss distribution
    ax1 = axes[0, 0]
    if "final_losses_list" in stability_results:
        final_losses = stability_results["final_losses_list"]
        mean_loss = stability_results["final_loss_mean"]
        std_loss = stability_results["final_loss_std"]
        n_runs = stability_results["n_runs"]
        ax1.hist(
            final_losses,
            bins=min(10, n_runs // 2 + 1),
            alpha=0.7,
            color="lightblue",
            edgecolor="black",
        )
        ax1.axvline(mean_loss, color="red", linestyle="--", label=f"Mean: {mean_loss:.4f}")
        ax1.axvline(mean_loss - std_loss, color="orange", linestyle=":", alpha=0.7)
        ax1.axvline(mean_loss + std_loss, color="orange", linestyle=":", alpha=0.7)
        ax1.legend()
    ax1.set_xlabel("Final Loss")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Final Loss Distribution")

    # 2. Stability metrics
    ax2 = axes[0, 1]
    ax2.axis("off")
    stability_text = [
        "Training Stability Metrics",
        "=" * 25,
        f"Runs analyzed: {stability_results['n_runs']}",
        f"Epochs per run: {stability_results['common_length']}",
        "",
        f"Final Loss:",
        f"  Mean: {stability_results['final_loss_mean']:.4f}",
        f"  Std:  {stability_results['final_loss_std']:.4f}",
        f"  CV:   {stability_results['final_loss_cv']:.4f}",
        "",
        f"Stability: {stability_results['stability_assessment'].upper()}",
        f"Convergence rate: {stability_results['convergence_rate']:.1%}",
    ]
    ax2.text(
        0.05,
        0.95,
        "\n".join(stability_text),
        transform=ax2.transAxes,
        fontfamily="monospace",
        fontsize=10,
        va="top",
    )

    # 3. Convergence status
    ax3 = axes[1, 0]
    converged = stability_results["converged_runs"]
    not_converged = stability_results["n_runs"] - converged
    ax3.pie(
        [converged, not_converged],
        labels=["Converged", "Not Converged"],
        colors=["green", "red"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax3.set_title("Convergence Analysis")

    # 4. Stability level
    ax4 = axes[1, 1]
    stability_levels = ["High", "Moderate", "Low"]
    current_stability = stability_results["stability_assessment"].title()
    colors = ["green" if level == current_stability else "lightgray" for level in stability_levels]
    bars = ax4.bar(stability_levels, [1, 1, 1], color=colors, alpha=0.7)
    if current_stability in stability_levels:
        bars[stability_levels.index(current_stability)].set_height(1.2)
        bars[stability_levels.index(current_stability)].set_alpha(1.0)
    ax4.set_ylabel("Stability Level")
    ax4.set_title(f"Overall Assessment: {current_stability}")
    ax4.set_ylim(0, 1.5)

    plt.tight_layout()

    return _finalize_plot(fig, show)


def plot_autocorr_vs_lag(
    data: Union[pd.Series, List[float]],
    max_lag: int = 20,
    title: str = "Autocorrelation of Loss",
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot autocorrelation of a metric series as a function of lag.

    Useful for diagnosing sequential dependence between runs — if
    autocorrelation is high at lag 1, consecutive runs may not be
    independent (e.g. due to incomplete GPU memory cleanup).

    Args:
        data: A ``pd.Series`` or list of metric values ordered by run.
        max_lag: Maximum lag to compute. Default 20.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_matplotlib()
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    if len(data) <= max_lag:
        return None

    autocorr_values = [data.autocorr(lag) for lag in range(1, max_lag + 1)]
    lags = range(1, max_lag + 1)

    fig = plt.figure(figsize=(10, 6))
    plt.stem(lags, autocorr_values)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.grid(True)

    return _finalize_plot(fig, show)


def plot_averaged_autocorr(
    lags: List[float],
    mean_autocorr: List[float],
    std_autocorr: List[float],
    title: str = "Averaged Autocorrelation of Loss",
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot averaged autocorrelation with error bands across multiple studies.

    Shows the mean autocorrelation at each lag with ±1 standard deviation
    shaded, summarizing sequential dependence across repeated studies.

    Args:
        lags: List of integer lag values.
        mean_autocorr: Mean autocorrelation at each lag.
        std_autocorr: Standard deviation of autocorrelation at each lag.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_matplotlib()
    fig = plt.figure(figsize=(10, 6))
    plt.plot(lags, mean_autocorr, "b-", label="Mean Autocorrelation")
    plt.fill_between(
        lags,
        np.array(mean_autocorr) - np.array(std_autocorr),
        np.array(mean_autocorr) + np.array(std_autocorr),
        color="b",
        alpha=0.2,
        label="Standard Deviation",
    )
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.legend()
    plt.grid(True)

    return _finalize_plot(fig, show)


def plot_pacf_vs_lag(
    data: Union[pd.Series, List[float]],
    max_lag: int = 20,
    title: str = "Partial Autocorrelation of Loss",
    alpha: float = 0.05,
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot partial autocorrelation function (PACF) with confidence bands.

    Requires ``statsmodels``. PACF isolates the direct correlation at
    each lag, removing the influence of intermediate lags.

    Args:
        data: A ``pd.Series`` or list of metric values ordered by run.
        max_lag: Maximum lag to compute. Default 20.
        title: Plot title. Default ``'Partial Autocorrelation of Loss'``.
        alpha: Confidence level for the bands. Default 0.05 (95% CI).
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if ``statsmodels``
        is not installed or the series is too short.
    """
    _check_matplotlib()
    try:
        from statsmodels.tsa.stattools import pacf
    except ImportError:
        settings.logger.warning("statsmodels required for PACF.")
        return None

    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    if len(data) <= max_lag + 1:
        return None

    pacf_values, conf_int = pacf(data, nlags=max_lag, alpha=alpha)
    pacf_values = pacf_values[1:]
    conf_int = conf_int[1:]
    lags = range(1, max_lag + 1)

    fig = plt.figure(figsize=(10, 6))
    plt.stem(lags, pacf_values, use_line_collection=True, label="PACF")
    plt.fill_between(
        lags, conf_int[:, 0] - pacf_values, conf_int[:, 1] - pacf_values, alpha=0.2, color="gray"
    )
    plt.title(title)
    plt.grid(True)

    return _finalize_plot(fig, show)


def plot_averaged_pacf(
    lags: List[float],
    mean_pacf: List[float],
    std_pacf: List[float],
    title: str = "Averaged Partial Autocorrelation of Loss",
    conf_level: float = 0.95,
    show: Optional[bool] = None,
) -> Optional["Figure"]:
    """Plot averaged partial autocorrelation with error bands across runs.

    Shows the mean PACF at each lag with ±1 standard deviation shaded.

    Args:
        lags: List of integer lag values.
        mean_pacf: Mean PACF at each lag.
        std_pacf: Standard deviation of PACF at each lag.
        title: Plot title. Default ``'Averaged Partial Autocorrelation of Loss'``.
        conf_level: Confidence level for reference lines. Default 0.95.
        show: Display behavior. See :func:`plot_confusion_matrix`.

    Returns:
        The ``matplotlib.figure.Figure``, or ``None`` if display is enabled.
    """
    _check_matplotlib()
    fig = plt.figure(figsize=(10, 6))
    plt.plot(lags, mean_pacf, "b-", label="Mean PACF", linewidth=2)
    plt.fill_between(
        lags,
        np.array(mean_pacf) - np.array(std_pacf),
        np.array(mean_pacf) + np.array(std_pacf),
        color="b",
        alpha=0.2,
        label="±1 Standard Deviation",
    )

    n_points = len(lags) * 10
    conf_bound = 1.96 / np.sqrt(n_points)
    plt.axhline(y=conf_bound, color="gray", linestyle=":", alpha=0.7)
    plt.axhline(y=-conf_bound, color="gray", linestyle=":", alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.grid(True)

    return _finalize_plot(fig, show)
