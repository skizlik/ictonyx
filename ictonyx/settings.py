# ictonyx/settings.py
import logging
import sys
from typing import Dict, Optional, Tuple, Union

# Global Configuration
_DISPLAY_PLOTS: bool = True
_VERBOSE: bool = True

# THEME CONFIGURATION
THEME: Dict[str, str] = {
    "train": "#1f77b4",  # Blue — training metrics
    "val": "#ff7f0e",  # Orange — validation metrics
    "test": "#2ca02c",  # Green — test-set metrics
    "baseline": "gray",  # Forest plot zero-reference line
    "significant": "crimson",  # Significant result highlight
    "grid": "#e6e6e6",  # Light gray gridlines
    "neutral": "#888888",  # Neutral / indeterminate
    "better": "#2ca02c",  # Positive direction (same as test)
    "worse": "crimson",  # Negative direction
    "point": "#333333",  # Scatter / strip plot individual points
    "positive": "#2ca02c",  # Alias for better
    "negative": "crimson",  # Alias for worse
    "sequential": "Blues",  # Seaborn colormap for heatmaps
    "diverging": "RdBu_r",  # Seaborn diverging colormap
}

# Setup Library Logger
logger = logging.getLogger("ictonyx")
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't bubble up to root logger

# Only attach handler if none exists — lets applications configure logging themselves.
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


def set_verbose(verbose: bool):
    """
    Global override for verbosity.

    Args:
        verbose: If True, log level is INFO. If False, WARNING.
    """
    global _VERBOSE
    _VERBOSE = verbose
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)


def set_display_plots(display: bool):
    """
    Global override for plot display.

    Args:
        display: If True, plotting functions will call plt.show().
                 If False, they will return the Figure without blocking.
    """
    global _DISPLAY_PLOTS
    _DISPLAY_PLOTS = display


# Original colours kept here so "default" can restore them.
_DEFAULT_THEME: Dict[str, str] = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
    "baseline": "gray",
    "significant": "crimson",
    "grid": "#e6e6e6",
    "neutral": "#888888",
    "better": "#2ca02c",
    "worse": "crimson",
    "point": "#333333",
    "positive": "#2ca02c",
    "negative": "crimson",
    "sequential": "Blues",
    "diverging": "RdBu_r",
}
_VALID_THEMES = ("default", "dark", "publication")


def set_theme(theme_name: str):
    """Switch the global colour theme used by all plotting functions.

    Args:
        theme_name: One of ``'default'``, ``'dark'``, or ``'publication'``.

    Raises:
        ValueError: If *theme_name* is not one of the valid options.
    """
    if theme_name == "default":
        THEME.update(_DEFAULT_THEME)
    elif theme_name == "dark":
        THEME.update(
            {
                "train": "#4db8ff",
                "val": "#ff9933",
                "test": "#66ff66",
                "baseline": "#888888",
                "significant": "#ff4d4d",
                "grid": "#444444",
                "neutral": "#888888",
                "better": "#66ff66",
                "worse": "#ff4d4d",
                "point": "#dddddd",
                "positive": "#66ff66",
                "negative": "#ff4d4d",
                "sequential": "Blues",
                "diverging": "RdBu_r",
            }
        )
    elif theme_name == "publication":
        # Greyscale / high-contrast for print
        THEME.update(
            {
                "train": "black",
                "val": "gray",
                "test": "black",
                "baseline": "black",
                "significant": "black",
                "grid": "#eeeeee",
                "neutral": "#888888",
                "better": "black",
                "worse": "black",
                "point": "black",
                "positive": "black",
                "negative": "black",
                "sequential": "Greys",
                "diverging": "RdGy",
            }
        )
    else:
        raise ValueError(
            f"Unknown theme '{theme_name}'. "
            f"Valid options are: {', '.join(repr(t) for t in _VALID_THEMES)}"
        )


# Figure size scaling — insert before should_display()
_FIGSIZE_SCALE: float = 1.0


def set_figsize_scale(factor: float) -> None:
    """Scale all figure sizes by a multiplier.

    Useful for high-DPI displays or publication figures requiring a
    specific physical size. Default scale is 1.0 (no scaling).

    Args:
        factor: Positive multiplier applied to all base figure sizes.

    Raises:
        ValueError: If *factor* is not positive.
    """
    global _FIGSIZE_SCALE
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    _FIGSIZE_SCALE = float(factor)


def get_figsize(base: Tuple[float, float] = (10.0, 6.0)) -> Tuple[float, float]:
    """Return a base figure size scaled by the current global scale factor.

    Args:
        base: ``(width, height)`` in inches before scaling. Default ``(10, 6)``.

    Returns:
        Scaled ``(width, height)`` tuple.
    """
    return (base[0] * _FIGSIZE_SCALE, base[1] * _FIGSIZE_SCALE)


def should_display() -> bool:
    """Internal check for plotting functions."""
    return _DISPLAY_PLOTS


def should_verbose() -> bool:
    """Return the current global verbosity setting.

    Use this instead of accessing ``_VERBOSE`` directly.
    """
    return _VERBOSE
