# ictonyx/settings.py
import logging
import sys
from typing import Dict, Optional, Union

# Global Configuration
_DISPLAY_PLOTS: bool = True
_VERBOSE: bool = True

# THEME CONFIGURATION
THEME: Dict[str, str] = {
    "train": "#1f77b4",  # Standard Blue
    "val": "#ff7f0e",  # Standard Orange
    "test": "#2ca02c",  # Standard Green
    "baseline": "gray",  # For forest plots
    "significant": "crimson",  # For highlighting significance
    "grid": "#e6e6e6",  # Light gray for grids
}

# Setup Library Logger
logger = logging.getLogger("ictonyx")
logger.setLevel(logging.INFO)

# Default handler (StreamHandler to stdout)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))  # Simple format like print()
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


def set_theme(theme_name: str):
    """Quickly switch between color presets."""
    if theme_name == "dark":
        THEME.update(
            {
                "train": "#4db8ff",
                "val": "#ff9933",
                "test": "#66ff66",
                "baseline": "#888888",
                "significant": "#ff4d4d",
                "grid": "#444444",
            }
        )
    elif theme_name == "publication":
        # Grayscale/High Contrast
        THEME.update(
            {
                "train": "black",
                "val": "gray",
                "test": "black",
                "baseline": "black",
                "significant": "black",
                "grid": "#eeeeee",
            }
        )


def should_display() -> bool:
    """Internal check for plotting functions."""
    return _DISPLAY_PLOTS
