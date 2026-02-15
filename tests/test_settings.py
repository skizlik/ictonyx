import logging

import pytest

from ictonyx.settings import (
    THEME,
    logger,
    set_display_plots,
    set_theme,
    set_verbose,
    should_display,
)


class TestSetVerbose:
    """Test set_verbose function."""

    def teardown_method(self):
        """Reset to default after each test."""
        set_verbose(True)

    def test_verbose_true(self):
        set_verbose(True)
        assert logger.level == logging.INFO

    def test_verbose_false(self):
        set_verbose(False)
        assert logger.level == logging.WARNING

    def test_toggle(self):
        set_verbose(False)
        assert logger.level == logging.WARNING
        set_verbose(True)
        assert logger.level == logging.INFO


class TestSetDisplayPlots:
    """Test set_display_plots and should_display."""

    def teardown_method(self):
        set_display_plots(True)

    def test_display_true(self):
        set_display_plots(True)
        assert should_display() is True

    def test_display_false(self):
        set_display_plots(False)
        assert should_display() is False

    def test_toggle(self):
        set_display_plots(False)
        assert should_display() is False
        set_display_plots(True)
        assert should_display() is True


class TestSetTheme:
    """Test set_theme function."""

    def teardown_method(self):
        """Reset to default theme."""
        THEME.update(
            {
                "train": "#1f77b4",
                "val": "#ff7f0e",
                "test": "#2ca02c",
                "baseline": "gray",
                "significant": "crimson",
                "grid": "#e6e6e6",
            }
        )

    def test_dark_theme(self):
        set_theme("dark")
        assert THEME["train"] == "#4db8ff"
        assert THEME["val"] == "#ff9933"
        assert THEME["grid"] == "#444444"

    def test_publication_theme(self):
        set_theme("publication")
        assert THEME["train"] == "black"
        assert THEME["val"] == "gray"
        assert THEME["grid"] == "#eeeeee"

    def test_unknown_theme_no_change(self):
        """Unknown theme name should leave THEME unchanged."""
        original_train = THEME["train"]
        set_theme("nonexistent_theme")
        assert THEME["train"] == original_train

    def test_theme_has_all_keys(self):
        """Verify all expected keys exist."""
        expected = {"train", "val", "test", "baseline", "significant", "grid"}
        assert expected == set(THEME.keys())
