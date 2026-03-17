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
        """Reset to default theme after each test using the canonical API."""
        set_theme("default")

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

    def test_default_theme_resets_to_originals(self):
        """set_theme('default') must restore the original palette exactly."""
        set_theme("dark")  # mutate away from defaults
        set_theme("default")
        assert THEME["train"] == "#1f77b4"
        assert THEME["val"] == "#ff7f0e"
        assert THEME["test"] == "#2ca02c"
        assert THEME["baseline"] == "gray"
        assert THEME["significant"] == "crimson"
        assert THEME["grid"] == "#e6e6e6"

    def test_unknown_theme_raises_value_error(self):
        """An unrecognised theme name must raise ValueError."""
        with pytest.raises(ValueError, match="nonexistent_theme"):
            set_theme("nonexistent_theme")

    def test_unknown_theme_does_not_mutate_theme(self):
        """THEME must be unchanged when ValueError is raised."""
        original = dict(THEME)  # snapshot before
        with pytest.raises(ValueError):
            set_theme("nonexistent_theme")
        assert THEME == original  # no side effects

    def test_error_message_lists_valid_options(self):
        """ValueError message must name all valid themes so users know what to pass."""
        with pytest.raises(ValueError) as exc_info:
            set_theme("typo")
        msg = str(exc_info.value)
        assert "default" in msg
        assert "dark" in msg
        assert "publication" in msg

    def test_theme_has_all_keys(self):
        """All six expected colour keys must be present after every theme switch."""
        expected = {"train", "val", "test", "baseline", "significant", "grid"}
        for theme in ("default", "dark", "publication"):
            set_theme(theme)
            assert expected == set(THEME.keys()), f"Missing keys after set_theme('{theme}')"
