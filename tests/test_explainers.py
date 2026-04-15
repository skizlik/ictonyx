"""Tests for SHAP explainability functions."""

import unittest.mock as mock
import warnings

import numpy as np
import pytest

try:
    from ictonyx.explainers import (
        _check_shap,
        _warn_if_deep_explainer_deprecated,
        get_shap_feature_importance,
        plot_shap_dependence,
        plot_shap_summary,
        plot_shap_waterfall,
    )

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

pytest.importorskip("shap", reason="shap not installed")
pytestmark = pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")


@pytest.fixture
def trained_tree_model():
    from sklearn.ensemble import GradientBoostingClassifier

    from ictonyx.core import ScikitLearnModelWrapper

    X = np.random.rand(80, 4)
    y = np.random.randint(0, 2, 80)
    wrapper = ScikitLearnModelWrapper(GradientBoostingClassifier(n_estimators=10))
    wrapper.fit((X, y))
    return wrapper, X, ["feat_a", "feat_b", "feat_c", "feat_d"]


@pytest.fixture
def trained_linear_model():
    """A Ridge regressor — takes the KernelExplainer code path."""
    from sklearn.linear_model import Ridge

    from ictonyx.core import ScikitLearnModelWrapper

    X = np.random.rand(60, 4)
    y = X[:, 0] * 2 + X[:, 1] - 1 + np.random.rand(60) * 0.1
    wrapper = ScikitLearnModelWrapper(Ridge())
    wrapper.fit((X, y))
    return wrapper, X, ["a", "b", "c", "d"]


def test_get_shap_feature_importance_returns_array(trained_tree_model):
    wrapper, X, feature_names = trained_tree_model
    importance = get_shap_feature_importance(wrapper, X, feature_names)
    assert isinstance(importance, np.ndarray)
    assert len(importance) == len(feature_names)
    assert all(v >= 0 for v in importance)


def test_get_shap_feature_importance_all_finite(trained_tree_model):
    wrapper, X, feature_names = trained_tree_model
    importance = get_shap_feature_importance(wrapper, X, feature_names)
    assert all(np.isfinite(v) for v in importance)


def test_plot_shap_summary_runs_without_error(trained_tree_model):
    import matplotlib

    matplotlib.use("Agg")
    wrapper, X, feature_names = trained_tree_model
    plot_shap_summary(wrapper, X, feature_names)


def test_plot_shap_waterfall_runs_without_error(trained_tree_model):
    import matplotlib

    matplotlib.use("Agg")
    wrapper, X, feature_names = trained_tree_model
    plot_shap_waterfall(wrapper, X, 0, feature_names)


class TestSHAPAvailabilityGuard:
    def test_check_shap_raises_when_unavailable(self, monkeypatch):
        import ictonyx.explainers as exp_mod

        monkeypatch.setattr(exp_mod, "HAS_SHAP", False)
        with pytest.raises(ImportError, match="SHAP is required"):
            _check_shap()

    def test_get_importance_raises_without_shap(self, monkeypatch):
        import ictonyx.explainers as exp_mod

        monkeypatch.setattr(exp_mod, "HAS_SHAP", False)
        X = np.random.rand(20, 3)
        with pytest.raises(ImportError):
            get_shap_feature_importance(object(), X, ["a", "b", "c"])


class TestNonTreeExplainerPath:
    def test_kernel_explainer_used_for_linear_model(self, trained_linear_model):
        wrapper, X, names = trained_linear_model
        importance = get_shap_feature_importance(wrapper, X, names)
        assert isinstance(importance, np.ndarray)
        assert len(importance) == len(names)

    def test_feature_importance_values_non_negative(self, trained_linear_model):
        wrapper, X, names = trained_linear_model
        importance = get_shap_feature_importance(wrapper, X, names)
        assert all(v >= 0 for v in importance)

    def test_plot_shap_summary_linear(self, trained_linear_model):
        import matplotlib

        matplotlib.use("Agg")
        wrapper, X, names = trained_linear_model
        plot_shap_summary(wrapper, X, names)


class TestDeepExplainerDeprecation:
    def test_warning_emitted_for_new_shap(self):
        import ictonyx.explainers as exp_mod

        mock_shap = mock.MagicMock()
        mock_shap.__version__ = "0.45.0"
        with mock.patch.object(exp_mod, "shap", mock_shap):
            with pytest.warns(DeprecationWarning, match="DeepExplainer is deprecated"):
                exp_mod._warn_if_deep_explainer_deprecated()

    def test_no_warning_for_old_shap(self):
        import ictonyx.explainers as exp_mod

        mock_shap = mock.MagicMock()
        mock_shap.__version__ = "0.44.1"
        with mock.patch.object(exp_mod, "shap", mock_shap):
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                exp_mod._warn_if_deep_explainer_deprecated()


class TestSHAPEdgeCases:
    def test_waterfall_out_of_bounds_raises(self, trained_tree_model):
        wrapper, X, names = trained_tree_model
        with pytest.raises(ValueError, match="out of range"):
            plot_shap_waterfall(wrapper, X, len(X) + 100, names)
