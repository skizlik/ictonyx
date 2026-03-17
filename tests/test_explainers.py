"""Tests for SHAP explainability functions."""

import numpy as np
import pytest

try:
    from ictonyx.explainers import (
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


def test_get_shap_feature_importance_returns_dict(trained_tree_model):
    wrapper, X, feature_names = trained_tree_model
    importance = get_shap_feature_importance(wrapper.model, X, feature_names)
    assert isinstance(importance, dict)
    assert set(importance.keys()) == set(feature_names)
    assert all(v >= 0 for v in importance.values())


def test_get_shap_feature_importance_all_finite(trained_tree_model):
    wrapper, X, feature_names = trained_tree_model
    importance = get_shap_feature_importance(wrapper.model, X, feature_names)
    assert all(np.isfinite(v) for v in importance.values())


def test_plot_shap_summary_returns_figure(trained_tree_model):
    import matplotlib

    matplotlib.use("Agg")
    wrapper, X, feature_names = trained_tree_model
    fig = plot_shap_summary(wrapper.model, X, feature_names)
    assert fig is not None


def test_plot_shap_waterfall_returns_figure(trained_tree_model):
    import matplotlib

    matplotlib.use("Agg")
    wrapper, X, feature_names = trained_tree_model
    fig = plot_shap_waterfall(wrapper.model, X, feature_names, sample_idx=0)
    assert fig is not None
