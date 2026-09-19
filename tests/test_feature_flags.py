"""v0.4.10 C5: feature flags reflect the packages they depend on (v12 1.23)."""

from importlib.util import find_spec

import pytest

import ictonyx as ix


@pytest.mark.parametrize(
    "flag,module",
    [
        ("process_isolation", "cloudpickle"),
        ("hyperparameter_tuning", "optuna"),
        ("explainability", "shap"),
        ("mlflow_logger", "mlflow"),
    ],
)
def test_feature_flags_track_real_packages(flag, module):
    assert ix.get_feature_availability()[flag] == (find_spec(module) is not None)


def test_feature_flag_keys_unchanged():
    expected = {
        "tensorflow_support",
        "sklearn_support",
        "pytorch_support",
        "huggingface_support",
        "statistical_functions",
        "bootstrap_ci",
        "plotting_functions",
        "mlflow_logger",
        "hyperparameter_tuning",
        "explainability",
        "data_handlers",
        "memory_management",
        "process_isolation",
    }
    assert set(ix.get_feature_availability()) == expected
