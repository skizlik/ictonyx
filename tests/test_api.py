# tests/test_api.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ictonyx import ModelConfig, api
from ictonyx.analysis import ModelComparisonResults, StatisticalTestResult
from ictonyx.api import _ensure_wrapper, _get_model_name
from ictonyx.core import TENSORFLOW_AVAILABLE, BaseModelWrapper

# --- Fixtures (Reusable Data) ---


@pytest.fixture
def sample_df():
    """Creates a minimal valid DataFrame."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_arrays():
    """Creates minimal valid (X, y) arrays."""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    return X, y


@pytest.fixture
def dummy_model_func():
    """A simple function that acts as a model builder."""

    def builder(config=None):
        mock = MagicMock(spec=BaseModelWrapper)
        mock.fit.return_value = None
        mock.predict.return_value = np.array([0, 1])
        return mock

    return builder


@pytest.fixture
def dummy_model_func_2():
    """A second dummy model builder with a different name."""

    def builder_2(config=None):
        mock = MagicMock(spec=BaseModelWrapper)
        mock.fit.return_value = None
        mock.predict.return_value = np.array([0, 1])
        return mock

    return builder_2


@pytest.fixture
def mock_runner_results():
    """Creates a fake VariabilityStudyResults object."""
    mock_results = MagicMock()
    mock_results.get_final_metrics.return_value = {"run_1": 0.95, "run_2": 0.96}
    return mock_results


# --- Tests for variability_study ---


@patch("ictonyx.api._run_study")  # FIX: Updated name from _run_study_internal
def test_variability_study_dataframe(mock_run, sample_df, dummy_model_func, mock_runner_results):
    """Test standard usage with DataFrame."""
    # Arrange
    mock_run.return_value = mock_runner_results

    # Act
    results = api.variability_study(
        model=dummy_model_func,
        data=sample_df,
        target_column="target",
        runs=3,
        epochs=5,
        learning_rate=0.01,  # Extra kwarg for config
    )

    # Assert
    assert results == mock_runner_results

    # Verify _run_study was called
    mock_run.assert_called_once()

    # Verify Config construction
    call_args = mock_run.call_args
    config_arg = call_args.kwargs["model_config"]

    assert isinstance(config_arg, ModelConfig)
    assert config_arg["epochs"] == 5
    assert config_arg["learning_rate"] == 0.01  # Verify kwargs passthrough


@patch("ictonyx.api._run_study")  # FIX: Updated name
def test_variability_study_arrays(mock_run, sample_arrays, dummy_model_func, mock_runner_results):
    """Test usage with numpy tuples."""
    mock_run.return_value = mock_runner_results

    api.variability_study(model=dummy_model_func, data=sample_arrays, runs=2)

    # Verify DataHandler was resolved correctly
    passed_handler = mock_run.call_args.kwargs["data_handler"]
    assert passed_handler.data_type == "arrays"


def test_variability_study_invalid_data():
    """Test that invalid data raises appropriate errors."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        api.variability_study(lambda x: x, data=12345)


def test_variability_study_missing_target(sample_df):
    """Test missing target column error for DataFrames."""
    with pytest.raises(ValueError, match="target_column is required"):
        api.variability_study(lambda x: x, data=sample_df)


def test_variability_study_verbose_false_suppresses_output(capsys, sample_df):
    """verbose=False must produce no stdout output."""
    from sklearn.ensemble import RandomForestClassifier

    with pytest.warns(UserWarning, match="runs=3"):
        api.variability_study(
            model=RandomForestClassifier,
            data=sample_df,
            target_column="target",
            runs=3,
            verbose=False,
            seed=42,
        )
    captured = capsys.readouterr()
    assert captured.out == ""


# --- Tests for compare_models ---


@patch("ictonyx.api._stat_compare")
@patch("ictonyx.api.variability_study")
def test_compare_models_flow(
    mock_var_study,
    mock_stat_compare,
    sample_df,
    dummy_model_func,
    dummy_model_func_2,
    mock_runner_results,
):
    """Test that compare_models correctly orchestrates multiple studies."""
    # Arrange
    mock_var_study.return_value = mock_runner_results
    mock_stat_compare.return_value = ModelComparisonResults(
        overall_test=MagicMock(spec=StatisticalTestResult),
        raw_data={},
        pairwise_comparisons={},
        significant_comparisons=[],
        correction_method="holm",
        n_models=2,
        metric=None,
    )

    models = [dummy_model_func, dummy_model_func_2]

    # Act
    result = api.compare_models(
        models=models, data=sample_df, target_column="target", runs=5, metric="val_accuracy"
    )

    # Assert
    # Should run study once for each model
    assert mock_var_study.call_count == 2

    # Should call stats comparison once
    mock_stat_compare.assert_called_once()

    # Check that it extracted the metrics correctly
    stats_call_args = mock_stat_compare.call_args[0][0]
    assert len(stats_call_args) == 2
    assert isinstance(stats_call_args[list(stats_call_args.keys())[0]], pd.Series)


@patch("ictonyx.api.variability_study")
def test_compare_models_insufficient_data(mock_var_study, sample_df):
    """Test error when models fail to return metrics."""
    # Arrange: Study returns empty metrics
    empty_results = MagicMock()
    empty_results.get_final_metrics.return_value = {}
    empty_results.get_metric_values.return_value = []
    mock_var_study.return_value = empty_results

    # Act / Assert — compare_models now raises ValueError instead of returning error dict
    with pytest.raises(ValueError, match="Insufficient valid results for comparison"):
        api.compare_models(
            models=[lambda x: x, lambda x: x], data=sample_df, target_column="target"
        )


# --- Tests for Model Wrapping Helpers ---


class DummyClassifier:
    """A mock class that looks like an sklearn estimator."""

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def test_get_model_builder_class():
    """Test wrapping a raw class."""
    # FIX: Updated function name to _get_model_builder
    wrapper = api._get_model_builder(DummyClassifier)
    model_instance = wrapper(ModelConfig({}))
    assert isinstance(model_instance, BaseModelWrapper)


def test_get_model_builder_instance_clones():
    """Test that passing an sklearn instance clones it per run."""
    instance = DummyClassifier()
    builder = api._get_model_builder(instance)

    model_a = builder(ModelConfig({}))
    model_b = builder(ModelConfig({}))

    # Each call should produce a distinct wrapper
    assert model_a is not model_b
    # And distinct underlying models (not the same object)
    assert model_a.model is not model_b.model


def test_get_model_builder_instance_preserves_params():
    """Test that cloned instances keep the original hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier

    instance = RandomForestClassifier(n_estimators=42, max_depth=7)
    builder = api._get_model_builder(instance)

    wrapped = builder(ModelConfig({}))
    assert wrapped.model.n_estimators == 42
    assert wrapped.model.max_depth == 7


def test_get_model_builder_invalid():
    """Test rejection of invalid model inputs."""
    with pytest.raises(ValueError, match="Invalid model input"):
        # FIX: Updated function name
        api._get_model_builder("not a model")


class TestEnsureWrapper:
    """_ensure_wrapper passthrough, auto-wrap, and rejection."""

    def test_passthrough_already_wrapped(self):
        from sklearn.ensemble import RandomForestClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(RandomForestClassifier())
        assert _ensure_wrapper(wrapper) is wrapper

    def test_auto_wraps_sklearn(self):
        from sklearn.ensemble import RandomForestClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        result = _ensure_wrapper(RandomForestClassifier())
        assert isinstance(result, ScikitLearnModelWrapper)

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="Cannot wrap"):
            _ensure_wrapper("not_a_model")

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="Cannot wrap"):
            _ensure_wrapper(42)

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_keras_model_not_wrapped_as_sklearn(self):
        """A Keras model must be wrapped as KerasModelWrapper, not ScikitLearnModelWrapper.

        This test catches the ordering bug where the duck-typing check
        (hasattr fit/predict) fires before the Keras string check, causing
        Keras models to be silently mis-wrapped.
        """
        import tensorflow as tf

        from ictonyx.core import KerasModelWrapper, ScikitLearnModelWrapper

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4, input_shape=(4,), activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        result = _ensure_wrapper(model)

        assert isinstance(result, KerasModelWrapper), (
            f"Expected KerasModelWrapper but got {type(result).__name__}. "
            "The duck-typing check is firing before the Keras check."
        )
        assert not isinstance(result, ScikitLearnModelWrapper)


class TestGetModelName:
    """_get_model_name from class, function, and instance."""

    def test_from_class(self):
        from sklearn.tree import DecisionTreeClassifier

        assert _get_model_name(DecisionTreeClassifier) == "DecisionTreeClassifier"

    def test_from_function(self):
        def my_builder(config):
            pass

        assert _get_model_name(my_builder) == "my_builder"

    def test_from_instance(self):
        from sklearn.tree import DecisionTreeClassifier

        assert _get_model_name(DecisionTreeClassifier()) == "DecisionTreeClassifier"


class TestCompareModelsConfigurationError:
    """compare_models() must raise ConfigurationError with an actionable message
    when the requested metric is absent from a model's results."""

    def test_val_accuracy_missing_raises_configuration_error(self):
        """When sklearn produces no val_accuracy (no validation data),
        ConfigurationError must be raised with a hint about val_split."""
        import warnings

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        import ictonyx as ix
        from ictonyx.data import ArraysDataHandler
        from ictonyx.exceptions import ConfigurationError

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)
        # val_split=0.0 means no validation data → sklearn produces only 'accuracy',
        # not 'val_accuracy'. compare_models(metric='val_accuracy') must raise.
        handler = ArraysDataHandler(X, y, val_split=0.0, test_split=0.0)

        with pytest.raises(ConfigurationError) as exc_info:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ix.compare_models(
                    models=[RandomForestClassifier, GradientBoostingClassifier],
                    data=handler,
                    runs=3,
                    metric="val_accuracy",
                    seed=42,
                )
        msg = str(exc_info.value)
        assert "val_accuracy" in msg
        assert "accuracy" in msg or "val_split" in msg


class TestGetModelBuilderEdgeCases:
    """Cover _build_instance_cloner branches."""

    def test_invalid_model_object_raises(self):
        """An object with no fit method must raise ValueError."""
        from ictonyx.api import _get_model_builder

        with pytest.raises((ValueError, TypeError)):
            _get_model_builder(42)

    def test_callable_not_class_returned_directly(self):
        """A plain callable (not a class) is wrapped and returned."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.api import _get_model_builder
        from ictonyx.core import ScikitLearnModelWrapper

        def my_builder(config):
            return ScikitLearnModelWrapper(DecisionTreeClassifier())

        builder = _get_model_builder(my_builder)
        assert callable(builder)


class TestVariabilityStudy:
    """VariabilityStudy tests."""

    def test_variability_study_warns_when_runs_lt_20(small_df):
        """runs=15 must trigger the low-power warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with (
                patch("ictonyx.api._run_study"),
                patch("ictonyx.api.auto_resolve_handler"),
                patch("ictonyx.api._get_model_builder"),
            ):
                try:
                    api.variability_study(
                        model=MagicMock,
                        data=small_df,
                        target_column="target",
                        runs=15,
                        seed=42,
                    )
                except Exception:
                    pass
        low_power = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "insufficient" in str(x.message).lower()
        ]
        assert len(low_power) > 0, "runs=15 should trigger the low-power warning"

    def test_variability_study_no_warning_at_runs_20(small_df):
        """runs=20 must NOT trigger the low-power warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with (
                patch("ictonyx.api._run_study"),
                patch("ictonyx.api.auto_resolve_handler"),
                patch("ictonyx.api._get_model_builder"),
            ):
                try:
                    api.variability_study(
                        model=MagicMock,
                        data=small_df,
                        target_column="target",
                        runs=20,
                        seed=42,
                    )
                except Exception:
                    pass
        low_power = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "insufficient" in str(x.message).lower()
        ]
        assert len(low_power) == 0, "runs=20 must not trigger the low-power warning"


class TestBuilderInstanceClonerSafety:
    def test_build_instance_cloner_raises_for_unknown_instance(self):
        """An unknown instance type must raise ValueError, not silently reuse state."""
        from ictonyx.api import _get_model_builder

        class UnknownModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return X

        instance = UnknownModel()
        with pytest.raises(ValueError, match="cannot be cloned"):
            _get_model_builder(instance)

    def test_build_instance_cloner_sklearn_clones_correctly(self):
        """sklearn instances must be cloned, not reused."""
        from sklearn.tree import DecisionTreeClassifier

        from ictonyx.api import _get_model_builder

        instance = DecisionTreeClassifier(max_depth=3)
        builder = _get_model_builder(instance)

        # Build twice — must produce different objects
        from ictonyx.config import ModelConfig

        config = ModelConfig({})
        wrapper_1 = builder(config)
        wrapper_2 = builder(config)
        assert wrapper_1 is not wrapper_2, (
            "Each builder call must produce a new wrapper instance. "
            "If the same instance is returned, weights would leak between runs."
        )


class TestGetFeatureAvailability:
    def test_returns_dict_with_expected_keys(self):
        import ictonyx as ix

        result = ix.get_feature_availability()
        expected_keys = [
            "tensorflow_support",
            "sklearn_support",
            "statistical_functions",
            "bootstrap_ci",
            "plotting_functions",
            "mlflow_logger",
            "hyperparameter_tuning",
            "explainability",
            "data_handlers",
            "memory_management",
            "process_isolation",
        ]
        for key in expected_keys:
            assert key in result

    def test_sklearn_support_is_bool(self):
        import ictonyx as ix

        result = ix.get_feature_availability()
        assert isinstance(result["sklearn_support"], bool)

    def test_data_handlers_is_list(self):
        import ictonyx as ix

        result = ix.get_feature_availability()
        assert isinstance(result["data_handlers"], list)
