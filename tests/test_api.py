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
        models=models,
        data=sample_df,
        target_column="target",
        runs=5,
        metric="val_accuracy",
        paired=False,
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
    empty_results.has_test_data = False
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


class TestCompareResultsPairing:
    def _make_results(self, seed, n=5):
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        data = load_wine()
        X = StandardScaler().fit_transform(data.data)
        return api.variability_study(
            model=RandomForestClassifier,
            data=(X, data.target),
            runs=n,
            seed=seed,
            verbose=False,
        )

    def test_paired_true_uses_wilcoxon(self):
        ra = self._make_results(seed=42)
        rb = self._make_results(seed=42)
        result = api.compare_results(ra, rb, paired=True)
        assert "Wilcoxon" in result.overall_test.test_name

    def test_paired_false_uses_mann_whitney(self):
        ra = self._make_results(seed=42)
        rb = self._make_results(seed=42)
        result = api.compare_results(ra, rb, paired=False)
        assert "Mann-Whitney" in result.overall_test.test_name

    def test_unequal_runs_warns_and_falls_back(self):
        ra = self._make_results(seed=42, n=5)
        rb = self._make_results(seed=99, n=4)
        with pytest.warns(UserWarning, match="equal run counts"):
            result = api.compare_results(ra, rb, paired=True)
        assert "Mann-Whitney" in result.overall_test.test_name

    def test_seed_parameter_accepted(self):
        ra = self._make_results(seed=42)
        rb = self._make_results(seed=42)
        assert api.compare_results(ra, rb, seed=0) is not None


def test_compare_models_model_kwargs_do_not_reach_data_handler():
    """Model hyperparameters must not be forwarded to the DataHandler."""
    from sklearn.datasets import load_wine
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    # Before the fix this raised TypeError because 'C' reached ArraysDataHandler.
    result = api.compare_models(
        models=[LogisticRegression(C=1.0), LogisticRegression(C=10.0)],
        data=(X, data.target),
        runs=5,
        seed=42,
        verbose=False,
    )
    assert result is not None


def test_build_from_class_passes_construction_kwargs():
    """X-3: wrapper classes whose __init__ requires args beyond random_state
    must receive those args from ModelConfig. Regression test for the
    HuggingFaceModelWrapper silent-failure bug in v0.4.4."""
    from ictonyx.api import _get_model_builder
    from ictonyx.config import ModelConfig

    class RequiresArg:
        def __init__(self, required_arg, optional=None):
            self.required_arg = required_arg
            self.optional = optional

        # The builder calls _ensure_wrapper() on the result. For this test,
        # we want the raw instance back, so we monkeypatch around that below.

    # Use a class that doesn't require wrapping by bypassing _ensure_wrapper.
    # Easier: use a class that _ensure_wrapper accepts as-is.
    # _ensure_wrapper raises on unknown types, so we patch it for this test.
    import ictonyx.api as api_mod

    original_ensure = api_mod._ensure_wrapper
    api_mod._ensure_wrapper = lambda x: x
    try:
        builder = _get_model_builder(RequiresArg)
        conf = ModelConfig(
            {
                "required_arg": "value",
                "optional": "other",
                "run_seed": 42,  # infra; must NOT be passed to constructor
            }
        )
        instance = builder(conf)
        assert instance.required_arg == "value"
        assert instance.optional == "other"
    finally:
        api_mod._ensure_wrapper = original_ensure


def test_build_from_class_with_random_state_and_extra_kwargs():
    """X-3: a class that accepts BOTH random_state and other required args
    must receive both."""
    import ictonyx.api as api_mod
    from ictonyx.api import _get_model_builder
    from ictonyx.config import ModelConfig

    class Combined:
        def __init__(self, required_arg, random_state=None):
            self.required_arg = required_arg
            self.random_state = random_state

    original_ensure = api_mod._ensure_wrapper
    api_mod._ensure_wrapper = lambda x: x
    try:
        builder = _get_model_builder(Combined)
        conf = ModelConfig(
            {
                "required_arg": "x",
                "run_seed": 123,
            }
        )
        instance = builder(conf)
        assert instance.required_arg == "x"
        assert instance.random_state == 123
    finally:
        api_mod._ensure_wrapper = original_ensure


def test_build_from_class_filters_unknown_kwargs_for_fixed_signature():
    """X-3: for a constructor WITHOUT **kwargs, only accepted parameters
    are splatted. ModelConfig keys not in the signature (e.g. 'epochs' for
    a wrapper that doesn't take epochs) must be filtered out."""
    import ictonyx.api as api_mod
    from ictonyx.api import _get_model_builder
    from ictonyx.config import ModelConfig

    class Strict:
        def __init__(self, wanted):
            self.wanted = wanted

    original_ensure = api_mod._ensure_wrapper
    api_mod._ensure_wrapper = lambda x: x
    try:
        builder = _get_model_builder(Strict)
        conf = ModelConfig(
            {
                "wanted": "yes",
                "epochs": 10,  # irrelevant to Strict.__init__
                "batch_size": 32,  # also irrelevant
                "run_seed": 1,
            }
        )
        # Must not raise TypeError about unexpected kwargs
        instance = builder(conf)
        assert instance.wanted == "yes"
    finally:
        api_mod._ensure_wrapper = original_ensure


def test_build_from_class_passes_all_kwargs_when_var_keyword():
    """X-3: for a constructor WITH **kwargs, all non-infra ModelConfig
    keys are forwarded (minus run_seed and random_state)."""
    import ictonyx.api as api_mod
    from ictonyx.api import _get_model_builder
    from ictonyx.config import ModelConfig

    class Flexible:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    original_ensure = api_mod._ensure_wrapper
    api_mod._ensure_wrapper = lambda x: x
    try:
        builder = _get_model_builder(Flexible)
        conf = ModelConfig(
            {
                "model_name_or_path": "distilbert-base-uncased",
                "num_labels": 4,
                "run_seed": 42,  # excluded
            }
        )
        instance = builder(conf)
        assert instance.kwargs["model_name_or_path"] == "distilbert-base-uncased"
        assert instance.kwargs["num_labels"] == 4
        assert "run_seed" not in instance.kwargs
        assert "random_state" not in instance.kwargs
    finally:
        api_mod._ensure_wrapper = original_ensure


def test_compare_models_k2_paired_carries_ci():
    """compare_models(paired=True) must surface the mean-difference CI
    from the analytic core. The ci_effect_size is None for paired Wilcoxon
    (per the earlier Wilcoxon-CI fix) but confidence_interval must be
    populated."""
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    import ictonyx as ix

    X, y = make_classification(n_samples=200, random_state=0)
    result = ix.compare_models(
        models=[LogisticRegression(max_iter=1000), DecisionTreeClassifier()],
        data=(X, y),
        runs=10,
        seed=42,
        paired=True,
    )
    assert len(result.pairwise_comparisons) == 1
    pair = next(iter(result.pairwise_comparisons.values()))
    # Mean-difference CI must be populated.
    assert pair.confidence_interval is not None
    # Paired Wilcoxon: effect-size CI is None by design (Commit 4).
    assert pair.ci_effect_size is None


def test_compare_models_k2_unpaired_carries_ci():
    """compare_models(paired=False) on two models routes to compare_two_models
    via the multi-model KW+MW path. Verify the pairwise comparison carries
    a mean-difference CI when one is computed."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    import ictonyx as ix

    X, y = make_classification(n_samples=200, random_state=0)
    result = ix.compare_models(
        models=[LogisticRegression(max_iter=1000), DecisionTreeClassifier()],
        data=(X, y),
        runs=10,
        seed=42,
        paired=False,
    )
    assert len(result.pairwise_comparisons) >= 1
    pair = next(iter(result.pairwise_comparisons.values()))
    # The pairwise_comparison object must have the CI attribute, populated
    # or None (None is acceptable for MW path at present; see Commit 4).
    assert hasattr(pair, "confidence_interval")
    assert hasattr(pair, "ci_effect_size")


def test_compare_results_paired_carries_ci():
    """compare_results(paired=True) must carry the CI from compare_two_models."""
    import numpy as np
    import pandas as pd

    import ictonyx as ix
    from ictonyx.runners import VariabilityStudyResults

    rng = np.random.default_rng(7)
    # Build two fake results objects with genuine paired variation.
    results_a = VariabilityStudyResults(
        all_runs_metrics=[pd.DataFrame({"val_accuracy": [v]}) for v in rng.normal(0.90, 0.02, 20)],
        final_metrics={"val_accuracy": list(rng.normal(0.90, 0.02, 20))},
        final_test_metrics=[],
        seed=42,
        run_seeds=list(range(20)),
    )
    results_b = VariabilityStudyResults(
        all_runs_metrics=[pd.DataFrame({"val_accuracy": [v]}) for v in rng.normal(0.87, 0.02, 20)],
        final_metrics={"val_accuracy": list(rng.normal(0.87, 0.02, 20))},
        final_test_metrics=[],
        seed=42,
        run_seeds=list(range(20)),
    )

    result = ix.compare_results(results_a, results_b, paired=True, metric="val_accuracy")
    pair = next(iter(result.pairwise_comparisons.values()))
    assert pair.confidence_interval is not None
