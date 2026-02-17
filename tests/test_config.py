"""Test ModelConfig functionality thoroughly."""

import numpy as np
import pytest

from ictonyx import ModelConfig


class TestModelConfig:
    """Test ModelConfig class."""

    def test_config_creation_empty(self):
        """Test creating empty config."""
        config = ModelConfig()
        assert config.params == {}
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    def test_config_creation_with_params(self):
        """Test creating config with parameters."""
        params = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}
        config = ModelConfig(params)
        assert config.params == params
        assert config["epochs"] == 10

    def test_dictionary_interface(self):
        """Test dict-like access."""
        config = ModelConfig({"a": 1, "b": 2})

        # __getitem__
        assert config["a"] == 1

        # __setitem__
        config["c"] = 3
        assert config["c"] == 3

        # __contains__
        assert "a" in config
        assert "d" not in config

        # KeyError for missing
        with pytest.raises(KeyError):
            _ = config["missing"]

    def test_property_validation(self):
        """Test property setters validate inputs."""
        config = ModelConfig()

        # Valid values
        config.epochs = 10
        config.batch_size = 32
        config.learning_rate = 0.001

        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001

        # Invalid epochs
        with pytest.raises(ValueError, match="positive integer"):
            config.epochs = -1
        with pytest.raises(ValueError, match="positive integer"):
            config.epochs = 0
        with pytest.raises(ValueError, match="positive integer"):
            config.epochs = 3.14

        # Invalid batch_size
        with pytest.raises(ValueError, match="positive integer"):
            config.batch_size = -32

        # Invalid learning_rate
        with pytest.raises(ValueError, match="positive number"):
            config.learning_rate = -0.001

    def test_factory_methods(self):
        """Test all factory methods create proper configs."""
        # Default config
        default = ModelConfig.from_defaults()
        assert default.epochs == 10
        assert default.batch_size == 32
        assert default.learning_rate == 0.001
        assert "cleanup_threshold" in default.params

        # CNN config
        cnn = ModelConfig.for_cnn(input_shape=(64, 64, 3), num_classes=10)
        assert cnn.params["input_shape"] == (64, 64, 3)
        assert cnn.params["num_classes"] == 10
        assert cnn.params["loss"] == "categorical_crossentropy"

        # XGBoost config
        xgb = ModelConfig.for_xgboost(num_classes=3)
        assert "n_estimators" in xgb.params
        assert xgb.params["objective"] == "multi:softprob"

        # Variability study config
        base = ModelConfig({"epochs": 5})
        study = ModelConfig.for_variability_study(base, num_runs=10)
        assert study.params["num_runs"] == 10
        assert study.params["epochs_per_run"] == 5

    def test_copy_independence(self):
        """Test copy creates independent configs."""
        config1 = ModelConfig({"epochs": 10, "nested": {"a": 1}})
        config2 = config1.copy()

        # Modify copy
        config2.epochs = 20
        config2.params["nested"]["a"] = 2
        config2.params["new"] = "value"

        # Original unchanged
        assert config1.epochs == 10
        assert config1.params["nested"]["a"] == 1
        assert "new" not in config1.params

    def test_validation_methods(self):
        """Test parameter validation methods."""
        config = ModelConfig(
            {"epochs": 10, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"}
        )

        # Required params check
        missing = config.validate_required(["epochs", "missing_param"])
        assert missing == ["missing_param"]

        # Type validation
        errors = config.validate_types({"epochs": int, "learning_rate": float, "optimizer": str})
        assert errors == []

        # Wrong type
        config.params["bad_type"] = [1, 2, 3]
        errors = config.validate_types({"bad_type": str})
        assert len(errors) == 1
        assert "bad_type" in errors[0]

    def test_method_chaining(self):
        """Test methods return self for chaining."""
        config = ModelConfig()
        result = config.set("a", 1).set("b", 2).update({"c": 3})

        assert result is config
        assert config.params == {"a": 1, "b": 2, "c": 3}

    def test_keys_values_items(self):
        """Test dictionary view methods."""
        config = ModelConfig({"a": 1, "b": 2})

        assert set(config.keys()) == {"a", "b"}
        assert set(config.values()) == {1, 2}
        assert set(config.items()) == {("a", 1), ("b", 2)}


class TestModelConfigExtended:
    """Additional ModelConfig tests."""

    def test_repr(self):
        config = ModelConfig({"epochs": 10})
        assert "epochs" in repr(config)
        assert "10" in repr(config)

    def test_merge_is_alias_for_update(self):
        config = ModelConfig({"a": 1})
        result = config.merge({"b": 2})
        assert result is config
        assert config.params == {"a": 1, "b": 2}

    def test_has(self):
        config = ModelConfig({"epochs": 10})
        assert config.has("epochs") is True
        assert config.has("missing") is False

    def test_for_xgboost_binary(self):
        config = ModelConfig.for_xgboost(num_classes=2)
        assert config.params["objective"] == "binary:logistic"

    def test_for_xgboost_multiclass(self):
        config = ModelConfig.for_xgboost(num_classes=5)
        assert config.params["objective"] == "multi:softprob"

    def test_for_cnn(self):
        config = ModelConfig.for_cnn(input_shape=(64, 64, 3), num_classes=10)
        assert config.params["loss"] == "categorical_crossentropy"
        assert config.params["num_classes"] == 10
        assert config.params["input_shape"] == (64, 64, 3)
        assert "accuracy" in config.params["metrics"]

    def test_from_defaults_all_keys(self):
        config = ModelConfig.from_defaults()
        assert "epochs" in config
        assert "batch_size" in config
        assert "learning_rate" in config


class TestConfigSettersUncovered:
    """Verbose, num_runs, epochs_per_run, cleanup_threshold setter validation."""

    def test_verbose_setter_rejects_negative(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="non-negative integer"):
            config.verbose = -1

    def test_verbose_setter_rejects_float(self):
        config = ModelConfig()
        with pytest.raises(ValueError):
            config.verbose = 1.5

    def test_verbose_setter_accepts_zero(self):
        config = ModelConfig()
        config.verbose = 0
        assert config.verbose == 0

    def test_num_runs_setter_rejects_zero(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="positive integer"):
            config.num_runs = 0

    def test_num_runs_setter_rejects_negative(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="positive integer"):
            config.num_runs = -5

    def test_num_runs_roundtrip(self):
        config = ModelConfig()
        config.num_runs = 10
        assert config.num_runs == 10

    def test_epochs_per_run_setter_rejects_zero(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="positive integer"):
            config.epochs_per_run = 0

    def test_epochs_per_run_roundtrip(self):
        config = ModelConfig()
        config.epochs_per_run = 50
        assert config.epochs_per_run == 50

    def test_cleanup_threshold_rejects_too_low(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="between 0.1 and 1.0"):
            config.cleanup_threshold = 0.05

    def test_cleanup_threshold_rejects_too_high(self):
        config = ModelConfig()
        with pytest.raises(ValueError, match="between 0.1 and 1.0"):
            config.cleanup_threshold = 1.5

    def test_cleanup_threshold_accepts_boundaries(self):
        config = ModelConfig()
        config.cleanup_threshold = 0.1
        assert config.cleanup_threshold == 0.1
        config.cleanup_threshold = 1.0
        assert config.cleanup_threshold == 1.0
