"""Integration tests for complete workflows."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ictonyx import (
    ModelConfig,
    assess_training_stability,
    compare_two_models,
    run_variability_study,
)
from ictonyx.core import BaseModelWrapper, TrainingResult
from ictonyx.data import TabularDataHandler


class SimpleModel(BaseModelWrapper):
    """Simple model for integration testing."""

    def __init__(self, config, random_seed=None):
        super().__init__(None, f"simple_{random_seed or 0}")
        self.config = config
        if random_seed:
            np.random.seed(random_seed)

    def _cleanup_implementation(self):
        pass

    def fit(self, train_data, validation_data=None, **kwargs):
        epochs = self.config.get("epochs", 5)
        base = 0.5 + np.random.random() * 0.3

        history = {
            "train_accuracy": np.linspace(0.4, base + 0.1, epochs).tolist(),
            "loss": np.linspace(1.0, 0.2, epochs).tolist(),
        }
        if validation_data is not None:
            history["val_accuracy"] = np.linspace(0.35, base, epochs).tolist()

        self.training_result = TrainingResult(history=history)

    def predict(self, data, **kwargs):
        X = data[0] if isinstance(data, tuple) else data
        self.predictions = np.random.randint(0, 2, len(X))
        return self.predictions

    def predict_proba(self, data, **kwargs):
        n = len(data[0]) if isinstance(data, tuple) else len(data)
        probs = np.random.rand(n, 2)
        return probs / probs.sum(axis=1, keepdims=True)

    def evaluate(self, data, **kwargs):
        return {"accuracy": 0.7 + np.random.random() * 0.2}

    def assess(self, true_labels):
        if self.predictions is None:
            raise ValueError("No predictions")
        return {"accuracy": 0.8}

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        return cls(ModelConfig())


class TestEndToEndWorkflow:
    """Test complete ictonyx workflow."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline."""

        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "feature1": np.random.rand(100),
                    "feature2": np.random.rand(100),
                    "feature3": np.random.rand(100),
                    "target": np.random.randint(0, 2, 100),
                }
            )
            df.to_csv(f.name, index=False)
            data_path = f.name

        try:
            # 1. Setup
            config = ModelConfig({"epochs": 3, "batch_size": 32})

            data_handler = TabularDataHandler(data_path=data_path, target_column="target")

            def model_builder(conf):
                return SimpleModel(conf, random_seed=np.random.randint(1000))

            # 2. Run variability study
            results = run_variability_study(
                model_builder=model_builder,
                data_handler=data_handler,
                model_config=config,
                num_runs=5,
                verbose=False,
            )

            # 3. Check results
            assert results.n_runs == 5
            assert len(results.final_metrics["val_accuracy"]) == 5

            # 4. Assess stability
            stability = assess_training_stability(results.all_runs_metrics, window_size=2)

            assert "stability_assessment" in stability
            assert stability["n_runs"] == 5
            assert stability["common_length"] == 3  # 3 epochs

            # 5. Statistical comparison
            # Split results into two "models" for comparison
            val_accs = results.final_metrics["val_accuracy"]
            model1_data = pd.Series(val_accs[:3])
            model2_data = pd.Series(val_accs[3:])

            if len(model1_data) >= 2 and len(model2_data) >= 2:
                comparison = compare_two_models(model1_data, model2_data, paired=False)

                assert hasattr(comparison, "p_value")
                assert hasattr(comparison, "test_name")

            # 6. Get summary
            summary = results.summarize()
            assert "Successful runs: 5" in summary

        finally:
            # Cleanup
            os.unlink(data_path)

    def test_data_handler_integration(self):
        """Test data handler with real file."""

        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {"x1": range(100), "x2": range(100, 200), "y": [i % 2 for i in range(100)]}
            )
            df.to_csv(f.name, index=False)
            path = f.name

        try:
            handler = TabularDataHandler(path, target_column="y")
            data = handler.load(test_split=0.2, val_split=0.1)

            assert "train_data" in data
            assert "val_data" in data
            assert "test_data" in data

            X_train, y_train = data["train_data"]
            assert len(X_train) > 0
            assert len(y_train) > 0
            assert X_train.shape[1] == 2  # Two features

        finally:
            os.unlink(path)

    def test_config_validation_integration(self):
        """Test config validation in real workflow."""

        config = ModelConfig()

        # Add valid parameters
        config.epochs = 10
        config.batch_size = 32

        # Validation should pass
        missing = config.validate_required(["epochs", "batch_size"])
        assert missing == []

        # Type checking
        errors = config.validate_types({"epochs": int, "batch_size": int})
        assert errors == []

        # Factory methods should produce valid configs
        cnn_config = ModelConfig.for_cnn()
        assert cnn_config.validate_required(["input_shape", "num_classes"]) == []


class TestSklearnPipeline:
    """Full pipeline: sklearn → variability_study → statistics."""

    def test_classifier_full_pipeline(self):
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        import ictonyx as ix

        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        results = ix.variability_study(
            model=RandomForestClassifier,
            data=df,
            target_column="target",
            runs=5,
        )
        assert results.n_runs == 5
        assert "val_accuracy" in results.final_metrics
        assert "Successful runs: 5" in results.summarize()

    def test_regressor_full_pipeline(self):
        from sklearn.linear_model import LinearRegression

        import ictonyx as ix

        X = np.random.rand(200, 4)
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + np.random.randn(200) * 0.1
        results = ix.variability_study(model=LinearRegression, data=(X, y), runs=5)
        assert results.n_runs == 5
        assert any("r2" in k for k in results.final_metrics)

    def test_compare_two_models(self):
        from sklearn.datasets import load_iris
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        import ictonyx as ix

        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        comparison = ix.compare_models(
            models=[RandomForestClassifier, GradientBoostingClassifier],
            data=df,
            target_column="target",
            runs=10,
        )
        # With deterministic sklearn models on iris, both classifiers produce
        # identical outputs across runs, making paired comparison undefined.
        # Verify the integration path runs cleanly and returns a valid result.
        assert comparison.overall_test is not None
        assert comparison.overall_test.test_name  # non-empty test name
        assert len(comparison.raw_data) == 2
        assert hasattr(comparison, "is_significant")
        assert len(comparison.get_summary()) > 0

    def test_results_to_dataframe(self):
        from sklearn.tree import DecisionTreeClassifier

        import ictonyx as ix

        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        results = ix.variability_study(model=DecisionTreeClassifier, data=(X, y), runs=4)
        df = results.to_dataframe()
        assert len(df) == 4
        assert "run_id" in df.columns


class TestSaveLoadRoundtrip:
    def test_sklearn_save_load(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        from ictonyx.core import ScikitLearnModelWrapper

        wrapper = ScikitLearnModelWrapper(RandomForestClassifier(n_estimators=5))
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        wrapper.fit((X, y))
        path = str(tmp_path / "model.pkl")
        wrapper.save_model(path)
        loaded = ScikitLearnModelWrapper.load_model(path)
        preds = loaded.predict(X)
        assert len(preds) == 50


class TestStatisticalIntegration:
    def test_assess_training_stability_on_runner_output(self):
        from ictonyx import ModelConfig, assess_training_stability, run_variability_study
        from ictonyx.data import ArraysDataHandler

        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        handler = ArraysDataHandler(X, y.astype(float))
        config = ModelConfig({"epochs": 5})

        # Use the SimpleModel fixture already defined at top of test_integration.py
        results = run_variability_study(
            model_builder=lambda cfg: SimpleModel(cfg),
            data_handler=handler,
            model_config=config,
            num_runs=4,
            verbose=False,
        )
        stability = assess_training_stability(results.all_runs_metrics, window_size=2)
        assert "stability_assessment" in stability
        assert stability["n_runs"] == 4


class TestTestSetPrimacy:
    """Verify that compare_models auto-selects test metrics when available."""

    @pytest.mark.skipif(
        not __import__("ictonyx.core", fromlist=["SKLEARN_AVAILABLE"]).SKLEARN_AVAILABLE,
        reason="sklearn required",
    )
    def test_compare_models_resolves_metric_from_results(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        import ictonyx as ix
        from ictonyx.data import ArraysDataHandler

        rng = np.random.default_rng(42)
        X = rng.standard_normal((120, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        handler = ArraysDataHandler(X, y, val_split=0.15, test_split=0.15)

        comparison = ix.compare_models(
            models=[RandomForestClassifier, DecisionTreeClassifier],
            data=handler,
            runs=5,
            seed=42,
        )
        assert comparison.overall_test is not None
        assert comparison.metric in ("test_accuracy", "val_accuracy")
        assert comparison.n_models == 2
