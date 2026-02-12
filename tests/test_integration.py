"""Integration tests for complete workflows."""
import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from ictonyx import (
    ModelConfig,
    run_variability_study,
    compare_two_models,
    assess_training_stability,
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
        epochs = self.config.get('epochs', 5)
        base = 0.5 + np.random.random() * 0.3

        history = {
            'train_accuracy': np.linspace(0.4, base + 0.1, epochs).tolist(),
            'loss': np.linspace(1.0, 0.2, epochs).tolist()
        }
        if validation_data is not None:
            history['val_accuracy'] = np.linspace(0.35, base, epochs).tolist()

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
        return {'accuracy': 0.7 + np.random.random() * 0.2}
    
    def assess(self, true_labels):
        if self.predictions is None:
            raise ValueError("No predictions")
        return {'accuracy': 0.8}
    
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'feature3': np.random.rand(100),
                'target': np.random.randint(0, 2, 100)
            })
            df.to_csv(f.name, index=False)
            data_path = f.name
        
        try:
            # 1. Setup
            config = ModelConfig({
                'epochs': 3,
                'batch_size': 32
            })
            
            data_handler = TabularDataHandler(
                data_path=data_path,
                target_column='target'
            )
            
            def model_builder(conf):
                return SimpleModel(conf, random_seed=np.random.randint(1000))
            
            # 2. Run variability study
            results = run_variability_study(
                model_builder=model_builder,
                data_handler=data_handler,
                model_config=config,
                num_runs=5,
                verbose=False
            )
            
            # 3. Check results
            assert results.n_runs == 5
            assert len(results.final_metrics['val_accuracy']) == 5
            
            # 4. Assess stability
            stability = assess_training_stability(
                results.all_runs_metrics,
                window_size=2
            )
            
            assert 'stability_assessment' in stability
            assert stability['n_runs'] == 5
            assert stability['common_length'] == 3  # 3 epochs
            
            # 5. Statistical comparison
            # Split results into two "models" for comparison
            val_accs = results.final_metrics['val_accuracy']
            model1_data = pd.Series(val_accs[:3])
            model2_data = pd.Series(val_accs[3:])
            
            if len(model1_data) >= 2 and len(model2_data) >= 2:
                comparison = compare_two_models(
                    model1_data,
                    model2_data,
                    paired=False
                )
                
                assert hasattr(comparison, 'p_value')
                assert hasattr(comparison, 'test_name')
            
            # 6. Get summary
            summary = results.summarize()
            assert "Successful runs: 5" in summary
            
        finally:
            # Cleanup
            os.unlink(data_path)
    
    def test_data_handler_integration(self):
        """Test data handler with real file."""
        
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'x1': range(100),
                'x2': range(100, 200),
                'y': [i % 2 for i in range(100)]
            })
            df.to_csv(f.name, index=False)
            path = f.name
        
        try:
            handler = TabularDataHandler(path, target_column='y')
            data = handler.load(test_split=0.2, val_split=0.1)
            
            assert 'train_data' in data
            assert 'val_data' in data
            assert 'test_data' in data
            
            X_train, y_train = data['train_data']
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
        missing = config.validate_required(['epochs', 'batch_size'])
        assert missing == []
        
        # Type checking
        errors = config.validate_types({
            'epochs': int,
            'batch_size': int
        })
        assert errors == []
        
        # Factory methods should produce valid configs
        cnn_config = ModelConfig.for_cnn()
        assert cnn_config.validate_required(['input_shape', 'num_classes']) == []
