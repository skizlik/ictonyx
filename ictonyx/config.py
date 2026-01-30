# ictonyx/config.py

from typing import Dict, Any, List, Optional, Union, KeysView, ValuesView, ItemsView

class ModelConfig:
    """
    A class to manage and store all model training parameters and hyperparameters.

    Provides explicit parameter access methods and common parameter properties
    for better IDE support and clearer error messages.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the config with a dictionary of parameters."""
        self.params = params if params is not None else {}

    def __repr__(self) -> str:
        """Provides a clean string representation."""
        return f"ModelConfig({self.params})"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access like `config['epochs']`."""
        if key not in self.params:
            raise KeyError(f"Parameter '{key}' not found in config. Available: {list(self.params.keys())}")
        return self.params[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting like `config['epochs'] = 10`."""
        self.params[key] = value

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: `'epochs' in config`."""
        return key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with optional default value."""
        return self.params.get(key, default)

    def set(self, key: str, value: Any) -> 'ModelConfig':
        """Set parameter and return self for chaining."""
        self.params[key] = value
        return self

    def update(self, other_params: Dict[str, Any]) -> 'ModelConfig':
        """Update multiple parameters at once and return self for chaining."""
        self.params.update(other_params)
        return self

    def merge(self, other_params: Dict[str, Any]) -> 'ModelConfig':
        """Alias for update() to maintain backward compatibility."""
        return self.update(other_params)

    def has(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self.params

    def keys(self) -> KeysView[str]:
        """Get all parameter keys."""
        return self.params.keys()

    def values(self) -> ValuesView[Any]:
        """Get all parameter values."""
        return self.params.values()

    def items(self) -> ItemsView[str, Any]:
        """Get all parameter key-value pairs."""
        return self.params.items()

    def copy(self) -> 'ModelConfig':
        """Create a deep copy of the configuration."""
        import copy
        return ModelConfig(copy.deepcopy(self.params))

    def validate_required(self, required_params: List[str]) -> List[str]:
        """
        Check if required parameters are present.

        Args:
            required_params: List of parameter names that must be present

        Returns:
            List of missing parameter names (empty if all present)
        """
        missing = [param for param in required_params if param not in self.params]
        return missing

    def validate_types(self, type_specs: Dict[str, type]) -> List[str]:
        """
        Validate parameter types.

        Args:
            type_specs: Dictionary mapping parameter names to expected types

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        for param_name, expected_type in type_specs.items():
            if param_name in self.params:
                value = self.params[param_name]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Parameter '{param_name}' should be {expected_type.__name__}, got {type(value).__name__}")
        return errors

    # Common parameter properties with validation
    @property
    def epochs(self) -> Optional[int]:
        """Get epochs parameter."""
        return self.params.get('epochs')

    @epochs.setter
    def epochs(self, value: int):
        """Set epochs parameter with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"epochs must be a positive integer, got {value}")
        self.params['epochs'] = value

    @property
    def batch_size(self) -> Optional[int]:
        """Get batch_size parameter."""
        return self.params.get('batch_size')

    @batch_size.setter
    def batch_size(self, value: int):
        """Set batch_size parameter with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {value}")
        self.params['batch_size'] = value

    @property
    def learning_rate(self) -> Optional[float]:
        """Get learning_rate parameter."""
        return self.params.get('learning_rate')

    @learning_rate.setter
    def learning_rate(self, value: Union[float, int]):
        """Set learning_rate parameter with validation."""
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError(f"learning_rate must be a positive number, got {value}")
        self.params['learning_rate'] = float(value)

    @property
    def verbose(self) -> Optional[int]:
        """Get verbose parameter."""
        return self.params.get('verbose')

    @verbose.setter
    def verbose(self, value: int):
        """Set verbose parameter with validation."""
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"verbose must be a non-negative integer, got {value}")
        self.params['verbose'] = value

    # NEW: Experiment-specific properties for runners.py
    @property
    def num_runs(self) -> Optional[int]:
        """Get num_runs parameter for variability studies."""
        return self.params.get('num_runs')

    @num_runs.setter
    def num_runs(self, value: int):
        """Set num_runs parameter with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"num_runs must be a positive integer, got {value}")
        self.params['num_runs'] = value

    @property
    def epochs_per_run(self) -> Optional[int]:
        """Get epochs_per_run parameter for variability studies."""
        return self.params.get('epochs_per_run')

    @epochs_per_run.setter
    def epochs_per_run(self, value: int):
        """Set epochs_per_run parameter with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"epochs_per_run must be a positive integer, got {value}")
        self.params['epochs_per_run'] = value

    # Factory Methods for Smart Defaults

    @classmethod
    def from_defaults(cls) -> 'ModelConfig':
        """Returns a config with general, reasonable defaults including memory management."""
        return cls({
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'verbose': 0,
            'cleanup_threshold': 0.8
        })

    @classmethod
    def for_cnn(cls, input_shape: tuple = (128, 128, 3), num_classes: int = 10) -> 'ModelConfig':
        """Returns a config with defaults suitable for a CNN model."""
        base_config = cls.from_defaults()
        base_config.update({
            'input_shape': input_shape,
            'num_classes': num_classes,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'cleanup_threshold': 0.75  # More aggressive for memory-intensive CNNs
        })
        return base_config

    @classmethod
    def for_xgboost(cls, num_classes: int = 10) -> 'ModelConfig':
        """Returns a config with defaults suitable for an XGBoost model."""
        base_config = cls.from_defaults()
        base_config.update({
            'n_estimators': 180,
            'max_depth': 7,
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        })
        return base_config

    @classmethod
    def for_variability_study(cls, base_config: 'ModelConfig', num_runs: int = 5) -> 'ModelConfig':
        """Create config specifically for variability studies."""
        study_config = base_config.copy()
        study_config.set('num_runs', num_runs)
        study_config.set('epochs_per_run', base_config.get('epochs', 10))
        return study_config

    @property
    def cleanup_threshold(self) -> Optional[float]:
        """Get cleanup_threshold parameter."""
        return self.params.get('cleanup_threshold')

    @cleanup_threshold.setter
    def cleanup_threshold(self, value: float):
        """Set cleanup threshold with validation."""
        if not isinstance(value, (float, int)) or not 0.1 <= value <= 1.0:
            raise ValueError("cleanup_threshold must be between 0.1 and 1.0")
        self.params['cleanup_threshold'] = float(value)