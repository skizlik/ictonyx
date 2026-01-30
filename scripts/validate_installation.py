# scripts/validate_installation.py
#!/usr/bin/env python3
"""Comprehensive validation of ictonyx installation and functionality."""
import sys
import os
import time
import json
from pathlib import Path

# Silence TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60)

def validate_core_functionality():
    """Test that core ictonyx features actually work, not just import."""
    print("\n🔧 Testing Core Functionality...")
    
    try:
        import ictonyx
        from ictonyx import ModelConfig, run_variability_study
        import numpy as np
        import pandas as pd
        
        # Test 1: Config system
        config = ModelConfig({'epochs': 5, 'batch_size': 32})
        assert config.epochs == 5, "Config failed"
        print("  ✓ ModelConfig works")
        
        # Test 2: Create minimal data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        print("  ✓ Can generate test data")
        
        # Test 3: Memory info
        memory_info = ictonyx.get_memory_info()
        print(f"  ✓ Memory tracking works: {memory_info.get('process_rss_mb', 'N/A')} MB")
        
        # Test 4: Statistical functions
        if hasattr(ictonyx, 'mann_whitney_test'):
            from ictonyx import mann_whitney_test
            result = mann_whitney_test(
                pd.Series(np.random.randn(20)),
                pd.Series(np.random.randn(20) + 0.5)
            )
            print(f"  ✓ Statistical tests work (p={result.p_value:.4f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Core functionality failed: {e}")
        return False

def benchmark_performance():
    """Quick performance benchmark."""
    print("\n⚡ Performance Benchmark...")
    
    try:
        import ictonyx
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create dataset
        X, y = make_classification(n_samples=1000, n_features=20)
        
        # Time model creation and training
        start = time.time()
        
        from ictonyx.core import ScikitLearnModelWrapper
        model = ScikitLearnModelWrapper(RandomForestClassifier(n_estimators=10))
        model.fit((X[:800], y[:800]), validation_data=(X[800:], y[800:]))
        
        elapsed = time.time() - start
        print(f"  ✓ Training completed in {elapsed:.2f}s")
        
        # Test prediction
        start = time.time()
        predictions = model.predict(X[800:])
        pred_time = time.time() - start
        print(f"  ✓ Prediction completed in {pred_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return False

def check_optional_dependencies():
    """Check status of all optional dependencies."""
    print("\n📦 Optional Dependencies Status...")
    
    deps = {
        'tensorflow': ('Deep learning support', 'pip install tensorflow'),
        'torch': ('PyTorch support', 'pip install torch'),
        'mlflow': ('Experiment tracking', 'pip install mlflow'),
        'hyperopt': ('Hyperparameter tuning', 'pip install hyperopt'),
        'shap': ('Model explainability', 'pip install shap'),
        'matplotlib': ('Plotting support', 'pip install matplotlib'),
        'seaborn': ('Enhanced plots', 'pip install seaborn'),
        'cloudpickle': ('Process isolation', 'pip install cloudpickle'),
    }
    
    installed = []
    missing = []
    
    for package, (description, install_cmd) in deps.items():
        try:
            __import__(package)
            print(f"  ✓ {package:<12} - {description}")
            installed.append(package)
        except ImportError:
            print(f"  ✗ {package:<12} - {description}")
            print(f"    → Install: {install_cmd}")
            missing.append(package)
    
    return len(installed), len(missing)

def generate_report(output_file='ictonyx_validation.json'):
    """Generate a JSON report of the validation."""
    import ictonyx
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': ictonyx.__version__,
        'features': ictonyx.get_feature_availability(),
        'python_version': sys.version,
        'platform': sys.platform,
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved to: {output_file}")

def main():
    print_header("ICTONYX INSTALLATION VALIDATOR")
    
    # Run all validation steps
    results = []
    
    # Basic import test
    from scripts.validate_imports import test_basic_import, test_all_exports
    print("\n📚 Import Validation...")
    results.append(test_basic_import())
    results.append(test_all_exports())
    
    # Functionality tests
    results.append(validate_core_functionality())
    results.append(benchmark_performance())
    
    # Dependency check
    installed, missing = check_optional_dependencies()
    
    # Generate report
    generate_report()
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    if all(results):
        print("✅ Core validation: PASSED")
    else:
        print("❌ Core validation: FAILED")
    
    print(f"📦 Dependencies: {installed} installed, {missing} missing")
    
    if missing > 0:
        print(f"\n💡 To enable all features, run:")
        print(f"   pip install ictonyx[all]")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
