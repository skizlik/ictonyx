# scripts/profile_memory.py
#!/usr/bin/env python3
"""Profile memory usage of ictonyx operations."""
import os
import sys
import tracemalloc
import psutil
import gc

def profile_variability_study():
    """Profile memory usage during a variability study."""
    print("Starting memory profiling...")
    
    # Start tracing
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        import ictonyx
        from ictonyx import run_variability_study, ModelConfig
        from ictonyx.core import ScikitLearnModelWrapper
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create data
        X, y = make_classification(n_samples=5000, n_features=50)
        
        # Save to temp file
        import pandas as pd
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(X).assign(target=y).to_csv(f.name, index=False)
            data_path = f.name
        
        # Run study
        from ictonyx.data import TabularDataHandler
        
        results = run_variability_study(
            model_builder=lambda c: ScikitLearnModelWrapper(
                RandomForestClassifier(n_estimators=100)
            ),
            data_handler=TabularDataHandler(data_path, 'target'),
            model_config=ModelConfig({'epochs': 5}),
            num_runs=3,
            verbose=False
        )
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\nMemory Profile:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Peak:     {peak / 1024 / 1024:.1f} MB")
        print(f"  Final:    {final_memory:.1f} MB")
        print(f"  Leaked:   {final_memory - baseline_memory:.1f} MB")
        
        # Cleanup
        os.unlink(data_path)
        gc.collect()
        
        return final_memory - baseline_memory < 50  # Less than 50MB leak
        
    except Exception as e:
        print(f"Profiling failed: {e}")
        tracemalloc.stop()
        return False

if __name__ == "__main__":
    success = profile_variability_study()
    sys.exit(0 if success else 1)
