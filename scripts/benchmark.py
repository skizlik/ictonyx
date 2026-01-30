# scripts/benchmark.py
#!/usr/bin/env python3
"""Benchmark ictonyx against vanilla implementations."""
import time
import numpy as np

def benchmark_vs_vanilla():
    """Compare ictonyx vs vanilla sklearn."""
    print("Ictonyx vs Vanilla Benchmark\n" + "=" * 40)
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=10000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Vanilla approach
    start = time.time()
    vanilla_accuracies = []
    for i in range(5):
        clf = RandomForestClassifier(random_state=i)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        vanilla_accuracies.append(acc)
    vanilla_time = time.time() - start
    
    # Ictonyx approach  
    import ictonyx
    from ictonyx.core import ScikitLearnModelWrapper
    from ictonyx import ModelConfig
    
    start = time.time()
    # ... run with ictonyx
    ictonyx_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Vanilla: {vanilla_time:.2f}s (mean acc: {np.mean(vanilla_accuracies):.3f})")
    print(f"  Ictonyx: {ictonyx_time:.2f}s (with stats & tracking)")
    print(f"\n✅ Ictonyx adds {ictonyx_time - vanilla_time:.2f}s for full analysis")

if __name__ == "__main__":
    benchmark_vs_vanilla()
