#!/usr/bin/env python3
"""Validate that all advertised functions can actually be imported."""
import sys
import os

# Silence TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_basic_import():
    """Test basic ictonyx import."""
    try:
        import ictonyx
        print(f"✓ Ictonyx v{ictonyx.__version__} imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ictonyx: {e}")
        return False

def test_all_exports():
    """Test that everything in __all__ actually exists."""
    import ictonyx
    
    missing = []
    for name in ictonyx.__all__:
        if not hasattr(ictonyx, name):
            missing.append(name)
    
    if missing:
        print(f"✗ Missing exports: {missing}")
        return False
    else:
        print(f"✓ All {len(ictonyx.__all__)} exports verified")
        return True


def test_common_imports():
    """Test common import patterns."""
    import ictonyx

    # Required imports (must always work)
    required = [
        ("ModelConfig", "from ictonyx import ModelConfig"),
        ("plot_variability_summary", "from ictonyx import plot_variability_summary"),
        ("compare_two_models", "from ictonyx import compare_two_models"),
    ]

    # Optional imports (depend on optional dependencies)
    optional = [
        ("KerasModelWrapper", "from ictonyx import KerasModelWrapper", ictonyx.TENSORFLOW_AVAILABLE),
    ]

    success = 0
    total = len(required)

    # Test required imports
    for name, import_stmt in required:
        try:
            exec(import_stmt)
            print(f"✓ {name}")
            success += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")

    # Test optional imports
    for name, import_stmt, available in optional:
        if available:
            total += 1
            try:
                exec(import_stmt)
                print(f"✓ {name}")
                success += 1
            except ImportError as e:
                print(f"✗ {name}: {e}")
        else:
            print(f"⊘ {name} (skipped - optional dependency not installed)")

    return success == total

def main():
    print("=" * 60)
    print("ICTONYX IMPORT VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("All Exports Exist", test_all_exports),
        ("Common Imports", test_common_imports),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
