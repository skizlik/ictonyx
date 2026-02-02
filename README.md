# Ictonyx

**Iteration Comparison Testing Over N-runs: Yield eXamination**

*Stop comparing lucky runs.*

![CI/CD](https://github.com/skizlik/ictonyx/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Ictonyx is a Python framework for rigorous statistical comparison of machine learning models. It automates multi-run experiments, performs proper hypothesis testing, and reports effect sizes—so you know whether Model A is *actually* better than Model B, or whether you just got lucky.

---

## The Problem

You train Model A. Accuracy: 94.2%. You train Model B. Accuracy: 93.8%. Model A wins.

**Retrain a week later:** Model A: 93.1%, Model B: 94.5%.

Machine learning training is stochastic. Random initialization, data shuffling, dropout—every run differs. Comparing single runs is like flipping a coin once and calling it biased.

## The Solution

```python
import ictonyx
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

comparison = ictonyx.compare_models(
    models=[RandomForestClassifier, DecisionTreeClassifier],
    data="dataset.csv",
    target_column="target",
    runs=10
)

print(comparison['overall_test'].conclusion)
# "Mann-Whitney U test indicates a statistically significant difference
#  (p=0.003). Effect size: d=1.31 (large)."
```

---

## Installation

```bash
# From GitHub
pip install git+https://github.com/skizlik/ictonyx.git

# With optional dependencies (TensorFlow, MLflow, SHAP)
pip install "ictonyx[all] @ git+https://github.com/skizlik/ictonyx.git"

# Development
git clone https://github.com/skizlik/ictonyx.git
cd ictonyx
pip install -e ".[all]"
```

---

## Quick Start

### Variability Study

```python
import ictonyx
from sklearn.ensemble import RandomForestClassifier

results = ictonyx.variability_study(
    model=RandomForestClassifier,
    data="dataset.csv",
    target_column="target",
    runs=10
)

print(results.summarize())
# Mean: 0.942, Std: 0.018, Min: 0.910, Max: 0.971

ictonyx.plot_variability_summary(results.all_runs_metrics, results.final_val_accuracies)
```

### Model Comparison

```python
comparison = ictonyx.compare_models(
    models=[ModelA, ModelB, ModelC],
    data="dataset.csv",
    target_column="target",
    runs=10
)

ictonyx.plot_comparison_boxplots(comparison)
ictonyx.plot_comparison_forest(comparison)  # Effect sizes with CIs
```

### Deep Learning with GPU Isolation

Keras training in a loop leaks GPU memory. Ictonyx runs each session in an isolated subprocess:

```python
results = ictonyx.variability_study(
    model=create_keras_model,
    data="dataset.csv",
    target_column="target",
    runs=10,
    epochs=50,
    use_process_isolation=True,
    gpu_memory_limit=4096
)
```

---

## Key Features

**Statistical Analysis**
- Automatic test selection (t-test, Mann-Whitney, ANOVA, Kruskal-Wallis)
- Effect sizes (Cohen's d, rank-biserial correlation, eta-squared)
- Multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg)

**Visualizations**
- `plot_variability_summary()` — Training curves + metric distributions
- `plot_comparison_boxplots()` — Side-by-side model comparison
- `plot_comparison_forest()` — Effect sizes with confidence intervals
- `plot_confusion_matrix()`, `plot_roc_curve()`, `plot_training_history()`

**Data Handling**
- CSV files, DataFrames, NumPy arrays, image directories
- Automatic format detection via `auto_resolve_handler()`

**Memory Management**
- Process isolation for GPU workloads
- `cleanup_gpu_memory()`, `get_memory_info()`, `managed_memory()` context manager

---

## GPU Development Environment

Ictonyx includes a Docker environment with CUDA 12.9, cuDNN, and TensorFlow pre-configured.

```bash
# Build the image
./build-gpu.sh

# Verify GPU access
./test-gpu.sh

# Launch JupyterLab
./run-gpu.sh
# Access at http://localhost:8888

# Run tests in container
./run-gpu.sh pytest tests/ -v
```

The container runs as your user ID—no root-owned files.

---

## Configuration

```python
import ictonyx

ictonyx.set_verbose(False)        # Suppress console output
ictonyx.set_display_plots(False)  # Non-blocking plots for scripts

# Check available features
print(ictonyx.get_feature_availability())
```

---

## Comparison with Other Tools

| Tool | Experiment Tracking | Statistical Comparison | Effect Sizes | GPU Isolation |
|------|---------------------|------------------------|--------------|---------------|
| MLflow | Yes | No | No | No |
| W&B | Yes | No | No | No |
| Optuna | No | No | No | No |
| **Ictonyx** | Via MLflow | Yes | Yes | Yes |

Ictonyx complements tracking tools. Use MLflow to log experiments, Ictonyx to determine if differences are real.

---

## Contributing

1. Fork and clone
2. `pip install -e ".[all]"`
3. `pytest tests/ -v`
4. `black ictonyx/ && isort ictonyx/`
5. Open a PR

---

## License

MIT
