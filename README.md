# Ictonyx: Iteration Comparison Testing Over N-runs: Yield eXamination. 
A library for variability analysis in ML training.

![CI/CD](https://github.com/skizlik/ictonyx/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Ictonyx** is a framework for conducting systematic **variability studies** and **statistical comparisons** of machine learning models.

Unlike standard training loops that produce a single "lucky" result, Ictonyx treats model training as a stochastic process. It automates the execution of multiple runs, handles GPU memory isolation, and performs rigorous hypothesis testing to determine if Model A is *actually* better than Model B.

---

## 🚀 Key Features

* **One-Line Variability Studies:** Automate $N$ runs of a model with a single function call.
* **Process Isolation:** Runs every training session in a separate system process to guarantee zero GPU memory leaks between runs.
* **Statistical Rigor:** Automatically selects the correct hypothesis test (T-test vs Mann-Whitney) based on data distribution assumptions.
* **Effect Sizes:** Reports *Cohen's d* and *Rank-Biserial Correlation* to distinguish statistical significance from practical significance.
* **Professional Visualization:** Publication-ready Boxplots, Forest Plots, and Training Trajectories.

---

## 📦 Installation

Ictonyx is currently installed directly from source:

```bash
# Basic installation (from GitHub)
pip install git+[https://github.com/skizlik/ictonyx.git](https://github.com/skizlik/ictonyx.git)

# With all optional dependencies (TensorFlow, MLflow, SHAP, Hyperopt)
# Note: Quotes are often required for brackets in Zsh/PowerShell
pip install "ictonyx[all] @ git+[https://github.com/skizlik/ictonyx.git](https://github.com/skizlik/ictonyx.git)"
```

**For Developers (Editable Install):**
```bash
# 1. Clone the repository
git clone [https://github.com/skizlik/ictonyx.git](https://github.com/skizlik/ictonyx.git)
cd ictonyx

# 2. Install in editable mode
pip install -e ".[all]"
```

---

## ⚡ Quick Start

### 1. The Variability Study
Run a model 5 times to see how stable it is. Ictonyx automatically handles DataFrames, CSV paths, or Numpy arrays.

```python
import ictonyx
from sklearn.ensemble import RandomForestClassifier

# 1. Run the study
results = ictonyx.variability_study(
    model=RandomForestClassifier,
    data="dataset.csv",
    target_column="label",
    runs=5,
    epochs=10
)

# 2. View statistical summary (Mean, Std, Min, Max)
print(results.summarize())

# 3. Visualize stability (Trajectories & Histogram)
ictonyx.plot_variability_summary(
    results.all_runs_metrics, 
    results.final_val_accuracies
)
```

### 2. Comparing Models (The Showdown)
Compare a Random Forest against a Decision Tree with statistical guarantees.

```python
from sklearn.tree import DecisionTreeClassifier

# 1. Run the comparison
comparison = ictonyx.compare_models(
    models=[RandomForestClassifier, DecisionTreeClassifier],
    data="dataset.csv",
    target_column="label",
    runs=10,
    metric="val_accuracy"
)

# 2. Get the verdict
print(comparison['overall_test'].conclusion)
# Output: "Mann-Whitney U test indicates a statistically significant difference (p=0.002)..."

# 3. Visualize the difference (Boxplot)
ictonyx.plot_comparison_boxplots(comparison)
```

---

## 🛠️ Advanced Configuration

### GPU Process Isolation
For Deep Learning (TensorFlow/Keras), memory leaks are common when retraining models in a loop. Ictonyx solves this by spawning a fresh process for every run.

```python
results = ictonyx.variability_study(
    model=create_keras_model,
    data=image_dir,
    runs=5,
    use_process_isolation=True,  # <--- The Magic Switch
    gpu_memory_limit=4096        # Limit to 4GB per run
)
```

### Logging & Theming
Ictonyx uses a centralized configuration system.

```python
import ictonyx.settings

# Silence console output
ictonyx.set_verbose(False)

# Prevent plots from blocking execution (for scripts)
ictonyx.set_display_plots(False)

# Change plot colors
ictonyx.settings.set_theme('publication') # Black & White high contrast
```

---

## 📊 Visualizations

Ictonyx provides three primary visualization tools:

1.  **`plot_variability_summary`**: Shows training curves, final loss distribution, and convergence stability.
2.  **`plot_comparison_boxplots`**: Side-by-side comparison of model performance distributions with significance annotations.
3.  **`plot_comparison_forest`**: Visualizes the effect size (difference from baseline) with confidence intervals.

---

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Run tests (`pytest tests/ -v`).
4.  Commit your changes.
5.  Push to the branch and open a Pull Request.

**License:** MIT
