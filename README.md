# Ictonyx

**Iteration Comparison Testing Over N-runs: Yield eXamination**

A Python framework for studying machine learning model variability and performing rigorous statistical comparisons.

[![CI/CD](https://github.com/skizlik/ictonyx/actions/workflows/test.yml/badge.svg)](https://github.com/skizlik/ictonyx/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/ictonyx)](https://pypi.org/project/ictonyx/)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skizlik/ictonyx/blob/main/examples/quickstart.ipynb)

---

## The problem

Training a machine learning model involves stochastic factors: random weight initialisation, data shuffling, dropout. Train the same architecture on the same dataset twice and you will get different weights, different predictions, and different evaluation metrics.

This means a model's performance is a random variable, not a constant — and treating it as a constant, as most practitioners do, leads to conclusions that cannot be replicated or trusted. Reporting a single accuracy from a single training run is not a measurement; it is a sample of size one.

Ictonyx trains a model N times under independent random seeds, collects the full distribution of outcomes, and provides the statistical machinery to reason about that distribution rigorously.

---

## Installation
```bash
pip install ictonyx
```

### Optional extras

| Extra | What it includes | Install |
|---|---|---|
| `sklearn` | scikit-learn, joblib | `pip install ictonyx[sklearn]` |
| `tensorflow` | TensorFlow, Keras | `pip install ictonyx[tensorflow]` |
| `torch` | PyTorch | `pip install ictonyx[torch]` |
| `mlflow` | MLflow tracking | `pip install ictonyx[mlflow]` |
| `explain` | SHAP explainability | `pip install ictonyx[explain]` |
| `tuning` | Optuna hyperparameter tuning | `pip install ictonyx[tuning]` |
| `isolation` | Process isolation for GPU runs | `pip install ictonyx[isolation]` |
| `progress` | tqdm progress bars | `pip install ictonyx[progress]` |
| `all` | Everything above | `pip install ictonyx[all]` |

Extras can be combined:
```bash
pip install "ictonyx[tensorflow,isolation]"
```

Requires Python 3.10+. Current release: **0.4.2** — [changelog](CHANGELOG.md) · [PyPI](https://pypi.org/project/ictonyx/)
---

## Quick start

Train a small feed-forward network on the wine data from sklearn twenty times and observe the distribution of outcomes.  Here, we'll use a simple Tensorflow model.

```python
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import ictonyx as ix

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target


def build_model(config):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return ix.KerasModelWrapper(model)

results = ix.variability_study(
    model=build_model,
    data=(X, y),
    runs=20,
    epochs=20,
    seed=42,
    verbose=False
)

print(results.summarize())
```

```
Variability Study Results
==============================
Successful runs: 20
Seed: 42

Test Set Metrics:
--------------------
accuracy:
  Mean:             0.9097
  SD (sample, N-1): 0.0467
  Min:              0.8333
  Max:              0.9722
loss:
  Mean:             0.5285
  SD (sample, N-1): 0.1085
  Min:              0.3791
  Max:              0.7748

Validation Metrics:
--------------------
train_accuracy:
  Mean:             0.8851
  SD (sample, N-1): 0.0436
  Min:              0.8065
  Max:              0.9355
train_loss:
  Mean:             0.5507
  SD (sample, N-1): 0.0987
  Min:              0.4288
  Max:              0.7760
val_accuracy:
  Mean:             0.9194
  SD (sample, N-1): 0.0709
  Min:              0.7222
  Max:              1.0000
val_loss:
  Mean:             0.4981
  SD (sample, N-1): 0.1105
  Min:              0.3793
  Max:              0.7476
```

On a 178-sample dataset, the same architecture achieves anywhere from 72% to 100% validation accuracy depending solely on the random seed — a 28-point range that a single-run evaluation would never reveal.  Ictonyx also allows plotting of training histories:

```python
ix.plot_variability_summary(results=results, metric='accuracy')
```

![Variability summary for keras CNN across 20 runs](images/variability_summary.png)

---

## Comparing two models

A routine question in applied ML: does one model actually outperform another,
or did it just draw a better random seed? A single run of each tells you nothing
— the difference could be real, or it could be noise. Ictonyx answers this by
running both models the same number of times and applying a statistical test to
the resulting distributions.

Ictonyx works equally well with sklearn estimators — pass a class or configured instance directly, no wrapper required.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

comparison = ix.compare_models(
    models=[
        MLPClassifier(hidden_layer_sizes=(64,), max_iter=200),
        RandomForestClassifier(n_estimators=40),
    ],
    data=(X, y),
    runs=20,
    metric='val_accuracy',
    seed=42,
    verbose=False,
)

print(comparison.get_summary())

ix.plot_comparison_boxplots(comparison)
```

```
Model Comparison Results (val_accuracy)
========================================
Models compared: 2
Omnibus test: Paired Wilcoxon Signed-Rank Test: 5.500, p=0.0005 ***, r (effect size)=0.815

Pairwise comparisons (none correction):
  MLPClassifier_vs_RandomForestClassifier: Paired Wilcoxon Signed-Rank Test: 5.500, p=0.0005 ***, r (effect size)=0.815 *

Significant pairs: MLPClassifier_vs_RandomForestClassifier
```
![Comparison boxplots for model comparison](images/comparison_boxplots.png)


Both models received identical per-run seeds, so each MLP run is directly paired with the corresponding Random Forest run. The Wilcoxon signed-rank test exploits this pairing to remove noise common to both models.

The MLP outperformed the Random Forest with an effect size of r=0.815 (p=0.0005). That is not a marginal result that might reverse on a different seed.

The 'none correction' label indicates that with only two models there is a single pair to test, so no multiple-comparison correction is applied

---

## Process isolation for GPU runs

Keras models accumulate GPU memory across training runs in a loop. For studies with many runs or large models, run each training session in a subprocess:

```python
results = ix.variability_study(
    model=build_model,
    data=(X, y),
    runs=20,
    epochs=20,
    use_process_isolation=True,
    gpu_memory_limit=4096,
    seed=42,
)
```

Each run executes in a child process and exits cleanly, releasing all GPU memory before the next run begins.

---

## PyTorch

```python
import torch
import torch.nn as nn
import ictonyx as ix
from ictonyx import PyTorchModelWrapper, ArraysDataHandler

def build_net(config: ModelConfig) -> PyTorchModelWrapper:
    model = nn.Sequential(
        nn.Linear(30, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    return PyTorchModelWrapper(
        model,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': config.get('learning_rate', 0.001)},
        task='classification',
    )

# Pass arrays directly — Ictonyx handles splitting
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.int64)

results = ix.variability_study(
    model=build_net,
    data=ArraysDataHandler(X, y, val_split=0.2, test_split=0.1),
    runs=20,
    seed=42,
)

ix.plot_variability_summary(results=results, metric='accuracy')


```
![Variability summary for PyTorch classifier across 20 runs](images/pytorch_variability.png)


---

## Working with results

```python
# Full distribution of any metric across runs
results.get_metric_values('val_accuracy')       # List[float]

# Per-epoch statistics across all runs
results.get_epoch_statistics('val_accuracy')    # DataFrame: epoch, mean, sd, se, ci_lower, ci_upper

# All per-run, per-epoch DataFrames
results.all_runs_metrics                        # List[pd.DataFrame]

# Seed for exact reproducibility
results.seed
```

---

## Examples

The `examples/` directory contains Jupyter notebooks:

- `quickstart.ipynb` — wine dataset variability study and three-model comparison using Keras and sklearn. No GPU required for the comparison sections.
- `01_mnist_variability_study.ipynb` — deep dive into Keras CNN variability on MNIST with full visualisation
- `02_mnist_model_comparison.ipynb` — comparing two CNN architectures statistically
- `03_learning_rate_variability.ipynb` — hyperparameter sweep across learning rates and batch sizes using `run_grid_study()`
- `04_pytorch_classification.ipynb` — PyTorch classification variability study with epoch-level diagnostics
- `05_pytorch_regression.ipynb` — PyTorch regression variability study with known ground truth
- `06_sklearn_models.ipynb` — sklearn classification and regression: single-model variability and multi-model comparison

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use Ictonyx in published work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff), or use the **Cite this repository** button
on the GitHub repository page.

```bibtex
@software{kizlik_ictonyx,
  author  = {Kizlik, Stephen},
  title   = {Ictonyx: A Framework for Variability Analysis in Machine Learning Training},
  version = {0.4.2},
  url     = {https://github.com/skizlik/ictonyx},
  license = {MIT},
}
```
