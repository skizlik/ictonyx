# Ictonyx

**Iteration Comparison Testing Over N-runs: Yield eXamination**

A Python framework for assessing machine learning model variability and performing rigorous statistical comparisons.

![CI/CD](https://github.com/skizlik/ictonyx/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## The Problem

Machine learning models are typically trained once on a given data set. We use a variety of metrics to assess and compare these models - but generally, on one training run.

But training a model involves stochastic factors: random initialization, data shuffling, dropout. Training the same model on the same data will often produce different predicted values and different assessment metrics. Our model parameters aren't constants; they are random variables, and we should treat them that way.

Ictonyx runs your model multiple times and provides complete distributions for your model metrics, along with measures of center and dispersion. This allows inferences to be made about the models we're training.

## Installation

```bash
pip install ictonyx
```

## Quick Start: sklearn

```python
import ictonyx as ix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

results = ix.variability_study(
    model=RandomForestClassifier,
    data=df,
    target_column='target',
    runs=20
)

print(results.summarize())
```

Output:
```
Study Summary:
  Successful runs: 20/20
  Val accuracy: 0.8800 (SD = 0.0267)
```

## Comparing Two Models

```python
import ictonyx as ix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

comparison = ix.compare_models(
    models=[RandomForestClassifier, GradientBoostingClassifier],
    data=df,
    target_column='target',
    runs=20
)

print(comparison['overall_test'].conclusion)
```

Output:
```
Kruskal-Wallis test indicates significant differences between group 
distributions (H=26.000, p=0.0000) with large effect size (epsilon-squared=0.658)
```

## Deep Learning: MNIST Example with Visualization

The library was originally built for TensorFlow/Keras workflows. This example trains a CNN 10 times and plots the variability:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Input

import ictonyx as ix
from ictonyx import (
    ModelConfig,
    KerasModelWrapper,
    ArraysDataHandler,
    run_variability_study,
    plot_variability_summary
)

# Load MNIST
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_train = X_train[..., np.newaxis]

# Use subset for speed
X = X_train[:5000]
y = y_train[:5000]

def create_cnn(config: ModelConfig) -> KerasModelWrapper:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return KerasModelWrapper(model)

config = ModelConfig({'epochs': 10, 'batch_size': 64, 'verbose': 0})
data_handler = ArraysDataHandler(X, y)

results = run_variability_study(
    model_builder=create_cnn,
    data_handler=data_handler,
    model_config=config,
    num_runs=10
)

print(results.summarize())

plot_variability_summary(
    all_runs_metrics_list=results.all_runs_metrics,
    final_metrics_series=results.final_val_accuracies,
    metric='accuracy'
)
```

## GPU Memory Isolation

Training Keras models in a loop leaks GPU memory. Ictonyx can run each training session in an isolated subprocess:

```python
results = run_variability_study(
    model_builder=create_cnn,
    data_handler=data_handler,
    model_config=config,
    num_runs=10,
    use_process_isolation=True,
    gpu_memory_limit=4096
)
```

## What It Does

- Runs N training iterations of the same model
- Computes mean, standard deviation, min, max of your chosen metric
- Performs statistical tests (Mann-Whitney, Kruskal-Wallis) to compare models
- Reports effect sizes so you know if differences are practically significant
- Visualizes training curves and metric distributions

## License

MIT
