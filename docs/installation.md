# Installation

## Basic
```bash
pip install ictonyx
```

## Optional extras
```bash
pip install ictonyx[sklearn]      # scikit-learn model wrappers
pip install ictonyx[torch]        # PyTorch model wrapper
pip install ictonyx[tensorflow]   # Keras/TF model wrapper
pip install ictonyx[huggingface]  # Transformer fine-tuning
pip install ictonyx[mlflow]       # MLflow run logging
pip install ictonyx[explain]      # SHAP feature importance
pip install ictonyx[tuning]       # Optuna hyperparameter tuning
pip install ictonyx[all]          # All supported extras
```

## Requirements
- Python ≥ 3.10
- numpy ≥ 1.24, pandas ≥ 2.0, scipy ≥ 1.10
- matplotlib ≥ 3.7, seaborn ≥ 0.13
