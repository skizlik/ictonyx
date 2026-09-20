# Contributing to Ictonyx

Thanks for considering contributing. Here's how to get started.

## Development Setup

Fork and clone the repo, then install in dev mode:

    git clone https://github.com/skizlik/ictonyx.git
    cd ictonyx

Choose your install depth depending on what you're working on:

**Core only** (statistics, analysis, sklearn — sufficient for most contributions):

    pip install -e ".[sklearn]"
    pip install pytest pytest-cov black isort flake8 pre-commit

**Full ML environment** (required if working on Keras, PyTorch, MLflow, or SHAP):

    pip install -e ".[sklearn]"
    pip install tensorflow
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install mlflow shap hyperopt jupyterlab tqdm cloudpickle psutil

Then set up pre-commit hooks:

    pre-commit install
    pytest tests/ -v

The `pre-commit install` step sets up git hooks that run black and isort automatically on every commit.

## Running Tests

The default run excludes the Monte-Carlo power tests, which take several
minutes:

    pytest -m "not slow"          # ~5-7 min on a bare sklearn environment

Run the slow tests separately (CI does this on one cell):

    pytest -m slow

A single file, or a subset that avoids subprocess-heavy tests:

    pytest tests/test_analysis.py
    pytest tests/test_api.py -k "not isolation and not parallel"

Coverage is measured in CI; locally it is opt-in:

    pytest -m "not slow" --cov=ictonyx --cov-report=term

Notes:

- `.flake8` sets the line length to 100 to match black. Run
  `flake8 ictonyx --select=E,W` for style; the `F` codes are a known backlog.
- Tests that need an optional framework (TensorFlow, PyTorch, transformers)
  are `skipif`-guarded and run on the Linux/Windows CI cells. macOS is the
  bare sklearn cell.
- `tests/test_tuning.py` requires optuna; hyperopt-specific tests in it
  skip without hyperopt.
- Process-isolation tests spawn subprocesses. If your machine is
  memory-constrained, close other heavy applications before a full run.

## Code Style

This project uses **black** for formatting and **isort** for import sorting. If you installed the pre-commit hooks, this happens automatically. To run manually:

    black ictonyx/ --line-length 100
    isort ictonyx/ --profile black --line-length 100

Flake8 runs in CI for syntax errors and undefined names:

    flake8 ictonyx/ --select=E9,F63,F7,F82 --show-source

## Pull Request Checklist

Before opening a PR:

1. All tests pass: `pytest tests/ -v`
2. Formatting is clean: `black --check ictonyx/ --line-length 100`
3. No flake8 errors: `flake8 ictonyx/ --select=E9,F63,F7,F82`
4. New features have tests
5. Docstrings follow the existing style (Google-style Args/Returns/Raises)

## Project Structure

    ictonyx/
        api.py          # High-level variability_study() and compare_models()
        runners.py      # ExperimentRunner, the core engine
        core.py         # Model wrappers (Keras, sklearn, PyTorch)
        data.py         # DataHandler hierarchy
        analysis.py     # Statistical tests and comparisons
        bootstrap.py    # Bootstrap confidence intervals
        plotting.py     # All visualization functions
        config.py       # ModelConfig
        settings.py     # Global settings (verbosity, themes)
        loggers.py      # BaseLogger, MLflowLogger
        memory.py       # GPU memory management
        exceptions.py   # Custom exception hierarchy

## Where to Start

Good first contributions:

- **Improve test coverage.** Several modules are below 50%. Pick one, read the code, write tests. `data.py` (38%) and `core.py` (59%) are the biggest gaps.
- **Add a regression example notebook.** The library supports regression but has no example showing it end to end.
- **Improve error messages.** Find a `raise` statement with a vague message and make it specific.

## Questions?

Open an issue on GitHub. There are no dumb questions.
