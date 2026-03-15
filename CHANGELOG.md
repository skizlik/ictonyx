# Changelog

All notable changes to Ictonyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Regression task support (MSE, MAE, R² as first-class metrics)
- Sphinx documentation hosted on ReadTheDocs
- Parallel execution for non-GPU models via `joblib`
- `VariabilityStudyResults.bootstrap_ci()` convenience method
- `VariabilityStudyResults.report()` for self-contained summaries
- Paired/blocked experimental designs for model comparison

---

## [0.3.5] - 2026-03-15

### Added
- `ModelComparisonResults` dataclass in `analysis.py` — typed return object for
  `compare_models()`, replacing the previous untyped `Dict[str, Any]`. Includes
  `is_significant()` and `get_summary()` convenience methods.
- `seed` parameter to `compare_models()` — all models in a comparison now use
  the same base seed, making comparisons reproducible. If `None`, a seed is
  generated and used consistently across all models.
- `cloudpickle` and `psutil` declared as optional `[isolation]` extra in
  `pyproject.toml`. Both were silently used in `memory.py` but undeclared,
  causing degraded process isolation behavior on clean installs.
- Python 3.13 classifier added to `pyproject.toml` (already tested in CI).

### Fixed
- `compare_models()` now raises `ValueError` (with a descriptive message) when
  fewer than 2 models produce valid results, replacing a silent error-dict return.
- `compare_models()` `raw_data` field was documented in the return dict but
  silently absent; now populated correctly in `ModelComparisonResults`.
- `stop_on_failure_rate` default in `ExperimentRunner.run_study()` corrected
  from `0.5` to `0.8` to match the documented value in the class docstring.
- `check_independence()` accepted an `alpha` parameter but ignored it, hardcoding
  `1.96` as the critical value. Now correctly derives the critical value via
  `norm.ppf(1 - alpha / 2)`.
- `check_independence()` docstring listed a phantom `threshold` parameter that
  did not exist in the function signature; removed.
- `ScikitLearnModelWrapper.fit()` kwarg allowlist contained `X_idx_sorted` and
  `check_input`, both removed from scikit-learn in v1.0 (2021). Only
  `sample_weight` remains.
- `utils.py` imported `sklearn.model_selection.train_test_split` at module level,
  causing `import ictonyx` to fail on installs without scikit-learn despite
  sklearn being optional. Import moved inside `train_val_test_split()`.
- `test_compare_models_insufficient_data` updated to match new `ValueError`
  behavior (previously asserted on a silent error-dict return).
- Dead `gc` re-imports removed from `KerasModelWrapper` and
  `PyTorchModelWrapper` cleanup methods (`gc` already imported at module level).
- Dead `_cleanup_model_references()` and `_cleanup_tensorflow_session()` methods
  removed from `KerasModelWrapper` (defined but never called; logic already
  present in `_cleanup_implementation()`).
- Dead `_check_matplotlib()` and `_check_seaborn()` alias functions removed from
  `plotting.py`; all call sites now use `_check_plotting()` directly. Removed
  associated `# FIX THIS` comment.
- Unnecessary `f`-string prefix removed from logger calls with no interpolation
  in `runners.py`, `explainers.py`, and `tuning.py`.

### Changed
- `requirements.txt` removed. It incorrectly listed all optional ML frameworks
  (TensorFlow, PyTorch, MLflow, SHAP) as mandatory dependencies, causing 1.5–3GB
  of unwanted installs for users who only need sklearn support. Dev environment
  setup instructions moved to `CONTRIBUTING.md`.
- `CONTRIBUTING.md` expanded with tiered dev setup: core-only install vs full
  ML environment, with explicit install commands for each.
- README corrected: version badge showed 0.3.3 (was 0.3.4).
- README `compare_models()` example updated to use `results.get_summary()`
  reflecting the new `ModelComparisonResults` return type.

## [0.3.4] - 2026-03-13

### Added
- `run_grid_study()` — runs a full variability study across a Cartesian
  product of parameter configurations. Each configuration executes via
  `run_variability_study()` with process isolation by default, preventing
  CUDA memory accumulation across configurations. Supports `dry_run=True`
  for execution planning before committing to long runs.
- `GridStudyResults` dataclass — holds one `VariabilityStudyResults` per
  configuration with summary methods: `to_dataframe()`, `summarize()`,
  `get_results_for_config()`, `list_configurations()`.
- Both names exported from `ictonyx` public API.
- 16 new tests in `test_runners.py`. runners.py coverage 58% → 72%.

## [0.3.3] - 2026-02-17

### Fixed
- `ScikitLearnModelWrapper.fit()` now reports `r2`/`val_r2` for regressors
  instead of mislabeling R² as `accuracy`/`val_accuracy`. Classifier
  history keys are unchanged.


## [0.3.2] - 2026-02-15

### Added
- tqdm progress bars for variability studies (optional dependency, graceful fallback)
- Pre-commit hooks for black and isort

### Changed
- scikit-learn is now an optional dependency; install with `pip install ictonyx[sklearn]`
- Linting (black, isort, flake8) now enforced in CI — previously ran but did not fail builds
- Rewrote CONTRIBUTING.md with development setup, style guide, and project map

### Fixed
- `auto_resolve_handler` crashed when passing (X, y) tuples due to extra kwargs
- `PyTorchModelWrapper.load_model()` returned a raw dict instead of a wrapper
- `_get_model_builder` leaked fitted state between runs when given sklearn instances
- Off-by-one in failure rate calculation in `ExperimentRunner.run_study()`
- Undefined name `pd` in `tuning.py`
- Unused `global THEME` declaration in `settings.py`
- Seaborn `FutureWarning` in comparison boxplots (compatible with v0.14)
- Pinned black version in CI to prevent formatting drift

---

## [0.3.1] - 2026-02-14

### Fixed
- `auto_resolve_handler` crashed when extra kwargs (e.g. `batch_size`) were
  passed with tuple data, due to `**kwargs` forwarded to `ArraysDataHandler`
- `PyTorchModelWrapper.load_model()` now returns a fully reconstructed
  wrapper instead of a raw checkpoint dict, matching the `BaseModelWrapper`
  contract
- `_get_model_builder` now clones sklearn instances per run via
  `sklearn.base.clone()`, preventing fitted weights from leaking between
  runs and silently corrupting variability studies
- Off-by-one in `ExperimentRunner.run_study()` failure rate calculation
- Documented `compare_models` data handler reuse for identical splits

## [0.3.0] - 2026-02-13

### Added
- `PyTorchModelWrapper` for PyTorch `nn.Module` models with built-in training
  loop, device management, and metric tracking for classification and regression
- `PYTORCH_AVAILABLE` flag for conditional import detection
- Auto-wrapping support for `nn.Module` in the high-level `variability_study()` API
- PyTorch CPU added to CI test matrix (all platforms)
- Two example notebooks demonstraing basic PyTorch use

### Changed
- `core.py` now conditionally imports PyTorch alongside TensorFlow and sklearn
- `validate_imports.py` checks `PyTorchModelWrapper` when torch is available

---

## [0.2.0] - 2026-02-12

### Breaking Changes
- **Metric-agnostic pipeline**: `ExperimentRunner` now tracks all metrics per run
  in `final_metrics: Dict[str, List[float]]`, replacing the hardcoded
  `final_val_accuracies: List[float]`. Code that accessed `final_val_accuracies`
  directly must switch to `final_metrics['val_accuracy']` or use
  `VariabilityStudyResults.get_metric_values('val_accuracy')`.
- **`TrainingResult` dataclass**: All model wrappers now set `self.training_result`
  (a `TrainingResult` with a `.history` dict) instead of `self.history`.
  `MockHistory` is deleted.
- **`run_study()` return type**: Returns `VariabilityStudyResults` directly instead
  of an unnamed `(List, List, List)` tuple.

### Added
- `TrainingResult` dataclass as the universal training output for all wrappers
- `VariabilityStudyResults.get_metric_values()` for direct metric access
- `ScikitLearnModelWrapper.evaluate()` now returns precision, recall, and F1
  in addition to accuracy; regression models get R², MSE, and MAE
- Python 3.12 added to CI test matrix and PyPI classifiers

### Changed
- Replaced all `print()` calls with structured `logger` output across
  `data.py`, `explainers.py`, `memory.py`, and `tuning.py`
- Replaced bare `except:` clauses with `except Exception:` in `memory.py`
- Bumped seaborn dependency to `^0.13.0`

### Fixed
- `ExperimentRunner.run_study()` now resets all accumulators at the start of
  each call; previously, calling `run_study()` twice on the same runner silently
  appended to existing results

---

## [0.1.0] - 2026-02-12

First release to PyPI.

### Added
- **Bootstrap confidence intervals** — new `ictonyx.bootstrap` module with:
  - `bootstrap_ci()`: general-purpose engine for arbitrary statistics
  - `bootstrap_mean_difference_ci()`: CI for difference in means between groups
  - `bootstrap_effect_size_ci()`: CI for Cohen's d
  - `bootstrap_paired_difference_ci()`: CI for paired comparisons
  - Both BCa (bias-corrected and accelerated) and percentile methods
- `compare_two_models()` now automatically computes 95% BCa confidence intervals
  for both the mean difference and Cohen's d effect size
- New fields on `StatisticalTestResult`: `confidence_interval`, `ci_confidence_level`,
  `ci_method`, `ci_effect_size`
- CI data included in `get_summary()`, `generate_statistical_summary()`,
  and `create_results_dataframe()` output
- 52 new bootstrap tests, 7 CI integration tests
- Expanded `analysis.py` test coverage from ~42% to 84%

### Fixed
- `mann_whitney_test` and `wilcoxon_signed_rank_test` silently overwrote
  `sample_sizes` and `assumptions_met` when constructing results
- `anova_test` and `shapiro_wilk_test` crashed on every call due to missing
  required `statistic` and `p_value` arguments in `StatisticalTestResult` constructor
- Indentation bug in `wilcoxon_signed_rank_test` that misaligned warning logic
- Misleading inline comment in `mann_whitney_test` that described wrong test direction
- `ScikitLearnModelWrapper` mock loss history produced constant values instead of
  realistic decaying loss curves

---

## [0.0.5] - 2026-01-29

### Added
- High-level API (`variability_study`, `compare_models`) for single-function usage
- `VariabilityStudyResults` object with `.summarize()` and `.get_final_metrics()` methods
- Process isolation for GPU memory management (`use_process_isolation=True`)
- Statistical comparison functions with automatic test selection
- Effect size calculations (Cohen's d, rank-biserial correlation, eta-squared)
- Multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg)
- Forest plot visualization for effect sizes with confidence intervals
- Comparison boxplot visualization
- Docker GPU environment (`build-gpu.sh`, `run-gpu.sh`, `test-gpu.sh`)

### Changed
- Refactored model wrappers with abstract base class (`BaseModelWrapper`)
- Improved scikit-learn compatibility with mock history objects
- Enhanced error messages with custom exception hierarchy

### Fixed
- GPU memory leaks in repeated Keras training runs
- Matplotlib backend issues in CI/CD environments (headless mode)

---

## [0.0.4] - 2025-12

### Added
- CI/CD pipeline with GitHub Actions
- Cross-platform testing (Ubuntu, macOS, Windows)
- Coverage reporting with Codecov integration
- pytest configuration with coverage thresholds

---

## [0.0.3] - 2025-11

### Added
- `TabularDataHandler` for CSV and DataFrame inputs
- `ImageDataHandler` for image classification tasks
- `TextDataHandler` for NLP datasets
- `TimeSeriesDataHandler` for sequential data
- `auto_resolve_handler()` for automatic data format detection

---

## [0.0.2] - 2025-10

### Added
- `KerasModelWrapper` for TensorFlow/Keras models
- `ScikitLearnModelWrapper` for sklearn estimators
- Basic plotting functions:
  - `plot_training_history()`
  - `plot_confusion_matrix()`
  - `plot_roc_curve()`
  - `plot_precision_recall_curve()`

---

## [0.0.1] - 2025-09

### Added
- Initial release
- Core `ExperimentRunner` functionality
- Basic variability study capability
- `ModelConfig` for hyperparameter management
- `BaseLogger` for experiment tracking
- MLflow integration (`MLflowLogger`)
