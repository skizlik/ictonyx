# Changelog

All notable changes to Ictonyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Sphinx documentation hosted on ReadTheDocs
- Parallel execution for non-GPU models via `joblib`
- `VariabilityStudyResults.bootstrap_ci()` convenience method
- `VariabilityStudyResults.report()` for self-contained summaries
- Paired/blocked experimental designs for model comparison


---

## [0.3.7] - 2026-03-17

### Fixed
- `ScikitLearnModelWrapper.assess()` raised `AttributeError` on any
  regression `predict → assess` call because `self.task` was never set
  in `__init__`. Fixed by adding `task: Optional[str] = None` to the
  constructor and assigning `self.task` at the end of `fit()` using the
  `is_classifier` flag already computed there.
- `HyperparameterTuner` silently returned the worst model when
  optimising on `r2`, `f1`, or `auc` because those metrics were absent
  from the negation condition (Hyperopt minimises). Fixed in all three
  locations: the `objective` function, the best-loss conversion in
  `tune()`, and `get_best_params()`.
- `logger.addHandler()` was called unconditionally at import time in
  `settings.py`, causing every ictonyx message to be emitted twice in
  any environment with an already-configured root logger (Jupyter,
  applications using `logging.basicConfig()`). Handler addition is now
  guarded by `if not logger.handlers`; `logger.propagate` set to
  `False`.
- `VariabilityStudyResults.compare_models_statistically()` passed a
  full per-epoch `pd.Series` as each observation to
  `compare_multiple_models()`. With `n_epochs > 1`, statistical tests
  received `n_runs × n_epochs` observations instead of `n_runs`,
  inflating the effective sample size and producing incorrect p-values.
  Fixed by reading directly from `self.final_metrics`, which stores
  exactly one scalar per run.
- `ExperimentRunner` seeded each run with `self.seed + run_id`,
  producing consecutive integers (e.g. 42, 43, 44). Many RNG
  implementations exhibit correlation between adjacent seeds,
  introducing systematic bias across runs. Replaced with
  `np.random.SeedSequence`, which uses a hash-based spawning algorithm
  to produce independent, uncorrelated child states. Child seeds are
  generated once at the start of `run_study()` and used in both the
  standard and process-isolated execution paths.
- `ScikitLearnModelWrapper.fit()` included fabricated `'loss'` and
  `'val_loss'` keys (`1 - accuracy`) in classifier training history.
  `1 - accuracy` is not a valid loss value, cannot be meaningfully
  compared to Keras `val_loss`, and produces values outside `[0, 1]`
  for regressors where `score()` returns R². Both keys removed.
  Classifier history now contains only `accuracy` and `val_accuracy`;
  regressor history unchanged (`r2`, `val_r2`).
- `assess_training_stability()` silently reported `converged=False` for
  every sklearn run after the loss key removal, because the convergence
  check looked only for `'loss'` and `'val_loss'`. The fallback now
  checks `'train_loss'`, `'val_loss'`, `'val_accuracy'`, and `'r2'` in
  order before giving up.

### Changed
- `ModelConfig.for_variability_study()` default `num_runs` corrected
  from `5` to `10` to match the documented behaviour. This is a
  **behavioural change**: studies created with
  `for_variability_study(base_config)` without an explicit `num_runs`
  argument will now run 10 times instead of 5.

### Added
- `tests/test_tuning.py` — first direct test coverage for
  `HyperparameterTuner`, including a regression test confirming `r2`
  is negated correctly (bug fixed above). Skipped automatically when
  `hyperopt` is not installed.
- `tests/test_integration.py` expanded from 3 tests to cover the full
  sklearn classifier and regressor pipelines end-to-end, two-model
  comparison via `compare_models()`, `results.to_dataframe()`,
  save/load roundtrip, and statistical stability analysis on real
  runner output.

---

## [0.3.6] - 2026-03-17

### Added
- `KerasModelWrapper` now accepts an explicit `task` parameter
  (`'classification'`, `'regression'`, or `None`). When set, task detection
  skips loss-function inspection entirely. Required for custom loss functions,
  uncompiled models, and any loss name not in the built-in recognition list.
  `load_model()` updated to accept and forward `task=` to the loaded wrapper.
- `tqdm` declared as an optional dependency. `pip install ictonyx[progress]`
  now installs it. Previously tqdm was used by `runners.py` for progress bars
  but was undeclared, so `pip install ictonyx[all]` silently omitted it.
  `[progress]` extra added; `tqdm` added to `[all]` extra.
- `set_theme()` exported from the public API — `ictonyx.set_theme()` now works.
  It was present in `settings.py` but missing from `__init__.py` imports and
  `__all__`, causing `AttributeError` on any direct use.
- `tests/conftest.py` — shared fixtures available to all test files without
  explicit import: `small_classification_arrays`, `small_multiclass_arrays`,
  `small_regression_arrays`, `minimal_config`, `tabular_classification_handler`,
  `tabular_regression_handler`.
- New regression tests for previously fixed bugs:
  - `TestScikitLearnWrapperExtended::test_assess_regression` — verifies
    `ScikitLearnModelWrapper.assess()` returns `{'r2','mse','mae'}` for
    regressors and never returns `'accuracy'`.
  - `TestPyTorchRegression::test_regression_assess_returns_full_metrics` —
    same contract for `PyTorchModelWrapper`.
  - `TestPyTorchUtilities::test_load_model_without_architecture_raises` —
    verifies `PyTorchModelWrapper.load_model(path)` raises `ValueError` with
    a descriptive message when `model=` is omitted.

### Fixed
- **`ScikitLearnModelWrapper.assess()` returned wrong metrics for regressors.**
  Always called `accuracy_score()` regardless of task type, producing 0.0 for
  regression models with no error or warning. Now uses the same
  classifier/regressor heuristic as `fit()` and `evaluate()`: classifiers get
  `{'accuracy'}`, regressors get `{'r2', 'mse', 'mae'}` via pure NumPy.
- **`KerasModelWrapper.assess()` had the same regression bug.** Fixed with
  identical approach, routing through `_is_classification_model()`.
- **`PyTorchModelWrapper.assess()` returned only `{'mse'}` for regression.**
  Inconsistent with `ScikitLearnModelWrapper` (which returns `{'r2','mse','mae'}`),
  silently causing `KeyError` when callers read `result['r2']` or `result['mae']`
  on a PyTorch regressor. Now returns `{'r2', 'mse', 'mae'}` using the same
  pure-NumPy formula across all three wrappers.
- **`KerasModelWrapper._is_classification_model()` silently returned `True` for
  unknown losses.** Uncompiled models, custom loss functions, and inspection
  exceptions all fell through to `return True`, treating regression models as
  classifiers and producing wrong predictions and metrics with no indication of
  the problem. Now raises `ValueError` with actionable guidance in all three
  cases, directing the user to set `task=` explicitly.
- **`PyTorchModelWrapper.load_model()` violated the Liskov Substitution
  Principle.** The base class signature is `load_model(cls, path: str)`. The
  PyTorch override required a non-defaulted `model: nn.Module` argument, making
  it impossible to call through the base class interface. `model` is now
  `Optional[nn.Module] = None`; passing `None` raises `ValueError` explaining
  the PyTorch state-dict constraint with a usage example.
- **`set_theme()` silently ignored unknown theme names.** A typo like
  `set_theme("pubication")` was a no-op with no error or warning. Now raises
  `ValueError` naming the invalid theme and listing all valid options
  (`'default'`, `'dark'`, `'publication'`). A `'default'` branch added to
  restore the original palette, backed by a `_DEFAULT_THEME` constant.
- **`HyperparameterTuner.tune()` accessed raw Keras history API.** The
  objective function assigned the return value of `wrapped_model.fit()` to
  `history` and then read `history.history` — treating it as a Keras `History`
  object. `fit()` returns `None` for all three wrapper types. The `hasattr`
  guard caught this and raised a generic `ValueError` on every non-Keras trial,
  making `HyperparameterTuner` effectively Keras-only. Now calls `fit()` for
  its side effect, then reads `wrapped_model.training_result.history`, which
  works for all wrappers.
- **PyTorch model detection in `api.py` used a fragile string heuristic.**
  Both `_build_instance_cloner()` and `_ensure_wrapper()` detected PyTorch
  models with `"torch" in str(type(obj).__mro__)` — a substring match on a
  stringified list of type objects. Could false-positive on any class whose
  module path contains "torch". Replaced with
  `PYTORCH_AVAILABLE and isinstance(obj, _torch_nn.Module)`, the same pattern
  used for sklearn. `torch.nn` imported once at module level as `_torch_nn`.
- **`test_unknown_theme_no_change` asserted old broken behavior and caused
  five consecutive CI failures.** The test called `set_theme("nonexistent_theme")`
  with no `pytest.raises` guard and asserted THEME was unchanged — correct for
  the old silent no-op, wrong after the `ValueError` fix. Replaced with
  `test_unknown_theme_raises_value_error`. Full `TestSetTheme` class rewritten:
  `teardown_method` now calls `set_theme("default")` instead of hardcoding
  palette values; three new tests added covering the `'default'` reset branch,
  THEME immutability on error, and error message content.
- **`check_normality()` had a misleading inline comment.** Comment read "Consider
  normal if *any* test fails to reject normality" but the code used `all()`,
  which requires *every* test to agree. The code is correct. Comment updated to
  accurately describe the conservative `all()` semantics and explain why that
  choice is appropriate for the small-sample ML regime.

### Changed
- `KerasModelWrapper._is_classification_model()` restructured for clarity. The
  `try` block now covers only the attribute access steps that can raise
  `AttributeError` or `TypeError`. Indicator list matching moved outside the
  `try` block; the final `raise ValueError` for unknown losses is now plainly
  outside any exception handler, making the control flow unambiguous.
- `settings.py` gains two module-level constants: `_DEFAULT_THEME` (the
  original colour palette, used by `set_theme('default')` to restore defaults)
  and `_VALID_THEMES` (used in `ValueError` messages).

### Removed
- "Regression task support (MSE, MAE, R² as first-class metrics)" removed from
  `[Unreleased]` planned list — fully delivered in this release across all three
  wrappers (`assess()`) and in `ScikitLearnModelWrapper.evaluate()`.

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
