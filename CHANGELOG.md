# Changelog

All notable changes to Ictonyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- 5×2cv paired t-test for classifier comparison
- PyTorch model wrapper
- Regression task support
- Sphinx documentation hosted on ReadTheDocs

---

## [0.1.0] - 2026-02-12

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
