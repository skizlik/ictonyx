# Changelog

All notable changes to Ictonyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- PyPI publication
- Sphinx documentation hosted on ReadTheDocs
- Additional real-world dataset examples

---

## [0.0.5] - 2025-01-29

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
