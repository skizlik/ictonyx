# Changelog

All notable changes to Ictonyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- `VariabilityStudyResults.report()` for self-contained HTML/markdown summaries
- `VariabilityStudyResults.bootstrap_ci()` convenience method
- Infrastructure sweep
- `PyTorchDataHandler`
- Paired/blocked experimental designs for model comparison
- `plot_run_independence_diagnostics`, `plot_paired_deltas`,
  `plot_epoch_run_heatmap`, `plot_sequential_ci`, `plot_stability_pareto`,
  `plot_run_metric_correlations`
- Migration Guide (prerequisite for v0.5.0 deprecated API removal)

---

## [0.4.5] — 2026-04-18

### Fixed

- `compare_two_models()` was attaching a Cohen's d bootstrap confidence
  interval to paired Wilcoxon results. The two estimators have
  incompatible scales: Wilcoxon's `r = |z|/sqrt(n)` is bounded by 1.0,
  Cohen's d is unbounded. On large-effect paired data the CI could
  exclude the point estimate. Paired Wilcoxon now returns
  `ci_effect_size=None`; a Wilcoxon-r bootstrap CI is planned for v0.4.9.
- `compare_models()` now routes pairwise comparisons through
  `compare_two_models()`, surfacing the bootstrap confidence intervals
  that the analytic core was already computing. Previously, the public
  API silently dropped `ci_effect_size` and `confidence_interval`.
  Multiple-comparison corrections continue to apply to p-values only.
- `_build_from_class()` now forwards kwargs from ModelConfig to wrapper
  constructors. Previously, user-supplied constructor arguments
  (including required ones like `model_name_or_path` for
  `HuggingFaceModelWrapper`) were silently dropped, causing every run
  to fail.
- `paired_wilcoxon_test` emits `UserWarning` and returns
  `conclusion="inconclusive"` when `std(differences) < 1e-10`.
  Previously, deterministic model pairs produced spurious p-values. The
  tolerance is exposed via the `deterministic_tol` parameter.

### Added

- `VariabilityStudyResults.save()` now includes a `_schema_version`
  field. `load()` warns via `UserWarning` on missing or mismatched
  schema versions.
- `VariabilityStudyResults.run_seeds: List[int]` captures the per-run
  child seeds spawned via `SeedSequence.spawn()`, enabling exact
  per-run reproduction of a study.

### Tests

- Several pre-existing paired-comparison tests used constant-offset
  data, which the new deterministic-pair guardrail correctly rejects.
  Test inputs updated to use genuine paired variation. One integration
  test (iris + sklearn) was rewritten to verify the integration path
  rather than assert significance on a mathematically undefined
  comparison.

### Documentation

- New README section *What variability does Ictonyx measure?* clarifies
  that the library characterizes training-stochastic variance under a
  fixed data split, not sampling variance.

---

## [0.4.4] — 2026-04-18

### Added
- `HuggingFaceModelWrapper` — text classification via `transformers.Trainer`.
  Accepts `(List[str], List[int])` or `datasets.Dataset`. Seed-controlled via
  `transformers.set_seed()` and `TrainingArguments(seed=..., data_seed=...)`.
  Temporary checkpoint directory cleaned up after each run to prevent disk
  accumulation. `pip install ictonyx[huggingface]`
- `plot_rank_correlation_over_epoch(results, metric, threshold, window, ax, show)` —
  Spearman correlation between epoch-k and final-epoch run rankings over training.
  Annotates the first epoch at which the smoothed correlation exceeds `threshold`.
  Requires `n_runs >= 15`.
- Sphinx documentation site with ReadTheDocs integration at
  `https://ictonyx.readthedocs.io`. API reference pages for all 13 modules.

### Fixed
- `PyTorchModelWrapper` regression history now records `val_mse`, `val_r2`,
  `val_mae`, and `val_rmse` per epoch via a new
  `_compute_epoch_regression_metrics()` helper. Previously only `val_loss` was
  recorded for regression tasks.
- `_standardize_history_df()` rename map extended to cover `r2 → train_r2`,
  `mse → train_mse`, `rmse → train_rmse`, `mae → train_mae`.
- `failed_runs` count is now persisted to and restored from checkpoint. Previously
  reset to zero on resume, under-counting failures.
- Failure rate denominator now uses attempted runs (successful + failed). A study
  losing 5 of 20 runs previously reported 0% failure rate.
- `assert` guards in public paths replaced with explicit `RuntimeError` or
  `ValueError`, and `# type: ignore[return-value]` where the assert was
  suppressing a mypy false positive. `assert` is silently stripped by `python -O`.
- `KerasModelWrapper.predict_proba()`: `rtol` changed to `atol` in probability
  sum check. Relative tolerance is semantically wrong for checking whether
  probabilities sum to 1.
- `HuggingFaceModelWrapper.fit()`: history buffer lengths now aligned before
  constructing `TrainingResult`. Mismatched lengths (val metrics captured per
  epoch; train loss captured once) caused silent run failures in `variability_study()`.
- `KerasModelWrapper` gains `clear_session: bool = True` constructor parameter.
  When `True`, calls `tf.keras.backend.clear_session()` at the start of each
  `fit()` to release GPU memory from previous runs.
- `summarize()` output now includes N (run count) and SE (standard error) for
  each metric block.
- `stability_weight` now raises `ValueError` at construction time for values
  outside `[0, 1]`.
- `os` moved to top-level import in `core.py`; redundant local imports removed.

### Deprecated
- `wilcoxon_signed_rank_test()` now emits `DeprecationWarning`. Use
  `compare_results()` or `test_against_null()` instead. Will be removed in v0.5.0.

### Compatibility notes
- `TrainingArguments`: `no_cuda` replaced with `use_cpu` (transformers ≥ 4.45).
- `Trainer.__init__()`: `tokenizer` parameter removed (transformers ≥ 4.46).
  Padding is handled entirely by `DataCollatorWithPadding`.
- Test model for CI: `prajjwal/bert-tiny` replaced with
  `google/bert_uncased_L-2_H-128_A-2` which includes a complete `tokenizer.json`.

---

## [0.4.3] — 2026-04-17

### Fixed — Critical correctness

- `VariabilityStudyResults.preferred_metric()`: key-prefix mismatch caused
  the method to always return `"val_accuracy"` even when a held-out test set
  was provided. `final_test_metrics` stores bare keys (`"accuracy"`) but the
  method was checking for `"test_accuracy"`, silently routing
  `test_against_null()` and `compare_results()` to validation metrics for
  every user who provides a test set. `get_test_metric_values()` now accepts
  both bare (`"accuracy"`) and prefixed (`"test_accuracy"`) forms
  transparently.
- `KerasModelWrapper.evaluate()`: no longer crashes with `TypeError` on
  loss-only Keras models. TF 2.x returns a plain `float` when no additional
  metrics are compiled; `dict(zip(metric_names, float))` raised because a
  float is not iterable. A scalar guard wraps the result before zipping.
- `PyTorchModelWrapper.assess()`: inline `from sklearn.metrics import
  accuracy_score` removed. The module-level symbol (which has a numpy
  fallback for sklearn-free environments) is used instead.
- `PyTorchModelWrapper.predict_proba()`: last-layer softmax detection changed
  from `list(self.model.modules())[-1]` to `list(self.model.children())[-1]`.
  The prior implementation performed a depth-first traversal returning the
  deepest leaf of the last sub-branch for non-Sequential architectures
  (ResNets, attention blocks), not the output layer.
- `compare_models()`: metric routing reverted to always default to
  `"val_accuracy"` when `metric=None`. The BUG-39 fix caused auto-routing to
  test metrics whenever `has_test_data` was true, including cases where the
  DataHandler created an internal test split not intended for cross-model
  comparison. Users who want test-set comparison pass `metric="test_accuracy"`
  explicitly.
- `check_independence()` docstring: `Note:` corrected — the function applies
  Bonferroni correction; the stale text claiming it did not was removed.

### Fixed — Plotting correctness

- `plot_pairwise_comparison_matrix()`: no longer crashes on
  `ModelComparisonResults` dataclass; `hasattr`/`isinstance` guards replace
  dict key access.
- `plot_roc_curve()` and `plot_precision_recall_curve()`:
  `_check_tensorflow_utils()` removed; sklearn users no longer blocked by a
  TF import check when a numpy fallback was already present.
- `plot_comparison_forest()`: CI now uses the pre-computed
  `confidence_interval` from `StatisticalTestResult` when available (paired,
  correct). Falls back to Welch unpaired approximation. Guard added to
  prevent unpacking non-sequence `confidence_interval` values (e.g. mocks).
  `ax=` parameter added.
- `plot_pacf_vs_lag()`: CI band no longer subtracted from PACF values; prior
  implementation centred the band at zero instead of at the PACF values.
- `plot_averaged_pacf()`: fabricated CI formula `1.96/sqrt(len(lags)*10)`
  replaced with correct `1.96/sqrt(n_series)`. New `n_series` parameter
  required for CI reference lines. Deprecated; will be removed in v0.5.0.
- `_finalize_plot()`: now returns `None` when showing, preventing Jupyter
  double-render where the returned figure object caused a second inline
  display.

### Changed — Style and infrastructure

- Added 8 new keys to `settings.THEME` and all three theme variants
  (`default`, `dark`, `publication`): `neutral`, `better`, `worse`, `point`,
  `positive`, `negative`, `sequential`, `diverging`.
- `set_figsize_scale(factor)` and `get_figsize(base)` added to `settings.py`
  for global figure size scaling.
- `_apply_style(ax)` helper added to `plotting.py` — single source of truth
  for grid, spines, and tick parameters across all plot functions.
- `import seaborn as sn` corrected to `import seaborn as sns` throughout
  `plotting.py` (10 call sites).
- Seven functions converted from `plt.figure()` + state-machine `plt.*`
  calls to `fig, ax = plt.subplots()` + `ax.*` methods:
  `plot_confusion_matrix`, `plot_roc_curve`, `plot_precision_recall_curve`,
  `plot_autocorr_vs_lag`, `plot_averaged_autocorr`, `plot_pacf_vs_lag`,
  `plot_averaged_pacf`. All now use `settings.get_figsize()` and `dpi=150`.
- Hardcoded colour strings in `plot_training_stability` replaced with
  `settings.THEME` references.
- `ax: Optional[Axes] = None` added to `plot_comparison_boxplots`,
  `plot_autocorr_vs_lag`, `plot_averaged_autocorr`, `plot_pacf_vs_lag`,
  and all three new variability functions.

### Added

- `plot_run_trajectories(results, metric, ax, show)` — overlaid per-run
  epoch trajectories with mean curve.
- `plot_run_distribution(results, metric, ax, show)` — violin plot of final
  metric values with individual run dots overlaid.
- `plot_run_strip(results, metric, ax, show)` — strip plot with mean and
  ±1 SD markers. Best for small n where a violin oversmooths.
- `plot_variability_summary()`: new `kind=` parameter dispatches to the three
  new functions when used with `results=`. Old positional-argument form
  deprecated and will be removed in v0.5.0.
- `plot_comparison_boxplots()`: significance bracket annotations (`*`) drawn
  above boxes for pairs in `significant_comparisons`.

---

## [0.4.2] — 2026-04-15

### Fixed

- `pyproject.toml`: relaxed numpy pin from `^1.24.0` (`>=1.24.0, <2.0.0`)
  to `>=1.24.0`, restoring compatibility with numpy 2.x. The prior cap
  caused `pip install ictonyx` to silently downgrade numpy, breaking shap
  0.46+, rasterio, and any other package requiring numpy 2. Also relaxed
  pandas, scipy, matplotlib, and seaborn to floor-only pins for the same
  reason. The shap optional dependency was similarly relaxed from `^0.43.0`
  to `>=0.43.0` (Poetry's caret notation caps at `<0.44.0` when the major
  version is 0).

---

## [0.4.1] — 2026-04-15

### Fixed — Statistical correctness

- `check_convergence()`: replaced OR-of-heuristics with slope-based linear
  regression test. Prior implementation declared convergence on virtually
  every decreasing loss curve; new test requires the recent window to be
  statistically flat by magnitude or by evidence.
- `rank_biserial_correlation()`: returns `(nan, "undefined")` with
  `RuntimeWarning` on exception instead of silently returning
  `(0.0, "negligible")`.
- `compare_results()`: now uses paired Wilcoxon signed-rank test by default
  (`paired=True`) when both inputs have equal run counts, consistent with
  `compare_models()`. Falls back to Mann-Whitney U with `UserWarning` when
  run counts differ. Pass `paired=False` to use Mann-Whitney U explicitly.
- `assumptions_met.update()` in `compare_two_models()`: replaced with
  `setdefault` so keys already set by the chosen test are not overwritten.
- `None` assumption values in `create_results_dataframe()`: no longer
  collapsed to `False`. A new `assumptions_untestable` column counts
  untestable assumptions.
- `ScikitLearnModelWrapper.evaluate()`: now returns `rmse` alongside `r2`,
  `mse`, and `mae` for regression tasks, consistent with `_regression_metrics()`.
- `PyTorchModelWrapper.assess()`: rewritten. Raises `ValueError` on NaN
  predictions and on `task=None` instead of silently falling through to the
  regression branch. Classification path now uses `accuracy_score` for
  consistency with Keras and sklearn wrappers. Full docstring added.

### Fixed — Checkpoint reliability

- Checkpoint writes are now atomic: data is written to a `.tmp` file and
  renamed into place with `os.replace()`, preventing corruption on interrupted
  runs.
- Checkpoints now include a `_schema_version` field sourced from
  `ictonyx/_version.py`. Loading a checkpoint from a different version emits
  `UserWarning` with guidance to delete and restart.

### Fixed — API consistency

- `run_grid_study()`, `ExperimentRunner.run_study()`,
  `run_variability_study()`: default `num_runs` raised to 20, consistent
  with `variability_study()` and `compare_models()`. `run_grid_study()` now
  emits `UserWarning` when `num_runs < 20`.
- `for_variability_study()` in `ModelConfig`: default `num_runs` raised
  from 10 to 20.
- `compare_results()`: new `seed` parameter threads random state to bootstrap
  CI computation for reproducible results.
- `compare_models()`: now applies `_INFRA_KWARGS` separation before calling
  `auto_resolve_handler()`, preventing model hyperparameters from reaching
  the DataHandler and raising `TypeError`.
- `compare_models()` two-model shortcut: aliased `StatisticalTestResult`
  replaced with `dataclasses.replace()`, preventing mutation of `overall_test`
  from affecting the `pairwise_comparisons` entry.
- `ModelConfig.merge()` and `.has()`: changed from `DeprecationWarning` to
  `UserWarning` so the notice is visible in Jupyter notebooks. Python silences
  `DeprecationWarning` by default in all non-`__main__` contexts; the identical
  fix was applied to `get_final_metrics()` in v0.3.15 but not here.
- `test_against_null()` docstring: corrected `num_runs >= 10` recommendation
  to `num_runs >= 20`.
- `ScikitLearnModelWrapper.load_model()`: now emits `logger.warning()` on
  pickle fallback, matching the security notice already present in `save_model()`.

### Fixed — Core wrappers

- `PyTorchModelWrapper.predict()`: raises `ValueError` on NaN outputs, checked
  on the raw tensor before `torch.max` discards NaN values into integer indices.
- `PyTorchModelWrapper.predict_proba()`: emits `UserWarning` when the model's
  final layer is `nn.Softmax` or `nn.LogSoftmax`.
- `ImageDataHandler`: per-class file listing now uses `sorted()` for consistent
  cross-platform ordering.
- `ImageDataHandler.load()`: emits `UserWarning` about fixed training shuffle
  across runs. Full per-run shuffle fix deferred to v0.5.0.
- All three `assess()` docstrings and `_regression_metrics()` docstring: `rmse`
  added to documented regression return keys.
- `get_epoch_statistics()` docstring: corrected "percentile CI band" to
  "t-based confidence interval on the per-epoch mean across runs".

### Fixed — MLflow and SHAP

- `MLflowLogger.log_model()`: replaced fragile `hasattr` duck-typing with
  `isinstance` checks for Keras, PyTorch, and sklearn model flavour selection.
- `MLflowLogger.get_run_url()` `file://` branch: now respects the
  `MLFLOW_UI_URL` environment variable instead of always returning
  `http://localhost:5000`.
- SHAP background dataset: now sampled randomly via `np.random.default_rng(42)`
  instead of always selecting the first N rows.
- `shap.DeepExplainer`: emits `DeprecationWarning` when SHAP >= 0.45.

### Deprecated

- `HyperparameterTuner.tune()` Hyperopt fallback: now emits
  `DeprecationWarning` when Optuna is not installed. Hyperopt backend will be
  removed in v0.5.0. Install Optuna: `pip install ictonyx[tuning]`.
- `ScikitLearnModelWrapper.fit()` `epochs`, `batch_size`, `verbose` kwargs:
  now emit `DeprecationWarning`. These parameters will raise `TypeError` in
  v0.5.0. sklearn models have always silently ignored them.

### Changed

- `ictonyx/_version.py` added as the single source of truth for the package
  version. `__init__.py` and `runners.py` both import from it directly,
  replacing the prior inline `__version__` string and avoiding stale metadata
  from `importlib.metadata` during development.
- `compare_multiple_models()` removed from `ictonyx.*` public namespace.
  Still callable as `ictonyx.analysis.compare_multiple_models`. Use
  `ix.compare_models()` instead.
- `hyperopt` removed from `all` and `ml` extras in `pyproject.toml`.
- `_SKLEARN_FIT_KWARGS` and `_KNOWN_IGNORED_KWARGS` promoted to class-level
  `frozenset` constants in `ScikitLearnModelWrapper`. Added `eval_set` and
  `cat_features` ahead of v0.5.0 XGBoost/LightGBM wrappers.

### Internal

- Redundant `mannwhitneyu` call eliminated in `mann_whitney_test()` by passing
  the already-computed U statistic to `rank_biserial_correlation()`.
- BCa two-sample bootstrap: added `UserWarning` when group size ratio exceeds
  3:1, complementing the existing `min(n1, n2) < 15` warning.
- `calculate_averaged_autocorr()`: parameter renamed `histories` → `series_list`.
- `kruskal_wallis_test()`: epsilon-squared formula clarified in a comment.
- `get_epoch_statistics()`: emits `UserWarning` when run counts vary across
  epoch positions.
- Redundant local `accuracy_score` import removed from
  `PyTorchModelWrapper.assess()`.
- `run_grid_study()` PEP 604 annotation: `list[int | None]` replaced with
  `List[Optional[int]]` for Python 3.10 runtime compatibility.
- Checkpoint docstring corrected: `checkpoint.joblib` → `checkpoint.pkl`.
- Test coverage for `explainers.py` expanded: SHAP availability guards,
  KernelExplainer path for non-tree models, `DeepExplainer` deprecation
  warning, and out-of-bounds sample index handling.


---

## [0.4.0] — 2026-04-10

### Fixed — Statistical correctness

- `_hedges_g()`: called `_interpret_variance_explained()` instead of
  `_interpret_cohens_d()` when labelling the effect size magnitude.
  g = 0.20 was labelled "large" (eta-squared scale) instead of "small"
  (Cohen's d scale). Numeric value was correct; verbal interpretation
  was wrong for every call site.

- `compare_two_models()` Welch path: the bootstrap confidence interval
  was computed via `bootstrap_effect_size_ci()`, which uses the Cohen's d
  (unpooled variance) formula internally. The point estimate used Hedges'
  g (bias-corrected, pooled). CI and point estimate were for different
  estimators of the same quantity. Fixed by adding `bootstrap_hedges_g_ci()`
  to `bootstrap.py`, which applies the Hedges' J correction factor
  `1 - 3/(4·df - 1)` inside each bootstrap resample. The Welch branch
  in `compare_two_models()` now calls this function.

- `paired_wilcoxon_test()`: effect size r was computed as
  `norm.ppf(p/2) / sqrt(n)` — deriving Z from the p-value via the
  normal quantile. This is invalid when scipy uses exact distribution
  tables at small n, which is the common case. Replaced with the
  W-statistic asymptotic formula (`z = (W - E[W]) / sqrt(Var[W])`,
  `r = |z| / sqrt(n)`), consistent with `wilcoxon_signed_rank_test()`.

### Fixed — API

- `VariabilityStudyResults.summarize()`: removed a spurious t-interval
  95% CI on the mean that appeared below each metric's SD. The interval
  was computed under an unvalidated normality assumption. Removing it
  avoids misleading users who would reasonably interpret it as a
  distribution interval rather than a CI on the population mean estimate.
  SD, Min, and Max remain.

- `compare_models()`: now defaults to `paired=True` for two-model
  comparisons. Both models receive identical per-run seeds by construction
  via `SeedSequence.spawn()`, making runs genuinely paired at the RNG
  level. The prior `paired=False` default discarded this pairing
  information and used a less powerful unpaired test. **Behavioural
  change:** two-model comparisons that previously used Mann-Whitney U
  will now use paired Wilcoxon signed-rank. Pass `paired=False`
  explicitly to restore prior behaviour.

### Fixed — Packaging

- `optuna` added to `[tool.poetry.dependencies]` as an optional
  dependency under the `tuning` extra. Previously declared in
  `[tool.poetry.extras]` only, causing `pip install ictonyx[tuning]`
  to install without optuna and fail at import time.

### Internal

- Regression tests added for all three statistical fixes in
  `test_analysis.py` and `test_bootstrap.py`. The BUG-2 test verifies
  that the Welch-path CI and point estimate are for the same estimator.
  The BUG-3 test verifies that effect size r is consistent between
  `paired_wilcoxon_test()` and `wilcoxon_signed_rank_test()` on the
  same paired data.

---

## [0.3.15] — 2026-03-31

### Fixed — Critical correctness

- `KerasModelWrapper.predict()`: regression branch now returns `self.predictions`
  instead of implicitly returning `None`. Any caller capturing the return value
  of `predict()` on a Keras regression model received `None` silently; `assess()`
  then raised `ValueError: Model has not generated predictions yet` even though
  predictions were computed and stored correctly.
- `PyTorchModelWrapper.predict()`: classification branch now returns
  `self.predictions` instead of implicitly returning `None`. Mirror of the Keras
  regression bug; affected all PyTorch classifiers. The regression branch was
  unaffected.

### Fixed — Statistical correctness

- `check_independence()`: short series (n < max_lag + 2) now returns `None`
  instead of `False` as the independence flag. `False` signals detected
  autocorrelation; `None` signals untestable. `mann_whitney_test()` updated to
  treat `None` as untestable rather than emitting a spurious autocorrelation
  warning on studies with fewer than 7 runs. Return type annotation updated to
  `Optional[bool]`.
- `check_independence()`: SE formula corrected from `1 / sqrt(n)` to
  `1 / sqrt(n - lag)`. The prior formula made the significance threshold too
  tight at higher lags, inflating false-positive autocorrelation detection.
- `calculate_averaged_autocorr()`: standard deviation now uses `ddof=1`
  (Bessel-corrected sample SD). Prior `ddof=0` produced systematically narrow
  error bands in `plot_averaged_autocorr()`.
- `kruskal_wallis_test()`: now performs a Levene equal-variance check and records
  the result in `assumptions_met["equal_variances"]`, mirroring the existing check
  in `anova_test()`. A warning is added when groups have unequal variances, since
  KW can reject H₀ for variance differences rather than location differences.

### Fixed — API and usability

- `variability_study()` and `compare_models()`: default `runs` raised from 10 to
  20. The prior default of 10 triggered the library's own `runs < 20` warning on
  every invocation that accepted the default — a self-contradictory design.
  **Behavioural change:** existing code that relies on the default will now run
  twice as many experiments. Pass `runs=10` explicitly to restore prior behaviour.
- `compare_models()`: docstring for `runs` parameter corrected from stale
  "Default 5" to "Default 20."
- `get_final_metrics()`: deprecation warning changed from `DeprecationWarning`
  to `UserWarning`. Python silences `DeprecationWarning` in all non-`__main__`
  contexts by default, making the warning invisible in notebooks and imported
  modules. Consistent with the identical fix applied to `TextDataHandler` and
  `TimeSeriesDataHandler` in v0.3.14.

### Fixed — Dead code removal

- `analysis.py`, `bootstrap.py`: removed `HAS_SCIPY = True` dead flag, the
  `_check_scipy()` guard function, and all 8 call sites in `analysis.py`. scipy
  is a mandatory hard dependency; the flag was always `True` and the guard could
  never raise. Removed the two dead `if method == "bca" and not HAS_SCIPY`
  branches in `bootstrap.py`.
- `_validate_process_isolation()`: removed a verbatim-duplicate data-size
  serialisation check that ran twice back-to-back. For large datasets this
  serialised training data twice unnecessarily; on failure it emitted the same
  warning twice.

### Fixed — Data handlers

- `TextDataHandler` and `TimeSeriesDataHandler`: now raise `ImportError` at
  construction when the required `tf.keras.preprocessing` APIs are absent (Keras
  3 / TF 2.16+), rather than emitting a warning and proceeding to crash in
  `load()`. Fail-fast behaviour at object construction.
- `TimeSeriesDataHandler`: corrected a copy-paste duplicate in the `ImportError`
  message body where the first line of the error string appeared twice.

### Fixed — Documentation

- `check_normality()`: removed verbatim-duplicate `require_all_tests` parameter
  entry from the docstring Args section.
- `README.md`: all quickstart examples updated from `runs=10` / `num_runs=10` to
  `runs=20` to match the library's own statistical recommendation. All
  `summarize()` output blocks updated from `Std:` to `SD (sample, N-1):` to
  match the v0.3.14 label change. All output blocks re-executed.

### Internal

- `core.py`: added `assert self.predictions is not None` before all four
  `predict()` return sites to satisfy mypy `Optional` narrowing. The assertion
  is true by construction — the preceding assignment guarantees it — and is
  checked at runtime.
- `runners.py`: hoisted `import pickle as _serializer` above the `try/except`
  block in `_validate_process_isolation()` to resolve a flake8 F821 false
  positive caused by assignment inside a `try` block.

---

## [0.3.14] — 2026-03-29

### Fixed — Silent correctness
- `ScikitLearnModelWrapper.evaluate()`: precision, recall, and F1 were
  computed on regression models due to a `try` block outside `if is_classifier:`.
  On regression models this produced nonsensical metrics or swallowed exceptions
  silently. The block is now correctly gated behind `is_classifier`.
- `_build_instance_cloner()`: unknown instance types now raise `ValueError`
  instead of silently reusing the same instance across all runs, which caused
  trained weights to leak between runs and invalidated study independence.
- `_run_single_fit_standard()`: `run_seed` is now injected into `model_config`
  before `model_builder()`, matching isolated mode. Previously, class-based
  sklearn builders received `random_state=None` in standard mode.
- `PyTorchModelWrapper.predict_proba()`: single-output sigmoid binary classifiers
  (output shape `(n, 1)`) now return valid probability pairs via `sigmoid`.
  `softmax` on `(n, 1)` produces all-ones, not probabilities.

### Fixed — PyTorch / GPU
- `ExperimentRunner`: `cudnn.deterministic` and `cudnn.benchmark` are now
  restored to their prior values after each run via a `_deterministic_cudnn()`
  context manager. Previously these were set permanently, silently degrading
  GPU throughput in any PyTorch code running in the same process afterwards.

### Fixed — Statistical correctness
- `paired_wilcoxon_test()`: added `method='auto'`, matching the fix applied
  to `wilcoxon_signed_rank_test()` in v0.3.13. Ensures exact computation at
  small n rather than the normal approximation.
- `compare_models()`: default `runs` corrected to 10, matching
  `variability_study()` and the v0.3.13 CHANGELOG.
- Warning threshold raised from `runs < 10` to `runs < 20` in both
  `variability_study()` and `compare_models()`, matching the `>= 20`
  recommendation stated in the warning message.

### Fixed — Config and process isolation
- `ModelConfig.copy()`: frozen state is now propagated to the copy. Previously
  a frozen config became mutable after copying.
- `_validate_process_isolation()`: now uses `cloudpickle` when available,
  matching the serialiser used during execution. Previously, lambdas and
  notebook-defined functions were incorrectly rejected by validation.

### Fixed — Packaging and output
- `shap`, `hyperopt`, and `joblib` added to pip-installable extras:
  `ictonyx[explain]`, `ictonyx[tuning]`, and `ictonyx[sklearn]` respectively.
- `TextDataHandler` and `TimeSeriesDataHandler`: deprecation warnings changed
  from `DeprecationWarning` (silenced by Python by default outside `__main__`)
  to `UserWarning`, ensuring visibility in notebooks and imported modules.
- `VariabilityStudyResults.summarize()`: SD label clarified to `SD (sample, N-1)`
  to distinguish Bessel-corrected sample SD from population SD.

### Internal
- Removed unreachable `if self.predictions is None` guards from `predict()`
  in all three model wrappers. The branch was dead by construction and the
  `# pragma: no branch` annotation was suppressing coverage tooling.

---

## [0.3.13] — 2026-03-28

### Fixed — Critical (statistical correctness)
- `wilcoxon_signed_rank_test()`: replaced `method='approx'` with `method='auto'`.
  At n=5 the normal approximation produced false positives (p=0.043 vs exact
  p=0.063), reversing significance decisions at α=0.05. Effect size is now
  derived from the p-value via inverse normal CDF, since scipy's `auto` mode
  does not expose `zstatistic` directly.
- `check_convergence()`: corrected `and` to `or` in the final return statement.
  The docstring specified OR semantics; the code used AND, causing plateaued
  training runs to be reported as not converged and deflating `convergence_rate`
  in `assess_training_stability()` output.
- `check_convergence()`: return values now wrapped in `bool()` to prevent
  `numpy.bool_` identity-check failures under NumPy 2.0.

### Fixed — Misleading output
- `StatisticalTestResult.get_summary()`: the corrected p-value is now displayed
  and labelled `p_corr` when a multiple-comparison correction has been applied.
  Previously the raw p-value was shown alongside a marker derived from the
  corrected value, producing contradictory output such as `p=0.0300 ns`.

### Fixed — Adoption
- `pyproject.toml`: ML framework extras are now installable via pip.
  `pip install ictonyx[tensorflow]`, `ictonyx[torch]`, `ictonyx[mlflow]`, and
  `ictonyx[all]` now install their respective dependencies. Previously all ML
  frameworks were in a Poetry dev group, inaccessible through pip extras.
- `variability_study()` and `compare_models()`: default `runs` raised from 5
  to 10. The previous default triggered the library's own `runs < 10` warning
  on every first call.
- `compare_models()`: raises `ConfigurationError` with an actionable message
  when the requested metric is absent from a model's results, instead of
  propagating a bare `KeyError`. Includes a targeted hint when `val_accuracy`
  is requested but only `accuracy` is available.

### Fixed — Stale claims from v0.3.12
- `GridStudyResults.to_dataframe()`: values now stored at full float precision.
  `round(..., 4)` was still applied to all five summary columns.
- `variability_study()` docstring: corrected seed derivation from the stale
  `seed + i` description to `SeedSequence.spawn()`.
- `exceptions.py`: `import datetime` moved to module top; timestamps now use
  UTC via `datetime.timezone.utc`.

### Fixed — Code quality
- `check_normality()`: misplaced comment moved out of unreachable position
  (after an early `return`) to before the line it documents.
- `check_normality()`: added missing `require_all_tests` parameter to the
  docstring Args section.
- `check_independence()`: removed redundant `n = len(data.dropna())`
  recomputation.
- `_ensure_wrapper()`: explanatory comments moved to before the `if` blocks
  they describe, rather than after the preceding `return` statements.
- `ScikitLearnModelWrapper.fit()`: removed inline `import warnings as _warnings`;
  uses the module-top import directly.

### Deprecated
- `TextDataHandler` and `TimeSeriesDataHandler` now emit `DeprecationWarning`
  at construction. Both handlers are inoperable under TF 2.16+ (the required
  version) and will be replaced with framework-agnostic implementations in
  v0.4.0.

### Changed
- README installation section rewritten with per-framework install commands.

---

## [0.3.12] — 2026-03-27

### Fixed — critical
- `variability_study()`: `verbose=False` now correctly suppresses output.
  The parameter was stored in `ModelConfig` but never forwarded to the
  runner — full output was always produced regardless.
- `PyTorchModelWrapper.fit()`: regression validation history no longer
  doubles in length per epoch. `val_loss` was appended twice — once
  unconditionally and once in the regression branch — corrupting every
  PyTorch regression variability study silently.
- `TextDataHandler` and `TimeSeriesDataHandler`: now raise `ImportError`
  with clear guidance when `tf.keras.preprocessing` APIs are unavailable
  (removed in Keras 3 / TF 2.16+). Full framework-agnostic rewrites
  of both handlers are planned for v0.4.0.

### Fixed — reproducibility
- `_set_seeds()`: `torch.backends.cudnn.deterministic = True` and
  `benchmark = False` now set when CUDA is available. GPU PyTorch
  studies were not reproducible across invocations without this.
- `run_grid_study()`: child seeds now derived via `SeedSequence.spawn()`
  for statistical independence, consistent with `run_study()`. Previously
  used `seed + i`, which can produce correlated RNG states.
- sklearn `random_state` now injected at wrapper construction when the
  estimator accepts it, making sklearn studies reproducible with the
  stored seed.
- `compare_models(paired=True)`: raises `ValueError` on unequal run
  counts instead of warning and proceeding into a scipy error.
- `MemoryManager`: warns when `cloudpickle` is absent and process
  isolation falls back to standard pickle, which cannot serialize
  notebook-defined functions or lambdas.
- `allow_memory_growth`: wrapped in try/except with a clear warning
  when TF was already initialized before `MemoryManager` construction.

### Fixed — statistical correctness
- `anova_test()`: emits `UserWarning` when any group has fewer than 30
  samples, noting that normality cannot be assumed at small n and
  suggesting `kruskal_wallis_test()` instead.

### Fixed — API and wrappers
- `_ensure_wrapper()`: dead comments moved out of unreachable block;
  refactored to clean `if/elif` structure.
- `KerasModelWrapper._cleanup_implementation()`: `clear_session()` now
  called before `del self.model`, preventing dangling session references.
- `ScikitLearnModelWrapper.fit()`: warning `stacklevel` corrected from
  3 to 2.
- `ScikitLearnModelWrapper.save_model()`: uses `joblib.dump()` instead
  of raw pickle; significantly faster for ensemble models.
- `ImageDataHandler`: now accepts `color_mode` (default `'rgb'`),
  `val_split` (default `0.2`), and `test_split` (default `0.1`)
  constructor parameters, consistent with other data handlers.
  Invalid `color_mode` values raise `ValueError` at construction time.
- `ModelConfig.__hash__ = None` made explicit.
- `ModelConfig.for_cnn()`: docstring clarifies that the `"loss"` key
  is advisory only and is not read by any runner or wrapper.

### Fixed — plotting
- All plot functions now unconditionally return `plt.Figure`. Previously
  `_finalize_plot()` returned `None` when display was enabled, making
  plots untestable and uncomposable programmatically.
- `plot_autocorr_vs_lag()`: emits `UserWarning` when series is too short
  to compute autocorrelation at the requested lag, instead of returning
  `None` silently.
- `_find_metric_columns()`: `None` removed from training column candidate
  list, preventing spurious column matches.

### Added
- `variability_study()` and `compare_models()`: emit `UserWarning` when
  `runs < 10`, noting low statistical power at small n.
- `paired_wilcoxon_test` exported from `ictonyx` top-level namespace.
- `plot_variability_summary()`: accepts a `VariabilityStudyResults`
  object directly via `results=` parameter, eliminating manual
  extraction of `all_runs_metrics` and `final_metrics_series`.
- `plot_grid_study_heatmap()`: heatmap visualization for 2D parameter
  sweeps from `GridStudyResults`, showing mean or SD across two swept
  parameters with annotated cells.
- `ModelConfig.freeze()`: makes a config instance read-only after
  construction. Attempting to set or update parameters on a frozen
  config raises `RuntimeError`.

### Deprecated
- `get_final_metrics()` now emits `DeprecationWarning`. Use
  `get_metric_values()` instead. Will be removed in v0.5.0.

### Internal
- `scipy`, `matplotlib`, and `seaborn` committed as unconditional core
  dependencies; dead try/except fallback paths removed from
  `analysis.py` and `bootstrap.py`.
- `loggers.py`: all `print()` calls replaced with `logger.info()` so
  output respects the global verbose setting.
- `verbose=False` now propagates through the full stack including data
  loading, via `set_verbose()` in the high-level API.
- `GridStudyResults.to_dataframe()`: values no longer rounded at storage
  time; full float precision preserved for downstream analysis.
- `HyperparameterTuner`: `eval_time` now stores wall-clock seconds;
  `_should_minimize()` helper extracted; deprecated `merge()` replaced
  with `update()`.
- Dead code removed from `bootstrap_mean_difference_ci`.
- `import warnings` and `import datetime` moved to module top in
  `core.py` and `exceptions.py` respectively; `exceptions.py` timestamps
  now use UTC.
- `joblib` removed from required dependencies in `pyproject.toml`.
- `ExperimentRunner`, `run_grid_study()`, and `_set_seeds()` docstrings
  corrected to describe `SeedSequence.spawn()` accurately.
- `_INFRA_KWARGS` documented in `variability_study()` docstring.
- README `summarize()` output block updated to match current format.

---

## [0.3.11] — 2026-03-25

### Added
- `VariabilityStudyResults.test_against_null(null_value, metric, alpha)`:
  one-sample Wilcoxon signed-rank test against a user-specified null value.
  The v0.3.10 removal stub for `compare_models_statistically()` directed
  users to this method; it did not exist. Requires `num_runs >= 6` for
  reliable results.
- `variability_study()` and `compare_models()`: `verbose` parameter
  (default `True`).
- `compare_models()`: `paired` parameter (default `False`). When `True`,
  uses Wilcoxon signed-rank — correct when models share the same seeds.
  Warns when series have unequal run counts.
- `run_grid_study()`: `seed` parameter; each configuration receives
  `seed + i` for reproducible grid studies.

### Fixed — critical
- `KerasModelWrapper.predict()`: regression branch now returns raw float
  values. Previously applied `>= 0.5` threshold and cast to integer,
  silently corrupting every Keras regression prediction.
- `KerasModelWrapper.predict()`: binary classification threshold corrected
  from `> 0.5` to `>= 0.5`. A sigmoid output of exactly 0.5 was assigned
  to class 0; correct behaviour is class 1, consistent with sklearn
  convention. The v0.3.10 CHANGELOG claimed this fix was shipped — it was not.

### Fixed — statistical correctness
- `summarize()`, `get_summary_stats()`, and `assess_training_stability()`:
  standard deviation now uses `ddof=1` (sample std). Reported std was
  systematically too small — approximately 5% at n=10, 11% at n=5.
- `get_epoch_statistics()`: `ci_lower`/`ci_upper` now contain a
  t-distribution CI on the epoch mean. Previously empirical percentiles,
  which at n=5 are essentially min and max.
- `get_epoch_statistics()`: raises `ValueError` if `confidence` is not
  in (0, 1). Passing `confidence=95` previously produced garbage silently.
- `plot_comparison_forest()`: CI now uses Welch t-multiplier per
  comparison instead of hardcoded z=1.96. CIs at small n were too narrow.
- `compare_two_models()`: paired path refactored to dedicated
  `paired_wilcoxon_test()` with natural conclusion text. Previously passed
  pre-computed differences to a single-sample test, producing misleading
  metadata.
- `compare_two_models()`: effect size CI no longer computed for the
  Mann-Whitney path, where Cohen's d CI is the wrong pairing.
- `check_convergence()`: secondary criterion now requires both criteria
  to agree. Previously fired on any decreasing curve. Falls back to
  variance criterion alone when autocorrelation cannot be computed.
- `check_independence()`: false-positive "independent" on short series
  corrected — returns early when `n < max_lag + 2`. Loop bound corrected
  from `n // 4` to `min(max_lag, n - 2)`; SE uses post-dropna count.
- `check_normality()`: `require_all_tests` parameter added (default
  `False`); n<3 return structure standardized; n<20 comment corrected.
- `cohens_d()`: returns `NaN` with `RuntimeWarning` when pooled std is
  undefined. Previously silently returned 0.0.
- `assess_training_stability()`: correct results for DataFrame histories;
  all std/variance computations now use `ddof=1`.
- `compare_multiple_models()`: warns when multi-value Series are passed,
  indicating accidental per-epoch data rather than per-run data.
- `ModelConfig.for_variability_study()`: `epochs_per_run` key now read
  by `run_study()` before falling back to `epochs`.
- R² returns `NaN` instead of `0.0` when `ss_tot == 0` in all three
  model wrappers.

### Fixed — API and runners
- `compare_models()`: two instances of the same class with different
  parameters no longer silently overwrite each other in results.
- `compare_models()`: uses `get_metric_values()` instead of deprecated
  `get_final_metrics()`.
- `variability_study()`: model kwargs no longer bleed into `ModelConfig`
  and experiment logs.
- `ScikitLearnModelWrapper.assess()`: reads `self.task` set during
  `fit()`; uses `accuracy_score` for consistency with other wrappers.
- `run_study()`: raises `ValueError` for `num_runs < 1`.
- `plot_pairwise_comparison_matrix()`: no longer crashes on model names
  containing underscores; returns `None` with a warning on empty input.
- `plot_roc_curve()` / `plot_pr_curve()`: TensorFlow dependency removed
  from label binarization; falls back to `np.eye()` when TF is absent.
- `BaseLogger`: debug-only messages removed from stdout.

---

## [0.3.10] — 2026-03-24

### Removed

- `VariabilityStudyResults.compare_models_statistically()` removed. The method applied
  Kruskal-Wallis to groups of one observation each (one per run), producing statistically
  incoherent results. A stub raises `AttributeError` with migration guidance. Use
  `ix.compare_models()` for cross-model comparison.

### Added

- `ArraysDataHandler` now accepts optional `X_test` and `y_test` parameters. When
  provided, `load()` bypasses internal test splitting and returns the supplied arrays as
  `test_data` directly, performing only a train/val split on the training arrays. Ensures
  deterministic, consistent test evaluation across variability study runs for users with
  existing held-out test sets. Fully backward compatible.

- `ArraysDataHandler.__init__()` now accepts `val_split` and `test_split` as constructor
  parameters, storing them as defaults for `load()`. Explicit arguments to `load()` still
  take precedence; existing call sites are unaffected.

### Fixed — data integrity (re-run may be required)

- `ScikitLearnModelWrapper.fit()`: `val_accuracy` and `val_r2` are no longer written to
  the training history when no validation data is provided. Previously `val_score` was
  silently set equal to `train_score`, fabricating a generalisation metric that propagated
  into `final_metrics` and any downstream comparison or summary. **Users who ran sklearn
  variability studies without an explicit validation split should re-run their experiments.**

### Fixed — statistical correctness

- `analysis.py` — `rank_biserial_correlation()` returned the wrong sign on every call.
  The formula `1 - 2U/(n₁n₂)` has been corrected to `2U/(n₁n₂) - 1`, consistent with
  the Kerby (2014) definition and scipy's convention. Previously, a result where model A
  outperformed model B would report a negative rank-biserial correlation. Effect size
  magnitude and all p-values were unaffected.

- `ScikitLearnModelWrapper.evaluate()`: precision, recall, and F1 averaging strategy
  (`'binary'` vs `'weighted'`) is now determined from the training label count rather
  than `np.unique(y_test)`. A test batch missing any class would silently apply the
  wrong averaging.

- `HyperparameterTuner.tune()`: now applies `space_eval()` before returning, decoding
  hyperopt-encoded indices back to actual hyperparameter values. Previously a search over
  `hp.choice("batch_size", [16, 32, 64])` could return `{"batch_size": 2}` (the index)
  instead of `{"batch_size": 64}` (the value).

- `KerasModelWrapper.predict()`: binary classification threshold corrected from `> 0.5`
  to `>= 0.5`. A sigmoid output of exactly 0.5 was previously assigned to class 0 instead
  of class 1, inconsistent with sklearn convention and `predict_proba`.

### Fixed — crashes and silent failures

- `_ensure_wrapper()`: Keras models are now correctly wrapped as `KerasModelWrapper`.
  Previously the duck-typing check (`hasattr(obj, "fit") and hasattr(obj, "predict")`)
  fired before the Keras check, silently mis-wrapping any Keras model as
  `ScikitLearnModelWrapper`.

- `ArraysDataHandler.__init__()`: now raises `ValueError` if exactly one of `X_test` /
  `y_test` is provided. Previously the asymmetric pair was silently accepted; `load()`
  then returned `test_data=(array, None)`, causing a confusing crash inside `evaluate()`.

- `ImageDataHandler._preprocess_image()`: replaced dead nested `try/except` blocks with
  a direct call to `tf.image.decode_image()`. Python exception handlers cannot intercept
  TF op errors inside `tf.data.Dataset.map()` in graph execution mode — the JPEG→PNG
  fallback chain was never triggered.

- `ImageDataHandler.load()`: added `_validate_image_files()` pre-flight scan using Pillow
  before the dataset is built. Previously, any unreadable image silently injected an
  all-zero tensor with its original class label into training data. Now raises
  `DataValidationError` listing the offending files. Requires `Pillow`; skips validation
  with a warning if not installed.

- `TabularDataHandler.load()`: NaN values in feature columns now emit a `logger.warning`
  identifying the affected columns and counts. Previously the check computed `null_counts`
  and discarded it — a commented-out `print` was the only remnant. Target-column NaN
  already warned; feature columns now do too.

- `predict()` in all three wrappers: `assert self.predictions is not None` replaced with
  an explicit `ModelError`. Python strips `assert` statements when running with `-O`,
  silently turning the invariant check into dead code.

- `PyTorchModelWrapper.load_model()`: `weights_only` parameter is now passed through to
  `torch.load()`. Previously hardcoded as `True`, silently overriding the caller's value
  and causing `UnpicklingError` for any legacy checkpoint loaded with `weights_only=False`.

- `BaseModelWrapper.__del__()`: cleanup is now skipped during interpreter shutdown.
  Previously, calling `tf.keras.backend.clear_session()` during teardown could trigger
  TF's shutdown sequence out of order, causing `AttributeError` or segfaults in some
  TF versions.

- `ModelConfig` property setters (`epochs`, `batch_size`, `num_runs`, `epochs_per_run`,
  `learning_rate`, `cleanup_threshold`): now accept `numpy.integer` values. Previously
  `isinstance(value, int)` rejected `np.int64` and similar types with a `ValueError`,
  silently breaking any grid search that iterated over a numpy parameter array.

### Fixed — data layer

- `data.py`: TF preprocessing import guard now catches `AttributeError` in addition to
  `ImportError`. In TF 2.16+ / Keras 3, `tensorflow.keras.preprocessing` exists as a
  module stub but its sub-attributes have moved, raising `AttributeError` instead of
  `ImportError`. Without this fix, importing `ArraysDataHandler` or `TabularDataHandler`
  would fail in any Keras 3 environment.

- `TabularDataHandler.get_data_info()`: no longer returns `data_path: "in_memory_dataframe"`
  and `path_exists: False` when the handler was initialised from a DataFrame. Now returns
  `source: "dataframe"` alongside the actual shape and column metadata.

### Internal

- Stale comment removed from `BaseModelWrapper.check_memory_and_cleanup_if_needed()`.
- `conftest.py` fixture `TypeError` resolved; shared test fixtures now work as intended.


---

## [0.3.9] - 2026-03-18

### Fixed
- `apply_multiple_comparison_correction()`: Holm step-down algorithm
  used `np.minimum.accumulate` instead of `np.maximum.accumulate` to
  enforce monotonicity. This collapsed all corrected p-values down to
  the smallest scaled value, making the correction more permissive than
  intended. With input `[0.001, 0.01, 0.03]` the old code produced
  `[0.003, 0.003, 0.003]`; the correct output is `[0.003, 0.02, 0.03]`.
- `kruskal_wallis_test()`: effect size interpretation now calls
  `_interpret_epsilon_squared()` rather than `_interpret_eta_squared()`.
  Docstring Returns section corrected to say "epsilon-squared".
- `wilcoxon_signed_rank_test()`: Z-score for effect size now uses
  `scipy.stats.wilcoxon(method='approx').zstatistic`, which applies
  scipy's internal tie-corrected variance. The previous manual formula
  ignored ties and could produce an incorrect effect size `r`.
- `VariabilityStudyResults.compare_models_statistically()`: minimum run
  guard raised from 2 to 3, matching the actual requirement of
  `kruskal_wallis_test()`.
- `VariabilityStudyResults.to_dataframe()`: test metrics now stored
  with `run_id` and joined by identity rather than list position.
  Previously, a failed test evaluation for one run silently assigned
  wrong metrics to all subsequent runs.
- SHAP tree detection replaced string-based class name matching with
  `isinstance()` checks against all sklearn tree types, plus
  name-fragment fallback for XGBoost, LightGBM, and CatBoost. Fixes
  silent fallback to `KernelExplainer` for `GradientBoostingClassifier`,
  `HistGradientBoostingClassifier`, `ExtraTreesClassifier`,
  `AdaBoostClassifier`, and others.
- `DataHandler` refactored: path validation moved to new
  `FileDataHandler` intermediate class. `ArraysDataHandler` no longer
  carries a dummy `"in_memory_arrays"` path or a no-op
  `_validate_data_path()` override.
  **Minor breaking change:** `ArraysDataHandler.data_path` no longer
  exists.
- `run_grid_study()`: replaced `print()` calls with
  `logger.info/warning`. Added `verbose` parameter (default `True`).
  `set_verbose(False)` now correctly suppresses grid study output.
- `ScikitLearnModelWrapper.fit()`: unrecognized keyword arguments now
  produce a `UserWarning` instead of being silently discarded.

### Deprecated
- `ModelConfig.merge()`: use `update()` instead. Removal in v0.5.0.
- `ModelConfig.has()`: use `'key' in config` instead. Removal in
  v0.5.0.

### Added
- `VariabilityStudyResults.get_epoch_statistics()`: computes per-epoch
  mean, SD, SE, and percentile confidence band across all runs. Returns
  a DataFrame suitable for use with `plt.fill_between()`.
- `ModelConfig.__iter__()`, `__len__()`, `__eq__()`, `to_dict()`:
  `dict(config)`, `len(config)`, and equality comparison now work as
  expected.
- Coverage floor set at 60% via `--cov-fail-under` in pytest config.
  Total coverage: 65%.

---

## [0.3.8] - 2026-03-18

### Fixed
- `_isolated_training_function()` no longer catches all exceptions
  internally. Exceptions now propagate to the subprocess worker, which
  returns `{"success": False}`. Training crashes inside subprocesses
  were previously silently treated as empty history with no error
  message surfaced to the caller.
- `cloudpickle` and `psutil` declared as `optional = true` in
  `[tool.poetry.dependencies]`. `pip install ictonyx[isolation]` now
  actually installs these packages.
- `check_independence()`: removed `from scipy.stats import norm` inside
  the function body; replaced with `stats.norm` from the module-level
  import. The naked import re-executed on every call.
- `apply_multiple_comparison_correction()`: removed phantom `alpha`
  parameter from docstring (the parameter does not exist in the
  signature).
- `settings.should_verbose()` added as a public accessor for `_VERBOSE`;
  `api.py` no longer accesses the private attribute directly.
- `ScikitLearnModelWrapper.evaluate()`: replaced two bare
  `except Exception: pass` blocks with `logger.warning()` calls.
  Failures computing precision/recall/f1 or R²/MSE/MAE were previously
  silent.
- `check_independence()` returned dict now includes `threshold` and `n`
  keys. `max_autocorr` now uses `abs()` so large negative
  autocorrelations are correctly captured. `Returns:` docstring
  completed with full key inventory.
- `protobuf = "<4.0.0"` constraint removed from ml dependency group;
  conflicted with TensorFlow 2.16+ and MLflow 2.x.
- `_standardize_history_df()` extracted as a shared static method on
  `ExperimentRunner`; removes duplicated column rename logic from both
  `_run_single_fit_isolated()` and `_run_single_fit_standard()`.
- `__version__` moved to the top of `__init__.py`, before imports.
- `PyTorchModelWrapper.load_model()` now defaults to
  `weights_only=True`, preventing arbitrary code execution when loading
  checkpoints from untrusted sources. A `weights_only` parameter is
  exposed for callers who need to load legacy checkpoints.
- Security warnings added to `save_object()`, `load_object()`, and
  `ScikitLearnModelWrapper.load_model()` docstrings. `load_object()`
  now raises `FileNotFoundError` with a clean message before attempting
  to open the file, matching what the docstring already promised.
- `MemoryManager` no longer calls `mp.set_start_method("spawn",
  force=True)`, which permanently mutated global process state and
  could break PyTorch `DataLoader` on Linux. Now uses
  `mp.get_context("spawn")` throughout.

### Added
- `SECURITY.md` documenting pickle risks, the `weights_only` default
  for PyTorch checkpoints, and a vulnerability reporting path.
- `mypy` added to CI lint job and to dev dependencies. All 74 type
  errors resolved; `py.typed` marker is now backed by actual type
  checking.

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
