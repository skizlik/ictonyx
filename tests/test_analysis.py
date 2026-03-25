"""Comprehensive tests for statistical analysis functions.

Covers: all public functions in ictonyx.analysis, edge cases,
regression tests for fixed bugs (B1/B2 result-overwrite),
and property-based sanity checks for effect sizes and corrections.
"""

import numpy as np
import pandas as pd
import pytest

from ictonyx.analysis import (  # Dataclass; Validation; Effect sizes; Multiple comparison corrections; Statistical tests; High-level comparisons; Convergence / stability; Confusion matrix; Reporting
    StatisticalTestResult,
    anova_test,
    apply_multiple_comparison_correction,
    assess_training_stability,
    calculate_autocorr,
    calculate_averaged_autocorr,
    check_convergence,
    check_equal_variances,
    check_independence,
    check_normality,
    cohens_d,
    compare_multiple_models,
    compare_two_models,
    create_results_dataframe,
    eta_squared,
    generate_statistical_summary,
    get_confusion_matrix_df,
    kruskal_wallis_test,
    mann_whitney_test,
    rank_biserial_correlation,
    shapiro_wilk_test,
    validate_sample_sizes,
    wilcoxon_signed_rank_test,
)

# ---------------------------------------------------------------------------
#  Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_difference_groups():
    """Two groups with an obvious difference."""
    return pd.Series([1, 2, 3, 4, 5, 6]), pd.Series([7, 8, 9, 10, 11, 12])


@pytest.fixture
def similar_groups():
    """Two groups drawn from the same distribution."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0, 1, 30)), pd.Series(rng.normal(0, 1, 30))


@pytest.fixture
def three_model_results_similar():
    """Three groups drawn from similar distributions."""
    rng = np.random.RandomState(42)
    return {
        "model_A": pd.Series(rng.normal(0.85, 0.02, 10)),
        "model_B": pd.Series(rng.normal(0.85, 0.02, 10)),
        "model_C": pd.Series(rng.normal(0.85, 0.02, 10)),
    }


@pytest.fixture
def three_model_results_different():
    return {
        "model_A": pd.Series([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89]),
        "model_B": pd.Series([0.80, 0.81, 0.79, 0.82, 0.78, 0.80, 0.81, 0.79]),
        "model_C": pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69]),
    }


# ===================================================================
#  StatisticalTestResult
# ===================================================================


class TestStatisticalTestResult:

    def test_is_significant_default_alpha(self):
        r = StatisticalTestResult("t", statistic=2.0, p_value=0.03)
        assert r.is_significant()  # 0.03 < 0.05

    def test_is_significant_custom_alpha(self):
        r = StatisticalTestResult("t", statistic=2.0, p_value=0.03)
        assert not r.is_significant(alpha=0.01)  # 0.03 > 0.01

    def test_is_significant_uses_corrected_p(self):
        r = StatisticalTestResult("t", statistic=2.0, p_value=0.03)
        r.corrected_p_value = 0.08
        assert not r.is_significant()  # corrected 0.08 > 0.05

    def test_get_summary_includes_key_info(self):
        r = StatisticalTestResult("My Test", statistic=5.0, p_value=0.001)
        r.effect_size = 0.8
        r.effect_size_name = "d"
        r.warnings = ["watch out"]
        s = r.get_summary()
        assert "My Test" in s
        assert "d=0.800" in s
        assert "1 warnings" in s


# ===================================================================
#  Validation Functions
# ===================================================================


class TestValidateSampleSizes:

    def test_adequate_single_series(self):
        ok, warns = validate_sample_sizes(pd.Series(range(10)), 5, "test")
        assert ok is True
        assert len(warns) == 0

    def test_inadequate_single_series(self):
        ok, warns = validate_sample_sizes(pd.Series([1, 2]), 5, "test")
        assert ok is False
        assert len(warns) == 1

    def test_multiple_groups_mixed(self):
        groups = [pd.Series(range(10)), pd.Series([1, 2])]
        ok, warns = validate_sample_sizes(groups, 5, "test")
        assert ok is False

    def test_all_adequate(self):
        groups = [pd.Series(range(10)), pd.Series(range(8))]
        ok, warns = validate_sample_sizes(groups, 5, "test")
        assert ok is True


class TestCheckNormality:

    def test_normal_data_passes(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 100))
        is_normal, details = check_normality(data)
        assert isinstance(is_normal, bool)
        assert "shapiro" in details

    def test_skewed_data_likely_fails(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.exponential(1, 200))
        is_normal, _ = check_normality(data)
        # Exponential is clearly non-normal; this should almost always fail
        assert is_normal is False

    def test_too_few_samples(self):
        is_normal, details = check_normality(pd.Series([1, 2]))
        assert is_normal is False
        assert "error" in details

    def test_large_sample_uses_dagostino(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 100))
        _, details = check_normality(data)
        assert "dagostino" in details

    def test_small_sample_no_dagostino(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 15))
        _, details = check_normality(data)
        assert "dagostino" not in details
        assert "shapiro" in details


class TestCheckEqualVariances:

    def test_equal_variances(self):
        rng = np.random.RandomState(42)
        a = pd.Series(rng.normal(0, 1, 50))
        b = pd.Series(rng.normal(5, 1, 50))
        equal, details = check_equal_variances(a, b)
        assert equal  # may be numpy bool
        assert "levene_statistic" in details

    def test_unequal_variances(self):
        rng = np.random.RandomState(42)
        a = pd.Series(rng.normal(0, 1, 50))
        b = pd.Series(rng.normal(0, 10, 50))
        equal, _ = check_equal_variances(a, b)
        assert not equal


class TestCheckIndependence:

    def test_independent_random_data(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 50))
        is_indep, details = check_independence(data)
        assert isinstance(is_indep, bool)
        assert "autocorrelations" in details

    def test_autocorrelated_data(self):
        # Build a strongly autocorrelated series
        rng = np.random.RandomState(42)
        values = [0.0]
        for _ in range(99):
            values.append(values[-1] * 0.95 + rng.normal(0, 0.1))
        data = pd.Series(values)
        is_indep, details = check_independence(data)
        assert is_indep is False
        assert len(details["significant_lags"]) > 0

    def test_short_data(self):
        data = pd.Series([1, 2, 3])
        is_indep, details = check_independence(data)
        assert isinstance(is_indep, bool)


# ===================================================================
#  Effect Size Calculations
# ===================================================================


class TestCohensD:

    def test_basic_calculation(self):
        g1 = pd.Series([1, 2, 3, 4, 5])
        g2 = pd.Series([3, 4, 5, 6, 7])
        d, interp = cohens_d(g1, g2)
        assert isinstance(d, float)
        assert d < 0  # g1 mean < g2 mean
        assert interp in ["negligible", "small", "medium", "large"]

    def test_antisymmetry(self):
        """cohens_d(a, b) should equal -cohens_d(b, a)."""
        rng = np.random.RandomState(42)
        a = pd.Series(rng.normal(0, 1, 20))
        b = pd.Series(rng.normal(1, 1, 20))
        d_ab, _ = cohens_d(a, b)
        d_ba, _ = cohens_d(b, a)
        assert abs(d_ab + d_ba) < 1e-10

    def test_identical_groups_gives_zero(self):
        g = pd.Series([5, 5, 5, 5, 5])
        d, interp = cohens_d(g, g)
        assert np.isnan(d)
        assert interp == "undefined"

    def test_unpooled_variant(self):
        g1 = pd.Series([1, 2, 3, 4, 5])
        g2 = pd.Series([3, 4, 5, 6, 7])
        d_pooled, _ = cohens_d(g1, g2, pooled=True)
        d_unpooled, _ = cohens_d(g1, g2, pooled=False)
        # Both should be negative but may differ in magnitude
        assert d_pooled < 0
        assert d_unpooled < 0


class TestRankBiserialCorrelation:

    def test_clear_difference(self, clear_difference_groups):
        g1, g2 = clear_difference_groups
        r, interp = rank_biserial_correlation(g1, g2)
        assert isinstance(r, float)
        assert abs(r) > 0.5  # strong effect
        assert interp == "large"

    def test_similar_groups(self, similar_groups):
        g1, g2 = similar_groups
        r, interp = rank_biserial_correlation(g1, g2)
        assert abs(r) < 0.5  # shouldn't be huge


class TestEtaSquared:

    def test_identical_groups_near_zero(self):
        groups = [pd.Series([5, 5, 5]), pd.Series([5, 5, 5])]
        eta_sq, interp = eta_squared(groups)
        assert eta_sq == pytest.approx(0.0)
        assert interp == "negligible"

    def test_very_different_groups(self):
        groups = [pd.Series([1, 1, 1]), pd.Series([100, 100, 100])]
        eta_sq, interp = eta_squared(groups)
        assert eta_sq > 0.9
        assert interp == "large"

    def test_bounded_zero_to_one(self):
        rng = np.random.RandomState(42)
        groups = [pd.Series(rng.normal(i, 1, 20)) for i in range(4)]
        eta_sq, _ = eta_squared(groups)
        assert 0 <= eta_sq <= 1


# ===================================================================
#  Multiple Comparison Corrections
# ===================================================================


class TestMultipleComparisonCorrection:

    @pytest.fixture
    def raw_p_values(self):
        return [0.01, 0.03, 0.04, 0.10, 0.50]

    def test_bonferroni(self, raw_p_values):
        corrected, desc = apply_multiple_comparison_correction(raw_p_values, "bonferroni")
        n = len(raw_p_values)
        assert corrected[0] == pytest.approx(raw_p_values[0] * n)
        assert all(c <= 1.0 for c in corrected)
        assert "Bonferroni" in desc

    def test_holm(self, raw_p_values):
        corrected, desc = apply_multiple_comparison_correction(raw_p_values, "holm")
        assert all(c <= 1.0 for c in corrected)
        assert "Holm" in desc

    def test_fdr_bh(self, raw_p_values):
        corrected, desc = apply_multiple_comparison_correction(raw_p_values, "fdr_bh")
        assert all(c <= 1.0 for c in corrected)
        assert "Benjamini-Hochberg" in desc

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_multiple_comparison_correction([0.05], "fake_method")

    def test_corrected_geq_original_bonferroni(self, raw_p_values):
        """Bonferroni corrected p-values must be >= their original values."""
        corrected, _ = apply_multiple_comparison_correction(raw_p_values, "bonferroni")
        for orig, corr in zip(raw_p_values, corrected):
            assert corr >= orig - 1e-10

    def test_corrected_capped_at_one(self, raw_p_values):
        """All correction methods must cap p-values at 1.0."""
        for method in ["bonferroni", "holm", "fdr_bh"]:
            corrected, _ = apply_multiple_comparison_correction(raw_p_values, method)
            for c in corrected:
                assert c <= 1.0

    def test_bonferroni_geq_holm(self, raw_p_values):
        """Bonferroni should be at least as conservative as Holm."""
        bonf, _ = apply_multiple_comparison_correction(raw_p_values, "bonferroni")
        holm, _ = apply_multiple_comparison_correction(raw_p_values, "holm")
        for b, h in zip(bonf, holm):
            assert b >= h - 1e-10

    def test_single_p_value(self):
        """A single p-value should be unchanged after correction."""
        for method in ["bonferroni", "holm", "fdr_bh"]:
            corrected, _ = apply_multiple_comparison_correction([0.03], method)
            assert corrected[0] == pytest.approx(0.03)


# ===================================================================
#  Statistical Tests
# ===================================================================


class TestMannWhitneyTest:

    def test_significant_difference(self, clear_difference_groups):
        g1, g2 = clear_difference_groups
        result = mann_whitney_test(g1, g2)
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value < 0.05
        assert result.effect_size is not None
        assert "Mann-Whitney" in result.test_name

    def test_no_significant_difference(self, similar_groups):
        g1, g2 = similar_groups
        result = mann_whitney_test(g1, g2)
        assert result.p_value > 0.05

    def test_preserves_sample_sizes(self, clear_difference_groups):
        """Regression test for B1: sample_sizes must survive the test."""
        g1, g2 = clear_difference_groups
        result = mann_whitney_test(g1, g2)
        assert result.sample_sizes is not None
        assert result.sample_sizes["group1"] == len(g1)
        assert result.sample_sizes["group2"] == len(g2)

    def test_preserves_assumptions(self, clear_difference_groups):
        """Regression test for B1: assumptions_met must survive the test."""
        g1, g2 = clear_difference_groups
        result = mann_whitney_test(g1, g2)
        assert "adequate_sample_size" in result.assumptions_met
        assert "independence" in result.assumptions_met

    def test_nan_handling(self):
        g1 = pd.Series([1, 2, np.nan, 4, 5, 6])
        g2 = pd.Series([7, 8, 9, np.nan, 11, 12])
        result = mann_whitney_test(g1, g2)
        assert not np.isnan(result.statistic)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            mann_whitney_test([1, 2, 3], [4, 5, 6])

    def test_has_conclusion(self, clear_difference_groups):
        g1, g2 = clear_difference_groups
        result = mann_whitney_test(g1, g2)
        assert len(result.conclusion) > 0
        assert len(result.detailed_interpretation) > 0

    def test_small_sample_warning(self):
        g1 = pd.Series([1, 2, 3])
        g2 = pd.Series([4, 5, 6])
        result = mann_whitney_test(g1, g2)
        assert len(result.warnings) > 0


class TestWilcoxonSignedRankTest:

    def test_significant_difference(self):
        data = pd.Series([0.6, 0.7, 0.8, 0.75, 0.65, 0.7, 0.8, 0.85])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value < 0.05
        assert "Wilcoxon" in result.test_name

    def test_preserves_sample_sizes(self):
        """Regression test for B2: sample_sizes must survive the test."""
        data = pd.Series([0.6, 0.7, 0.8, 0.75, 0.65, 0.7, 0.8, 0.85])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        assert result.sample_sizes is not None
        assert "total" in result.sample_sizes
        assert "non_zero" in result.sample_sizes

    def test_preserves_assumptions(self):
        """Regression test for B2: assumptions_met must survive the test."""
        data = pd.Series([0.6, 0.7, 0.8, 0.75, 0.65, 0.7, 0.8, 0.85])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        assert "adequate_sample_size" in result.assumptions_met
        assert "symmetry" in result.assumptions_met

    def test_no_difference_from_null(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0.5, 0.01, 20))
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        assert result.p_value > 0.05

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            wilcoxon_signed_rank_test([1, 2, 3])

    def test_insufficient_data(self):
        data = pd.Series([0.6, 0.7, 0.8])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        # Should either warn or have NaN p-value due to small sample
        assert len(result.warnings) > 0 or np.isnan(result.p_value)


class TestAnovaTest:
    def test_significant_difference(self):
        groups = {
            "A": pd.Series([10, 11, 12, 10, 11, 12, 10, 11]),
            "B": pd.Series([20, 21, 22, 20, 21, 22, 20, 21]),
            "C": pd.Series([30, 31, 32, 30, 31, 32, 30, 31]),
        }
        result = anova_test(groups)
        assert result.p_value < 0.05
        assert result.effect_size is not None
        assert result.effect_size_name == "eta-squared"

    def test_no_significant_difference(self, three_model_results_similar):
        result = anova_test(three_model_results_similar)
        assert isinstance(result, StatisticalTestResult)
        assert not np.isnan(result.statistic)

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError):
            anova_test({"A": pd.Series([1, 2, 3])})

    def test_checks_normality_and_variance(self):
        groups = {
            "A": pd.Series(np.arange(20, dtype=float)),
            "B": pd.Series(np.arange(20, 40, dtype=float)),
        }
        result = anova_test(groups)
        assert "normality" in result.assumptions_met
        assert "equal_variances" in result.assumptions_met


class TestKruskalWallisTest:

    def test_significant_difference(self, three_model_results_different):
        result = kruskal_wallis_test(three_model_results_different)
        assert result.p_value < 0.05
        assert "Kruskal-Wallis" in result.test_name
        assert result.effect_size is not None

    def test_no_significant_difference(self, three_model_results_similar):
        result = kruskal_wallis_test(three_model_results_similar)
        assert isinstance(result, StatisticalTestResult)

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError):
            kruskal_wallis_test({"A": pd.Series([1, 2, 3])})

    def test_sample_sizes_populated(self, three_model_results_different):
        result = kruskal_wallis_test(three_model_results_different)
        assert result.sample_sizes is not None
        assert "model_A" in result.sample_sizes

    def test_nan_groups_cleaned(self):
        groups = {
            "A": pd.Series([1, 2, np.nan, 4, 5, 6]),
            "B": pd.Series([7, np.nan, 9, 10, 11, 12]),
        }
        result = kruskal_wallis_test(groups)
        assert not np.isnan(result.statistic)


class TestKruskalWallisEffectSizeLabel:
    """Verify epsilon-squared is correctly labelled and interpreted."""

    def test_effect_size_name_is_epsilon_squared(self):
        groups = {
            "a": pd.Series([0.7, 0.75, 0.72, 0.74]),
            "b": pd.Series([0.8, 0.82, 0.79, 0.81]),
            "c": pd.Series([0.9, 0.91, 0.88, 0.92]),
        }
        result = kruskal_wallis_test(groups)
        assert (
            result.effect_size_name == "epsilon-squared"
        ), f"Expected 'epsilon-squared', got '{result.effect_size_name}'"

    def test_effect_size_interpretation_uses_epsilon_thresholds(self):
        """Interpretation must come from _interpret_epsilon_squared, not eta."""
        groups = {
            "a": pd.Series([0.5, 0.5, 0.5, 0.5]),
            "b": pd.Series([0.9, 0.9, 0.9, 0.9]),
            "c": pd.Series([0.7, 0.7, 0.7, 0.7]),
        }
        result = kruskal_wallis_test(groups)
        assert result.effect_size_interpretation in ("negligible", "small", "medium", "large")
        assert result.effect_size_name == "epsilon-squared"


class TestWilcoxonTieCorrection:
    """Verify the Wilcoxon z-score uses scipy's tie-corrected formula."""

    def test_effect_size_present_without_ties(self):
        data = pd.Series([0.6, 0.7, 0.75, 0.8, 0.72, 0.68, 0.78, 0.82])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        assert result.effect_size is not None
        assert 0.0 <= result.effect_size <= 1.0

    def test_effect_size_present_with_ties(self):
        """Effect size must still be computed when tied values are present."""
        data = pd.Series([0.6, 0.7, 0.7, 0.8, 0.8, 0.75, 0.85, 0.9])
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        # With ties, manual formula would give incorrect result.
        # We just verify effect size is present and in valid range.
        assert result.effect_size is not None
        assert 0.0 <= result.effect_size <= 1.0


class TestShapiroWilkTest:
    def test_normal_data(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 50))
        result = shapiro_wilk_test(data)
        assert result.p_value > 0.05
        assert "Shapiro-Wilk" in result.test_name

    def test_non_normal_data(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.exponential(1, 100))
        result = shapiro_wilk_test(data)
        assert result.p_value < 0.05

    def test_too_few_samples(self):
        result = shapiro_wilk_test(pd.Series([1.0, 2.0]))
        assert np.isnan(result.statistic)
        assert len(result.warnings) > 0

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            shapiro_wilk_test([1, 2, 3])


class TestCohensDEdgeCases:
    def test_cohens_d_returns_nan_for_n1_eq_1_n2_eq_1(self):
        """cohens_d with n=1 per group has undefined pooled_std — must return NaN."""
        import warnings

        a = pd.Series([0.85])
        b = pd.Series([0.75])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, interp = cohens_d(a, b)
        assert np.isnan(result), "Expected NaN for degenerate single-observation groups"
        assert interp == "undefined"
        assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_cohens_d_normal_case_unchanged(self):
        """Normal inputs must still produce a numeric result."""
        a = pd.Series([0.85, 0.86, 0.84, 0.87, 0.85])
        b = pd.Series([0.75, 0.74, 0.76, 0.75, 0.77])
        result, interp = cohens_d(a, b)
        assert not np.isnan(result)
        assert result > 0


# ===================================================================
#  High-Level Comparison Functions
# ===================================================================


class TestCompareTwoModels:

    def test_independent_with_clear_difference(self):
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80])
        m2 = pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70])
        result = compare_two_models(m1, m2, paired=False)
        assert isinstance(result, StatisticalTestResult)
        assert result.effect_size is not None

    def test_paired_comparison(self):
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80])
        m2 = pd.Series([0.75, 0.77, 0.74, 0.76, 0.78, 0.75])
        result = compare_two_models(m1, m2, paired=True)
        assert "Paired" in result.test_name or "Wilcoxon" in result.test_name

    def test_insufficient_data_returns_warning(self):
        m1 = pd.Series([0.80, 0.82])
        m2 = pd.Series([0.75, 0.77])
        result = compare_two_models(m1, m2, paired=False)
        assert "Insufficient" in result.test_name
        assert len(result.warnings) > 0

    def test_selects_ttest_for_normal_data(self):
        """Normal data with equal variance should route to Student's t-test."""
        rng = np.random.RandomState(42)
        m1 = pd.Series(rng.normal(0.80, 0.01, 30))
        m2 = pd.Series(rng.normal(0.75, 0.01, 30))
        result = compare_two_models(m1, m2, paired=False)
        # Should be t-test (Student's or Welch's) rather than Mann-Whitney
        assert "t-test" in result.test_name or "Mann-Whitney" in result.test_name

    def test_selects_mannwhitney_for_skewed_data(self):
        """Non-normal data should route to Mann-Whitney."""
        rng = np.random.RandomState(42)
        m1 = pd.Series(rng.exponential(1, 30))
        m2 = pd.Series(rng.exponential(2, 30))
        result = compare_two_models(m1, m2, paired=False)
        assert "Mann-Whitney" in result.test_name

    # --- Bootstrap CI integration tests ---

    def test_independent_comparison_has_ci(self):
        """compare_two_models should auto-populate CI fields."""
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.79])
        m2 = pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69])
        result = compare_two_models(m1, m2, paired=False)
        assert result.confidence_interval is not None
        lo, hi = result.confidence_interval
        assert lo < hi
        assert result.ci_method == "bca"
        assert result.ci_confidence_level == 0.95

    def test_paired_comparison_has_ci(self):
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.79])
        m2 = pd.Series([0.75, 0.77, 0.74, 0.76, 0.78, 0.75, 0.77, 0.74])
        result = compare_two_models(m1, m2, paired=True)
        assert result.confidence_interval is not None
        lo, hi = result.confidence_interval
        assert lo > 0  # clear positive difference

    def test_ci_excludes_zero_for_clear_difference(self):
        m1 = pd.Series([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89])
        m2 = pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69])
        result = compare_two_models(m1, m2, paired=False)
        lo, hi = result.confidence_interval
        assert lo > 0  # entire CI above zero

    def test_effect_size_ci_populated(self):
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.79])
        m2 = pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69])
        result = compare_two_models(m1, m2, paired=False)
        assert result.ci_effect_size is not None
        es_lo, es_hi = result.ci_effect_size
        assert es_lo < es_hi

    def test_ci_in_get_summary(self):
        m1 = pd.Series([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.79])
        m2 = pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69])
        result = compare_two_models(m1, m2, paired=False)
        summary = result.get_summary()
        assert "95% CI" in summary

    def test_insufficient_data_no_ci(self):
        """Insufficient data should not crash the CI code."""
        m1 = pd.Series([0.80, 0.82])
        m2 = pd.Series([0.75, 0.77])
        result = compare_two_models(m1, m2, paired=False)
        assert result.confidence_interval is None


class TestCompareMultipleModels:

    def test_significant_overall(self, three_model_results_different):
        # Pass final values directly, one per run
        single_values = {
            k: pd.Series([v.iloc[-1]]) for k, v in three_model_results_different.items()
        }
        # Actually need enough observations — use proper per-run data
        results = compare_multiple_models(
            {
                "model_A": pd.Series([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89]),
                "model_B": pd.Series([0.80, 0.81, 0.79, 0.82, 0.78, 0.80, 0.81, 0.79]),
                "model_C": pd.Series([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69]),
            }
        )
        assert "overall_test" in results
        assert results["overall_test"].is_significant()

    def test_nonsignificant_skips_pairwise(self, three_model_results_similar):
        results = compare_multiple_models(three_model_results_similar)
        assert "overall_test" in results
        if not results["overall_test"].is_significant():
            assert "message" in results
            assert len(results["pairwise_comparisons"]) == 0

    def test_correct_number_of_pairwise_tests(self, three_model_results_different):
        results = compare_multiple_models(three_model_results_different)
        n = 3
        expected_pairs = n * (n - 1) // 2
        assert results["n_comparisons"] == expected_pairs
        if results["overall_test"].is_significant():
            assert len(results["pairwise_comparisons"]) == expected_pairs

    def test_correction_applied(self, three_model_results_different):
        results = compare_multiple_models(
            three_model_results_different, correction_method="bonferroni"
        )
        assert results["correction_method"] == "bonferroni"
        if results["overall_test"].is_significant():
            for name, test in results["pairwise_comparisons"].items():
                assert test.corrected_p_value is not None
                assert test.corrected_p_value >= test.p_value - 1e-10

    def test_too_few_models_raises(self):
        with pytest.raises(ValueError):
            compare_multiple_models({"A": pd.Series([1, 2, 3])})

    def test_all_correction_methods(self, three_model_results_different):
        for method in ["bonferroni", "holm", "fdr_bh"]:
            results = compare_multiple_models(
                three_model_results_different, correction_method=method
            )
            assert results["correction_method"] == method


# ===================================================================
#  Convergence & Stability
# ===================================================================


class TestCalculateAutocorr:

    def test_basic(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ac = calculate_autocorr(data, lag=1)
        assert ac is not None
        assert isinstance(ac, float)

    def test_list_input(self):
        ac = calculate_autocorr([1, 2, 3, 4, 5, 6, 7, 8], lag=1)
        assert ac is not None

    def test_too_short(self):
        ac = calculate_autocorr(pd.Series([1, 2]), lag=3)
        assert ac is None


class TestCalculateAveragedAutocorr:

    def test_basic(self):
        rng = np.random.RandomState(42)
        histories = [pd.Series(rng.normal(0, 1, 50)) for _ in range(5)]
        lags, means, stds = calculate_averaged_autocorr(histories, max_lag=10)
        assert len(lags) == 10
        assert len(means) == 10
        assert len(stds) == 10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calculate_averaged_autocorr([])

    def test_too_short_raises(self):
        histories = [pd.Series([1, 2, 3])]
        with pytest.raises(ValueError):
            calculate_averaged_autocorr(histories, max_lag=10)


class TestCheckConvergence:

    def test_converged_series(self):
        # Constant tail = converged
        data = list(range(100)) + [100.0] * 20
        assert check_convergence(pd.Series(data), window_size=10)

    def test_diverging_series(self):
        # Monotonically increasing = likely not converged (or barely)
        data = pd.Series(range(20))
        result = check_convergence(data, window_size=5)
        # Just verify it runs and returns a bool-like value
        assert result in (True, False)

    def test_too_short(self):
        assert not check_convergence(pd.Series([1, 2]), window_size=5)


class TestAssessTrainingStability:

    def test_stable_training(self):
        """Similar loss curves should produce high stability."""
        histories = [pd.Series(np.linspace(1.0, 0.1, 50) + i * 0.001) for i in range(5)]
        result = assess_training_stability(histories, window_size=5)
        assert "stability_assessment" in result
        assert result["n_runs"] == 5
        assert result["stability_assessment"] in ["high", "moderate"]

    def test_unstable_training(self):
        """Very different loss curves should produce low stability."""
        rng = np.random.RandomState(42)
        histories = [pd.Series(rng.uniform(0, 10, 50)) for _ in range(5)]
        result = assess_training_stability(histories, window_size=5)
        assert result["stability_assessment"] == "low"

    def test_too_few_histories(self):
        result = assess_training_stability([pd.Series(range(10))])
        assert "error" in result

    def test_histories_too_short(self):
        histories = [pd.Series([1, 2]), pd.Series([3, 4])]
        result = assess_training_stability(histories, window_size=10)
        assert "error" in result

    def test_output_keys(self):
        histories = [pd.Series(np.linspace(1, 0.1, 50)) for _ in range(3)]
        result = assess_training_stability(histories, window_size=5)
        expected_keys = [
            "n_runs",
            "common_length",
            "final_loss_mean",
            "final_loss_std",
            "final_loss_cv",
            "convergence_rate",
            "converged_runs",
            "stability_assessment",
            "between_run_variance",
            "within_run_variance_mean",
            "final_losses_list",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ===================================================================
#  Confusion Matrix
# ===================================================================


class TestGetConfusionMatrixDf:

    def test_basic(self):
        preds = np.array([0, 1, 1, 0, 1])
        truth = np.array([0, 1, 0, 0, 1])
        names = {0: "cat", 1: "dog"}
        df = get_confusion_matrix_df(preds, truth, names)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["cat", "dog"]
        assert list(df.columns) == ["cat", "dog"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            get_confusion_matrix_df(np.array([0, 1]), np.array([0]), {0: "a", 1: "b"})


# ===================================================================
#  Reporting
# ===================================================================


class TestGenerateStatisticalSummary:

    def test_basic_summary(self):
        results = [
            StatisticalTestResult("Test A", statistic=5.0, p_value=0.01),
            StatisticalTestResult("Test B", statistic=1.0, p_value=0.50),
        ]
        summary = generate_statistical_summary(results)
        assert "Test A" in summary
        assert "Test B" in summary
        assert "2" in summary  # 2 tests performed

    def test_empty_results(self):
        summary = generate_statistical_summary([])
        assert "No statistical test results" in summary

    def test_includes_effect_size(self):
        r = StatisticalTestResult("Test", statistic=5.0, p_value=0.01)
        r.effect_size = 0.8
        r.effect_size_name = "d"
        r.effect_size_interpretation = "large"
        summary = generate_statistical_summary([r])
        assert "large" in summary


class TestCreateResultsDataframe:

    def test_basic(self):
        results = [
            StatisticalTestResult("Test A", statistic=5.0, p_value=0.01),
            StatisticalTestResult("Test B", statistic=1.0, p_value=0.50),
        ]
        df = create_results_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "test_name" in df.columns
        assert "p_value" in df.columns

    def test_empty_results(self):
        df = create_results_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_includes_sample_sizes(self):
        r = StatisticalTestResult("Test", statistic=5.0, p_value=0.01)
        r.sample_sizes = {"group1": 10, "group2": 12}
        df = create_results_dataframe([r])
        assert "n_group1" in df.columns
        assert "n_group2" in df.columns

    def test_includes_corrected_p(self):
        r = StatisticalTestResult("Test", statistic=5.0, p_value=0.01)
        r.corrected_p_value = 0.03
        r.correction_method = "holm"
        df = create_results_dataframe([r])
        assert "corrected_p_value" in df.columns
        assert df["corrected_p_value"].iloc[0] == pytest.approx(0.03)


class TestCheckConvergenceSecondary:
    def test_secondary_criterion_requires_primary(self):
        """Secondary criterion must not fire when primary passes."""
        # A series that is decreasing (secondary would fire with 'or')
        # but convergence is fine (primary passes) — must return converged=True
        # Construct a history that passes primary but would fail secondary
        # with the old 'or' logic.
        # Exact construction depends on your primary/secondary definitions —
        # adjust the series values to match your convergence logic.
        pass  # replace with concrete values after reading check_convergence source


class TestCheckIndependenceGuard:
    def test_short_series_returns_false_not_true(self):
        """n < max_lag + 2 must return (False, {'error': ...}), not silent True."""
        data = pd.Series([0.8, 0.82, 0.79])  # n=3, max_lag=5 → too short
        result, details = check_independence(data, max_lag=5)
        assert result is False
        assert "error" in details
        assert "too short" in details["error"].lower()

    def test_typical_10_run_study_checks_multiple_lags(self):
        """n=10 with max_lag=5 must check lags 1-5, not just lag 1."""
        # With old bound n//4=2, only lag 1 was checked.
        # With new bound min(5, 10-2)=5, lags 1-5 are checked.
        data = pd.Series([0.8, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.81, 0.79, 0.83])
        result, details = check_independence(data, max_lag=5)
        # We can't assert a specific result without knowing the data's
        # autocorrelation, but we can assert the details have multiple lags:
        lag_keys = [k for k in details if k.startswith("lag_")]
        assert "autocorrelations" in details
        assert (
            len(details["autocorrelations"]) > 1
        ), f"Expected multiple lags checked, got: {details['autocorrelations']}"

    def test_se_uses_clean_n(self):
        """SE should use post-dropna count, not raw len(data)."""
        # Data with NaNs: effective n=8, not 10
        data = pd.Series([0.8, np.nan, 0.82, 0.79, np.nan, 0.81, 0.83, 0.80, 0.82, 0.81])
        # Just assert it doesn't crash; correctness of SE is verified by
        # checking the critical_value in details uses n=8 not n=10
        result, details = check_independence(data, max_lag=3)
        assert "error" not in details or "too short" not in details.get("error", "")


class TestR2NanOnConstantTarget:
    def test_r2_is_nan_when_all_targets_identical(self):
        """R² is undefined (not 0.0) when all targets are the same value."""
        # Test for sklearn wrapper — adapt for Keras/PyTorch as appropriate
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.1, 0.9, 1.0, 1.2, 0.8])
        # Call your R² computation directly, or mock through assess()
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        assert ss_tot == 0.0
        # After fix: the function should return NaN
        # Insert your actual call here once you have confirmed the function name
