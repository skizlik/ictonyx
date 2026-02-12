"""Comprehensive tests for bootstrap confidence interval functions.

Tests cover: input validation, mathematical properties, both CI methods,
all convenience functions, edge cases, and reproducibility.
"""
import pytest
import numpy as np
import pandas as pd
from ictonyx.bootstrap import (
    BootstrapCIResult,
    bootstrap_ci,
    bootstrap_mean_difference_ci,
    bootstrap_effect_size_ci,
    bootstrap_paired_difference_ci,
    _to_clean_array,
    _percentile_ci,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_difference():
    """Two groups with an obvious, large difference."""
    g1 = np.array([0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.90, 0.91])
    g2 = np.array([0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71])
    return g1, g2


@pytest.fixture
def no_difference():
    """Two groups from the same distribution."""
    rng = np.random.RandomState(42)
    g1 = rng.normal(0.80, 0.02, 20)
    g2 = rng.normal(0.80, 0.02, 20)
    return g1, g2


@pytest.fixture
def paired_data():
    """Paired observations with a consistent small improvement."""
    rng = np.random.RandomState(42)
    base = rng.normal(0.80, 0.05, 15)
    g1 = base + 0.03  # consistently ~3% better
    g2 = base
    return g1, g2


# ===================================================================
#  _to_clean_array
# ===================================================================

class TestToCleanArray:

    def test_numpy_array(self):
        arr = _to_clean_array(np.array([1.0, 2.0, 3.0]))
        assert arr.dtype == np.float64
        assert len(arr) == 3

    def test_pandas_series(self):
        arr = _to_clean_array(pd.Series([1.0, np.nan, 3.0]))
        assert len(arr) == 2  # NaN dropped

    def test_python_list(self):
        arr = _to_clean_array([1.0, 2.0, 3.0])
        assert isinstance(arr, np.ndarray)

    def test_nan_removal(self):
        arr = _to_clean_array(np.array([1.0, np.nan, np.inf, 3.0]))
        assert len(arr) == 2
        assert np.all(np.isfinite(arr))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _to_clean_array("not an array")

    def test_2d_array_flattened(self):
        arr = _to_clean_array(np.array([[1, 2], [3, 4]]))
        assert arr.ndim == 1
        assert len(arr) == 4


# ===================================================================
#  Input Validation
# ===================================================================

class TestBootstrapInputValidation:

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci(
                np.array([1.0]), statistic_fn=np.mean, n_bootstrap=100
            )

    def test_invalid_confidence_low(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            bootstrap_ci(
                np.array([1, 2, 3]), statistic_fn=np.mean,
                confidence=0.0, n_bootstrap=100
            )

    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            bootstrap_ci(
                np.array([1, 2, 3]), statistic_fn=np.mean,
                confidence=1.0, n_bootstrap=100
            )

    def test_too_few_bootstrap_samples(self):
        with pytest.raises(ValueError, match="n_bootstrap must be >= 100"):
            bootstrap_ci(
                np.array([1, 2, 3]), statistic_fn=np.mean,
                n_bootstrap=10
            )

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_ci(
                np.array([1, 2, 3]), statistic_fn=np.mean,
                method='fake', n_bootstrap=100
            )


class TestTwoSampleInputValidation:

    def test_mean_diff_too_few_group1(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_mean_difference_ci(
                np.array([1.0]), np.array([1, 2, 3]), n_bootstrap=100
            )

    def test_effect_size_too_few_group2(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_effect_size_ci(
                np.array([1, 2, 3]), np.array([1.0]), n_bootstrap=100
            )

    def test_paired_unequal_lengths(self):
        with pytest.raises(ValueError, match="equal-length"):
            bootstrap_paired_difference_ci(
                np.array([1, 2, 3]), np.array([1, 2]), n_bootstrap=100
            )

    def test_paired_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_paired_difference_ci(
                np.array([1.0]), np.array([2.0]), n_bootstrap=100
            )


# ===================================================================
#  Core Properties (mathematical invariants)
# ===================================================================

class TestBootstrapCoreProperties:

    def test_ci_contains_point_estimate(self, clear_difference):
        """The point estimate should (almost always) be inside the CI."""
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_wider_ci_at_lower_confidence(self, clear_difference):
        """A 99% CI must be wider than a 90% CI on the same data."""
        g1, g2 = clear_difference
        ci_90 = bootstrap_mean_difference_ci(
            g1, g2, confidence=0.90, n_bootstrap=5000, random_state=42
        )
        ci_99 = bootstrap_mean_difference_ci(
            g1, g2, confidence=0.99, n_bootstrap=5000, random_state=42
        )
        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_ci_lower_less_than_upper(self, clear_difference):
        g1, g2 = clear_difference
        for method in ['percentile', 'bca']:
            result = bootstrap_mean_difference_ci(
                g1, g2, method=method, n_bootstrap=5000, random_state=42
            )
            assert result.ci_lower <= result.ci_upper

    def test_positive_se(self, clear_difference):
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        assert result.se_bootstrap > 0

    def test_reproducibility(self, clear_difference):
        """Same random_state must produce identical results."""
        g1, g2 = clear_difference
        r1 = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=123
        )
        r2 = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=123
        )
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.point_estimate == r2.point_estimate

    def test_different_seeds_differ(self, clear_difference):
        g1, g2 = clear_difference
        r1 = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=1
        )
        r2 = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=999
        )
        # Extremely unlikely to be identical with different seeds
        assert r1.ci_lower != r2.ci_lower


# ===================================================================
#  bootstrap_ci (single-sample engine)
# ===================================================================

class TestBootstrapCiEngine:

    def test_mean_of_known_distribution(self):
        """CI of the mean of [1..100] should bracket 50.5."""
        data = np.arange(1, 101, dtype=float)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=5000, confidence=0.95, method='percentile',
            random_state=42
        )
        assert result.ci_lower < 50.5 < result.ci_upper

    def test_median_ci(self):
        rng = np.random.RandomState(42)
        data = rng.normal(10, 2, 50)
        result = bootstrap_ci(
            data, statistic_fn=np.median,
            n_bootstrap=5000, method='bca', random_state=42
        )
        assert result.ci_lower < np.median(data) < result.ci_upper

    def test_return_distribution(self):
        data = np.arange(1, 21, dtype=float)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=500, method='percentile',
            random_state=42, return_distribution=True
        )
        assert result.bootstrap_distribution is not None
        assert len(result.bootstrap_distribution) >= 100

    def test_no_distribution_by_default(self):
        data = np.arange(1, 21, dtype=float)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=500, method='percentile', random_state=42
        )
        assert result.bootstrap_distribution is None

    def test_accepts_pandas_series(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=500, method='percentile', random_state=42
        )
        assert result.ci_lower < result.ci_upper

    def test_accepts_list(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=500, method='percentile', random_state=42
        )
        assert result.ci_lower < result.ci_upper

    def test_bca_method(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 30)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=5000, method='bca', random_state=42
        )
        assert result.method == 'bca'
        assert result.ci_lower < result.ci_upper

    def test_percentile_method(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 30)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=5000, method='percentile', random_state=42
        )
        assert result.method == 'percentile'


# ===================================================================
#  bootstrap_mean_difference_ci
# ===================================================================

class TestBootstrapMeanDifferenceCi:

    def test_clear_difference_excludes_zero(self, clear_difference):
        """CI should not contain zero when groups are clearly different."""
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        # g1 > g2 by about 0.20, so entire CI should be positive
        assert result.ci_lower > 0

    def test_no_difference_contains_zero(self, no_difference):
        """CI should contain zero when groups are from same distribution."""
        g1, g2 = no_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, confidence=0.95, random_state=42
        )
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_sign_matches_direction(self, clear_difference):
        """If g1 > g2, point estimate should be positive."""
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=42
        )
        assert result.point_estimate > 0

    def test_reversed_groups_flip_sign(self, clear_difference):
        """Swapping groups should negate the point estimate."""
        g1, g2 = clear_difference
        r1 = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=42
        )
        r2 = bootstrap_mean_difference_ci(
            g2, g1, n_bootstrap=2000, random_state=42
        )
        assert abs(r1.point_estimate + r2.point_estimate) < 1e-10

    def test_percentile_method(self, clear_difference):
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, method='percentile', n_bootstrap=2000, random_state=42
        )
        assert result.method == 'percentile'
        assert result.ci_lower > 0  # still clearly different

    def test_bca_method(self, clear_difference):
        g1, g2 = clear_difference
        result = bootstrap_mean_difference_ci(
            g1, g2, method='bca', n_bootstrap=2000, random_state=42
        )
        assert result.method == 'bca'
        assert result.ci_lower > 0

    def test_handles_nan_in_input(self):
        g1 = pd.Series([0.90, 0.91, np.nan, 0.92, 0.88, 0.90])
        g2 = pd.Series([0.70, np.nan, 0.69, 0.72, 0.68, 0.70])
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=42
        )
        assert np.isfinite(result.ci_lower)
        assert np.isfinite(result.ci_upper)

    def test_unequal_group_sizes(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(0.9, 0.02, 15)
        g2 = rng.normal(0.7, 0.02, 8)
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=2000, random_state=42
        )
        assert result.ci_lower > 0


# ===================================================================
#  bootstrap_effect_size_ci
# ===================================================================

class TestBootstrapEffectSizeCi:

    def test_large_effect_excludes_zero(self, clear_difference):
        g1, g2 = clear_difference
        result = bootstrap_effect_size_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        # Cohen's d should be very large and positive
        assert result.ci_lower > 0
        assert result.point_estimate > 1.0  # huge effect

    def test_no_effect_contains_zero(self, no_difference):
        g1, g2 = no_difference
        result = bootstrap_effect_size_ci(
            g1, g2, n_bootstrap=5000, confidence=0.95, random_state=42
        )
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_antisymmetry(self, clear_difference):
        """Cohen's d(g1, g2) should negate Cohen's d(g2, g1)."""
        g1, g2 = clear_difference
        r1 = bootstrap_effect_size_ci(
            g1, g2, n_bootstrap=2000, random_state=42
        )
        r2 = bootstrap_effect_size_ci(
            g2, g1, n_bootstrap=2000, random_state=42
        )
        assert abs(r1.point_estimate + r2.point_estimate) < 1e-10

    def test_unpooled_variant(self, clear_difference):
        g1, g2 = clear_difference
        r_pooled = bootstrap_effect_size_ci(
            g1, g2, pooled=True, n_bootstrap=2000, random_state=42
        )
        r_unpooled = bootstrap_effect_size_ci(
            g1, g2, pooled=False, n_bootstrap=2000, random_state=42
        )
        # Both should be large and positive, but may differ
        assert r_pooled.ci_lower > 0
        assert r_unpooled.ci_lower > 0


# ===================================================================
#  bootstrap_paired_difference_ci
# ===================================================================

class TestBootstrapPairedDifferenceCi:

    def test_consistent_improvement_excludes_zero(self, paired_data):
        g1, g2 = paired_data
        result = bootstrap_paired_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        # g1 is consistently ~0.03 above g2
        assert result.ci_lower > 0
        assert result.point_estimate == pytest.approx(0.03, abs=0.005)

    def test_no_improvement_contains_zero(self):
        rng = np.random.RandomState(42)
        base = rng.normal(0.80, 0.05, 20)
        g1 = base + rng.normal(0, 0.001, 20)
        g2 = base + rng.normal(0, 0.001, 20)
        result = bootstrap_paired_difference_ci(
            g1, g2, n_bootstrap=5000, confidence=0.95, random_state=42
        )
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_paired_narrower_than_independent(self, paired_data):
        """Paired CI should be narrower than independent CI on same data."""
        g1, g2 = paired_data
        paired_r = bootstrap_paired_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        indep_r = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        paired_width = paired_r.ci_upper - paired_r.ci_lower
        indep_width = indep_r.ci_upper - indep_r.ci_lower
        # Paired should be narrower because it removes between-fold variance
        assert paired_width < indep_width


# ===================================================================
#  BootstrapCIResult
# ===================================================================

class TestBootstrapCIResult:

    def test_str_representation(self):
        r = BootstrapCIResult(
            ci_lower=0.10, ci_upper=0.25, point_estimate=0.18,
            confidence_level=0.95, method='bca',
            n_bootstrap=10000, se_bootstrap=0.04
        )
        s = str(r)
        assert "95%" in s
        assert "bca" in s
        assert "0.1000" in s
        assert "0.2500" in s

    def test_distribution_optional(self):
        r = BootstrapCIResult(
            ci_lower=0.1, ci_upper=0.2, point_estimate=0.15,
            confidence_level=0.95, method='percentile',
            n_bootstrap=1000, se_bootstrap=0.03
        )
        assert r.bootstrap_distribution is None


# ===================================================================
#  Edge Cases
# ===================================================================

class TestEdgeCases:

    def test_constant_data(self):
        """All identical values: CI should collapse to a point."""
        data = np.array([5.0] * 20)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=1000, method='percentile', random_state=42
        )
        assert result.ci_lower == pytest.approx(5.0)
        assert result.ci_upper == pytest.approx(5.0)

    def test_constant_groups(self):
        g1 = np.array([0.9] * 10)
        g2 = np.array([0.8] * 10)
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=1000, method='percentile', random_state=42
        )
        assert result.point_estimate == pytest.approx(0.1)
        assert result.ci_lower == pytest.approx(0.1)
        assert result.ci_upper == pytest.approx(0.1)

    def test_minimum_sample_size(self):
        """Two observations is the minimum — should work, not crash."""
        g1 = np.array([0.9, 0.91])
        g2 = np.array([0.7, 0.71])
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=1000, method='percentile', random_state=42
        )
        assert np.isfinite(result.ci_lower)
        assert np.isfinite(result.ci_upper)

    def test_highly_skewed_data(self):
        """BCa should handle skewed data better than percentile."""
        rng = np.random.RandomState(42)
        data = rng.exponential(1.0, 30)
        result = bootstrap_ci(
            data, statistic_fn=np.mean,
            n_bootstrap=5000, method='bca', random_state=42
        )
        assert result.ci_lower < result.ci_upper
        assert result.ci_lower < np.mean(data) < result.ci_upper

    def test_custom_statistic_function(self):
        """User can pass any statistic function."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        def iqr(x):
            return float(np.percentile(x, 75) - np.percentile(x, 25))

        result = bootstrap_ci(
            data, statistic_fn=iqr,
            n_bootstrap=2000, method='percentile', random_state=42
        )
        assert result.ci_lower <= iqr(data) <= result.ci_upper

    def test_very_small_effect(self):
        """Near-zero effect: CI should be narrow and contain zero."""
        rng = np.random.RandomState(42)
        g1 = rng.normal(0.800, 0.02, 20)
        g2 = rng.normal(0.801, 0.02, 20)
        result = bootstrap_mean_difference_ci(
            g1, g2, n_bootstrap=5000, random_state=42
        )
        # Difference is essentially zero
        assert abs(result.point_estimate) < 0.02
