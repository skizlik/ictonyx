import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Bootstrap confidence intervals
try:
    from .bootstrap import (
        BootstrapCIResult,
        bootstrap_effect_size_ci,
        bootstrap_mean_difference_ci,
        bootstrap_paired_difference_ci,
    )

    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

# Optional scipy imports
try:
    from scipy import stats
    from scipy.stats import (
        beta,
        f_oneway,
        jarque_bera,
        kruskal,
        levene,
        mannwhitneyu,
        normaltest,
        shapiro,
        ttest_ind,
        ttest_rel,
        wilcoxon,
    )

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional sklearn import (only for confusion matrix)
try:
    from sklearn.metrics import confusion_matrix

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class StatisticalTestResult:
    """Comprehensive result from a statistical comparison test.

    Returned by all statistical test functions in this module. Contains
    the test statistic, p-value, effect size, assumption checks, and
    human-readable interpretation.

    Attributes:
        test_name: Name of the test performed (e.g. ``'Mann-Whitney U'``).
        statistic: The test statistic value.
        p_value: The raw (uncorrected) p-value.
        effect_size: Effect size estimate (Cohen's d, rank-biserial, or
            eta-squared, depending on the test).
        effect_size_name: Label for the effect size metric.
        effect_size_interpretation: Qualitative label (``'small'``,
            ``'medium'``, ``'large'``) per conventional thresholds.
        sample_sizes: Dict mapping group names to sample counts.
        assumptions_met: Dict mapping assumption names (e.g.
            ``'normality_group1'``) to boolean pass/fail.
        assumption_details: Dict with detailed assumption check outputs
            (e.g. Shapiro-Wilk statistics).
        warnings: List of warning strings (e.g. small sample size caveats).
        recommendations: List of suggested next steps.
        corrected_p_value: P-value after multiple comparison correction,
            if applicable.
        correction_method: Name of correction (``'bonferroni'``, ``'holm'``,
            ``'benjamini-hochberg'``), if applied.
        conclusion: One-sentence summary of the result.
        detailed_interpretation: Multi-sentence explanation suitable for
            a report.
        confidence_interval: Bootstrap CI for the mean difference, as a
            ``(lower, upper)`` tuple.
        ci_confidence_level: Confidence level (e.g. ``0.95``).
        ci_method: CI construction method (``'bca'`` or ``'percentile'``).
        ci_effect_size: Bootstrap CI for the effect size, as a
            ``(lower, upper)`` tuple.
    """

    test_name: str
    statistic: float
    p_value: float

    # Effect size information
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_size_interpretation: Optional[str] = None

    # Sample and power information
    sample_sizes: Optional[Dict[str, int]] = None

    # Assumption checking
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    assumption_details: Dict[str, Any] = field(default_factory=dict)

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Multiple comparison correction
    corrected_p_value: Optional[float] = None
    correction_method: Optional[str] = None

    # Interpretation
    conclusion: str = ""
    detailed_interpretation: str = ""

    # Confidence intervals (populated by bootstrap)
    confidence_interval: Optional[Tuple[float, float]] = None
    ci_confidence_level: Optional[float] = None
    ci_method: Optional[str] = None
    ci_effect_size: Optional[Tuple[float, float]] = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        p_val = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        return p_val < alpha

    def get_summary(self) -> str:
        """Get a concise summary of the test result."""
        sig_marker = (
            "***"
            if self.is_significant(0.001)
            else "**" if self.is_significant(0.01) else "*" if self.is_significant(0.05) else "ns"
        )

        summary = f"{self.test_name}: {self.statistic:.3f}, p={self.p_value:.4f} {sig_marker}"

        if self.effect_size is not None:
            summary += f", {self.effect_size_name}={self.effect_size:.3f}"

        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            summary += f", {self.ci_confidence_level*100:.0f}% CI [{lo:.4f}, {hi:.4f}]"

        if self.warnings:
            summary += f" (⚠ {len(self.warnings)} warnings)"

        return summary


def _check_scipy():
    """Check scipy availability."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for statistical tests. Install with: pip install scipy")


def _check_sklearn():
    """Check sklearn availability."""
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn required for confusion matrix. Install with: pip install scikit-learn"
        )


# VALIDATION FUNCTIONS


def validate_sample_sizes(
    data: Union[pd.Series, List[pd.Series]], min_size: int, test_name: str
) -> Tuple[bool, List[str]]:
    """Validate that sample sizes are adequate for a statistical test.

    Checks each group against a minimum size threshold and generates
    warnings for small samples.

    Args:
        data: A single ``pd.Series`` or list of ``pd.Series`` to check.
        min_size: Minimum acceptable sample size per group.
        test_name: Name of the test, used in warning messages.

    Returns:
        Tuple of ``(adequate, warnings)`` where ``adequate`` is ``True``
        if all groups meet the minimum size.
    """
    warnings_list = []

    if isinstance(data, pd.Series):
        sizes = [len(data)]
    else:
        sizes = [len(series) for series in data]

    inadequate_sizes = [i for i, size in enumerate(sizes) if size < min_size]

    if inadequate_sizes:
        warnings_list.append(
            f"{test_name} requires at least {min_size} samples per group. "
            f"Groups {inadequate_sizes} have insufficient data."
        )
        return False, warnings_list

    return True, warnings_list


def check_normality(data: pd.Series, alpha: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    """Test for normality using Shapiro-Wilk (and D'Agostino-Pearson if n >= 20).

    Used internally by assumption-checking logic in statistical tests.

    Args:
        data: A ``pd.Series`` of metric values.
        alpha: Significance level. Default 0.05.

    Returns:
        Tuple of ``(is_normal, details_dict)`` where ``is_normal`` is
        ``True`` if the sample does not reject normality, and
        ``details_dict`` contains test statistics and p-values.
    """

    _check_scipy()

    results = {}

    if len(data) < 3:
        return False, {"error": "Insufficient data for normality testing"}

    # Shapiro-Wilk test (best for small samples)
    if len(data) <= 5000:  # Shapiro-Wilk has sample size limits
        try:
            shapiro_stat, shapiro_p = shapiro(data)
            results["shapiro"] = {"statistic": shapiro_stat, "p_value": shapiro_p}
        except Exception:
            pass

    # D'Agostino and Pearson test (better for larger samples)
    if len(data) >= 20:
        try:
            dagostino_stat, dagostino_p = normaltest(data)
            results["dagostino"] = {"statistic": dagostino_stat, "p_value": dagostino_p}
        except Exception:
            pass

    if not results:
        return False, {"error": "Could not perform normality tests"}

    # Consider normal if any test fails to reject normality
    is_normal = all(test["p_value"] > alpha for test in results.values())

    return is_normal, results


def check_equal_variances(
    data1: pd.Series, data2: pd.Series, alpha: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """Test for equal variances between two groups using Levene's test.

    Used internally to choose between Student's t-test (equal variance)
    and Welch's t-test (unequal variance).

    Args:
        data1: First group's metric values.
        data2: Second group's metric values.
        alpha: Significance level. Default 0.05.

    Returns:
        Tuple of ``(variances_equal, details_dict)`` where
        ``variances_equal`` is ``True`` if Levene's test does not reject
        the null hypothesis of equal variances.
    """

    _check_scipy()

    try:
        stat, p_val = levene(data1, data2)
        equal_vars = p_val > alpha

        return equal_vars, {
            "levene_statistic": stat,
            "levene_p": p_val,
            "conclusion": "equal" if equal_vars else "unequal",
        }
    except Exception as e:
        return False, {"error": str(e)}


def check_independence(
    data: pd.Series, max_lag: int = 5, alpha: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """Check for autocorrelation that would violate the independence assumption.

    Computes autocorrelation at lags 1 through ``max_lag`` and flags the
    data as non-independent if any lag exceeds the threshold.

    Args:
        data: A ``pd.Series`` of metric values ordered by run.
        max_lag: Maximum lag to check. Default 5.
        threshold: Autocorrelation magnitude above which dependence is
            flagged. Default 0.3.

    Returns:
        Tuple of ``(is_independent, details_dict)`` where
        ``is_independent`` is ``True`` if no significant autocorrelation
        was detected.
    """

    autocorr_results = {}
    significant_lags = []

    for lag in range(1, min(max_lag + 1, len(data) // 4)):
        try:
            autocorr = data.autocorr(lag)
            if not np.isnan(autocorr):
                autocorr_results[f"lag_{lag}"] = autocorr

                # Rough significance test (assuming normal distribution)
                se = 1.0 / np.sqrt(len(data))
                if abs(autocorr) > 1.96 * se:  # 95% confidence
                    significant_lags.append(lag)
        except Exception:
            continue

    is_independent = len(significant_lags) == 0

    details = {
        "autocorrelations": autocorr_results,
        "significant_lags": significant_lags,
        "max_autocorr": max(autocorr_results.values()) if autocorr_results else 0,
    }

    return is_independent, details


# EFFECT SIZE CALCULATIONS


def cohens_d(group1: pd.Series, group2: pd.Series, pooled: bool = True) -> Tuple[float, str]:
    """Calculate Cohen's d effect size between two groups.

    Uses pooled standard deviation by default. Thresholds: small (0.2),
    medium (0.5), large (0.8).

    Args:
        group1: First group's metric values.
        group2: Second group's metric values.
        pooled: If ``True`` (default), use pooled standard deviation.

    Returns:
        Tuple of ``(d_value, interpretation)`` where ``interpretation``
        is one of ``'negligible'``, ``'small'``, ``'medium'``, ``'large'``.
    """

    mean1, mean2 = group1.mean(), group2.mean()

    if pooled:
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    else:
        # Control group standard deviation
        std2 = group2.std()
        d = (mean1 - mean2) / std2 if std2 > 0 else 0

    interpretation = _interpret_cohens_d(abs(d))

    return d, interpretation


def _interpret_cohens_d(abs_d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def rank_biserial_correlation(group1: pd.Series, group2: pd.Series) -> Tuple[float, str]:
    """Calculate rank-biserial correlation as the effect size for the Mann-Whitney U test.

    Ranges from -1 to 1, where 0 indicates no effect. Thresholds:
    small (0.1), medium (0.3), large (0.5).

    Args:
        group1: First group's metric values.
        group2: Second group's metric values.

    Returns:
        Tuple of ``(r_value, interpretation)`` where ``interpretation``
        is ``'negligible'``, ``'small'``, ``'medium'``, or ``'large'``.
    """

    _check_scipy()

    n1, n2 = len(group1), len(group2)

    try:
        U, _ = mannwhitneyu(group1, group2, alternative="two-sided")
        # Convert to rank-biserial correlation
        r = 1 - (2 * U) / (n1 * n2)
    except Exception:
        r = 0.0

    interpretation = _interpret_rank_biserial(abs(r))

    return r, interpretation


def _interpret_rank_biserial(abs_r: float) -> str:
    """Interpret rank-biserial correlation magnitude."""
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def eta_squared(groups: List[pd.Series]) -> Tuple[float, str]:
    """Calculate eta-squared effect size for ANOVA or Kruskal-Wallis.

    Represents the proportion of variance in the dependent variable
    explained by group membership. Thresholds: small (0.01),
    medium (0.06), large (0.14).

    Args:
        groups: List of ``pd.Series``, one per group.

    Returns:
        Tuple of ``(eta_sq, interpretation)`` where ``interpretation``
        is ``'negligible'``, ``'small'``, ``'medium'``, or ``'large'``.
    """

    # Calculate sums of squares
    all_data = pd.concat(groups, ignore_index=True)
    grand_mean = all_data.mean()

    # Total sum of squares
    ss_total = ((all_data - grand_mean) ** 2).sum()

    # Between-group sum of squares
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)

    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    interpretation = _interpret_eta_squared(eta_sq)

    return eta_sq, interpretation


def _interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared effect size magnitude."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


# MULTIPLE COMPARISON CORRECTIONS


def apply_multiple_comparison_correction(
    p_values: List[float], method: str = "holm"
) -> Tuple[List[float], str]:
    """Apply multiple comparison correction to a list of p-values.

    Adjusts p-values to control the family-wise error rate when
    performing many pairwise comparisons.

    Args:
        p_values: List of raw p-values from pairwise tests.
        method: Correction method — ``'bonferroni'``, ``'holm'``
            (recommended), or ``'fdr_bh'`` (Benjamini-Hochberg).
            Default ``'holm'``.
        alpha: Significance level. Default 0.05.

    Returns:
        List of corrected p-values in the same order as the input.
    """

    p_array = np.array(p_values)
    n = len(p_array)

    if method == "bonferroni":
        corrected = p_array * n
        corrected = np.minimum(corrected, 1.0)
        description = f"Bonferroni correction (α adjusted by factor of {n})"

    elif method == "holm":
        # Sort p-values with original indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Apply Holm correction
        corrected_sorted = np.minimum.accumulate(sorted_p * (n - np.arange(n)))
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        # Restore original order
        corrected = np.empty_like(corrected_sorted)
        corrected[sorted_indices] = corrected_sorted

        description = "Holm step-down correction (less conservative than Bonferroni)"

    elif method == "fdr_bh":  # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # BH procedure
        corrected_sorted = sorted_p * n / (np.arange(n) + 1)
        corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        corrected = np.empty_like(corrected_sorted)
        corrected[sorted_indices] = corrected_sorted

        description = "Benjamini-Hochberg FDR correction (controls false discovery rate)"

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist(), description


# STATISTICAL TESTS


def mann_whitney_test(
    model1_metrics: pd.Series,
    model2_metrics: pd.Series,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Mann-Whitney U test for comparing two independent groups.

    A non-parametric test that does not assume normal distributions.
    Includes sample-size validation, independence checks, rank-biserial
    effect size, and auto-generated interpretation text.

    Args:
        model1_metrics: Metric values for the first model.
        model2_metrics: Metric values for the second model.
        alternative: ``'two-sided'``, ``'less'``, or ``'greater'``.
            Default ``'two-sided'``.
        alpha: Significance level for assumption checks. Default 0.05.

    Returns:
        :class:`StatisticalTestResult` with test statistic, p-value,
        rank-biserial correlation effect size, and interpretation.

    Raises:
        TypeError: If inputs are not ``pd.Series``.
    """
    _check_scipy()

    result = StatisticalTestResult(
        test_name="Mann-Whitney U Test", statistic=float("nan"), p_value=float("nan")
    )

    # Validate inputs
    if not isinstance(model1_metrics, pd.Series) or not isinstance(model2_metrics, pd.Series):
        raise TypeError("Inputs must be pandas Series.")

    # Clean data
    clean1 = model1_metrics.dropna()
    clean2 = model2_metrics.dropna()

    result.sample_sizes = {"group1": len(clean1), "group2": len(clean2)}

    # Sample size validation
    adequate_size, size_warnings = validate_sample_sizes([clean1, clean2], 5, "Mann-Whitney U test")
    result.warnings.extend(size_warnings)
    result.assumptions_met["adequate_sample_size"] = adequate_size

    # Independence check
    is_indep1, indep_details1 = check_independence(clean1)
    is_indep2, indep_details2 = check_independence(clean2)

    if not is_indep1 or not is_indep2:
        result.warnings.append(
            "Data shows evidence of autocorrelation, violating independence assumption"
        )

    result.assumptions_met["independence"] = is_indep1 and is_indep2
    result.assumption_details["independence"] = {"group1": indep_details1, "group2": indep_details2}

    # Perform test
    try:
        statistic, p_value = mannwhitneyu(clean1, clean2, alternative=alternative)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

        # Effect size
        r, r_interpretation = rank_biserial_correlation(clean1, clean2)
        result.effect_size = r
        result.effect_size_name = "rank-biserial correlation"
        result.effect_size_interpretation = r_interpretation

    except Exception as e:
        result.warnings.append(f"Test failed: {str(e)}")
        result.statistic = float("nan")
        result.p_value = float("nan")
        return result

    # Generate interpretation
    result.conclusion = _generate_mann_whitney_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    # Recommendations
    if not adequate_size:
        result.recommendations.append("Collect more data for more reliable results")
    if result.effect_size is not None and abs(result.effect_size) < 0.1:
        result.recommendations.append(
            "Consider if this small effect size is practically meaningful"
        )

    return result


def wilcoxon_signed_rank_test(
    model_metrics: pd.Series,
    null_value: float = 0.5,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Wilcoxon signed-rank test for a single sample against a null value.

    Tests whether the median of the sample differs from ``null_value``.
    Useful for testing whether a model's accuracy is significantly
    different from chance.

    Args:
        model_metrics: Metric values for the model.
        null_value: Hypothesized median to test against. Default 0.5.
        alternative: ``'two-sided'``, ``'less'``, or ``'greater'``.
            Default ``'two-sided'``.
        alpha: Significance level. Default 0.05.

    Returns:
        :class:`StatisticalTestResult` with test statistic, p-value,
        effect size (Wilcoxon r), and interpretation.
    """
    _check_scipy()

    result = StatisticalTestResult(
        test_name="Wilcoxon Signed-Rank Test", statistic=float("nan"), p_value=float("nan")
    )

    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    # Clean data and center on null value
    clean_data = model_metrics.dropna()
    centered_data = clean_data - null_value

    # Remove zeros (ties at the null hypothesis value)
    non_zero_data = centered_data[centered_data != 0]

    result.sample_sizes = {"total": len(clean_data), "non_zero": len(non_zero_data)}

    # Sample size validation
    adequate_size, size_warnings = validate_sample_sizes(
        [non_zero_data], 6, "Wilcoxon signed-rank test"
    )
    result.warnings.extend(size_warnings)
    result.assumptions_met["adequate_sample_size"] = adequate_size

    # Symmetry assumption check (approximate)
    if len(centered_data) > 0:
        skewness = centered_data.skew()
        if abs(skewness) > 1:
            result.warnings.append("Data appears highly skewed, violating symmetry assumption")
            result.assumptions_met["symmetry"] = False
        else:
            result.assumptions_met["symmetry"] = True

        result.assumption_details["skewness"] = skewness

        # Perform test
        try:
            if len(non_zero_data) < 6:
                raise ValueError("Insufficient non-zero differences for Wilcoxon test")

            statistic, p_value = wilcoxon(non_zero_data, alternative=alternative)
            result.statistic = float(statistic)
            result.p_value = float(p_value)

            # Effect size (r = Z / sqrt(N))
            n = len(non_zero_data)
            if n > 0:
                z_score = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                r = abs(z_score) / np.sqrt(n)

                result.effect_size = r
                result.effect_size_name = "r (effect size)"
                result.effect_size_interpretation = _interpret_wilcoxon_r(r)

        except Exception as e:
            result.warnings.append(f"Test failed: {str(e)}")
            result.statistic = float("nan")
            result.p_value = float("nan")
            return result

    # Generate interpretation
    result.conclusion = _generate_wilcoxon_conclusion(result, null_value, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def anova_test(model_metrics: Dict[str, pd.Series], alpha: float = 0.05) -> StatisticalTestResult:
    """One-way ANOVA for comparing three or more independent groups.

    A parametric test that assumes normality and equal variances. If
    assumptions are violated, consider :func:`kruskal_wallis_test` instead.
    Includes eta-squared effect size.

    Args:
        model_metrics: Dict mapping model names to ``pd.Series`` of metric
            values.
        alpha: Significance level. Default 0.05.

    Returns:
        :class:`StatisticalTestResult` with F-statistic, p-value,
        eta-squared effect size, and interpretation.
    """
    _check_scipy()

    result = StatisticalTestResult(
        test_name="One-Way ANOVA", statistic=float("nan"), p_value=float("nan")
    )

    if len(model_metrics) < 2:
        raise ValueError("ANOVA requires at least two groups to compare.")

    # Clean data
    clean_groups = []
    group_names = []
    for name, series in model_metrics.items():
        clean_data = series.dropna()
        if len(clean_data) > 0:
            clean_groups.append(clean_data)
            group_names.append(name)

    result.sample_sizes = {name: len(group) for name, group in zip(group_names, clean_groups)}

    # Validation
    adequate_size, size_warnings = validate_sample_sizes(clean_groups, 3, "ANOVA")
    result.warnings.extend(size_warnings)
    result.assumptions_met["adequate_sample_size"] = adequate_size

    # Normality check for each group
    normality_results = {}
    all_normal = True
    for i, (name, group) in enumerate(zip(group_names, clean_groups)):
        is_normal, norm_details = check_normality(group)
        normality_results[name] = norm_details
        if not is_normal:
            all_normal = False

    result.assumptions_met["normality"] = all_normal
    result.assumption_details["normality"] = normality_results

    if not all_normal:
        result.warnings.append(
            "Some groups violate normality assumption - consider Kruskal-Wallis test"
        )
        result.recommendations.append("Consider using Kruskal-Wallis test for non-normal data")

    # Equal variance check (Levene's test for multiple groups)
    try:
        levene_stat, levene_p = levene(*clean_groups)
        equal_vars = levene_p > alpha
        result.assumptions_met["equal_variances"] = equal_vars
        result.assumption_details["variance_test"] = {
            "levene_statistic": levene_stat,
            "levene_p": levene_p,
        }

        if not equal_vars:
            result.warnings.append("Groups have unequal variances")
    except Exception:
        result.warnings.append("Could not test equal variances assumption")

    # Perform ANOVA
    try:
        statistic, p_value = f_oneway(*clean_groups)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

        # Effect size (eta-squared)
        eta_sq, eta_interpretation = eta_squared(clean_groups)
        result.effect_size = eta_sq
        result.effect_size_name = "eta-squared"
        result.effect_size_interpretation = eta_interpretation

    except Exception as e:
        result.warnings.append(f"ANOVA failed: {str(e)}")
        result.statistic = float("nan")
        result.p_value = float("nan")
        return result

    # Generate interpretation
    result.conclusion = _generate_anova_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def kruskal_wallis_test(
    model_metrics: Dict[str, pd.Series], alpha: float = 0.05
) -> StatisticalTestResult:
    """Kruskal-Wallis H-test for comparing three or more independent groups.

    A non-parametric alternative to ANOVA that does not assume normality
    or equal variances. Recommended when sample sizes are small or
    distributions are skewed.

    Args:
        model_metrics: Dict mapping model names to ``pd.Series`` of metric
            values.
        alpha: Significance level. Default 0.05.

    Returns:
        :class:`StatisticalTestResult` with H-statistic, p-value,
        eta-squared effect size, and interpretation.
    """
    _check_scipy()

    # Validate input
    if len(model_metrics) < 2:
        raise ValueError("Kruskal-Wallis test requires at least two groups to compare.")

    # Clean data - remove NaN values from each group
    clean_groups = []
    group_names = []
    for name, series in model_metrics.items():
        clean_data = series.dropna()
        if len(clean_data) > 0:
            clean_groups.append(clean_data)
            group_names.append(name)

    # Check for sufficient groups after cleaning
    if not clean_groups or len(clean_groups) < 2:
        raise ValueError(
            "Insufficient data after cleaning. Kruskal-Wallis test requires "
            "at least two non-empty groups."
        )

    # Validate sample sizes
    adequate_size, size_warnings = validate_sample_sizes(clean_groups, 3, "Kruskal-Wallis test")

    try:
        # Perform the Kruskal-Wallis test
        statistic, p_value = kruskal(*clean_groups)

        # Create result object with required parameters
        result = StatisticalTestResult(
            test_name="Kruskal-Wallis H-Test", statistic=float(statistic), p_value=float(p_value)
        )

        # Add sample size information
        result.sample_sizes = {name: len(group) for name, group in zip(group_names, clean_groups)}

        # Add validation warnings
        result.warnings.extend(size_warnings)
        result.assumptions_met["adequate_sample_size"] = adequate_size

        # Calculate effect size (epsilon-squared)
        N = sum(len(group) for group in clean_groups)
        k = len(clean_groups)

        if N > k:
            epsilon_sq = (statistic - k + 1) / (N - k)
            epsilon_sq = max(0.0, min(1.0, epsilon_sq))  # Ensure non-negative
        else:
            epsilon_sq = 0

        result.effect_size = epsilon_sq
        result.effect_size_name = "epsilon-squared"
        result.effect_size_interpretation = _interpret_eta_squared(epsilon_sq)

    except Exception as e:
        # Create a failure result object for consistent return type
        result = StatisticalTestResult(
            test_name="Kruskal-Wallis H-Test", statistic=float("nan"), p_value=float("nan")
        )
        result.warnings.append(f"Kruskal-Wallis test failed: {str(e)}")
        result.sample_sizes = {name: len(group) for name, group in zip(group_names, clean_groups)}
        return result

    # Generate interpretation
    result.conclusion = _generate_kruskal_wallis_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def shapiro_wilk_test(model_metrics: pd.Series, alpha: float = 0.05) -> StatisticalTestResult:
    """Shapiro-Wilk test for normality.

    Tests whether a sample comes from a normally distributed population.
    Used internally by :func:`compare_two_models` to select between
    parametric and non-parametric tests.

    Args:
        model_metrics: Metric values to test.
        alpha: Significance level. Default 0.05. A significant result
            (p < alpha) indicates the data is *not* normally distributed.

    Returns:
        :class:`StatisticalTestResult` with W-statistic, p-value, and
        interpretation.
    """
    _check_scipy()

    result = StatisticalTestResult(
        test_name="Shapiro-Wilk Normality Test", statistic=float("nan"), p_value=float("nan")
    )

    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    clean_data = model_metrics.dropna()
    result.sample_sizes = {"total": len(clean_data)}

    if len(clean_data) < 3:
        result.warnings.append("Shapiro-Wilk test requires at least 3 observations")
        result.statistic = float("nan")
        result.p_value = float("nan")
        return result

    if len(clean_data) > 5000:
        result.warnings.append(
            "Shapiro-Wilk test may not be reliable for very large samples (n>5000)"
        )

    try:
        statistic, p_value = shapiro(clean_data)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

    except Exception as e:
        result.warnings.append(f"Shapiro-Wilk test failed: {str(e)}")
        result.statistic = float("nan")
        result.p_value = float("nan")
        return result

    # Generate interpretation
    result.conclusion = _generate_shapiro_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    # Add recommendation about sample size
    if len(clean_data) < 20:
        result.recommendations.append(
            "Consider collecting more data for more reliable normality assessment"
        )
    elif len(clean_data) > 1000:
        result.recommendations.append(
            "For large samples, consider visual methods (Q-Q plots) alongside statistical tests"
        )

    return result


# HIGH-LEVEL COMPARISON FUNCTIONS


def compare_two_models(
    model1_results: pd.Series, model2_results: pd.Series, paired: bool = False, alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Compares two models using intelligent, assumption-driven test selection.

    This function automatically selects the correct statistical test based
    on the data's properties and whether the samples are paired.

    - If `paired=True`, performs a Wilcoxon signed-rank test on the differences.
    - If `paired=False`, it performs an "intelligent independent test":
        1. Checks for normality in both groups (using `check_normality`).
        2. If both are normal, checks for equal variance (using `check_equal_variances`).
        3.  - Normal + Equal Variance: Runs a Student's t-test.
        4.  - Normal + Unequal Variance: Runs a Welch's t-test.
        5.  - Not Normal: Runs a Mann-Whitney U test.

    Args:
        model1_results (pd.Series): A Series of metric results for model 1.
        model2_results (pd.Series): A Series of metric results for model 2.
        paired (bool, optional): Whether the samples are paired (e.g.,
            results from the same k-folds). Defaults to False.
        alpha (float, optional): Significance level used for assumption checks
            (normality, variance). Defaults to 0.05.

    Returns:
        StatisticalTestResult: A rich object containing the test name,
        statistic, p-value, effect size, and details on which
        assumptions were met.
    """

    # Clean data first
    if paired:
        # For paired data, remove rows where either value is missing
        combined = pd.DataFrame({"model1": model1_results, "model2": model2_results})
        combined_clean = combined.dropna()
        clean1 = combined_clean["model1"]
        clean2 = combined_clean["model2"]
    else:
        clean1 = model1_results.dropna()
        clean2 = model2_results.dropna()

    # Check sample sizes
    if len(clean1) < 6 or len(clean2) < 6:
        result = StatisticalTestResult(
            test_name="Insufficient Data",
            statistic=float("nan"),  # Required argument
            p_value=float("nan"),  # Required argument
        )
        result.warnings.append("Insufficient data for reliable statistical testing")
        result.recommendations.append("Collect more data (at least 6 samples per group)")
        return result

    if paired:
        # For paired comparisons, test the differences
        differences = clean1 - clean2
        result = wilcoxon_signed_rank_test(differences, null_value=0, alpha=alpha)
        result.test_name = "Paired Comparison (Wilcoxon Signed-Rank)"
    else:
        # For independent comparisons, use intelligent test selection

        # 1. Check assumptions
        is_normal1, norm_details1 = check_normality(clean1, alpha=alpha)
        is_normal2, norm_details2 = check_normality(clean2, alpha=alpha)

        assumptions_met = {"normality_group1": is_normal1, "normality_group2": is_normal2}
        assumption_details = {"normality_group1": norm_details1, "normality_group2": norm_details2}

        if is_normal1 and is_normal2:
            # 2. Both are normal: Check variances
            equal_vars, var_details = check_equal_variances(clean1, clean2, alpha=alpha)
            assumptions_met["equal_variances"] = equal_vars
            assumption_details["variance_test"] = var_details

            if equal_vars:
                # 3a. Normal + Equal Variance = Student's t-test
                test_name = "Independent Comparison (Student's t-test)"
                statistic, p_value = ttest_ind(clean1, clean2, equal_var=True)
                effect_size, es_interp = cohens_d(clean1, clean2, pooled=True)
                es_name = "Cohen's d"
            else:
                # 3b. Normal + Unequal Variance = Welch's t-test
                test_name = "Independent Comparison (Welch's t-test)"
                statistic, p_value = ttest_ind(clean1, clean2, equal_var=False)
                effect_size, es_interp = cohens_d(clean1, clean2, pooled=False)
                es_name = "Cohen's d (unpooled)"

            result = StatisticalTestResult(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_name=es_name,
                effect_size_interpretation=es_interp,
            )

        else:
            # 4. Not normal: Use Mann-Whitney U test
            result = mann_whitney_test(clean1, clean2, alpha=alpha)
            result.test_name = "Independent Comparison (Mann-Whitney U)"

        # Add assumption info to the final result
        result.assumptions_met.update(assumptions_met)
        result.assumption_details.update(assumption_details)

    # --- Bootstrap confidence intervals ---
    if HAS_BOOTSTRAP and not np.isnan(result.p_value):
        try:
            if paired:
                ci_result = bootstrap_paired_difference_ci(
                    clean1, clean2, n_bootstrap=10000, confidence=1 - alpha, method="bca"
                )
            else:
                ci_result = bootstrap_mean_difference_ci(
                    clean1, clean2, n_bootstrap=10000, confidence=1 - alpha, method="bca"
                )

            result.confidence_interval = (ci_result.ci_lower, ci_result.ci_upper)
            result.ci_confidence_level = ci_result.confidence_level
            result.ci_method = ci_result.method

            # Also compute CI for the effect size (Cohen's d)
            if result.effect_size is not None:
                es_ci = bootstrap_effect_size_ci(
                    clean1,
                    clean2,
                    n_bootstrap=10000,
                    confidence=1 - alpha,
                    method="bca",
                    pooled=True,
                )
                result.ci_effect_size = (es_ci.ci_lower, es_ci.ci_upper)

        except Exception:
            # Bootstrap is best-effort — never break an otherwise valid result
            pass

    return result


def compare_multiple_models(
    model_results: Dict[str, pd.Series], alpha: float = 0.05, correction_method: str = "holm"
) -> Dict[str, Any]:
    """
    Compares three or more models using a robust, two-step procedure.

    This function is the standard way to compare multiple groups in a
    statistically sound manner, protecting against inflating the error rate
    by running too many tests.

    The procedure is:
    1.  **Omnibus Test:** First, it runs a single "overall" test
        (Kruskal-Wallis) to see if *any* significant difference exists
        among all the model groups.
    2.  **Post-Hoc Tests:** If, and only if, the omnibus test is significant,
        it then performs pairwise Mann-Whitney U tests for every possible
        combination of models to find out *which specific pairs* are
        different.
    3.  **Correction:** The p-values from all post-hoc tests are automatically
        corrected using the specified `correction_method` (e.g., 'holm')
        to control the family-wise error rate.

    Args:
        model_results (Dict[str, pd.Series]): A dictionary where keys are
            model names (str) and values are the metric results (pd.Series)
            for that model.
        alpha (float, optional): The significance level to use for the tests.
            Defaults to 0.05.
        correction_method (str, optional): The method to use for correcting
            p-values in the post-hoc pairwise comparisons.
            Options: 'holm', 'bonferroni', 'fdr_bh'. Defaults to 'holm'.

    Returns:
        Dict[str, Any]: A dictionary containing the complete analysis, with
        keys:
            - 'overall_test' (StatisticalTestResult): The result of the
              Kruskal-Wallis omnibus test.
            - 'pairwise_comparisons' (Dict[str, StatisticalTestResult]):
              A dictionary of results for each pair, (e.g., "ModelA_vs_ModelB").
            - 'significant_comparisons' (List[str]): A list of the names
              of pairs that were significant after correction.
            - 'correction_method' (str): The correction method used.
            - 'message' (str): A message if no pairwise tests were
              performed (because the overall test was not significant).

    Raises:
        ValueError: If fewer than two models are provided in `model_results`.
    """

    model_names = list(model_results.keys())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError("Need at least 2 models to compare")

    # Overall test first (Kruskal-Wallis - more robust than ANOVA)
    overall_result = kruskal_wallis_test(model_results, alpha=alpha)

    results = {
        "overall_test": overall_result,
        "pairwise_comparisons": {},
        "correction_method": correction_method,
        "family_wise_error_rate": alpha,
        "n_comparisons": n_models * (n_models - 1) // 2,
    }

    if overall_result.is_significant(alpha):
        # Perform pairwise comparisons
        pairwise_tests = []
        pairwise_names = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                name1, name2 = model_names[i], model_names[j]
                comparison_name = f"{name1}_vs_{name2}"

                pairwise_result = mann_whitney_test(
                    model_results[name1], model_results[name2], alpha=alpha
                )

                pairwise_tests.append(pairwise_result)
                pairwise_names.append(comparison_name)

                results["pairwise_comparisons"][comparison_name] = pairwise_result

        # Apply multiple comparison correction
        p_values = [test.p_value for test in pairwise_tests]
        corrected_p_values, correction_description = apply_multiple_comparison_correction(
            p_values, method=correction_method
        )

        # Update results with corrected p-values
        for i, (name, test) in enumerate(zip(pairwise_names, pairwise_tests)):
            test.corrected_p_value = corrected_p_values[i]
            test.correction_method = correction_method

        results["correction_description"] = correction_description
        results["significant_comparisons"] = [
            name
            for name, test in results["pairwise_comparisons"].items()
            if test.is_significant(alpha)
        ]
    else:
        results["message"] = "Overall test not significant - no pairwise comparisons performed"
        results["significant_comparisons"] = []

    return results


# UTILITY FUNCTIONS FOR INTERPRETATION


def _generate_mann_whitney_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Mann-Whitney test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Mann-Whitney U test indicates a statistically significant difference between groups (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Mann-Whitney U test shows no statistically significant difference between groups (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_wilcoxon_conclusion(
    result: StatisticalTestResult, null_value: float, alpha: float
) -> str:
    """Generate conclusion for Wilcoxon test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Wilcoxon signed-rank test indicates the median differs significantly from {null_value} (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Wilcoxon signed-rank test shows no significant difference from {null_value} (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_anova_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for ANOVA test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"One-way ANOVA indicates significant differences between group means (F={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"One-way ANOVA shows no significant differences between group means (F={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_kruskal_wallis_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Kruskal-Wallis test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Kruskal-Wallis test indicates significant differences between group distributions (H={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Kruskal-Wallis test shows no significant differences between group distributions (H={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_shapiro_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Shapiro-Wilk test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Shapiro-Wilk test indicates the sample is not normally distributed (W={result.statistic:.4f}, p={p_val:.4f})"
    else:
        conclusion = f"Shapiro-Wilk test is consistent with normal distribution (W={result.statistic:.4f}, p={p_val:.4f})"

    return conclusion


def _generate_detailed_interpretation(result: StatisticalTestResult, alpha: float) -> str:
    """Generate detailed interpretation of statistical test result."""
    interpretation = [result.conclusion]

    # Sample size information
    if result.sample_sizes:
        sample_info = ", ".join([f"{k}: {v}" for k, v in result.sample_sizes.items()])
        interpretation.append(f"Sample sizes: {sample_info}")

    # Assumption violations
    if result.warnings:
        interpretation.append("Warnings:")
        for warning in result.warnings:
            interpretation.append(f"  - {warning}")

    # Multiple comparison correction
    if result.corrected_p_value is not None:
        interpretation.append(
            f"P-value corrected for multiple comparisons using {result.correction_method}"
        )

    # Recommendations
    if result.recommendations:
        interpretation.append("Recommendations:")
        for rec in result.recommendations:
            interpretation.append(f"  - {rec}")

    return "\n".join(interpretation)


def _interpret_wilcoxon_r(r: float) -> str:
    """Interpret Wilcoxon r effect size."""
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


# CONVERGENCE ASSESSMENT (enhanced existing functions)


def calculate_autocorr(data: Union[pd.Series, List[float]], lag: int = 1) -> Optional[float]:
    """Calculate autocorrelation of a metric series at a given lag.

    Args:
        data: A ``pd.Series`` or list of metric values ordered by run.
        lag: The lag at which to compute autocorrelation. Default 1.

    Returns:
        The autocorrelation coefficient as a float, or ``None`` if the
        series is too short or the computation fails.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) <= lag:
        return None

    try:
        return data.autocorr(lag)
    except Exception:
        return None


def calculate_averaged_autocorr(
    histories: List[pd.Series], max_lag: int = 20
) -> Tuple[List[float], List[float], List[float]]:
    """Compute mean and std of autocorrelation across multiple run histories.

    Args:
        histories: List of ``pd.Series``, each representing a training
            history (e.g. per-epoch loss values from one run).
        max_lag: Maximum lag to compute. Default 20.

    Returns:
        Tuple of ``(lags, mean_autocorr, std_autocorr)`` — three lists
        of equal length.

    Raises:
        ValueError: If ``histories`` is empty or none are long enough for
            the requested ``max_lag``.
    """
    if not histories:
        raise ValueError("The list of histories cannot be empty.")

    # Filter out histories that are too short
    valid_histories = [h for h in histories if len(h) > max_lag]

    if not valid_histories:
        raise ValueError(f"No histories long enough for max_lag={max_lag}")

    autocorrs = []
    for history in valid_histories:
        autocorr_values = []
        for lag in range(1, max_lag + 1):
            autocorr = calculate_autocorr(history, lag)
            autocorr_values.append(autocorr if autocorr is not None else np.nan)
        autocorrs.append(autocorr_values)

    autocorrs_np = np.array(autocorrs)
    mean_autocorr = np.nanmean(autocorrs_np, axis=0)
    std_autocorr = np.nanstd(autocorrs_np, axis=0)

    lags = list(range(1, max_lag + 1))

    return lags, mean_autocorr.tolist(), std_autocorr.tolist()


def check_convergence(
    data: Union[pd.Series, List[float]], window_size: int = 5, autocorr_threshold: float = 0.1
) -> bool:
    """Check whether a metric series has converged.

    Uses two criteria (either sufficient): low autocorrelation in a
    recent window, or low recent variance relative to overall variance.

    Args:
        data: A ``pd.Series`` or list of metric values.
        window_size: Number of recent values to examine. Default 5.
        autocorr_threshold: Maximum acceptable autocorrelation at lag 1
            for convergence. Default 0.1.

    Returns:
        ``True`` if either convergence criterion is met.
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) < window_size + 1:
        return False

    # Get recent window
    recent_data = data.iloc[-window_size:]

    # Primary criterion: low autocorrelation
    autocorr = calculate_autocorr(recent_data, lag=1)
    if autocorr is None:
        return False

    low_autocorr = abs(autocorr) < autocorr_threshold

    # Secondary criterion: low variance in recent window relative to overall
    if len(data) > window_size:
        recent_variance = recent_data.var()
        overall_variance = data.var()

        if overall_variance > 0:
            low_recent_variance = recent_variance < 0.1 * overall_variance
        else:
            low_recent_variance = True
    else:
        low_recent_variance = False

    return low_autocorr or low_recent_variance


# CONFUSION MATRIX FUNCTION (enhanced)


def get_confusion_matrix_df(
    predictions: np.ndarray, true_labels: np.ndarray, class_names: Dict[int, str]
) -> pd.DataFrame:
    """Generate confusion matrix DataFrame from predictions and true labels.""" """Generate a confusion matrix as a labeled DataFrame.

Args:
    predictions: Array of predicted class indices.
    true_labels: Array of true class indices.
    class_names: Dict mapping class indices to human-readable names,
        e.g. ``{0: 'cat', 1: 'dog'}``.

Returns:
    A square ``pd.DataFrame`` with class names as both row and column
    labels, suitable for :func:`~ictonyx.plotting.plot_confusion_matrix`.

Raises:
    ValueError: If ``predictions`` and ``true_labels`` differ in length.
"""

    _check_sklearn()

    if len(predictions) != len(true_labels):
        raise ValueError("predictions and true_labels must have the same length")

    class_indices = sorted(class_names.keys())
    cm = confusion_matrix(true_labels, predictions, labels=class_indices)

    return pd.DataFrame(
        cm,
        index=[class_names[i] for i in class_indices],
        columns=[class_names[i] for i in class_indices],
    )


# SUMMARY AND REPORTING FUNCTIONS


def generate_statistical_summary(results: List[StatisticalTestResult]) -> str:
    """
    Generates a human-readable, multi-line string summary of test results.

    This function takes a list of `StatisticalTestResult` objects and formats
    them into a clean, printable report. The report includes:
    - An overall summary (total tests, number significant).
    - A one-line summary for each test (e.g., "Mann-Whitney U: ... p=0.002 *").
    - The effect size and interpretation for each test.
    - Any major warnings (e.g., assumption violations) for each test.
    - A de-duplicated list of key recommendations.

    Args:
        results (List[StatisticalTestResult]): A list of one or more
            `StatisticalTestResult` objects to summarize.

    Returns:
        str: A formatted, multi-line string ready to be printed to the
        console or saved to a file.
    """
    if not results:
        return "No statistical test results to summarize."

    summary_lines = ["Statistical Analysis Summary", "=" * 30, ""]

    # Count significant results
    significant_results = [r for r in results if r.is_significant()]

    summary_lines.append(f"Tests performed: {len(results)}")
    summary_lines.append(
        f"Significant results: {len(significant_results)} ({100 * len(significant_results) / len(results):.1f}%)"
    )
    summary_lines.append("")

    # Individual test summaries
    for result in results:
        summary_lines.append(result.get_summary())

        # Add effect size context
        if result.effect_size is not None:
            effect_context = f"  Effect size ({result.effect_size_name}): {result.effect_size:.3f} ({result.effect_size_interpretation})"
            summary_lines.append(effect_context)

        # Add confidence interval
        if result.confidence_interval is not None:
            lo, hi = result.confidence_interval
            ci_line = f"  {result.ci_confidence_level*100:.0f}% CI for difference: [{lo:.4f}, {hi:.4f}] ({result.ci_method})"
            summary_lines.append(ci_line)
            if result.ci_effect_size is not None:
                es_lo, es_hi = result.ci_effect_size
                summary_lines.append(
                    f"  {result.ci_confidence_level*100:.0f}% CI for effect size: [{es_lo:.4f}, {es_hi:.4f}]"
                )

        # Add major warnings
        if result.warnings:
            major_warnings = [
                w
                for w in result.warnings
                if "assumption" in w.lower() or "insufficient" in w.lower()
            ]
            for warning in major_warnings:
                summary_lines.append(f"  ⚠ {warning}")

        summary_lines.append("")

    # Overall recommendations
    all_recommendations = [r for result in results for r in result.recommendations]

    if all_recommendations:
        summary_lines.append("Key Recommendations:")
        # Get unique recommendations
        unique_recs = list(set(all_recommendations))
        for rec in unique_recs:
            summary_lines.append(f"  - {rec}")
        summary_lines.append("")

    return "\n".join(summary_lines)


def create_results_dataframe(results: List[StatisticalTestResult]) -> pd.DataFrame:
    """
    Converts a list of statistical results into a structured pandas DataFrame.

    This function is ideal for programmatic analysis or for exporting results
    to a file (e.g., a CSV). Each row in the returned DataFrame represents
    a single test result.

    Args:
        results (List[StatisticalTestResult]): A list of one or more
            `StatisticalTestResult` objects.

    Returns:
        pd.DataFrame: A DataFrame where each row is a test. Key columns
        include:
            - 'test_name': The name of the test performed.
            - 'statistic': The test statistic (e..g, U, H, or F-value).
            - 'p_value': The original, uncorrected p-value.
            - 'significant': Boolean (True/False) if the test is significant.
            - 'effect_size': The calculated effect size.
            - 'effect_size_interpretation': (e.g., "small", "large").
            - 'n_warnings': Count of warnings (e.g., assumption violations).
            - 'assumptions_met': Boolean (True/False) if all assumptions passed.
            - 'n_group1', 'n_group2', etc.: Sample sizes for each group.
            - 'corrected_p_value': The p-value after correction (if applied).
            - 'correction_method': The correction method used (if applied).
    """
    if not results:
        return pd.DataFrame()

    data = []
    for result in results:
        row = {
            "test_name": result.test_name,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "significant": result.is_significant(),
            "effect_size": result.effect_size,
            "effect_size_name": result.effect_size_name,
            "effect_size_interpretation": result.effect_size_interpretation,
            "n_warnings": len(result.warnings),
            "assumptions_met": (
                all(result.assumptions_met.values()) if result.assumptions_met else None
            ),
        }

        # Add sample sizes
        if result.sample_sizes:
            for key, value in result.sample_sizes.items():
                row[f"n_{key}"] = value

        # Add corrected p-value if available
        if result.corrected_p_value is not None:
            row["corrected_p_value"] = result.corrected_p_value
            row["correction_method"] = result.correction_method

        # Add confidence interval if available
        if result.confidence_interval is not None:
            row["ci_lower"] = result.confidence_interval[0]
            row["ci_upper"] = result.confidence_interval[1]
            row["ci_method"] = result.ci_method
        if result.ci_effect_size is not None:
            row["ci_effect_size_lower"] = result.ci_effect_size[0]
            row["ci_effect_size_upper"] = result.ci_effect_size[1]

        data.append(row)

    return pd.DataFrame(data)


# TRAINING STABILITY ASSESSMENT


def assess_training_stability(
    loss_histories: List[pd.Series], window_size: int = 10
) -> Dict[str, Any]:
    """
    Assesses training stability by analyzing loss histories from multiple runs.

    This function calculates a suite of metrics to quantify the consistency
    and convergence of a model's training process. It truncates all
    histories to the shortest common length for a fair comparison.

    Args:
        loss_histories (List[pd.Series]): A list where each item is a
            pandas Series representing the loss history of a single training run.
        window_size (int, optional): The number of recent epochs to use for
            calculating convergence and final window statistics. Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary containing key stability metrics:
            - 'n_runs': Number of training runs analyzed.
            - 'common_length': The number of epochs used for analysis
              (based on the shortest run).
            - 'final_loss_mean': The average loss value at the final epoch
              across all runs.
            - 'final_loss_std': The standard deviation of the final epoch loss.
            - 'final_loss_cv': Coefficient of Variation (Std / Mean) of the
              final loss. A key stability metric; lower is more stable.
            - 'final_losses_list': The raw list of all final loss values,
              used for plotting the true distribution.
            - 'convergence_rate': The percentage (0.0 to 1.0) of runs that
              were flagged as "converged" by `check_convergence`.
            - 'converged_runs': The absolute number of converged runs.
            - 'stability_assessment': A qualitative judgment ("high",
              "moderate", "low") based on the final loss CV.
            - 'between_run_variance': The variance *between* the average
              loss of each run's final window. High values mean runs
              ended in different places.
            - 'within_run_variance_mean': The average variance *within*
              the final window of each run. High values mean the runs
              were still oscillating at the end.
    """
    if not loss_histories or len(loss_histories) < 2:
        return {"error": "Need at least 2 training histories for stability assessment"}

    # Find common length (minimum across all histories)
    min_length = min(len(history) for history in loss_histories)
    if min_length < window_size:
        return {"error": f"Training histories too short for window_size={window_size}"}

    # Truncate all histories to same length
    truncated_histories = [history.iloc[:min_length] for history in loss_histories]

    # Convert to array for easier computation
    loss_array = np.array([history.values for history in truncated_histories])

    # Compute stability metrics
    final_losses = loss_array[:, -1]
    final_window_losses = loss_array[:, -window_size:]

    results = {
        "n_runs": len(loss_histories),
        "common_length": min_length,
        "final_loss_mean": np.mean(final_losses),
        "final_loss_std": np.std(final_losses),
        "final_loss_cv": (
            np.std(final_losses) / np.mean(final_losses)
            if np.mean(final_losses) > 0
            else float("inf")
        ),
        "final_window_std_mean": np.mean(
            [np.std(run_window) for run_window in final_window_losses]
        ),
        "between_run_variance": np.var(np.mean(final_window_losses, axis=1)),
        "within_run_variance_mean": np.mean(
            [np.var(run_window) for run_window in final_window_losses]
        ),
        "final_losses_list": final_losses.tolist(),
    }

    # Convergence assessment for each run
    convergence_results = []
    for i, history in enumerate(truncated_histories):
        # Handle both Series and DataFrame inputs
        if isinstance(history, pd.DataFrame):
            # For DataFrames, use the loss column for convergence check
            if "loss" in history.columns:
                converged = check_convergence(history["loss"], window_size=window_size)
            elif "val_loss" in history.columns:
                converged = check_convergence(history["val_loss"], window_size=window_size)
            else:
                # No loss column found, can't check convergence
                converged = False
        elif isinstance(history, pd.Series):
            # Already a Series, use directly
            converged = check_convergence(history, window_size=window_size)
        else:
            # Unknown type, assume not converged
            converged = False
        convergence_results.append(converged)

    results["convergence_rate"] = sum(convergence_results) / len(convergence_results)
    results["converged_runs"] = sum(convergence_results)

    # Stability interpretation
    cv = results["final_loss_cv"]
    if cv < 0.05:
        results["stability_assessment"] = "high"
    elif cv < 0.15:
        results["stability_assessment"] = "moderate"
    else:
        results["stability_assessment"] = "low"

    return results
