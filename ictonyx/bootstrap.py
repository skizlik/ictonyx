"""
Bootstrap confidence interval estimation for Ictonyx.

Provides non-parametric confidence intervals for effect sizes, mean
differences, and arbitrary statistics. Designed for the small-sample
regime (5-30 runs) typical of ML model comparison experiments.

Methods implemented:
    - Percentile: Simple quantile-based CI. Adequate for symmetric distributions.
    - BCa (Bias-Corrected and Accelerated): Gold-standard bootstrap CI.
      Corrects for both median bias and skewness in the bootstrap distribution.
      Recommended default for most use cases.

References:
    - Efron, B. (1987). "Better Bootstrap Confidence Intervals."
      Journal of the American Statistical Association, 82(397), 171-185.
    - Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap."
      Chapman & Hall.
    - DiCiccio, T.J. & Efron, B. (1996). "Bootstrap Confidence Intervals."
      Statistical Science, 11(3), 189-228.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from scipy.stats import norm as _norm
    from scipy.special import ndtri as _ndtri
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCIResult:
    """Result of a bootstrap confidence interval estimation.

    Attributes:
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        point_estimate: The statistic computed on the original (non-resampled) data.
        confidence_level: The confidence level (e.g. 0.95 for a 95% CI).
        method: The CI construction method ('percentile' or 'bca').
        n_bootstrap: Number of bootstrap resamples used.
        se_bootstrap: Standard error of the bootstrap distribution.
        bootstrap_distribution: The full array of bootstrap replications
            (available for plotting or diagnostics).
    """
    ci_lower: float
    ci_upper: float
    point_estimate: float
    confidence_level: float
    method: str
    n_bootstrap: int
    se_bootstrap: float
    bootstrap_distribution: Optional[np.ndarray] = None

    def __str__(self) -> str:
        pct = self.confidence_level * 100
        return (
            f"{pct:.0f}% CI ({self.method}): "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(point estimate: {self.point_estimate:.4f})"
        )


# ---------------------------------------------------------------------------
#  Core bootstrap engine
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: Union[np.ndarray, pd.Series, List[float]],
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'bca',
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """Compute a bootstrap confidence interval for an arbitrary statistic.

    This is the general-purpose engine. For common use cases, prefer the
    convenience functions ``bootstrap_mean_difference_ci`` and
    ``bootstrap_effect_size_ci``, which handle two-sample data and call
    this function internally.

    Args:
        data: 1-D array-like of observations.
        statistic_fn: A callable that takes a 1-D numpy array and returns
            a scalar float. This function is applied to each bootstrap
            resample.
        n_bootstrap: Number of bootstrap resamples (default 10 000).
            Higher values give more precise CIs at the cost of compute time.
        confidence: Confidence level in (0, 1). Default 0.95.
        method: CI construction method.
            ``'percentile'`` — simple quantile method.
            ``'bca'`` — bias-corrected and accelerated (recommended).
        random_state: Seed for reproducibility. Pass an integer for
            deterministic results.
        return_distribution: If True, the full bootstrap distribution is
            stored in the result (useful for plotting). Default False.

    Returns:
        BootstrapCIResult with CI bounds, point estimate, and diagnostics.

    Raises:
        ValueError: If data has fewer than 2 observations, confidence is
            outside (0, 1), n_bootstrap < 100, or method is unrecognised.
    """
    # --- Input validation ---
    data = _to_clean_array(data)

    if len(data) < 2:
        raise ValueError(
            f"Bootstrap requires at least 2 observations, got {len(data)}."
        )
    if not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be between 0 and 1 exclusive, got {confidence}."
        )
    if n_bootstrap < 100:
        raise ValueError(
            f"n_bootstrap must be >= 100 for meaningful CIs, got {n_bootstrap}."
        )
    method = method.lower()
    if method not in ('percentile', 'bca'):
        raise ValueError(
            f"Unknown method '{method}'. Use 'percentile' or 'bca'."
        )
    if method == 'bca' and not HAS_SCIPY:
        raise ImportError(
            "scipy is required for BCa confidence intervals. "
            "Install with: pip install scipy"
        )

    rng = np.random.RandomState(random_state)
    n = len(data)

    # --- Point estimate on original data ---
    point_estimate = float(statistic_fn(data))

    # --- Generate bootstrap distribution ---
    boot_indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        boot_sample = data[boot_indices[i]]
        try:
            boot_stats[i] = statistic_fn(boot_sample)
        except Exception:
            boot_stats[i] = np.nan

    # Remove any failed bootstrap replicates
    valid_mask = np.isfinite(boot_stats)
    n_valid = valid_mask.sum()

    if n_valid < 100:
        raise ValueError(
            f"Only {n_valid} of {n_bootstrap} bootstrap replicates produced "
            f"finite values. The statistic function may be unstable for this data."
        )

    boot_stats_clean = boot_stats[valid_mask]

    # --- Compute CI ---
    alpha = 1 - confidence

    if method == 'percentile':
        ci_lower, ci_upper = _percentile_ci(boot_stats_clean, alpha)
    elif method == 'bca':
        ci_lower, ci_upper = _bca_ci(
            data, statistic_fn, boot_stats_clean, point_estimate, alpha
        )

    return BootstrapCIResult(
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        point_estimate=point_estimate,
        confidence_level=confidence,
        method=method,
        n_bootstrap=n_valid,
        se_bootstrap=float(np.std(boot_stats_clean, ddof=1)),
        bootstrap_distribution=boot_stats_clean if return_distribution else None,
    )


# ---------------------------------------------------------------------------
#  CI construction methods
# ---------------------------------------------------------------------------

def _percentile_ci(
    boot_stats: np.ndarray, alpha: float
) -> Tuple[float, float]:
    """Simple percentile confidence interval."""
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _bca_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    boot_stats: np.ndarray,
    point_estimate: float,
    alpha: float,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval.

    The BCa method adjusts the percentile interval for:
    1. Bias: the proportion of bootstrap values below the point estimate
       (z0 correction). If the bootstrap distribution is centered on the
       point estimate, z0 ≈ 0 and the correction vanishes.
    2. Acceleration: the skewness of the jackknife distribution (a
       correction). If the statistic is perfectly symmetric under
       resampling, a ≈ 0 and the method reduces to bias-corrected only.

    Falls back to percentile method if numerical issues arise (e.g.
    all bootstrap values are identical).
    """
    n = len(data)
    n_boot = len(boot_stats)

    # --- Bias correction factor (z0) ---
    # Proportion of bootstrap estimates below the point estimate
    prop_below = np.sum(boot_stats < point_estimate) / n_boot
    # Clamp to avoid infinite z-scores at 0 or 1
    prop_below = np.clip(prop_below, 1 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = _ndtri(prop_below)

    # --- Acceleration factor (a) via jackknife ---
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        try:
            jackknife_stats[i] = statistic_fn(jack_sample)
        except Exception:
            jackknife_stats[i] = np.nan

    valid_jack = jackknife_stats[np.isfinite(jackknife_stats)]

    if len(valid_jack) < 2:
        # Can't compute acceleration — fall back to percentile
        return _percentile_ci(boot_stats, alpha)

    jack_mean = np.mean(valid_jack)
    jack_diff = jack_mean - valid_jack

    denom = np.sum(jack_diff ** 2)
    if denom == 0:
        # All jackknife estimates identical — no acceleration needed
        a = 0.0
    else:
        a = np.sum(jack_diff ** 3) / (6 * denom ** 1.5)

    # --- Adjusted percentiles ---
    z_alpha_lower = _ndtri(alpha / 2)
    z_alpha_upper = _ndtri(1 - alpha / 2)

    # BCa formula for adjusted quantile positions
    def _bca_quantile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if abs(denominator) < 1e-10:
            # Degenerate case — fall back to unadjusted
            return _norm.cdf(z_alpha)
        adjusted_z = z0 + numerator / denominator
        return _norm.cdf(adjusted_z)

    q_lower = _bca_quantile(z_alpha_lower)
    q_upper = _bca_quantile(z_alpha_upper)

    # Clamp to valid percentile range
    q_lower = np.clip(q_lower, 0.5 / n_boot, 1 - 0.5 / n_boot)
    q_upper = np.clip(q_upper, 0.5 / n_boot, 1 - 0.5 / n_boot)

    ci_lower = float(np.percentile(boot_stats, 100 * q_lower))
    ci_upper = float(np.percentile(boot_stats, 100 * q_upper))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
#  Two-sample convenience functions
# ---------------------------------------------------------------------------

def bootstrap_mean_difference_ci(
    group1: Union[np.ndarray, pd.Series, List[float]],
    group2: Union[np.ndarray, pd.Series, List[float]],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'bca',
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """Bootstrap CI for the difference in means (group1 - group2).

    Resamples each group independently, computes the difference in means
    for each bootstrap replicate, and constructs a CI on that distribution.

    This is the most directly actionable CI for model comparison: it tells
    you the plausible range of the true performance gap.

    Args:
        group1: Metric values for model 1 (e.g. accuracies from 10 runs).
        group2: Metric values for model 2.
        n_bootstrap: Number of bootstrap resamples (default 10 000).
        confidence: Confidence level (default 0.95).
        method: 'percentile' or 'bca' (default 'bca').
        random_state: Seed for reproducibility.
        return_distribution: If True, store the full bootstrap distribution.

    Returns:
        BootstrapCIResult. The point_estimate is mean(group1) - mean(group2).
    """
    g1 = _to_clean_array(group1)
    g2 = _to_clean_array(group2)

    if len(g1) < 2 or len(g2) < 2:
        raise ValueError(
            f"Both groups need at least 2 observations. "
            f"Got {len(g1)} and {len(g2)}."
        )

    # Stack into single array for the engine, with a length marker
    # so the statistic function knows where to split.
    combined = np.concatenate([g1, g2])
    split_at = len(g1)

    def _mean_diff(data: np.ndarray) -> float:
        return float(np.mean(data[:split_at]) - np.mean(data[split_at:]))

    # For two-sample bootstrap, we resample each group independently.
    # This requires a custom bootstrap loop rather than the single-sample
    # engine, because resampling the combined array would lose the group
    # structure.
    return _two_sample_bootstrap(
        g1, g2,
        statistic_fn=lambda a, b: float(np.mean(a) - np.mean(b)),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        method=method,
        random_state=random_state,
        return_distribution=return_distribution,
    )


def bootstrap_effect_size_ci(
    group1: Union[np.ndarray, pd.Series, List[float]],
    group2: Union[np.ndarray, pd.Series, List[float]],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'bca',
    pooled: bool = True,
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """Bootstrap CI for Cohen's d effect size.

    Computes Cohen's d on each bootstrap resample and constructs a CI.
    This pairs naturally with the effect sizes already reported by
    ``compare_two_models`` — it adds the uncertainty range around that
    point estimate.

    Args:
        group1: Metric values for model 1.
        group2: Metric values for model 2.
        n_bootstrap: Number of bootstrap resamples (default 10 000).
        confidence: Confidence level (default 0.95).
        method: 'percentile' or 'bca' (default 'bca').
        pooled: If True (default), use pooled standard deviation. If False,
            use group2's std (Glass's delta variant).
        random_state: Seed for reproducibility.
        return_distribution: If True, store the full bootstrap distribution.

    Returns:
        BootstrapCIResult. The point_estimate is Cohen's d on the original data.
    """
    g1 = _to_clean_array(group1)
    g2 = _to_clean_array(group2)

    if len(g1) < 2 or len(g2) < 2:
        raise ValueError(
            f"Both groups need at least 2 observations. "
            f"Got {len(g1)} and {len(g2)}."
        )

    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        mean_diff = np.mean(a) - np.mean(b)
        if pooled:
            n1, n2 = len(a), len(b)
            var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            denom = np.sqrt(pooled_var)
        else:
            denom = np.std(b, ddof=1)
        return float(mean_diff / denom) if denom > 0 else 0.0

    return _two_sample_bootstrap(
        g1, g2,
        statistic_fn=_cohens_d,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        method=method,
        random_state=random_state,
        return_distribution=return_distribution,
    )


def bootstrap_paired_difference_ci(
    group1: Union[np.ndarray, pd.Series, List[float]],
    group2: Union[np.ndarray, pd.Series, List[float]],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'bca',
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """Bootstrap CI for paired mean differences.

    For paired data (e.g. two models evaluated on the same folds), this
    computes the CI on the mean of the pairwise differences. This is more
    powerful than the independent two-sample version because it removes
    between-fold variance.

    Args:
        group1: Metric values for model 1.
        group2: Metric values for model 2 (same length, paired with group1).
        n_bootstrap: Number of bootstrap resamples (default 10 000).
        confidence: Confidence level (default 0.95).
        method: 'percentile' or 'bca' (default 'bca').
        random_state: Seed for reproducibility.
        return_distribution: If True, store the full bootstrap distribution.

    Returns:
        BootstrapCIResult. The point_estimate is mean(group1 - group2).

    Raises:
        ValueError: If groups differ in length (pairing is impossible).
    """
    g1 = _to_clean_array(group1)
    g2 = _to_clean_array(group2)

    if len(g1) != len(g2):
        raise ValueError(
            f"Paired bootstrap requires equal-length groups. "
            f"Got {len(g1)} and {len(g2)}."
        )
    if len(g1) < 2:
        raise ValueError(
            f"Need at least 2 paired observations, got {len(g1)}."
        )

    differences = g1 - g2

    return bootstrap_ci(
        data=differences,
        statistic_fn=lambda d: float(np.mean(d)),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        method=method,
        random_state=random_state,
        return_distribution=return_distribution,
    )


# ---------------------------------------------------------------------------
#  Two-sample bootstrap engine (independent resampling)
# ---------------------------------------------------------------------------

def _two_sample_bootstrap(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'bca',
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """Internal engine for two-sample independent bootstrap.

    Resamples each group independently, which preserves the group structure
    and is correct for independent (unpaired) comparisons.
    """
    if not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be between 0 and 1 exclusive, got {confidence}."
        )
    if n_bootstrap < 100:
        raise ValueError(
            f"n_bootstrap must be >= 100, got {n_bootstrap}."
        )
    method = method.lower()
    if method not in ('percentile', 'bca'):
        raise ValueError(
            f"Unknown method '{method}'. Use 'percentile' or 'bca'."
        )
    if method == 'bca' and not HAS_SCIPY:
        raise ImportError(
            "scipy is required for BCa confidence intervals. "
            "Install with: pip install scipy"
        )

    rng = np.random.RandomState(random_state)
    n1, n2 = len(group1), len(group2)

    # Point estimate on original data
    point_estimate = float(statistic_fn(group1, group2))

    # Generate bootstrap distribution
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot1 = group1[rng.randint(0, n1, size=n1)]
        boot2 = group2[rng.randint(0, n2, size=n2)]
        try:
            boot_stats[i] = statistic_fn(boot1, boot2)
        except Exception:
            boot_stats[i] = np.nan

    valid_mask = np.isfinite(boot_stats)
    n_valid = valid_mask.sum()

    if n_valid < 100:
        raise ValueError(
            f"Only {n_valid} of {n_bootstrap} bootstrap replicates produced "
            f"finite values."
        )

    boot_stats_clean = boot_stats[valid_mask]
    alpha = 1 - confidence

    if method == 'percentile':
        ci_lower, ci_upper = _percentile_ci(boot_stats_clean, alpha)
    elif method == 'bca':
        ci_lower, ci_upper = _two_sample_bca_ci(
            group1, group2, statistic_fn,
            boot_stats_clean, point_estimate, alpha
        )

    return BootstrapCIResult(
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        point_estimate=point_estimate,
        confidence_level=confidence,
        method=method,
        n_bootstrap=n_valid,
        se_bootstrap=float(np.std(boot_stats_clean, ddof=1)),
        bootstrap_distribution=boot_stats_clean if return_distribution else None,
    )


def _two_sample_bca_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    boot_stats: np.ndarray,
    point_estimate: float,
    alpha: float,
) -> Tuple[float, float]:
    """BCa CI for two-sample statistics using combined jackknife."""
    n1, n2 = len(group1), len(group2)
    n_boot = len(boot_stats)

    # --- Bias correction (z0) ---
    prop_below = np.sum(boot_stats < point_estimate) / n_boot
    prop_below = np.clip(prop_below, 1 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = _ndtri(prop_below)

    # --- Acceleration via combined jackknife ---
    # Delete-one from each group in turn
    n_total = n1 + n2
    jackknife_stats = np.empty(n_total)

    for i in range(n1):
        jack1 = np.delete(group1, i)
        try:
            jackknife_stats[i] = statistic_fn(jack1, group2)
        except Exception:
            jackknife_stats[i] = np.nan

    for j in range(n2):
        jack2 = np.delete(group2, j)
        try:
            jackknife_stats[n1 + j] = statistic_fn(group1, jack2)
        except Exception:
            jackknife_stats[n1 + j] = np.nan

    valid_jack = jackknife_stats[np.isfinite(jackknife_stats)]

    if len(valid_jack) < 2:
        return _percentile_ci(boot_stats, alpha)

    jack_mean = np.mean(valid_jack)
    jack_diff = jack_mean - valid_jack
    denom = np.sum(jack_diff ** 2)

    if denom == 0:
        a = 0.0
    else:
        a = np.sum(jack_diff ** 3) / (6 * denom ** 1.5)

    # --- Adjusted percentiles ---
    z_alpha_lower = _ndtri(alpha / 2)
    z_alpha_upper = _ndtri(1 - alpha / 2)

    def _bca_quantile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if abs(denominator) < 1e-10:
            return _norm.cdf(z_alpha)
        adjusted_z = z0 + numerator / denominator
        return _norm.cdf(adjusted_z)

    q_lower = np.clip(_bca_quantile(z_alpha_lower), 0.5 / n_boot, 1 - 0.5 / n_boot)
    q_upper = np.clip(_bca_quantile(z_alpha_upper), 0.5 / n_boot, 1 - 0.5 / n_boot)

    ci_lower = float(np.percentile(boot_stats, 100 * q_lower))
    ci_upper = float(np.percentile(boot_stats, 100 * q_upper))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _to_clean_array(data: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """Convert input to a clean 1-D numpy float array, dropping NaNs."""
    if isinstance(data, pd.Series):
        arr = data.dropna().to_numpy(dtype=np.float64)
    elif isinstance(data, list):
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float64).ravel()
        arr = arr[np.isfinite(arr)]
    else:
        raise TypeError(
            f"Expected array-like, pd.Series, or list. Got {type(data).__name__}."
        )
    return arr
