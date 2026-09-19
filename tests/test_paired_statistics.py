"""v0.4.10 C6: one sample, one scale, one sign for paired statistics.

Closes v12 1.22, 1.25, 3.8, 3.14, 3.22, 2.41, 2.66.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import ictonyx.analysis
from ictonyx.analysis import (
    _signed_rank_effect_sizes,
    anova_test,
    compare_two_models,
    paired_wilcoxon_test,
)


def test_paired_effect_size_reaches_one_on_perfect_separation():
    r = paired_wilcoxon_test(pd.Series(np.arange(1, 21.0)), pd.Series(np.zeros(20)))
    assert r.effect_size == pytest.approx(1.0)
    assert r.effect_size_name.startswith("matched-pairs")
    assert 0 < r.effect_size_secondary < 1  # Rosenthal r_z cannot reach 1


def test_paired_effect_size_is_signed_and_antisymmetric():
    a = pd.Series([0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.75])
    b = pd.Series([0.5] * 7)
    assert paired_wilcoxon_test(a, b).effect_size == pytest.approx(
        -paired_wilcoxon_test(b, a).effect_size
    )
    assert paired_wilcoxon_test(a, b).effect_size > 0


def test_reviewer1_triple_is_pratt_consistent():
    # 4 positive differences, 3 zeros. Pratt ranks zeros 1-3 and positives 4-7,
    # so r = (4+5+6+7) / (7*8/2) = 22/28 -- not 1.0 (v12 1.22, 3.8).
    a = pd.Series([0.80, 0.80, 0.80, 0.81, 0.82, 0.83, 0.84])
    b = pd.Series([0.80] * 7)
    r = paired_wilcoxon_test(a, b)
    assert r.sample_sizes == {"n_pairs": 7, "non_zero_differences": 4}
    assert r.effect_size == pytest.approx(22 / 28)
    res = compare_two_models(a, b, paired=True, random_state=0)
    assert res.sample_sizes["n_pairs"] == 7  # CI and test on the same 7 pairs


def test_paired_test_is_positional_not_index_aligned():
    a = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], index=[6, 5, 4, 3, 2, 1, 0])
    b = pd.Series([0.5] * 7)
    positional = paired_wilcoxon_test(a.reset_index(drop=True), b)
    labelled = paired_wilcoxon_test(a, b)
    assert labelled.statistic == positional.statistic
    assert labelled.effect_size == positional.effect_size


def test_paired_unequal_lengths_raise():
    with pytest.raises(ValueError, match="equal-length"):
        paired_wilcoxon_test(pd.Series([1.0, 2, 3]), pd.Series([1.0, 2]))
    with pytest.raises(ValueError, match="equal-length"):
        compare_two_models(pd.Series([1.0, 2, 3]), pd.Series([1.0, 2]), paired=True)


def test_required_runs_paired_scale_round_trip():
    # The mapping r -> shift must invert the large-sample r_mp of Normal(shift, 1).
    rng = np.random.default_rng(0)
    for r_in in (0.3, 0.5):
        shift = stats.norm.ppf((r_in + 1) / 2) / np.sqrt(2)
        r_mp, *_ = _signed_rank_effect_sizes(rng.normal(shift, 1.0, 4000))
        assert r_mp == pytest.approx(r_in, abs=0.04)


def test_effect_size_variance_formula_has_one_copy():
    # The tie-corrected variance must live in exactly one place (v12 7.39).
    src = inspect.getsource(ictonyx.analysis)
    assert src.count("(2 * m + 1) / 24.0") == 1
    assert src.count("(2 * n + 1) / 24.0") == 0


def test_anova_untestable_normality_is_not_met():
    r = anova_test(
        {
            "a": pd.Series([1.0, 2.0]),
            "b": pd.Series([1.5, 2.5, 3.5, 2.0]),
            "c": pd.Series([2.0, 2.5, 3.0, 3.5]),
        }
    )
    assert r.assumptions_met["normality"] is False


def test_pairwise_matrix_is_antisymmetric(monkeypatch):
    """Signed effect sizes must satisfy r_ji = -r_ij in the matrix plot (v12 2.66)."""
    import ictonyx.plotting as plotting
    from ictonyx.analysis import StatisticalTestResult

    def res(r):
        s = StatisticalTestResult(test_name="t", statistic=0.0, p_value=0.5)
        s.effect_size = r
        return s

    captured = []
    real_heatmap = plotting.sns.heatmap

    def spy(*args, **kwargs):
        captured.append(kwargs.get("annot"))
        return real_heatmap(*args, **kwargs)

    monkeypatch.setattr(plotting.sns, "heatmap", spy)
    pairwise = {"A_vs_B": res(0.4), "A_vs_C": res(-0.2), "B_vs_C": res(0.6)}
    plotting.plot_pairwise_comparison_matrix({"pairwise_comparisons": pairwise}, show=False)

    # The effect-size panel is the heatmap whose annotations parse as floats.
    # Panels are drawn in order p-values, significance, effect sizes; take the last.
    annot = captured[-1]
    assert _is_float_grid(annot)
    m = np.array([[float(x) if x != "" else 0.0 for x in row] for row in annot])
    assert np.allclose(m, -m.T)


def _is_float_grid(annot):
    try:
        for row in annot:
            for x in row:
                if x != "":
                    float(x)
        return any(x != "" for row in annot for x in row)
    except (TypeError, ValueError):
        return False
