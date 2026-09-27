"""Invariants every comparison entry point must satisfy (0.5.0; v20 principle 12.3).

The degenerate-input contract (v20 section 11, rows D-1 to D-8) and symmetry.
Runs per push. Warnings are recorded and asserted where they are part of the
contract, so this file can run under ``-W error::UserWarning``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ictonyx.analysis import compare_multiple_models, compare_two_models

PAIRED = [True, False]


def _call(a, b, paired, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = compare_two_models(pd.Series(a), pd.Series(b), paired=paired, **kw)
    return r, [str(w.message) for w in rec]


def _ci_ok(r):
    ci = r.confidence_interval
    return ci is None or np.isnan(ci[0]) or ci[1] > ci[0]


@pytest.mark.parametrize("paired", PAIRED)
def test_d1_both_constant_equal(paired):
    r, _ = _call([0.9] * 20, [0.9] * 20, paired)
    assert np.isnan(r.p_value) and not r.is_significant() and _ci_ok(r)


@pytest.mark.parametrize("paired", PAIRED)
def test_d2_both_constant_different(paired):
    r, _ = _call([0.9167] * 20, [0.8889] * 20, paired)
    assert np.isnan(r.p_value) and not r.is_significant() and _ci_ok(r)


def test_d3_constant_difference_both_varying():
    a = np.linspace(0.80, 0.90, 20)
    r, msgs = _call(a, a - 0.05, True)
    assert np.isnan(r.p_value) and r.sample_sizes["effective_n"] == 1
    assert any("effective n = 1" in m for m in msgs)


@pytest.mark.parametrize("paired", PAIRED)
def test_d4_one_constant_one_varying_is_finite(paired):
    rng = np.random.default_rng(4)
    r, _ = _call([0.9] * 20, 0.85 + rng.normal(0, 0.01, 20), paired)
    assert np.isfinite(r.p_value)


@pytest.mark.parametrize("paired", PAIRED)
def test_d7_too_few_runs_is_refused(paired):
    rng = np.random.default_rng(5)
    r, _ = _call(rng.normal(0.9, 0.01, 5), rng.normal(0.8, 0.01, 5), paired)
    assert r.test_name == "Insufficient Data"


def test_d7_k2_multiple_models_refuses_too_few_runs():
    rng = np.random.default_rng(6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = compare_multiple_models(
            {"a": pd.Series(rng.normal(0.9, 0.01, 5)), "b": pd.Series(rng.normal(0.8, 0.01, 5))}
        )
    assert m.overall_test.test_name == "Insufficient Data"


@pytest.mark.parametrize("paired", PAIRED)
@pytest.mark.parametrize("seed", range(5))
def test_d8_no_interval_is_zero_width(paired, seed):
    rng = np.random.default_rng(100 + seed)
    a = rng.binomial(18, 0.9, 12) / 18
    b = rng.binomial(18, 0.9, 12) / 18
    r, _ = _call(a, b, paired)
    assert _ci_ok(r)


@pytest.mark.parametrize("paired", PAIRED)
def test_symmetry_swapping_models(paired):
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.80, 0.02, 20), rng.normal(0.78, 0.02, 20)
    r1, _ = _call(a, b, paired, random_state=0)
    r2, _ = _call(b, a, paired, random_state=0)
    assert r1.p_value == pytest.approx(r2.p_value)
    assert r1.effect_size == pytest.approx(-r2.effect_size)
    assert r1.point_estimate == pytest.approx(-r2.point_estimate)
