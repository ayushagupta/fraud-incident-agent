import numpy as np
import pytest

from src.monitoring.metrics import (
    compute_null_rate,
    compute_psi,
    compute_prediction_drift,
    count_unseen_categories,
)

# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, 2000)
    psi = compute_psi(arr, arr.copy())
    assert psi < 0.01


def test_psi_shifted_distributions_above_threshold():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 2000)
    production = rng.normal(5, 1, 2000)  # completely separated from reference
    psi = compute_psi(reference, production)
    assert psi > 0.2


def test_psi_is_always_non_negative():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 1000)
    production = rng.normal(3, 1, 1000)
    assert compute_psi(reference, production) >= 0.0
    assert compute_psi(production, reference) >= 0.0


def test_psi_is_symmetric_ish_for_large_shift():
    # Both directions should identify a shift as substantial (same order of magnitude).
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    production = rng.normal(4, 1, 2000)
    psi_fwd = compute_psi(reference, production)
    psi_rev = compute_psi(production, reference)
    # Neither direction should be more than 10x the other.
    assert max(psi_fwd, psi_rev) / min(psi_fwd, psi_rev) < 10.0


def test_psi_returns_zero_for_empty_arrays():
    arr = np.array([1.0, 2.0, 3.0])
    assert compute_psi(arr, np.array([])) == 0.0
    assert compute_psi(np.array([]), arr) == 0.0


def test_psi_ignores_nan_in_inputs():
    rng = np.random.default_rng(4)
    base = rng.uniform(0, 1, 500)
    with_nans = np.concatenate([base, np.full(100, np.nan)])
    # PSI computed with and without NaNs should be nearly identical.
    psi_clean = compute_psi(base, base.copy())
    psi_nan = compute_psi(with_nans, with_nans.copy())
    assert abs(psi_clean - psi_nan) < 0.05


# ---------------------------------------------------------------------------
# compute_null_rate
# ---------------------------------------------------------------------------


def test_null_rate_exact_fraction():
    arr = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    assert compute_null_rate(arr) == pytest.approx(0.4)


def test_null_rate_no_nulls():
    arr = np.array([1.0, 2.0, 3.0])
    assert compute_null_rate(arr) == pytest.approx(0.0)


def test_null_rate_all_nulls():
    arr = np.array([np.nan, np.nan, np.nan])
    assert compute_null_rate(arr) == pytest.approx(1.0)


def test_null_rate_empty_array():
    assert compute_null_rate(np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# count_unseen_categories
# ---------------------------------------------------------------------------


def test_count_unseen_categories_basic():
    production = np.array(["A", "B", "C", "D"])
    known = {"A", "B"}
    assert count_unseen_categories(production, known) == 2


def test_count_unseen_categories_excludes_nulls():
    production = np.array(["A", "B", "C", None], dtype=object)
    known = {"A", "B"}
    # None is excluded; only "C" is unseen
    assert count_unseen_categories(production, known) == 1


def test_count_unseen_categories_excludes_nan_floats():
    # Numeric arrays can contain NaN when values come from a DataFrame column.
    production = np.array([1.0, 2.0, np.nan, 3.0])
    known = {1.0, 2.0}
    # NaN excluded; only 3.0 is unseen
    assert count_unseen_categories(production, known) == 1


def test_count_unseen_categories_all_known():
    production = np.array(["X", "Y", "X"])
    known = {"X", "Y", "Z"}
    assert count_unseen_categories(production, known) == 0


def test_count_unseen_categories_empty():
    assert count_unseen_categories(np.array([]), set()) == 0


# ---------------------------------------------------------------------------
# compute_prediction_drift
# ---------------------------------------------------------------------------


def test_prediction_drift_delegates_to_psi():
    rng = np.random.default_rng(5)
    ref_probs = rng.beta(2, 5, 1000)       # skewed toward 0
    prod_probs = rng.beta(5, 2, 1000)      # skewed toward 1
    drift = compute_prediction_drift(ref_probs, prod_probs)
    assert drift > 0.2


def test_prediction_drift_stable_is_near_zero():
    rng = np.random.default_rng(6)
    probs = rng.beta(2, 5, 1000)
    assert compute_prediction_drift(probs, probs.copy()) < 0.01
