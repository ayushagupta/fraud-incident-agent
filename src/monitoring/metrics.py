import numpy as np

_EPSILON = 1e-6


def compute_psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between reference and production distributions.

    Uses quantile-based bucketing on the reference data so bucket boundaries are
    always data-driven. NaN values are excluded before bucketing; track missing
    rates separately with compute_null_rate. Returns 0.0 when either array is
    empty or all reference values are identical.
    """
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)

    ref_clean = ref[~np.isnan(ref)]
    prod_clean = prod[~np.isnan(prod)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0

    quantile_pts = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(ref_clean, quantile_pts))

    # After deduplication we need at least two distinct values to form any bucket.
    if len(edges) < 2:
        return 0.0

    # Replace the min/max reference extremes with -inf/+inf so production values
    # that exceed the reference range still fall inside a bucket.
    hist_edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])

    ref_counts, _ = np.histogram(ref_clean, bins=hist_edges)
    prod_counts, _ = np.histogram(prod_clean, bins=hist_edges)

    ref_pcts = ref_counts / len(ref_clean)
    prod_pcts = prod_counts / len(prod_clean)

    ref_pcts = np.where(ref_pcts == 0, _EPSILON, ref_pcts)
    prod_pcts = np.where(prod_pcts == 0, _EPSILON, prod_pcts)

    return float(np.sum((prod_pcts - ref_pcts) * np.log(prod_pcts / ref_pcts)))


def compute_null_rate(values: np.ndarray) -> float:
    """Return the fraction of null/NaN values in the array."""
    if len(values) == 0:
        return 0.0
    try:
        return float(np.sum(np.isnan(np.asarray(values, dtype=float))) / len(values))
    except (ValueError, TypeError):
        # Object array that cannot be cast to float (e.g., strings mixed with None).
        return float(sum(v is None for v in values) / len(values))


def count_unseen_categories(production: np.ndarray, known_categories: set) -> int:
    """Return the count of production values not present in known_categories.

    Null and NaN values are excluded; track those separately via compute_null_rate.
    """
    count = 0
    for v in production:
        if v is None:
            continue
        try:
            if np.isnan(float(v)):
                continue
        except (ValueError, TypeError):
            pass
        if v not in known_categories:
            count += 1
    return count


def compute_prediction_drift(
    reference_probs: np.ndarray, production_probs: np.ndarray, buckets: int = 10
) -> float:
    """PSI on model output probabilities. Delegates to compute_psi."""
    return compute_psi(reference_probs, production_probs, buckets)
