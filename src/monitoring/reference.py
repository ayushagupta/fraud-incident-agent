"""Build and persist the reference distribution profile from month-0 training data.

Run once after training v1 to produce the file that the monitoring job reads on
every subsequent check. Re-run whenever the production model is retrained on a
new window, since prediction probabilities in the profile come from the live model.
"""
import logging
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from src.common.config import settings
from src.common.mlflow_helpers import get_production_model

logger = logging.getLogger(__name__)

# Features the model treats as categorical (must match the serving layer).
CAT_FEATURES: list[str] = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# All model input features that carry a numeric distribution worth tracking.
NUMERIC_FEATURES: list[str] = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
]

_PSI_BUCKETS = 10


class NumericFeatureRef(TypedDict):
    bucket_edges: list  # floats including -inf and +inf sentinels
    expected_proportions: list[float]


class ReferenceProfile(TypedDict):
    numeric: dict[str, NumericFeatureRef]
    categorical: dict[str, list[str]]
    reference_probs: list[float]


def _build_numeric_ref(values: np.ndarray) -> NumericFeatureRef:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return {"bucket_edges": [-np.inf, np.inf], "expected_proportions": [1.0]}

    quantile_pts = np.linspace(0, 100, _PSI_BUCKETS + 1)
    edges = np.unique(np.percentile(clean, quantile_pts))

    if len(edges) < 2:
        return {"bucket_edges": [-np.inf, np.inf], "expected_proportions": [1.0]}

    hist_edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    counts, _ = np.histogram(clean, bins=hist_edges)
    expected_proportions = (counts / len(clean)).tolist()

    return {
        "bucket_edges": hist_edges.tolist(),
        "expected_proportions": expected_proportions,
    }


def build_reference_profile() -> None:
    """Load month-0 data, compute distribution profiles, run the production model,
    and write the result to the path configured in settings.REFERENCE_PROFILE_PATH.
    """
    data_path = Path(settings.RAW_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {data_path}")

    df = pd.read_csv(data_path)
    month0 = df[df["month"] == 0].reset_index(drop=True)
    logger.info("Loaded %d month-0 rows from %s", len(month0), data_path)

    numeric_ref: dict[str, NumericFeatureRef] = {}
    for col in NUMERIC_FEATURES:
        values = month0[col].to_numpy(dtype=float, na_value=np.nan)
        numeric_ref[col] = _build_numeric_ref(values)

    categorical_ref: dict[str, list[str]] = {}
    for col in CAT_FEATURES:
        known = [v for v in month0[col].dropna().unique().tolist()]
        categorical_ref[col] = known

    model, version = get_production_model()
    logger.info("Running production model version %s over month-0 data", version)

    feature_df = month0.drop(columns=["fraud_bool", "month"])
    for col in CAT_FEATURES:
        feature_df[col] = feature_df[col].astype("category")
    for col in NUMERIC_FEATURES:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    reference_probs: list[float] = model.predict_proba(feature_df)[:, 1].tolist()

    profile: ReferenceProfile = {
        "numeric": numeric_ref,
        "categorical": categorical_ref,
        "reference_probs": reference_probs,
    }

    out_path = Path(settings.REFERENCE_PROFILE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(profile, fh)

    logger.info(
        "Reference profile saved to %s (%d numeric, %d categorical features)",
        out_path,
        len(numeric_ref),
        len(categorical_ref),
    )
    print(f"Reference profile written to {out_path}")


def load_reference_profile() -> ReferenceProfile:
    """Load the persisted reference profile from disk."""
    path = Path(settings.REFERENCE_PROFILE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference profile not found at {path}. "
            "Run: python -m src.monitoring.reference"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_reference_profile()
