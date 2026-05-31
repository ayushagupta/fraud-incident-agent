"""Retrain the fraud detection model on a specified window of months and register it in MLflow.

The candidate version is registered but not promoted to Production. The deployment gate
in src/deployment/ handles that decision separately.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.models import infer_signature
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.config import settings
from src.common.mlflow_helpers import register_new_version

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "Base.csv"
TARGET = "fraud_bool"
DROP_COLS = [TARGET, "month"]
CAT_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]
MIN_ROWS = 10_000
BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "is_unbalance": True,
    "verbosity": -1,
}


class RetrainResult(BaseModel):
    candidate_version: str
    train_rows: int
    holdout_rows: int
    train_auc: float
    train_pr_auc: float
    holdout_auc: float
    holdout_pr_auc: float
    training_months: list[int]
    holdout_month: int
    duration_seconds: float


def _load_months(months: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    subset = df[df["month"].isin(months)].copy()
    feature_cols = [c for c in subset.columns if c not in DROP_COLS]
    X = subset[feature_cols].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    y = subset[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def _load_month_with_index(month: int) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Return features, labels, and original dataframe index for a single month."""
    df = pd.read_csv(DATA_PATH)
    subset = df[df["month"] == month].copy()
    feature_cols = [c for c in subset.columns if c not in DROP_COLS]
    X = subset[feature_cols].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    y = subset[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, subset.index


def _compute_metrics(
    model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series
) -> tuple[float, float]:
    proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))
    pr_auc = float(average_precision_score(y, proba))
    return auc, pr_auc


def retrain(
    months: list[int],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> RetrainResult:
    """Train a new LightGBM model on the given months and register it in MLflow.

    The holdout is taken as the last holdout_fraction of rows from the highest
    month in the window, preserving temporal order. The candidate version is
    registered but not promoted to Production.
    """
    started_at = time.monotonic()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    holdout_month = max(months)
    training_months = [m for m in months if m != holdout_month]

    # Load holdout month and split time-ordered
    X_hm, y_hm, _ = _load_month_with_index(holdout_month)
    n_holdout = max(1, int(len(X_hm) * holdout_fraction))
    X_holdout = X_hm.iloc[-n_holdout:].copy()
    y_holdout = y_hm.iloc[-n_holdout:].copy()
    X_from_holdout_month = X_hm.iloc[:-n_holdout].copy()
    y_from_holdout_month = y_hm.iloc[:-n_holdout].copy()

    # Load remaining months for training
    if training_months:
        X_other, y_other = _load_months(training_months)
        X_train = pd.concat([X_other, X_from_holdout_month], ignore_index=True)
        y_train = pd.concat([y_other, y_from_holdout_month], ignore_index=True)
    else:
        X_train = X_from_holdout_month.reset_index(drop=True)
        y_train = y_from_holdout_month.reset_index(drop=True)

    total_rows = len(X_train) + len(X_holdout)
    if total_rows < MIN_ROWS:
        raise ValueError(
            f"Filtered data has only {total_rows} rows across months {months}; "
            f"minimum required is {MIN_ROWS}."
        )

    params = {**BASE_PARAMS, "random_state": seed}

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "training_months": ",".join(str(m) for m in sorted(months)),
                "holdout_month": holdout_month,
                "train_rows": len(X_train),
                "holdout_rows": len(X_holdout),
                "holdout_fraction": holdout_fraction,
                "cat_features": ",".join(CAT_FEATURES),
            }
        )
        mlflow.set_tag("training_months", ",".join(str(m) for m in sorted(months)))

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

        train_auc, train_pr_auc = _compute_metrics(model, X_train, y_train)
        holdout_auc, holdout_pr_auc = _compute_metrics(model, X_holdout, y_holdout)

        mlflow.log_metrics(
            {
                "train_auc": train_auc,
                "train_pr_auc": train_pr_auc,
                "holdout_auc": holdout_auc,
                "holdout_pr_auc": holdout_pr_auc,
            }
        )

        sample = X_train.head(100)
        signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
        mlflow.lightgbm.log_model(model, name="model", signature=signature)

        run_id = run.info.run_id

    candidate_version = register_new_version(run_id)
    duration = time.monotonic() - started_at

    logger.info(
        "Retrain complete: version=%s holdout_auc=%.4f holdout_pr_auc=%.4f duration=%.1fs",
        candidate_version,
        holdout_auc,
        holdout_pr_auc,
        duration,
    )

    return RetrainResult(
        candidate_version=candidate_version,
        train_rows=len(X_train),
        holdout_rows=len(X_holdout),
        train_auc=train_auc,
        train_pr_auc=train_pr_auc,
        holdout_auc=holdout_auc,
        holdout_pr_auc=holdout_pr_auc,
        training_months=sorted(months),
        holdout_month=holdout_month,
        duration_seconds=round(duration, 2),
    )


def _default_months() -> list[int]:
    df = pd.read_csv(DATA_PATH, usecols=["month"])
    latest = int(df["month"].max())
    # default: 0 through latest-1
    return list(range(latest))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain fraud-detector on a month window.")
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Months to include (e.g. 0 1 2 3). Defaults to 0 through latest-1.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    months = args.months if args.months is not None else _default_months()
    result = retrain(months=months, holdout_fraction=args.holdout_fraction, seed=args.seed)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
