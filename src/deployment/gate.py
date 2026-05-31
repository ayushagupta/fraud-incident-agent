"""Deployment validation gate for candidate model versions.

Loads the current production model and the candidate, scores both on a holdout
that neither has seen, and decides whether the candidate should replace production
based on a PR-AUC improvement margin. Flips the Production alias only when the
candidate passes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.config import settings
from src.common.mlflow_helpers import get_production_model, promote_to_production

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


class GateResult(BaseModel):
    candidate_version: str
    production_version: str
    holdout_months: list[int]
    holdout_rows: int
    production_auc: float
    production_pr_auc: float
    candidate_auc: float
    candidate_pr_auc: float
    pr_auc_delta: float
    passed: bool
    reason: str
    promoted: bool


def _load_holdout(months: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    subset = df[df["month"].isin(months)].copy()
    feature_cols = [c for c in subset.columns if c not in DROP_COLS]
    X = subset[feature_cols].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    y = subset[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def _get_training_months_from_run(run: mlflow.entities.Run) -> list[int]:
    raw = run.data.tags.get("training_months") or run.data.params.get("training_months")
    if raw:
        return [int(m) for m in raw.split(",")]
    return []


def _get_holdout_month_from_run(run: mlflow.entities.Run) -> int | None:
    raw = run.data.params.get("holdout_month")
    return int(raw) if raw is not None else None


def _resolve_holdout_months(
    candidate_version: str,
    holdout_months: list[int] | None,
    client: MlflowClient,
) -> list[int]:
    if holdout_months is not None:
        return holdout_months

    mv = client.get_model_version(settings.MLFLOW_MODEL_NAME, candidate_version)
    run = client.get_run(mv.run_id)
    training_months = _get_training_months_from_run(run)

    all_months = sorted(
        pd.read_csv(DATA_PATH, usecols=["month"])["month"].unique().tolist()
    )
    unseen = [m for m in all_months if m not in training_months]
    if unseen:
        return [max(unseen)]

    fallback = _get_holdout_month_from_run(run)
    if fallback is not None:
        return [fallback]
    return [max(all_months)]


def _score_model(model, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))
    pr_auc = float(average_precision_score(y, proba))
    return auc, pr_auc


def evaluate_candidate(
    candidate_version: str,
    holdout_months: list[int] | None = None,
) -> GateResult:
    """Evaluate a candidate model against production on a shared holdout.

    Loads both models from the MLflow registry, resolves a holdout set that
    neither was trained on, scores them with AUC and PR-AUC, and decides pass
    or fail based on settings.PROMOTION_MARGIN. On pass, the Production alias
    is atomically flipped to the candidate version.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    prod_model, production_version = get_production_model()
    logger.info("Loaded production model version %s", production_version)

    candidate_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{candidate_version}"
    candidate_model = mlflow.lightgbm.load_model(candidate_uri)
    logger.info("Loaded candidate model version %s", candidate_version)

    resolved_months = _resolve_holdout_months(candidate_version, holdout_months, client)
    logger.info("Holdout months: %s", resolved_months)

    X_holdout, y_holdout = _load_holdout(resolved_months)

    prod_auc, prod_pr_auc = _score_model(prod_model, X_holdout, y_holdout)
    cand_auc, cand_pr_auc = _score_model(candidate_model, X_holdout, y_holdout)
    pr_auc_delta = cand_pr_auc - prod_pr_auc

    margin = settings.PROMOTION_MARGIN
    if pr_auc_delta >= margin:
        passed = True
        reason = f"candidate PR-AUC improvement {pr_auc_delta:.4f} meets margin {margin}"
    elif pr_auc_delta >= 0:
        passed = False
        reason = f"candidate PR-AUC improvement {pr_auc_delta:.4f} below margin {margin}"
    else:
        passed = False
        reason = "candidate PR-AUC worse than production"

    promoted = False
    if passed:
        promote_to_production(candidate_version)
        promoted = True
        logger.info(
            "Promoted version %s to production (pr_auc_delta=%.4f)",
            candidate_version,
            pr_auc_delta,
        )
    else:
        logger.info(
            "Candidate version %s did not pass gate (pr_auc_delta=%.4f): %s",
            candidate_version,
            pr_auc_delta,
            reason,
        )

    return GateResult(
        candidate_version=candidate_version,
        production_version=production_version,
        holdout_months=resolved_months,
        holdout_rows=len(X_holdout),
        production_auc=prod_auc,
        production_pr_auc=prod_pr_auc,
        candidate_auc=cand_auc,
        candidate_pr_auc=cand_pr_auc,
        pr_auc_delta=pr_auc_delta,
        passed=passed,
        reason=reason,
        promoted=promoted,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deployment gate against a candidate model version."
    )
    parser.add_argument("candidate_version", help="MLflow model version to evaluate.")
    parser.add_argument(
        "--holdout-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Months to use as holdout (e.g. 5 6 7). Defaults to most recent unseen month.",
    )
    args = parser.parse_args()
    result = evaluate_candidate(
        args.candidate_version,
        holdout_months=args.holdout_months,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
