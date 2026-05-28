"""Train the baseline LightGBM fraud detection model on BAF month 0 and register it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.config import settings
from src.common.mlflow_helpers import promote_to_production, register_new_version

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "Base.csv"
TRAIN_MONTH = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET = "fraud_bool"
DROP_COLS = [TARGET, "month"]
CAT_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]
PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "is_unbalance": True,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}


def load_month(month: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    subset = df[df["month"] == month].copy()
    feature_cols = [c for c in subset.columns if c not in DROP_COLS]
    X = subset[feature_cols].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    y = subset[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def main() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    print(f"Loading data for month {TRAIN_MONTH}...")
    X, y = load_month(TRAIN_MONTH)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows  Holdout: {len(X_holdout)} rows")
    print(f"Fraud rate  train={y_train.mean():.4f}  holdout={y_holdout.mean():.4f}")

    with mlflow.start_run() as run:
        mlflow.log_params(PARAMS)
        mlflow.log_params(
            {
                "train_month": TRAIN_MONTH,
                "train_rows": len(X_train),
                "holdout_rows": len(X_holdout),
                "cat_features": ",".join(CAT_FEATURES),
            }
        )

        print("Training...")
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

        proba = model.predict_proba(X_holdout)[:, 1]
        auc = roc_auc_score(y_holdout, proba)
        pr_auc = average_precision_score(y_holdout, proba)
        mlflow.log_metrics({"auc": auc, "pr_auc": pr_auc})

        sample = X_train.head(100)
        signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
        mlflow.lightgbm.log_model(model, name="model", signature=signature)

        run_id = run.info.run_id

    version = register_new_version(run_id)
    promote_to_production(version)

    print(f"Model version : {version}")
    print(f"AUC           : {auc:.4f}")
    print(f"PR-AUC        : {pr_auc:.4f}")


if __name__ == "__main__":
    main()
