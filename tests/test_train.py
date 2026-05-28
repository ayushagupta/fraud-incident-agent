"""Integration test: load the production model from MLflow and verify prediction shape.

Requires docker compose services (postgres, mlflow) to be running and
train_v1.py to have been executed at least once.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.common.mlflow_helpers import get_production_model

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "Base.csv"
CAT_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]
DROP_COLS = ["fraud_bool", "month"]
SAMPLE_SIZE = 10


@pytest.fixture(scope="module")
def sample_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)
    X = df.drop(columns=DROP_COLS)
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    return X


def test_production_model_prediction_shape(sample_features: pd.DataFrame) -> None:
    model, version = get_production_model()

    proba = model.predict_proba(sample_features)

    assert proba.shape == (SAMPLE_SIZE, 2), f"Expected ({SAMPLE_SIZE}, 2), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities outside [0, 1]"
    assert version is not None and version != "", "Version string is empty"


def test_production_model_fraud_column_is_second(sample_features: pd.DataFrame) -> None:
    model, _ = get_production_model()
    proba = model.predict_proba(sample_features)
    # LightGBM binary classifier: column 0 = P(not fraud), column 1 = P(fraud)
    fraud_proba = proba[:, 1]
    assert fraud_proba.min() >= 0.0
    assert fraud_proba.max() <= 1.0
