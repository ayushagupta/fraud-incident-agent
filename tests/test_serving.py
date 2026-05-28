"""Integration tests for the FastAPI serving layer.

Requires Postgres to be running (docker compose up -d postgres).
MLflow is replaced by a fixture-local dummy LGBMClassifier so the tests
do not need a trained production model in the registry.
"""

from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.common.db import Prediction, SessionLocal
from src.serving.app import app

CAT_FEATURES = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

SAMPLE_RECORD = {
    "income": 0.8,
    "name_email_similarity": 0.617,
    "prev_address_months_count": -1,
    "current_address_months_count": 89,
    "customer_age": 20,
    "days_since_request": 0.01,
    "intended_balcon_amount": -0.85,
    "payment_type": "AD",
    "zip_count_4w": 1658,
    "velocity_6h": 9223.28,
    "velocity_24h": 5745.25,
    "velocity_4w": 5941.66,
    "bank_branch_count_8w": 3,
    "date_of_birth_distinct_emails_4w": 18,
    "employment_status": "CA",
    "credit_risk_score": 154,
    "email_is_free": 1,
    "housing_status": "BC",
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "bank_months_count": 2,
    "has_other_cards": 0,
    "proposed_credit_limit": 1500.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 3.36,
    "device_os": "other",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
}

TEST_MODEL_VERSION = "test-v0"


def _make_dummy_model() -> lgb.LGBMClassifier:
    rows = [
        {**SAMPLE_RECORD},
        {**SAMPLE_RECORD, "income": 0.1, "credit_risk_score": 50, "device_fraud_count": 5},
        {**SAMPLE_RECORD, "customer_age": 35, "velocity_6h": 100.0},
        {**SAMPLE_RECORD, "income": 0.2, "zip_count_4w": 10},
    ]
    labels = np.array([0, 1, 0, 1])
    df = pd.DataFrame(rows)
    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    model = lgb.LGBMClassifier(n_estimators=2, num_leaves=2, verbosity=-1, random_state=42)
    model.fit(df, labels, categorical_feature=CAT_FEATURES)
    return model


@pytest.fixture(scope="module")
def client() -> TestClient:
    dummy_model = _make_dummy_model()
    with patch("src.serving.app.get_production_model", return_value=(dummy_model, TEST_MODEL_VERSION)):
        with TestClient(app) as c:
            yield c


def test_predict_response_and_db_row(client: TestClient) -> None:
    with SessionLocal() as session:
        before = session.query(Prediction).filter_by(model_version=TEST_MODEL_VERSION).count()

    resp = client.post("/predict", json=SAMPLE_RECORD)

    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == TEST_MODEL_VERSION

    with SessionLocal() as session:
        after = session.query(Prediction).filter_by(model_version=TEST_MODEL_VERSION).count()
    assert after == before + 1


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == TEST_MODEL_VERSION
