import datetime
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from src.common.db import Prediction, SessionLocal
from src.common.mlflow_helpers import get_production_model

logger = logging.getLogger(__name__)

CAT_FEATURES = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


class PredictRequest(BaseModel):
    income: float | None = None
    name_email_similarity: float | None = None
    prev_address_months_count: int | None = None
    current_address_months_count: int | None = None
    customer_age: int | None = None
    days_since_request: float | None = None
    intended_balcon_amount: float | None = None
    payment_type: str | None = None
    zip_count_4w: int | None = None
    velocity_6h: float | None = None
    velocity_24h: float | None = None
    velocity_4w: float | None = None
    bank_branch_count_8w: int | None = None
    date_of_birth_distinct_emails_4w: int | None = None
    employment_status: str | None = None
    credit_risk_score: int | None = None
    email_is_free: int | None = None
    housing_status: str | None = None
    phone_home_valid: int | None = None
    phone_mobile_valid: int | None = None
    bank_months_count: int | None = None
    has_other_cards: int | None = None
    proposed_credit_limit: float | None = None
    foreign_request: int | None = None
    source: str | None = None
    session_length_in_minutes: float | None = None
    device_os: str | None = None
    keep_alive_session: int | None = None
    device_distinct_emails_8w: int | None = None
    device_fraud_count: int | None = None


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


class ReloadResponse(BaseModel):
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, version = get_production_model()
    app.state.model = model
    app.state.model_version = version
    logger.info("Loaded production model version %s", version)
    yield
    logger.info("Serving layer shut down")


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    features = request.model_dump()
    df = pd.DataFrame([features])
    for col in df.columns:
        if col in CAT_FEATURES:
            df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    proba = float(app.state.model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)

    row = Prediction(
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        model_version=app.state.model_version,
        input_features=features,
        prediction=pred,
        probability=proba,
    )
    with SessionLocal() as session:
        session.add(row)
        session.commit()

    return PredictResponse(prediction=pred, probability=proba, model_version=app.state.model_version)


@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unreachable")
    return {"status": "ok", "model_version": app.state.model_version}


@app.post("/reload", response_model=ReloadResponse)
def reload() -> ReloadResponse:
    model, version = get_production_model()
    app.state.model = model
    app.state.model_version = version
    logger.info("Reloaded production model version %s", version)
    return ReloadResponse(model_version=version)
