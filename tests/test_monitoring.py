"""Tests for monitoring trigger logic and the full monitoring cycle.

Unit tests exercise evaluate_triggers in isolation (no DB required).
Integration tests require Postgres (docker compose up -d postgres).
"""
import datetime

import numpy as np
import pytest

from src.common.db import Incident, MonitoringMetric, Prediction, SessionLocal
from src.monitoring.reference import CAT_FEATURES, NUMERIC_FEATURES, _build_numeric_ref
from src.monitoring.run import evaluate_triggers, run_monitoring_cycle


# ---------------------------------------------------------------------------
# Unit tests: evaluate_triggers (no DB)
# ---------------------------------------------------------------------------


def test_no_triggers_on_clean_metrics():
    result = evaluate_triggers(
        feature_psi={"income": 0.05, "customer_age": 0.02},
        feature_null_rates={"income": 0.0, "customer_age": 0.0},
        unseen_counts={},
        prediction_drift=0.05,
    )
    assert result == []


def test_psi_alarm_fires_above_threshold():
    result = evaluate_triggers(
        feature_psi={"income": 0.8},
        feature_null_rates={"income": 0.0},
        unseen_counts={},
        prediction_drift=0.0,
    )
    alarm = [t for t in result if t["type"] == "psi_alarm"]
    assert len(alarm) == 1
    assert alarm[0]["feature"] == "income"


def test_psi_alarm_does_not_fire_below_threshold():
    result = evaluate_triggers(
        feature_psi={"income": 0.45},
        feature_null_rates={},
        unseen_counts={},
        prediction_drift=0.0,
    )
    assert not any(t["type"] == "psi_alarm" for t in result)


def test_psi_moderate_cluster_fires_when_enough_features():
    result = evaluate_triggers(
        feature_psi={"a": 0.25, "b": 0.30, "c": 0.22},
        feature_null_rates={},
        unseen_counts={},
        prediction_drift=0.0,
    )
    assert any(t["type"] == "psi_moderate_cluster" for t in result)


def test_psi_moderate_cluster_does_not_fire_below_count():
    result = evaluate_triggers(
        feature_psi={"a": 0.25, "b": 0.30},
        feature_null_rates={},
        unseen_counts={},
        prediction_drift=0.0,
    )
    assert not any(t["type"] == "psi_moderate_cluster" for t in result)


def test_null_rate_fires_above_threshold():
    result = evaluate_triggers(
        feature_psi={},
        feature_null_rates={"income": 0.5},
        unseen_counts={},
        prediction_drift=0.0,
    )
    null_triggers = [t for t in result if t["type"] == "null_rate"]
    assert len(null_triggers) == 1
    assert null_triggers[0]["feature"] == "income"


def test_null_rate_does_not_fire_at_threshold():
    # Exactly at threshold should not trigger (strictly greater than).
    result = evaluate_triggers(
        feature_psi={},
        feature_null_rates={"income": 0.2},
        unseen_counts={},
        prediction_drift=0.0,
    )
    assert not any(t["type"] == "null_rate" for t in result)


def test_unseen_category_fires_on_any_count():
    result = evaluate_triggers(
        feature_psi={},
        feature_null_rates={},
        unseen_counts={"payment_type": 3},
        prediction_drift=0.0,
    )
    cat_triggers = [t for t in result if t["type"] == "unseen_category"]
    assert len(cat_triggers) == 1
    assert cat_triggers[0]["feature"] == "payment_type"


def test_prediction_drift_fires_above_threshold():
    result = evaluate_triggers(
        feature_psi={},
        feature_null_rates={},
        unseen_counts={},
        prediction_drift=0.9,
    )
    assert any(t["type"] == "prediction_drift" for t in result)


def test_prediction_drift_does_not_fire_below_threshold():
    result = evaluate_triggers(
        feature_psi={},
        feature_null_rates={},
        unseen_counts={},
        prediction_drift=0.1,
    )
    assert not any(t["type"] == "prediction_drift" for t in result)


# ---------------------------------------------------------------------------
# Integration tests: full Postgres cycle
# ---------------------------------------------------------------------------


def _synthetic_profile(n_ref: int = 1000) -> dict:
    """Build a ReferenceProfile from N(0,1) data so tests don't need the real dataset."""
    rng = np.random.default_rng(42)
    numeric = {f: _build_numeric_ref(rng.normal(0, 1, n_ref)) for f in NUMERIC_FEATURES}
    categorical = {f: ["A", "B", "C"] for f in CAT_FEATURES}
    reference_probs = rng.beta(2, 5, n_ref).tolist()
    return {"numeric": numeric, "categorical": categorical, "reference_probs": reference_probs}


def _insert_predictions(
    n: int,
    null_income_rate: float = 0.0,
    rng_seed: int = 0,
) -> None:
    """Insert synthetic prediction rows into Postgres."""
    rng = np.random.default_rng(rng_seed)
    now = datetime.datetime.now(datetime.timezone.utc)
    rows = []
    for i in range(n):
        income = None if rng.random() < null_income_rate else float(rng.normal(0, 1))
        features: dict = {f: float(rng.normal(0, 1)) for f in NUMERIC_FEATURES}
        features["income"] = income
        for f in CAT_FEATURES:
            features[f] = rng.choice(["A", "B", "C"])
        rows.append(Prediction(
            timestamp=now - datetime.timedelta(seconds=i),
            model_version="test-v1",
            input_features=features,
            prediction=0,
            probability=float(rng.beta(2, 5)),
        ))
    with SessionLocal() as session:
        session.add_all(rows)
        session.commit()


@pytest.fixture
def reset_tables():
    """Wipe all monitoring-related tables before and after each integration test.

    Deletes ALL predictions (not just test rows) so old entries from other test
    modules do not bleed into the monitoring window query.
    """
    def _clean():
        with SessionLocal() as s:
            s.query(MonitoringMetric).delete(synchronize_session=False)
            s.query(Incident).delete(synchronize_session=False)
            s.query(Prediction).delete(synchronize_session=False)
            s.commit()
    _clean()
    yield
    _clean()


def test_monitoring_cycle_opens_incident_on_null_spike(reset_tables):
    """50% null rate on income should trigger a null_rate incident."""
    profile = _synthetic_profile()
    _insert_predictions(n=200, null_income_rate=0.5, rng_seed=1)

    triggers = run_monitoring_cycle(profile=profile)

    assert any(t["type"] == "null_rate" and t["feature"] == "income" for t in triggers)

    with SessionLocal() as session:
        incidents = session.query(Incident).all()
    assert len(incidents) == 1
    assert incidents[0].status == "open"
    assert "null_rate" in incidents[0].trigger_metric


def test_monitoring_cycle_no_incident_on_clean_data(reset_tables):
    """Clean data from the same N(0,1) distribution as the reference should not trigger."""
    profile = _synthetic_profile(n_ref=1000)
    _insert_predictions(n=500, null_income_rate=0.0, rng_seed=2)

    triggers = run_monitoring_cycle(profile=profile)

    assert triggers == []

    with SessionLocal() as session:
        incidents = session.query(Incident).all()
        metrics = session.query(MonitoringMetric).all()
    assert len(incidents) == 0
    assert len(metrics) > 0


def test_monitoring_cycle_no_duplicate_incident(reset_tables):
    """A second cycle with triggers should not open a second incident if one is already open."""
    profile = _synthetic_profile()
    _insert_predictions(n=200, null_income_rate=0.8, rng_seed=3)

    run_monitoring_cycle(profile=profile)
    run_monitoring_cycle(profile=profile)

    with SessionLocal() as session:
        count = session.query(Incident).filter(Incident.status == "open").count()
    assert count == 1
