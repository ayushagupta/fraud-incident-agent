"""Tests for agent tools.

All tests that touch Postgres use real DB rows; no mocking of SQLAlchemy.
MLflow-dependent tests mock MlflowClient to avoid needing a running registry.
Run: pytest tests/test_agent_tools.py
"""
import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.common.db import Incident, MonitoringMetric, Prediction, SessionLocal
from src.agent.tools import (
    compare_feature_distributions,
    get_incident_details,
    get_incident_history,
    get_null_rates,
    get_prediction_distribution,
    get_recent_deploys,
    get_recent_metrics,
    get_unseen_categories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_tables():
    """Wipe all relevant rows before and after each test."""
    def _clean():
        with SessionLocal() as s:
            s.query(MonitoringMetric).delete(synchronize_session=False)
            s.query(Incident).delete(synchronize_session=False)
            s.query(Prediction).delete(synchronize_session=False)
            s.commit()

    _clean()
    yield
    _clean()


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_prediction(
    probability: float = 0.3,
    income: float | None = 500.0,
    age: float = 30.0,
    timestamp: datetime.datetime | None = None,
    model_version: str = "test-v1",
) -> Prediction:
    ts = timestamp or _now()
    row = Prediction(
        timestamp=ts,
        model_version=model_version,
        input_features={"income": income, "customer_age": age, "payment_type": "AB"},
        prediction=0,
        probability=probability,
    )
    with SessionLocal() as s:
        s.add(row)
        s.commit()
        s.refresh(row)
    return row


def _insert_metric(
    metric_name: str,
    value: float,
    feature_name: str | None = None,
    timestamp: datetime.datetime | None = None,
) -> MonitoringMetric:
    ts = timestamp or _now()
    row = MonitoringMetric(
        timestamp=ts,
        metric_name=metric_name,
        feature_name=feature_name,
        value=value,
        window_start=ts - datetime.timedelta(minutes=5),
        window_end=ts,
    )
    with SessionLocal() as s:
        s.add(row)
        s.commit()
        s.refresh(row)
    return row


def _insert_incident(
    status: str = "open",
    trigger_metric: str = '{"types":["null_rate"],"features":["income"]}',
    agent_diagnosis: dict | None = None,
    agent_action: str | None = None,
    human_decision: str | None = None,
    opened_at: datetime.datetime | None = None,
) -> Incident:
    ts = opened_at or _now()
    row = Incident(
        opened_at=ts,
        closed_at=_now() if status == "closed" else None,
        status=status,
        trigger_metric=trigger_metric,
        agent_diagnosis=agent_diagnosis,
        agent_action=agent_action,
        human_decision=human_decision,
    )
    with SessionLocal() as s:
        s.add(row)
        s.commit()
        s.refresh(row)
    return row


# ---------------------------------------------------------------------------
# get_incident_details
# ---------------------------------------------------------------------------


def test_get_incident_details_returns_correct_fields():
    inc = _insert_incident(status="open", agent_action="alert_human")
    result = get_incident_details(inc.id)
    assert result["id"] == inc.id
    assert result["status"] == "open"
    assert result["agent_action"] == "alert_human"
    assert result["closed_at"] is None
    assert "trigger_metric" in result
    assert "agent_diagnosis" in result


def test_get_incident_details_raises_on_missing():
    with pytest.raises(KeyError):
        get_incident_details(999999)


def test_get_incident_details_closed_incident_has_closed_at():
    inc = _insert_incident(status="closed")
    result = get_incident_details(inc.id)
    assert result["closed_at"] is not None


# ---------------------------------------------------------------------------
# get_recent_metrics
# ---------------------------------------------------------------------------


def test_get_recent_metrics_returns_matching_rows():
    _insert_metric("psi", 0.42, feature_name="income")
    _insert_metric("psi", 0.10, feature_name="customer_age")
    rows = get_recent_metrics("psi", hours_back=1)
    assert len(rows) == 2
    assert all(r["value"] in (0.42, 0.10) for r in rows)


def test_get_recent_metrics_filters_by_feature_name():
    _insert_metric("psi", 0.42, feature_name="income")
    _insert_metric("psi", 0.10, feature_name="customer_age")
    rows = get_recent_metrics("psi", feature_name="income", hours_back=1)
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(0.42)


def test_get_recent_metrics_excludes_old_rows():
    old_ts = _now() - datetime.timedelta(hours=48)
    _insert_metric("psi", 0.99, feature_name="income", timestamp=old_ts)
    rows = get_recent_metrics("psi", hours_back=1)
    assert rows == []


def test_get_recent_metrics_ordered_by_timestamp():
    t1 = _now() - datetime.timedelta(minutes=30)
    t2 = _now() - datetime.timedelta(minutes=10)
    _insert_metric("psi", 0.3, feature_name="income", timestamp=t2)
    _insert_metric("psi", 0.1, feature_name="income", timestamp=t1)
    rows = get_recent_metrics("psi", feature_name="income", hours_back=1)
    assert rows[0]["value"] == pytest.approx(0.1)
    assert rows[1]["value"] == pytest.approx(0.3)


def test_get_recent_metrics_returns_required_keys():
    _insert_metric("psi", 0.2, feature_name="income")
    rows = get_recent_metrics("psi", hours_back=1)
    assert set(rows[0].keys()) >= {"timestamp", "value", "window_start", "window_end", "feature_name"}


# ---------------------------------------------------------------------------
# compare_feature_distributions
# ---------------------------------------------------------------------------


def test_compare_feature_distributions_returns_both_windows():
    now = _now()
    # Window A: 1h ago, Window B: 3h ago, size 1h each
    t_a = now - datetime.timedelta(hours=1, minutes=30)
    t_b = now - datetime.timedelta(hours=3, minutes=30)
    for _ in range(5):
        _insert_prediction(income=100.0, timestamp=t_a)
    for _ in range(5):
        _insert_prediction(income=500.0, timestamp=t_b)

    result = compare_feature_distributions(
        feature_name="income",
        window_a_hours_back=1,
        window_b_hours_back=3,
        window_size_hours=1,
    )
    assert "window_a" in result
    assert "window_b" in result
    assert result["feature_name"] == "income"
    assert result["window_a"]["mean"] == pytest.approx(100.0)
    assert result["window_b"]["mean"] == pytest.approx(500.0)
    assert result["mean_diff"] == pytest.approx(400.0)


def test_compare_feature_distributions_null_rate():
    now = _now()
    t_a = now - datetime.timedelta(hours=1, minutes=30)
    _insert_prediction(income=None, timestamp=t_a)
    _insert_prediction(income=None, timestamp=t_a)
    _insert_prediction(income=200.0, timestamp=t_a)

    result = compare_feature_distributions("income", window_a_hours_back=1, window_b_hours_back=3)
    # 2 of 3 are null
    assert result["window_a"]["null_rate"] == pytest.approx(2 / 3, rel=0.01)


def test_compare_feature_distributions_empty_window():
    result = compare_feature_distributions("income", window_a_hours_back=1, window_b_hours_back=3)
    assert result["window_a"]["row_count"] == 0
    assert result["window_a"]["mean"] is None
    assert result["mean_diff"] is None


# ---------------------------------------------------------------------------
# get_null_rates
# ---------------------------------------------------------------------------


def test_get_null_rates_returns_per_feature():
    _insert_metric("null_rate", 0.05, feature_name="income")
    _insert_metric("null_rate", 0.30, feature_name="customer_age")
    result = get_null_rates(hours_back=1)
    assert result["income"] == pytest.approx(0.05)
    assert result["customer_age"] == pytest.approx(0.30)


def test_get_null_rates_picks_latest_row_per_feature():
    t_old = _now() - datetime.timedelta(minutes=40)
    t_new = _now() - datetime.timedelta(minutes=5)
    _insert_metric("null_rate", 0.9, feature_name="income", timestamp=t_old)
    _insert_metric("null_rate", 0.1, feature_name="income", timestamp=t_new)
    result = get_null_rates(hours_back=1)
    assert result["income"] == pytest.approx(0.1)


def test_get_null_rates_excludes_old_rows():
    old_ts = _now() - datetime.timedelta(hours=3)
    _insert_metric("null_rate", 0.99, feature_name="income", timestamp=old_ts)
    result = get_null_rates(hours_back=1)
    assert "income" not in result


def test_get_null_rates_empty_when_no_data():
    result = get_null_rates(hours_back=1)
    assert result == {}


# ---------------------------------------------------------------------------
# get_unseen_categories
# ---------------------------------------------------------------------------


def test_get_unseen_categories_returns_counts():
    _insert_metric("unseen_category_count", 7.0, feature_name="payment_type")
    result = get_unseen_categories(hours_back=1)
    assert result["payment_type"] == 7


def test_get_unseen_categories_excludes_old_rows():
    old_ts = _now() - datetime.timedelta(hours=5)
    _insert_metric("unseen_category_count", 3.0, feature_name="payment_type", timestamp=old_ts)
    result = get_unseen_categories(hours_back=1)
    assert "payment_type" not in result


def test_get_unseen_categories_empty_when_no_data():
    result = get_unseen_categories(hours_back=1)
    assert result == {}


def test_get_unseen_categories_picks_latest_row():
    t_old = _now() - datetime.timedelta(minutes=40)
    t_new = _now() - datetime.timedelta(minutes=5)
    _insert_metric("unseen_category_count", 10.0, feature_name="device_os", timestamp=t_old)
    _insert_metric("unseen_category_count", 2.0, feature_name="device_os", timestamp=t_new)
    result = get_unseen_categories(hours_back=1)
    assert result["device_os"] == 2


# ---------------------------------------------------------------------------
# get_prediction_distribution
# ---------------------------------------------------------------------------


def test_get_prediction_distribution_returns_expected_keys():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        _insert_prediction(probability=p)
    result = get_prediction_distribution(hours_back=1)
    assert "mean" in result
    assert "std" in result
    assert "percentiles" in result
    assert "row_count" in result
    assert "histogram" in result


def test_get_prediction_distribution_correct_mean():
    for p in [0.2, 0.4, 0.6]:
        _insert_prediction(probability=p)
    result = get_prediction_distribution(hours_back=1)
    assert result["mean"] == pytest.approx(0.4, rel=0.001)
    assert result["row_count"] == 3


def test_get_prediction_distribution_percentile_keys():
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        _insert_prediction(probability=p)
    result = get_prediction_distribution(hours_back=1)
    assert set(result["percentiles"].keys()) == {"p5", "p25", "p50", "p75", "p95"}


def test_get_prediction_distribution_histogram_structure():
    for p in [0.1, 0.5, 0.9]:
        _insert_prediction(probability=p)
    result = get_prediction_distribution(hours_back=1)
    for bucket in result["histogram"]:
        assert "range" in bucket
        assert "count" in bucket
        assert len(bucket["range"]) == 2


def test_get_prediction_distribution_empty_window():
    result = get_prediction_distribution(hours_back=1)
    assert result["row_count"] == 0
    assert result["mean"] is None
    assert result["histogram"] == []


# ---------------------------------------------------------------------------
# get_recent_deploys (mocked MlflowClient)
# ---------------------------------------------------------------------------


def _make_mock_version(version: str, created_ms: int, prod_version: str | None) -> MagicMock:
    mv = MagicMock()
    mv.version = version
    mv.creation_timestamp = created_ms
    return mv


def test_get_recent_deploys_returns_versions_within_window():
    now_ms = int(_now().timestamp() * 1000)
    old_ms = now_ms - 8 * 24 * 3600 * 1000  # 8 days ago, outside 7-day window

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _make_mock_version("2", now_ms - 3600 * 1000, "2"),
        _make_mock_version("1", old_ms, "2"),
    ]
    prod_mv = MagicMock()
    prod_mv.version = "2"
    mock_client.get_model_version_by_alias.return_value = prod_mv

    with patch("src.agent.tools.MlflowClient", return_value=mock_client):
        result = get_recent_deploys(days_back=7)

    assert len(result) == 1
    assert result[0]["version"] == "2"
    assert result[0]["is_production"] is True


def test_get_recent_deploys_marks_production_correctly():
    now_ms = int(_now().timestamp() * 1000)

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _make_mock_version("3", now_ms - 1000, "2"),
        _make_mock_version("2", now_ms - 2000, "2"),
    ]
    prod_mv = MagicMock()
    prod_mv.version = "2"
    mock_client.get_model_version_by_alias.return_value = prod_mv

    with patch("src.agent.tools.MlflowClient", return_value=mock_client):
        result = get_recent_deploys(days_back=7)

    by_version = {r["version"]: r for r in result}
    assert by_version["3"]["is_production"] is False
    assert by_version["2"]["is_production"] is True


def test_get_recent_deploys_ordered_most_recent_first():
    now_ms = int(_now().timestamp() * 1000)

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _make_mock_version("1", now_ms - 2000, None),
        _make_mock_version("2", now_ms - 1000, None),
    ]
    mock_client.get_model_version_by_alias.side_effect = Exception("no prod")

    with patch("src.agent.tools.MlflowClient", return_value=mock_client):
        result = get_recent_deploys(days_back=7)

    assert result[0]["version"] == "2"
    assert result[1]["version"] == "1"


def test_get_recent_deploys_returns_empty_on_registry_error():
    mock_client = MagicMock()
    mock_client.search_model_versions.side_effect = Exception("connection refused")

    with patch("src.agent.tools.MlflowClient", return_value=mock_client):
        result = get_recent_deploys(days_back=7)

    assert result == []


# ---------------------------------------------------------------------------
# get_incident_history
# ---------------------------------------------------------------------------


def test_get_incident_history_returns_only_closed():
    _insert_incident(status="open")
    _insert_incident(status="closed", agent_action="suppress", human_decision="approved")
    result = get_incident_history(days_back=30)
    assert len(result) == 1
    assert result[0]["agent_action"] == "suppress"


def test_get_incident_history_returns_correct_fields():
    _insert_incident(
        status="closed",
        trigger_metric='{"types":["psi_alarm"],"features":["income"]}',
        agent_diagnosis={"type": "genuine_drift"},
        agent_action="retrain",
        human_decision="approved",
    )
    result = get_incident_history(days_back=30)
    assert len(result) == 1
    r = result[0]
    assert r["agent_diagnosis"] == {"type": "genuine_drift"}
    assert r["agent_action"] == "retrain"
    assert r["human_decision"] == "approved"
    assert "opened_at" in r
    assert "trigger_metric" in r


def test_get_incident_history_excludes_old_incidents():
    old_ts = _now() - datetime.timedelta(days=60)
    _insert_incident(status="closed", opened_at=old_ts)
    result = get_incident_history(days_back=30)
    assert result == []


def test_get_incident_history_respects_limit():
    for _ in range(5):
        _insert_incident(status="closed")
    result = get_incident_history(days_back=30, limit=3)
    assert len(result) == 3


def test_get_incident_history_ordered_most_recent_first():
    t1 = _now() - datetime.timedelta(hours=2)
    t2 = _now() - datetime.timedelta(hours=1)
    _insert_incident(status="closed", agent_action="suppress", opened_at=t1)
    _insert_incident(status="closed", agent_action="retrain", opened_at=t2)
    result = get_incident_history(days_back=30)
    assert result[0]["agent_action"] == "retrain"
    assert result[1]["agent_action"] == "suppress"
