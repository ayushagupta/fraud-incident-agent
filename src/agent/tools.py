"""Agent tool implementations.

Each function is a pure query against Postgres or MLflow -- no side effects,
no LLM calls. These become the tool definitions the agent loop will expose to
the model. Docstrings are written as plain prose because they will be lifted
verbatim as tool descriptions.
"""
import datetime
import logging
from typing import Any

import mlflow
from mlflow.client import MlflowClient
from sqlalchemy import text

from src.common.config import settings
from src.common.db import Incident, MonitoringMetric, SessionLocal

logger = logging.getLogger(__name__)

_HISTOGRAM_BUCKETS = 10


def get_incident_details(incident_id: int) -> dict[str, Any]:
    """Return the full record for a single incident.

    Fetches the incident row with the given id and returns its id, opened_at,
    closed_at, status, trigger_metric (the raw JSON string stored in the column),
    agent_diagnosis, agent_action, human_decision, and reasoning_trace. Raises
    KeyError if no incident with that id exists. Use this as the first tool call
    when the agent is invoked so it knows what triggered the investigation.
    """
    with SessionLocal() as session:
        row = session.get(Incident, incident_id)
        if row is None:
            raise KeyError(f"Incident {incident_id} not found")
        return {
            "id": row.id,
            "opened_at": row.opened_at.isoformat(),
            "closed_at": row.closed_at.isoformat() if row.closed_at else None,
            "status": row.status,
            "trigger_metric": row.trigger_metric,
            "agent_diagnosis": row.agent_diagnosis,
            "agent_action": row.agent_action,
            "human_decision": row.human_decision,
            "reasoning_trace": row.reasoning_trace,
        }


def get_recent_metrics(
    metric_name: str,
    feature_name: str | None = None,
    hours_back: int = 24,
) -> list[dict[str, Any]]:
    """Return monitoring metric rows for a given metric over the recent window.

    Queries monitoring_metrics for rows matching metric_name (and optionally
    feature_name) whose timestamp falls within hours_back hours of now. Returns
    rows in ascending timestamp order. Each dict has timestamp, value,
    window_start, and window_end. Use this to see how a metric has trended over
    time, for example to distinguish a sudden spike from a gradual drift.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)
    with SessionLocal() as session:
        q = session.query(MonitoringMetric).filter(
            MonitoringMetric.metric_name == metric_name,
            MonitoringMetric.timestamp >= cutoff,
        )
        if feature_name is not None:
            q = q.filter(MonitoringMetric.feature_name == feature_name)
        rows = q.order_by(MonitoringMetric.timestamp.asc()).all()
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "feature_name": r.feature_name,
                "value": r.value,
                "window_start": r.window_start.isoformat(),
                "window_end": r.window_end.isoformat(),
            }
            for r in rows
        ]


def compare_feature_distributions(
    feature_name: str,
    window_a_hours_back: int,
    window_b_hours_back: int,
    window_size_hours: int = 1,
) -> dict[str, Any]:
    """Compare the distribution of a feature between two historical time windows.

    window_a_hours_back is how many hours ago the end of window A is. Window A
    then spans from that endpoint backward by window_size_hours. Window B is
    defined the same way using window_b_hours_back. Window A should be more
    recent (smaller hours_back) and window B the older reference. Aggregation
    is done in Postgres
    via JSONB extraction so no raw rows are returned to Python. Each window
    section of the returned dict contains mean, std, min, max, median, null_rate,
    and row_count. The top-level mean_diff key is abs(window_a.mean -
    window_b.mean). Returns None for any stat when the window contains no rows.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = datetime.timedelta(hours=window_size_hours)

    # window_a_hours_back is how far back the *end* of the window is from now.
    # The window then extends backward by window_size_hours from that endpoint.
    window_a_end = now - datetime.timedelta(hours=window_a_hours_back)
    window_a_start = window_a_end - delta
    window_b_end = now - datetime.timedelta(hours=window_b_hours_back)
    window_b_start = window_b_end - delta

    sql = text("""
        SELECT
            AVG((input_features->>:feat)::numeric)                          AS mean,
            STDDEV((input_features->>:feat)::numeric)                       AS std,
            MIN((input_features->>:feat)::numeric)                          AS min,
            MAX((input_features->>:feat)::numeric)                          AS max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (input_features->>:feat)::numeric
            )                                                                AS median,
            COUNT(*) FILTER (WHERE input_features->>:feat IS NULL)::float
                / NULLIF(COUNT(*), 0)                                        AS null_rate,
            COUNT(*)                                                          AS row_count
        FROM predictions
        WHERE timestamp >= :start AND timestamp < :end
    """)

    def _run(start: datetime.datetime, end: datetime.datetime) -> dict[str, Any]:
        with SessionLocal() as session:
            row = session.execute(sql, {"feat": feature_name, "start": start, "end": end}).one()
        return {
            "mean": float(row.mean) if row.mean is not None else None,
            "std": float(row.std) if row.std is not None else None,
            "min": float(row.min) if row.min is not None else None,
            "max": float(row.max) if row.max is not None else None,
            "median": float(row.median) if row.median is not None else None,
            "null_rate": float(row.null_rate) if row.null_rate is not None else None,
            "row_count": int(row.row_count),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
        }

    stats_a = _run(window_a_start, window_a_end)
    stats_b = _run(window_b_start, window_b_end)

    mean_a = stats_a["mean"]
    mean_b = stats_b["mean"]
    mean_diff = abs(mean_a - mean_b) if (mean_a is not None and mean_b is not None) else None

    return {
        "feature_name": feature_name,
        "window_a": stats_a,
        "window_b": stats_b,
        "mean_diff": mean_diff,
    }


def get_null_rates(hours_back: int = 1) -> dict[str, float]:
    """Return the most recent null rate per feature from the monitoring metrics table.

    Looks up monitoring_metrics rows where metric_name is 'null_rate' and whose
    timestamp falls within hours_back hours of now. For each feature_name, returns
    the value from the row with the latest timestamp. Null rates come from the
    monitoring job and represent the fraction of predictions in that job's window
    where the feature value was missing. A sudden increase in null rate for a
    specific feature is the primary signal for pipeline bugs upstream.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)
    sql = text("""
        SELECT DISTINCT ON (feature_name)
            feature_name,
            value
        FROM monitoring_metrics
        WHERE metric_name = 'null_rate'
          AND timestamp >= :cutoff
          AND feature_name IS NOT NULL
        ORDER BY feature_name, timestamp DESC
    """)
    with SessionLocal() as session:
        rows = session.execute(sql, {"cutoff": cutoff}).all()
    return {r.feature_name: float(r.value) for r in rows}


def get_unseen_categories(hours_back: int = 1) -> dict[str, int]:
    """Return the most recent unseen category count per categorical feature.

    Looks up monitoring_metrics rows where metric_name is 'unseen_category_count'
    and whose timestamp falls within hours_back hours of now. For each feature,
    returns the count from the latest row. A non-zero count means that production
    traffic contains category values the model was never trained on, which is a
    strong signal of a schema change or bad upstream encoding.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)
    sql = text("""
        SELECT DISTINCT ON (feature_name)
            feature_name,
            value
        FROM monitoring_metrics
        WHERE metric_name = 'unseen_category_count'
          AND timestamp >= :cutoff
          AND feature_name IS NOT NULL
        ORDER BY feature_name, timestamp DESC
    """)
    with SessionLocal() as session:
        rows = session.execute(sql, {"cutoff": cutoff}).all()
    return {r.feature_name: int(r.value) for r in rows}


def get_prediction_distribution(hours_back: int = 1) -> dict[str, Any]:
    """Return summary statistics and a histogram of model output probabilities.

    Queries the predictions table for rows within hours_back hours of now and
    computes mean, std, the 5th/25th/50th/75th/95th percentiles, row count, and
    a 10-bucket histogram over [0, 1]. Aggregation runs in Postgres. Use this to
    detect whether the model's score distribution has shifted, which can indicate
    concept drift, a model regression after retraining, or a data pipeline issue
    that caused inputs to cluster in an unusual region.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)

    stats_sql = text("""
        SELECT
            AVG(probability)                                                           AS mean,
            STDDEV(probability)                                                        AS std,
            PERCENTILE_CONT(ARRAY[0.05, 0.25, 0.5, 0.75, 0.95])
                WITHIN GROUP (ORDER BY probability)                                    AS percentiles,
            COUNT(*)                                                                   AS row_count
        FROM predictions
        WHERE timestamp >= :cutoff
    """)

    hist_sql = text("""
        SELECT
            width_bucket(probability, 0, 1, :buckets) AS bucket,
            COUNT(*)                                   AS count
        FROM predictions
        WHERE timestamp >= :cutoff
          AND probability IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """)

    with SessionLocal() as session:
        stats_row = session.execute(stats_sql, {"cutoff": cutoff}).one()
        hist_rows = session.execute(hist_sql, {"cutoff": cutoff, "buckets": _HISTOGRAM_BUCKETS}).all()

    raw_pcts = stats_row.percentiles
    if raw_pcts is not None:
        percentiles = {
            "p5": float(raw_pcts[0]),
            "p25": float(raw_pcts[1]),
            "p50": float(raw_pcts[2]),
            "p75": float(raw_pcts[3]),
            "p95": float(raw_pcts[4]),
        }
    else:
        percentiles = None

    histogram: list[dict[str, Any]] = []
    for r in hist_rows:
        bucket_idx = int(r.bucket) if r.bucket is not None else _HISTOGRAM_BUCKETS + 1
        low = (bucket_idx - 1) / _HISTOGRAM_BUCKETS
        high = bucket_idx / _HISTOGRAM_BUCKETS
        histogram.append({"range": [round(low, 2), round(high, 2)], "count": int(r.count)})

    return {
        "mean": float(stats_row.mean) if stats_row.mean is not None else None,
        "std": float(stats_row.std) if stats_row.std is not None else None,
        "percentiles": percentiles,
        "row_count": int(stats_row.row_count),
        "histogram": histogram,
    }


def get_recent_deploys(days_back: int = 7) -> list[dict[str, Any]]:
    """Return the model version history from the MLflow registry over the last N days.

    Queries the registry for all versions of the configured model name and
    filters to those created within days_back days. Returns them ordered most
    recent first. Each entry has version, registered_at (ISO string), and
    is_production (True if this version currently holds the Production alias).
    Use this to check whether a model change coincides with the start of an
    incident, or to confirm which version is currently live.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    cutoff_ms = int(
        (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)).timestamp()
        * 1000
    )

    try:
        versions = client.search_model_versions(f"name='{settings.MLFLOW_MODEL_NAME}'")
    except Exception:
        logger.warning("MLflow registry unavailable, returning empty deploy history.")
        return []

    try:
        prod_mv = client.get_model_version_by_alias(
            settings.MLFLOW_MODEL_NAME, settings.MLFLOW_PRODUCTION_ALIAS
        )
        prod_version = prod_mv.version
    except Exception:
        prod_version = None

    result: list[dict[str, Any]] = []
    for mv in versions:
        if mv.creation_timestamp < cutoff_ms:
            continue
        registered_at = datetime.datetime.fromtimestamp(
            mv.creation_timestamp / 1000, tz=datetime.timezone.utc
        )
        result.append({
            "version": mv.version,
            "registered_at": registered_at.isoformat(),
            "is_production": mv.version == prod_version,
        })

    result.sort(key=lambda x: x["registered_at"], reverse=True)
    return result


def get_incident_history(
    days_back: int = 30,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return past closed incidents ordered most recent first.

    Queries the incidents table for rows with status 'closed' whose opened_at
    falls within days_back days of now. Returns at most limit rows. Each dict
    has id, opened_at, closed_at, trigger_metric, agent_diagnosis,
    agent_action, and human_decision. Use this to check whether the current
    incident resembles a pattern the agent has already diagnosed, or to measure
    how often a given action was approved or rejected by humans.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
    with SessionLocal() as session:
        rows = (
            session.query(Incident)
            .filter(Incident.status == "closed", Incident.opened_at >= cutoff)
            .order_by(Incident.opened_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "opened_at": r.opened_at.isoformat(),
                "closed_at": r.closed_at.isoformat() if r.closed_at else None,
                "trigger_metric": r.trigger_metric,
                "agent_diagnosis": r.agent_diagnosis,
                "agent_action": r.agent_action,
                "human_decision": r.human_decision,
            }
            for r in rows
        ]
