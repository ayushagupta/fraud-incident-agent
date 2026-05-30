"""Monitoring job: compute distribution metrics for recent predictions and open incidents.

Run once with --once for testing and the eval harness, or --loop to run continuously.
"""
import argparse
import datetime
import json
import logging
import time
from typing import Any

import numpy as np
from sqlalchemy import select

from src.common.config import settings
from src.common.db import Incident, MonitoringMetric, Prediction, SessionLocal
from src.monitoring.metrics import compute_null_rate, compute_psi, count_unseen_categories
from src.monitoring.reference import CAT_FEATURES, NUMERIC_FEATURES, ReferenceProfile, load_reference_profile

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


def _psi_from_profile(
    production: np.ndarray,
    bucket_edges: list[float],
    expected_proportions: list[float],
) -> float:
    """Compute PSI for production data against a pre-bucketed reference distribution.

    Uses the bucket edges and expected proportions stored in the reference profile
    rather than recomputing bucket boundaries from production data.
    """
    prod_clean = production[~np.isnan(production)]
    if len(prod_clean) == 0:
        return 0.0
    edges = np.array(bucket_edges, dtype=float)
    if len(edges) < 2:
        return 0.0
    prod_counts, _ = np.histogram(prod_clean, bins=edges)
    prod_pcts = prod_counts / len(prod_clean)
    ref_pcts = np.array(expected_proportions, dtype=float)
    ref_pcts = np.where(ref_pcts == 0, _EPSILON, ref_pcts)
    prod_pcts = np.where(prod_pcts == 0, _EPSILON, prod_pcts)
    return float(np.sum((prod_pcts - ref_pcts) * np.log(prod_pcts / ref_pcts)))


def evaluate_triggers(
    feature_psi: dict[str, float],
    feature_null_rates: dict[str, float],
    unseen_counts: dict[str, int],
    prediction_drift: float,
) -> list[dict[str, Any]]:
    """Pure function: evaluate all trigger conditions and return one dict per condition that fired.

    An empty list means no incident should be opened.
    """
    triggers: list[dict[str, Any]] = []

    for feature, psi in feature_psi.items():
        if psi > settings.PSI_ALARM_THRESHOLD:
            triggers.append({"type": "psi_alarm", "feature": feature, "value": round(psi, 4)})

    moderate = [(f, psi) for f, psi in feature_psi.items() if psi > settings.PSI_MODERATE_THRESHOLD]
    if len(moderate) >= settings.PSI_MODERATE_COUNT:
        triggers.append({
            "type": "psi_moderate_cluster",
            "count": len(moderate),
            "features": [f for f, _ in moderate[:5]],
        })

    for feature, rate in feature_null_rates.items():
        if rate > settings.NULL_RATE_THRESHOLD:
            triggers.append({"type": "null_rate", "feature": feature, "value": round(rate, 4)})

    for feature, count in unseen_counts.items():
        if count >= 1:
            triggers.append({"type": "unseen_category", "feature": feature, "count": count})

    if prediction_drift > settings.PREDICTION_DRIFT_THRESHOLD:
        triggers.append({"type": "prediction_drift", "value": round(prediction_drift, 4)})

    return triggers


def _summarize_triggers(triggers: list[dict[str, Any]]) -> str:
    """Return a compact JSON string summarizing the fired triggers, within 128 chars."""
    types = list({t["type"] for t in triggers})
    features = list({t["feature"] for t in triggers if "feature" in t})
    summary = {"types": types, "features": features[:4]}
    return json.dumps(summary, separators=(",", ":"))[:128]


def run_monitoring_cycle(profile: ReferenceProfile | None = None) -> list[dict[str, Any]]:
    """Execute one full monitoring cycle.

    Queries the most recent MONITORING_WINDOW_ROWS predictions, computes all metrics,
    writes them to monitoring_metrics, evaluates trigger conditions, and opens an
    incident row if any trigger fires and no incident is already open.

    Returns the list of trigger dicts (empty if nothing fired).
    """
    if profile is None:
        profile = load_reference_profile()

    now = datetime.datetime.now(datetime.timezone.utc)

    with SessionLocal() as session:
        rows = (
            session.execute(
                select(Prediction)
                .order_by(Prediction.timestamp.desc())
                .limit(settings.MONITORING_WINDOW_ROWS)
            )
            .scalars()
            .all()
        )

    if not rows:
        logger.info("No predictions in the recent window, skipping cycle.")
        return []

    window_end = max(r.timestamp for r in rows)
    window_start = min(r.timestamp for r in rows)

    all_features = NUMERIC_FEATURES + CAT_FEATURES
    feature_values: dict[str, list] = {f: [] for f in all_features}
    probabilities: list[float] = []

    for row in rows:
        feats = row.input_features
        for f in all_features:
            feature_values[f].append(feats.get(f))
        probabilities.append(row.probability)

    prod_probs = np.array(probabilities, dtype=float)

    feature_psi: dict[str, float] = {}
    for f in NUMERIC_FEATURES:
        ref = profile["numeric"].get(f)
        if ref is None:
            continue
        arr = np.array(
            [float(v) if v is not None else np.nan for v in feature_values[f]],
            dtype=float,
        )
        feature_psi[f] = _psi_from_profile(arr, ref["bucket_edges"], ref["expected_proportions"])

    ref_probs_arr = np.array(profile["reference_probs"], dtype=float)
    prediction_drift = compute_psi(ref_probs_arr, prod_probs)

    feature_null_rates: dict[str, float] = {}
    for f in all_features:
        arr = np.array(feature_values[f], dtype=object)
        feature_null_rates[f] = compute_null_rate(arr)

    unseen_counts: dict[str, int] = {}
    for f in CAT_FEATURES:
        known = set(profile["categorical"].get(f, []))
        arr = np.array(feature_values[f], dtype=object)
        unseen_counts[f] = count_unseen_categories(arr, known)

    metrics_rows: list[MonitoringMetric] = []
    for f in NUMERIC_FEATURES:
        if f in feature_psi:
            metrics_rows.append(MonitoringMetric(
                timestamp=now,
                metric_name="psi",
                feature_name=f,
                value=feature_psi[f],
                window_start=window_start,
                window_end=window_end,
            ))
    for f in all_features:
        metrics_rows.append(MonitoringMetric(
            timestamp=now,
            metric_name="null_rate",
            feature_name=f,
            value=feature_null_rates[f],
            window_start=window_start,
            window_end=window_end,
        ))
    for f in CAT_FEATURES:
        metrics_rows.append(MonitoringMetric(
            timestamp=now,
            metric_name="unseen_category_count",
            feature_name=f,
            value=float(unseen_counts[f]),
            window_start=window_start,
            window_end=window_end,
        ))
    metrics_rows.append(MonitoringMetric(
        timestamp=now,
        metric_name="prediction_drift",
        feature_name=None,
        value=prediction_drift,
        window_start=window_start,
        window_end=window_end,
    ))

    with SessionLocal() as session:
        session.add_all(metrics_rows)
        session.commit()

    logger.info("Wrote %d metric rows for window [%s, %s].", len(metrics_rows), window_start, window_end)

    triggers = evaluate_triggers(feature_psi, feature_null_rates, unseen_counts, prediction_drift)

    if not triggers:
        logger.info("All metrics within thresholds. No incident triggered.")
        return []

    with SessionLocal() as session:
        open_incident = session.execute(
            select(Incident).where(Incident.status == "open").limit(1)
        ).scalar_one_or_none()

        if open_incident is not None:
            logger.info(
                "Triggers fired but incident %d is already open. Skipping duplicate.",
                open_incident.id,
            )
            return triggers

        summary = _summarize_triggers(triggers)
        incident = Incident(
            opened_at=now,
            status="open",
            trigger_metric=summary,
        )
        session.add(incident)
        session.commit()
        logger.info("Incident opened (id=%d): %s", incident.id, summary)

    return triggers


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection monitoring job")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    group.add_argument("--loop", action="store_true", help="Run continuously on MONITORING_INTERVAL_SECONDS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.once:
        run_monitoring_cycle()
        return

    logger.info("Starting monitoring loop, interval=%ds.", settings.MONITORING_INTERVAL_SECONDS)
    while True:
        try:
            run_monitoring_cycle()
        except Exception:
            logger.exception("Monitoring cycle failed.")
        time.sleep(settings.MONITORING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
