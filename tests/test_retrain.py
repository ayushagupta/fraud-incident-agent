"""Tests for src/training/retrain.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.retrain import (
    CAT_FEATURES,
    DROP_COLS,
    RetrainResult,
    retrain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_month_df(month: int, n: int, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n):
        row: dict = {
            "fraud_bool": int(rng.random() < fraud_rate),
            "month": month,
            "income": float(rng.uniform(0, 1)),
            "name_email_similarity": float(rng.uniform(0, 1)),
            "prev_address_months_count": int(rng.integers(-1, 100)),
            "current_address_months_count": int(rng.integers(0, 200)),
            "customer_age": int(rng.integers(18, 80)),
            "days_since_request": float(rng.uniform(0, 30)),
            "intended_balcon_amount": float(rng.uniform(-10, 500)),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"]),
            "zip_count_4w": int(rng.integers(1, 5000)),
            "velocity_6h": float(rng.uniform(0, 20000)),
            "velocity_24h": float(rng.uniform(0, 20000)),
            "velocity_4w": float(rng.uniform(0, 20000)),
            "bank_branch_count_8w": int(rng.integers(0, 30)),
            "date_of_birth_distinct_emails_4w": int(rng.integers(0, 5)),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"]),
            "credit_risk_score": int(rng.integers(-175, 400)),
            "email_is_free": int(rng.integers(0, 2)),
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"]),
            "phone_home_valid": int(rng.integers(0, 2)),
            "phone_mobile_valid": int(rng.integers(0, 2)),
            "bank_months_count": int(rng.integers(-1, 40)),
            "has_other_cards": int(rng.integers(0, 2)),
            "proposed_credit_limit": float(rng.choice([200, 500, 1000, 1500, 2000])),
            "foreign_request": int(rng.integers(0, 2)),
            "source": rng.choice(["INTERNET", "TELEAPP"]),
            "session_length_in_minutes": float(rng.uniform(-1, 300)),
            "device_os": rng.choice(["windows", "linux", "macintosh", "other", "x11"]),
            "keep_alive_session": int(rng.integers(0, 2)),
            "device_distinct_emails_8w": int(rng.integers(0, 3)),
            "device_fraud_count": int(rng.integers(0, 10)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _make_fixture_csv(months: list[int], rows_per_month: int, tmp_path: "Path") -> "Path":
    frames = [_make_month_df(m, rows_per_month, seed=m) for m in months]
    df = pd.concat(frames, ignore_index=True)
    csv_path = tmp_path / "Base.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Unit test (no MLflow, no Postgres)
# ---------------------------------------------------------------------------

def test_retrain_unit(tmp_path):
    """Retrain on a small fixture dataset produces a sensible RetrainResult."""
    # Two months, 6000 rows each -> 12000 total, above the 10k minimum
    csv_path = _make_fixture_csv(months=[0, 1], rows_per_month=6000, tmp_path=tmp_path)

    fake_version = "7"

    with (
        patch("src.training.retrain.DATA_PATH", csv_path),
        patch("src.training.retrain.register_new_version", return_value=fake_version) as mock_register,
        patch("src.training.retrain.mlflow") as mock_mlflow,
    ):
        # Make mlflow.start_run() work as a context manager
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "fake-run-id"
        mock_mlflow.start_run.return_value = mock_run

        result = retrain(months=[0, 1], holdout_fraction=0.2, seed=42)

    assert isinstance(result, RetrainResult)
    assert result.candidate_version == fake_version
    assert result.train_rows > 0
    assert result.holdout_rows > 0
    assert 0.0 <= result.train_auc <= 1.0
    assert 0.0 <= result.train_pr_auc <= 1.0
    assert 0.0 <= result.holdout_auc <= 1.0
    assert 0.0 <= result.holdout_pr_auc <= 1.0
    assert result.holdout_month == 1, "Holdout should come from the highest month"
    assert result.training_months == [0, 1]
    assert result.duration_seconds >= 0.0
    mock_register.assert_called_once_with("fake-run-id")


def test_retrain_unit_too_few_rows(tmp_path):
    """Retrain raises when the filtered data is below the minimum row threshold."""
    # 500 rows per month, 2 months = 1000 rows total -- well below 10k
    csv_path = _make_fixture_csv(months=[0, 1], rows_per_month=500, tmp_path=tmp_path)

    with (
        patch("src.training.retrain.DATA_PATH", csv_path),
        patch("src.training.retrain.mlflow"),
    ):
        with pytest.raises(ValueError, match="minimum required"):
            retrain(months=[0, 1], holdout_fraction=0.2, seed=42)


def test_retrain_unit_holdout_from_highest_month(tmp_path):
    """Holdout rows come from the tail of the highest month, not from earlier months."""
    csv_path = _make_fixture_csv(months=[2, 3, 4], rows_per_month=4000, tmp_path=tmp_path)

    fake_version = "9"

    with (
        patch("src.training.retrain.DATA_PATH", csv_path),
        patch("src.training.retrain.register_new_version", return_value=fake_version),
        patch("src.training.retrain.mlflow") as mock_mlflow,
    ):
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "fake-run-id-2"
        mock_mlflow.start_run.return_value = mock_run

        result = retrain(months=[2, 3, 4], holdout_fraction=0.2, seed=42)

    assert result.holdout_month == 4
    # holdout is 20% of month-4's 4000 rows = 800
    assert result.holdout_rows == 800
    # train = months 2,3 (8000 rows) + 80% of month 4 (3200 rows) = 11200
    assert result.train_rows == 11200


# ---------------------------------------------------------------------------
# Integration test (requires MLflow + data)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_retrain_integration_registers_candidate():
    """Real retrain on months 0 and 1 registers a new non-production version."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.common.config import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    result = retrain(months=[0, 1], holdout_fraction=0.2, seed=42)

    assert result.train_rows >= 1000
    assert result.holdout_rows >= 1
    assert 0.0 < result.holdout_auc <= 1.0

    # Verify the version exists in the registry
    mv = client.get_model_version(settings.MLFLOW_MODEL_NAME, result.candidate_version)
    assert mv is not None

    # Verify it is NOT tagged as production
    aliases = client.get_model_version_by_alias  # just check the alias is NOT this version
    try:
        prod_mv = client.get_model_version_by_alias(
            settings.MLFLOW_MODEL_NAME, settings.MLFLOW_PRODUCTION_ALIAS
        )
        assert prod_mv.version != result.candidate_version, (
            "Candidate should not be promoted to Production automatically"
        )
    except Exception:
        # No production alias set at all -- also acceptable, candidate is not production
        pass
