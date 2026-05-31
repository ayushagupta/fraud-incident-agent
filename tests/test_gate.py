"""Tests for src/deployment/gate.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.deployment.gate import GateResult, evaluate_candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Returns a fixed probability array regardless of X, tiling as needed."""

    def __init__(self, proba_1d: np.ndarray) -> None:
        self._p = np.asarray(proba_1d, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        p = np.tile(self._p, int(np.ceil(n / len(self._p))))[:n]
        return np.column_stack([1.0 - p, p])


def _make_holdout(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Return a balanced holdout: first half positive, second half negative."""
    half = n // 2
    y = pd.Series([1] * half + [0] * half)
    X = pd.DataFrame({"dummy": np.zeros(n)})
    return X, y


# ---------------------------------------------------------------------------
# Shared fixture: patch everything except the models and their scoring
# ---------------------------------------------------------------------------


@pytest.fixture
def _base_mocks():
    """Patch MLflow connectivity and I/O; yield a dict with the promote mock."""
    X, y = _make_holdout()
    with (
        patch("src.deployment.gate.mlflow.set_tracking_uri"),
        patch("src.deployment.gate.MlflowClient"),
        patch("src.deployment.gate._resolve_holdout_months", return_value=[7]),
        patch("src.deployment.gate._load_holdout", return_value=(X, y)),
        patch("src.deployment.gate.promote_to_production") as mock_promote,
    ):
        yield {"mock_promote": mock_promote, "X": X, "y": y}


# ---------------------------------------------------------------------------
# Case 1: candidate clearly better -- should pass and promote
# ---------------------------------------------------------------------------


def test_gate_candidate_clearly_better(_base_mocks):
    """Candidate with near-perfect PR-AUC beats a random-baseline production model."""
    half = 100
    # Production: same prob for all rows -> random baseline, AUC ~ 0.5
    prod_model = _FakeModel(np.full(200, 0.5))
    # Candidate: high prob for positives, low for negatives -> near-perfect
    cand_model = _FakeModel(np.array([0.9] * half + [0.1] * half))

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "1")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("2", holdout_months=[7])

    assert isinstance(result, GateResult)
    assert result.production_version == "1"
    assert result.candidate_version == "2"
    assert result.pr_auc_delta > 0.01, "Expected a large positive PR-AUC delta"
    assert result.passed
    assert result.promoted
    assert "meets margin" in result.reason
    _base_mocks["mock_promote"].assert_called_once_with("2")


# ---------------------------------------------------------------------------
# Case 2: candidate marginally better but below margin -- should fail
# ---------------------------------------------------------------------------


def test_gate_candidate_below_margin(_base_mocks):
    """Candidate is identical to production (delta=0); delta >= 0 but < margin."""
    half = 100
    # Both models return the same predictions -> PR-AUC delta exactly 0
    same_proba = np.array([0.9] * half + [0.1] * half)
    prod_model = _FakeModel(same_proba)
    cand_model = _FakeModel(same_proba)

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "1")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("3", holdout_months=[7])

    assert not result.passed
    assert not result.promoted
    assert result.pr_auc_delta == pytest.approx(0.0, abs=1e-6)
    assert "below margin" in result.reason
    _base_mocks["mock_promote"].assert_not_called()


# ---------------------------------------------------------------------------
# Case 3: candidate worse than production -- should fail
# ---------------------------------------------------------------------------


def test_gate_candidate_worse_than_production(_base_mocks):
    """Candidate with inverted predictions (AUC 0) is worse than production."""
    half = 100
    # Production: near-perfect
    prod_model = _FakeModel(np.array([0.9] * half + [0.1] * half))
    # Candidate: inverted (positives get low prob, negatives get high prob)
    cand_model = _FakeModel(np.array([0.1] * half + [0.9] * half))

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "1")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("4", holdout_months=[7])

    assert not result.passed
    assert not result.promoted
    assert result.pr_auc_delta < 0
    assert result.reason == "candidate PR-AUC worse than production"
    _base_mocks["mock_promote"].assert_not_called()


# ---------------------------------------------------------------------------
# Explicit promotion / no-promotion assertions
# ---------------------------------------------------------------------------


def test_promote_not_called_when_fails(_base_mocks):
    """promote_to_production must not be invoked when the gate fails."""
    half = 100
    prod_model = _FakeModel(np.array([0.9] * half + [0.1] * half))
    cand_model = _FakeModel(np.array([0.1] * half + [0.9] * half))

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "3")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("5", holdout_months=[7])

    assert result.promoted is False
    _base_mocks["mock_promote"].assert_not_called()


def test_promote_called_when_passes(_base_mocks):
    """promote_to_production must be called with the candidate version when gate passes."""
    half = 100
    prod_model = _FakeModel(np.full(200, 0.5))
    cand_model = _FakeModel(np.array([0.9] * half + [0.1] * half))

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "3")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("6", holdout_months=[7])

    assert result.promoted is True
    _base_mocks["mock_promote"].assert_called_once_with("6")


# ---------------------------------------------------------------------------
# GateResult field completeness
# ---------------------------------------------------------------------------


def test_gate_result_fields(_base_mocks):
    """GateResult contains all expected fields with sensible types."""
    half = 100
    prod_model = _FakeModel(np.full(200, 0.5))
    cand_model = _FakeModel(np.array([0.9] * half + [0.1] * half))

    with (
        patch("src.deployment.gate.get_production_model", return_value=(prod_model, "1")),
        patch("src.deployment.gate.mlflow.lightgbm.load_model", return_value=cand_model),
    ):
        result = evaluate_candidate("2", holdout_months=[7])

    assert isinstance(result.holdout_months, list)
    assert result.holdout_rows == 200
    assert 0.0 <= result.production_auc <= 1.0
    assert 0.0 <= result.production_pr_auc <= 1.0
    assert 0.0 <= result.candidate_auc <= 1.0
    assert 0.0 <= result.candidate_pr_auc <= 1.0
    assert isinstance(result.reason, str) and len(result.reason) > 0


# ---------------------------------------------------------------------------
# Integration test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gate_integration():
    """Full gate against the actual MLflow registry.

    Set GATE_TEST_CANDIDATE to the candidate version string to run this.
    Verifies that the Production alias was updated iff result.promoted is True.
    """
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    from src.common.config import settings

    candidate_version = os.environ.get("GATE_TEST_CANDIDATE")
    if not candidate_version:
        pytest.skip("Set GATE_TEST_CANDIDATE=<version> to run integration test")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    result = evaluate_candidate(candidate_version)

    current_prod = client.get_model_version_by_alias(
        settings.MLFLOW_MODEL_NAME, settings.MLFLOW_PRODUCTION_ALIAS
    )
    if result.promoted:
        assert current_prod.version == candidate_version, (
            "Gate reported promoted=True but alias was not updated"
        )
    else:
        assert current_prod.version != candidate_version, (
            "Gate reported promoted=False but alias was flipped anyway"
        )
