"""Tests for the agent loop.

Unit test: mocks the Anthropic API to verify the loop correctly handles a
tool_use response followed by a final text response, without touching Postgres
or making real API calls.

Integration test: inserts a synthetic incident with genuine-drift fingerprint
(high PSI on multiple features, zero null rates) into Postgres, runs
investigate_incident, and asserts the returned diagnosis is genuine_drift with
action retrain. Requires a running Postgres and ANTHROPIC_API_KEY. Skipped
otherwise. Marked @pytest.mark.integration.
"""
import datetime
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agent.loop import AgentResult, _extract_json_object, investigate_incident
from src.common.db import Incident, MonitoringMetric, SessionLocal


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Unit test -- no DB, no real API
# ---------------------------------------------------------------------------


def test_loop_handles_tool_use_then_final_text():
    """Verify the loop executes a tool call and then parses the final JSON answer."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_01abc"
    tool_block.name = "get_incident_details"
    tool_block.input = {"incident_id": 42}

    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [tool_block]

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps({
        "diagnosis": "genuine_drift",
        "action": "retrain",
        "confidence": 0.9,
        "reasoning_trace": "High PSI on income and velocity_6h, zero null rates.",
    })

    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [text_block]

    tool_result_payload = json.dumps({
        "id": 42,
        "status": "open",
        "trigger_metric": '{"types":["psi_alarm"],"features":["income"]}',
    })

    with patch("src.agent.loop.anthropic.Anthropic") as mock_cls, \
         patch("src.agent.loop.execute_tool") as mock_exec:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [first_response, second_response]
        mock_exec.return_value = tool_result_payload

        result = investigate_incident(42)

    assert isinstance(result, AgentResult)
    assert result.diagnosis == "genuine_drift"
    assert result.action == "retrain"
    assert result.confidence == pytest.approx(0.9)
    assert result.iterations == 2
    assert result.tools_called == ["get_incident_details"]
    mock_exec.assert_called_once_with("get_incident_details", {"incident_id": 42})


def test_loop_returns_alert_human_at_max_iterations():
    """Verify the loop returns a safe fallback when max_iterations is reached."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_01"
    tool_block.name = "get_incident_details"
    tool_block.input = {"incident_id": 1}

    stuck_response = MagicMock()
    stuck_response.stop_reason = "tool_use"
    stuck_response.content = [tool_block]

    with patch("src.agent.loop.anthropic.Anthropic") as mock_cls, \
         patch("src.agent.loop.execute_tool") as mock_exec:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = stuck_response
        mock_exec.return_value = json.dumps({"id": 1})

        result = investigate_incident(1, max_iterations=3)

    assert result.diagnosis == "no_action_needed"
    assert result.action == "alert_human"
    assert result.confidence == 0.0
    assert result.iterations == 3
    assert "maximum iteration limit" in result.reasoning_trace


def test_loop_handles_json_inside_markdown_fences():
    """Verify the brace-counter finds JSON even when wrapped in code fences."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        "```json\n"
        '{"diagnosis": "pipeline_bug", "action": "alert_human", '
        '"confidence": 0.8, "reasoning_trace": "Null rate spiked on income."}\n'
        "```"
    )

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]

    with patch("src.agent.loop.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = response

        result = investigate_incident(1)

    assert result.diagnosis == "pipeline_bug"
    assert result.action == "alert_human"


def test_extract_json_object_with_prose_prefix():
    """Verify _extract_json_object finds the JSON object even after leading prose."""
    prose = (
        "Here is my analysis based on the evidence gathered.\n\n"
        '{"diagnosis": "schema_change", "action": "alert_human", '
        '"confidence": 0.75, "reasoning_trace": "Unseen categories detected."}'
        "\n\nHope that helps."
    )
    result = _extract_json_object(prose)
    assert result is not None
    assert result["diagnosis"] == "schema_change"
    assert result["confidence"] == pytest.approx(0.75)


def test_loop_parses_final_text_with_prose_surrounding_json():
    """Verify investigate_incident succeeds when the model wraps JSON in prose."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        "Based on the investigation, here is my conclusion:\n\n"
        '{"diagnosis": "no_action_needed", "action": "suppress", '
        '"confidence": 0.85, "reasoning_trace": "Metrics within noise."}'
        "\n\nNo further action required."
    )

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]

    with patch("src.agent.loop.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = response

        result = investigate_incident(1)

    assert result.diagnosis == "no_action_needed"
    assert result.action == "suppress"
    assert result.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Integration test -- real DB, real API
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_tables():
    def _clean():
        with SessionLocal() as s:
            s.query(MonitoringMetric).delete(synchronize_session=False)
            s.query(Incident).delete(synchronize_session=False)
            s.commit()

    _clean()
    yield
    _clean()


@pytest.mark.integration
def test_investigate_incident_genuine_drift(clean_tables):
    """Agent correctly diagnoses genuine drift given high PSI on multiple features."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    now = _now()

    with SessionLocal() as s:
        incident = Incident(
            opened_at=now,
            status="open",
            trigger_metric='{"types":["psi_alarm"],"features":["income","velocity_6h","velocity_24h"]}',
        )
        s.add(incident)
        s.flush()
        incident_id = incident.id

        # High PSI on three correlated features across three consecutive windows.
        for feature, base_psi in [("income", 0.72), ("velocity_6h", 0.65), ("velocity_24h", 0.61)]:
            for i in range(3):
                ts = now - datetime.timedelta(hours=i)
                s.add(MonitoringMetric(
                    timestamp=ts,
                    metric_name="psi",
                    feature_name=feature,
                    value=round(base_psi - i * 0.04, 3),
                    window_start=ts - datetime.timedelta(hours=1),
                    window_end=ts,
                ))

        # Zero null rates on all features.
        for feature in ["income", "velocity_6h", "velocity_24h", "customer_age"]:
            s.add(MonitoringMetric(
                timestamp=now,
                metric_name="null_rate",
                feature_name=feature,
                value=0.0,
                window_start=now - datetime.timedelta(hours=1),
                window_end=now,
            ))

        s.commit()

    result = investigate_incident(incident_id)

    assert result.diagnosis == "genuine_drift", (
        f"Expected genuine_drift, got {result.diagnosis}. "
        f"Reasoning: {result.reasoning_trace}"
    )
    assert result.action == "retrain", (
        f"Expected retrain, got {result.action}"
    )
