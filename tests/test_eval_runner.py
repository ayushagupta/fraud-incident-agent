"""Unit tests for the eval scenario runner.

Patches out subprocess, the monitoring cycle, DB operations, and the agent
to verify pass/fail logic for all expected/actual combinations. No live
services are required.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.agent.loop import AgentResult
from src.eval.runner import ScenarioResult, run_scenario
from src.eval.scenario import ExpectedConfig, Scenario, SetupConfig, TrafficConfig


def _make_scenario(expected_diagnosis: str, expected_action: str) -> Scenario:
    return Scenario(
        name="test_scenario",
        description="unit test scenario",
        setup=SetupConfig(traffic=TrafficConfig(month=2, rows=100, speed=1000.0)),
        expected=ExpectedConfig(diagnosis=expected_diagnosis, action=expected_action),
    )


def _mock_agent_result(diagnosis: str, action: str) -> AgentResult:
    return AgentResult(
        diagnosis=diagnosis,
        action=action,
        confidence=0.9,
        reasoning_trace="test trace",
        tools_called=["get_incident_details", "get_null_rates"],
        iterations=2,
    )


def _mock_incident(incident_id: int = 1) -> MagicMock:
    m = MagicMock()
    m.id = incident_id
    return m


def _incident(n: int) -> MagicMock:
    return _mock_incident(incident_id=n)


@patch("src.eval.runner.investigate_incident")
@patch("src.eval.runner.run_monitoring_cycle")
@patch("src.eval.runner.subprocess")
@patch("src.eval.runner._query_open_incident")
@patch("src.eval.runner._truncate_tables")
def test_no_incident_expected_and_none_opened(
    mock_truncate, mock_query, mock_subprocess, mock_monitoring, mock_investigate
):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_monitoring.return_value = []
    mock_query.return_value = None

    result = run_scenario(_make_scenario("no_incident", "no_action"))

    assert result.passed is True
    assert result.monitor_opened_incident is False
    assert result.failure_reason is None
    assert result.actual_diagnosis is None
    mock_investigate.assert_not_called()


@patch("src.eval.runner.investigate_incident")
@patch("src.eval.runner.run_monitoring_cycle")
@patch("src.eval.runner.subprocess")
@patch("src.eval.runner._query_open_incident")
@patch("src.eval.runner._truncate_tables")
def test_no_incident_expected_but_one_opened(
    mock_truncate, mock_query, mock_subprocess, mock_monitoring, mock_investigate
):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_monitoring.return_value = []
    mock_query.return_value = _mock_incident()

    result = run_scenario(_make_scenario("no_incident", "no_action"))

    assert result.passed is False
    assert result.monitor_opened_incident is True
    assert result.failure_reason == "monitor opened an unexpected incident"
    mock_investigate.assert_not_called()


@patch("src.eval.runner.investigate_incident")
@patch("src.eval.runner.run_monitoring_cycle")
@patch("src.eval.runner.subprocess")
@patch("src.eval.runner._query_open_incident")
@patch("src.eval.runner._truncate_tables")
def test_incident_expected_but_none_opened(
    mock_truncate, mock_query, mock_subprocess, mock_monitoring, mock_investigate
):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_monitoring.return_value = []
    mock_query.return_value = None

    result = run_scenario(_make_scenario("genuine_drift", "retrain"))

    assert result.passed is False
    assert result.monitor_opened_incident is False
    assert result.failure_reason == "monitor did not open an incident"
    mock_investigate.assert_not_called()


@patch("src.eval.runner.investigate_incident")
@patch("src.eval.runner.run_monitoring_cycle")
@patch("src.eval.runner.subprocess")
@patch("src.eval.runner._query_open_incident")
@patch("src.eval.runner._truncate_tables")
def test_agent_diagnosis_matches_expected(
    mock_truncate, mock_query, mock_subprocess, mock_monitoring, mock_investigate
):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_monitoring.return_value = [{"type": "psi_alarm"}]
    mock_query.return_value = _incident(5)
    mock_investigate.return_value = _mock_agent_result("genuine_drift", "retrain")

    result = run_scenario(_make_scenario("genuine_drift", "retrain"))

    assert result.passed is True
    assert result.actual_diagnosis == "genuine_drift"
    assert result.actual_action == "retrain"
    assert result.failure_reason is None
    assert result.tools_called == ["get_incident_details", "get_null_rates"]
    assert result.confidence == pytest.approx(0.9)
    mock_investigate.assert_called_once_with(5)


@patch("src.eval.runner.investigate_incident")
@patch("src.eval.runner.run_monitoring_cycle")
@patch("src.eval.runner.subprocess")
@patch("src.eval.runner._query_open_incident")
@patch("src.eval.runner._truncate_tables")
def test_agent_diagnosis_does_not_match_expected(
    mock_truncate, mock_query, mock_subprocess, mock_monitoring, mock_investigate
):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_monitoring.return_value = [{"type": "null_rate"}]
    mock_query.return_value = _incident(3)
    mock_investigate.return_value = _mock_agent_result("pipeline_bug", "alert_human")

    result = run_scenario(_make_scenario("genuine_drift", "retrain"))

    assert result.passed is False
    assert result.actual_diagnosis == "pipeline_bug"
    assert "diagnosis" in result.failure_reason
    assert "genuine_drift" in result.failure_reason
    assert "pipeline_bug" in result.failure_reason
