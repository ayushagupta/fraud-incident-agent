"""Eval scenario runner.

Executes a single scenario end-to-end: truncates state, replays traffic via
the simulator, runs one monitoring cycle, optionally invokes the agent, and
returns a ScenarioResult.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, text

from src.agent.loop import AgentResult, investigate_incident
from src.common.db import Incident, SessionLocal
from src.eval.scenario import Scenario
from src.monitoring.run import run_monitoring_cycle


class ScenarioResult(BaseModel):
    scenario_name: str
    expected_diagnosis: str
    expected_action: str
    actual_diagnosis: str | None
    actual_action: str | None
    confidence: float | None
    tools_called: list[str]
    iterations: int
    reasoning_trace: str | None
    duration_seconds: float
    passed: bool
    failure_reason: str | None
    monitor_opened_incident: bool


def _truncate_tables() -> None:
    with SessionLocal() as session:
        session.execute(
            text("TRUNCATE TABLE predictions, monitoring_metrics, incidents CASCADE")
        )
        session.commit()


def _query_open_incident() -> Any | None:
    with SessionLocal() as session:
        return session.execute(
            select(Incident).where(Incident.status == "open").limit(1)
        ).scalar_one_or_none()


def _build_simulator_cmd(scenario: Scenario) -> list[str]:
    traffic = scenario.setup.traffic
    cmd = [
        "python",
        "scripts/simulate_traffic.py",
        "--month", str(traffic.month),
        "--speed", str(traffic.speed),
        "--limit", str(traffic.rows),
    ]
    inj = scenario.setup.injection
    if inj is None:
        return cmd
    if inj.type == "nulls":
        cmd += ["--inject-nulls", inj.feature, str(inj.rate)]
    elif inj.type == "unseen_category":
        cmd += ["--inject-unseen-category", inj.column, inj.value]
    elif inj.type == "seasonal":
        cmd += ["--inject-seasonal", str(inj.multiplier)]
    return cmd


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Execute a full eval scenario and return the result.

    Resets Postgres state, replays traffic, runs one monitoring cycle, and
    optionally runs the agent. Compares the agent output to expected values
    to determine pass or fail.
    """
    start = time.monotonic()

    _truncate_tables()

    cmd = _build_simulator_cmd(scenario)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Simulator exited with code {proc.returncode}.\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
        )

    run_monitoring_cycle()

    incident = _query_open_incident()
    monitor_opened_incident = incident is not None

    exp_diag = scenario.expected.diagnosis
    exp_action = scenario.expected.action

    # No incident opened.
    if not monitor_opened_incident:
        duration = time.monotonic() - start
        if exp_diag == "no_incident":
            return ScenarioResult(
                scenario_name=scenario.name,
                expected_diagnosis=exp_diag,
                expected_action=exp_action,
                actual_diagnosis=None,
                actual_action=None,
                confidence=None,
                tools_called=[],
                iterations=0,
                reasoning_trace=None,
                duration_seconds=duration,
                passed=True,
                failure_reason=None,
                monitor_opened_incident=False,
            )
        return ScenarioResult(
            scenario_name=scenario.name,
            expected_diagnosis=exp_diag,
            expected_action=exp_action,
            actual_diagnosis=None,
            actual_action=None,
            confidence=None,
            tools_called=[],
            iterations=0,
            reasoning_trace=None,
            duration_seconds=duration,
            passed=False,
            failure_reason="monitor did not open an incident",
            monitor_opened_incident=False,
        )

    # Incident opened but we expected none.
    if exp_diag == "no_incident":
        duration = time.monotonic() - start
        return ScenarioResult(
            scenario_name=scenario.name,
            expected_diagnosis=exp_diag,
            expected_action=exp_action,
            actual_diagnosis=None,
            actual_action=None,
            confidence=None,
            tools_called=[],
            iterations=0,
            reasoning_trace=None,
            duration_seconds=duration,
            passed=False,
            failure_reason="monitor opened an unexpected incident",
            monitor_opened_incident=True,
        )

    # Run the agent.
    result: AgentResult = investigate_incident(incident.id)
    duration = time.monotonic() - start

    passed = result.diagnosis == exp_diag and result.action == exp_action
    failure_reason: str | None = None
    if not passed:
        parts = []
        if result.diagnosis != exp_diag:
            parts.append(f"diagnosis: expected {exp_diag!r}, got {result.diagnosis!r}")
        if result.action != exp_action:
            parts.append(f"action: expected {exp_action!r}, got {result.action!r}")
        failure_reason = "; ".join(parts)

    return ScenarioResult(
        scenario_name=scenario.name,
        expected_diagnosis=exp_diag,
        expected_action=exp_action,
        actual_diagnosis=result.diagnosis,
        actual_action=result.action,
        confidence=result.confidence,
        tools_called=result.tools_called,
        iterations=result.iterations,
        reasoning_trace=result.reasoning_trace,
        duration_seconds=duration,
        passed=passed,
        failure_reason=failure_reason,
        monitor_opened_incident=True,
    )
