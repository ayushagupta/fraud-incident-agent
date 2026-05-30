"""Agent loop for ML incident investigation.

Runs the Anthropic tool-use loop against a live incident and produces a
structured diagnosis. The agent calls tools in src/agent/tools.py to gather
evidence from Postgres and MLflow before committing to a final answer.
"""
import json
import logging
import sys
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from src.agent import tools as _tools
from src.common.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    diagnosis: Literal[
        "genuine_drift",
        "pipeline_bug",
        "schema_change",
        "seasonal_pattern",
        "no_action_needed",
    ]
    action: Literal["retrain", "rollback", "alert_human", "suppress"]
    confidence: float
    reasoning_trace: str
    tools_called: list[str]
    iterations: int


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ML incident response agent for a fraud detection system. Your job is to investigate incidents triggered by the monitoring system and produce a structured diagnosis with a recommended action.

The fraud detection model is a LightGBM classifier trained on month 1 of the Bank Account Fraud dataset. Production traffic replays months 2-8 through a FastAPI serving layer. Each prediction is logged to Postgres. A monitoring job computes PSI per feature, null rates, unseen category counts, and prediction distribution drift against a training reference, then opens incidents when thresholds are exceeded.

DIAGNOSES AND THEIR FINGERPRINTS

genuine_drift: Elevated PSI (above 0.2) on multiple features simultaneously, spread across semantically related features like income, velocity, or age. The drift is gradual, appearing across consecutive monitoring windows rather than as a single sudden spike. Null rates are normal. The prediction distribution may shift over time. This reflects natural population change and requires retraining on recent data.

pipeline_bug: Sudden spike in null rate on one or a few specific features while PSI on other features and the prediction distribution remain normal. The null rate increase is abrupt, appearing in one monitoring window. This reflects an upstream data pipeline failure that stopped sending certain features.

schema_change: Non-zero unseen category counts on categorical features. The model receives category values it was never trained on. PSI may also be elevated. This reflects a change in an upstream encoding system or new product categories being introduced.

seasonal_pattern: Drift metrics are elevated but the pattern matches historical incidents marked as seasonal. PSI may be high on temporal or volume-related features. The pattern repeats on a predictable cycle. Prior human decisions or agent actions of suppress are strong signals.

no_action_needed: The trigger fired but evidence is within noise. PSI is mildly elevated on one feature, null rates are normal, no unseen categories, prediction distribution is stable, no recent deploy coincides with the trigger.

ACTIONS AND WHEN TO RECOMMEND THEM

retrain: Recommend for genuine_drift. Recent data covers the shifted population.

rollback: Recommend when a recent model deploy (within the last 24-48 hours) coincides exactly with the start of the incident and metrics were normal before the deploy.

alert_human: Recommend for ambiguous evidence, confidence below 0.6, schema_change with unclear cause, or any situation that does not fit the known patterns.

suppress: Recommend for seasonal_pattern when historical incidents with the same fingerprint were suppressed. Also for no_action_needed.

INVESTIGATIVE APPROACH

1. Start by calling get_incident_details to read what triggered the incident.
2. Gather targeted evidence based on the trigger. Do not call every tool.
3. For PSI triggers: check get_recent_metrics for PSI trends, compare_feature_distributions to quantify shift, get_null_rates to rule out pipeline bug.
4. For null rate triggers: check get_null_rates to identify affected features, check whether PSI is also elevated.
5. For unseen category triggers: call get_unseen_categories.
6. Always check get_recent_deploys to rule out a recent model change.
7. Check get_incident_history if you suspect a seasonal pattern.
8. Reach a confident diagnosis before writing your final answer.

OUTPUT CONTRACT

Your final response (when you stop calling tools) MUST contain only a single JSON object. These rules are strict and non-negotiable.

The very first character of your final response MUST be {. The very last character MUST be }. No prose, no preamble, no commentary, no explanation may appear before or after the JSON. No markdown code fences, no triple backticks, no language tags. Put ALL your reasoning and analysis inside the "reasoning_trace" field of the JSON, not outside it. Any text outside the JSON object will cause a parse failure and force a human escalation.

Required format exactly:
{"diagnosis": "<genuine_drift|pipeline_bug|schema_change|seasonal_pattern|no_action_needed>", "action": "<retrain|rollback|alert_human|suppress>", "confidence": <0.0-1.0>, "reasoning_trace": "<your full reasoning here>"}"""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_incident_details",
        "description": (
            "Return the full record for a single incident. "
            "Fetches the incident row with the given id and returns its id, opened_at, "
            "closed_at, status, trigger_metric, agent_diagnosis, agent_action, "
            "human_decision, and reasoning_trace. Use this as the first tool call "
            "when the agent is invoked so it knows what triggered the investigation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "incident_id": {
                    "type": "integer",
                    "description": "The id of the incident to fetch.",
                }
            },
            "required": ["incident_id"],
        },
    },
    {
        "name": "get_recent_metrics",
        "description": (
            "Return monitoring metric rows for a given metric over the recent window. "
            "Queries monitoring_metrics for rows matching metric_name (and optionally "
            "feature_name) within hours_back hours of now. Returns rows in ascending "
            "timestamp order with timestamp, feature_name, value, window_start, and "
            "window_end. Use this to see how a metric has trended over time, for "
            "example to distinguish a sudden spike from a gradual drift."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "The metric_name to filter by, e.g. 'psi', 'null_rate', 'prediction_drift'.",
                },
                "feature_name": {
                    "type": "string",
                    "description": "Optional feature name to further filter rows.",
                },
                "hours_back": {
                    "type": "integer",
                    "description": "How many hours back to look. Default 24.",
                    "default": 24,
                },
            },
            "required": ["metric_name"],
        },
    },
    {
        "name": "compare_feature_distributions",
        "description": (
            "Compare the distribution of a feature between two historical time windows. "
            "window_a_hours_back and window_b_hours_back define how far back the end of "
            "each window is from now; window_size_hours sets the width of each window. "
            "Returns mean, std, min, max, median, null_rate, and row_count for each "
            "window plus the absolute mean difference. Use to quantify distribution "
            "shift on a specific feature."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "The feature column to compare.",
                },
                "window_a_hours_back": {
                    "type": "integer",
                    "description": "Hours ago for the end of the more recent window A.",
                },
                "window_b_hours_back": {
                    "type": "integer",
                    "description": "Hours ago for the end of the older reference window B.",
                },
                "window_size_hours": {
                    "type": "integer",
                    "description": "Width of each window in hours. Default 1.",
                    "default": 1,
                },
            },
            "required": ["feature_name", "window_a_hours_back", "window_b_hours_back"],
        },
    },
    {
        "name": "get_null_rates",
        "description": (
            "Return the most recent null rate per feature from the monitoring metrics table. "
            "Looks up monitoring_metrics rows where metric_name is 'null_rate' within "
            "hours_back hours of now. Returns the latest value for each feature_name. "
            "A sudden increase in null rate for a specific feature is the primary signal "
            "for pipeline bugs upstream."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours_back": {
                    "type": "integer",
                    "description": "How many hours back to look. Default 1.",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_unseen_categories",
        "description": (
            "Return the most recent unseen category count per categorical feature. "
            "Looks up monitoring_metrics rows where metric_name is 'unseen_category_count' "
            "within hours_back hours of now. A non-zero count means production traffic "
            "contains category values the model was never trained on, a strong signal of "
            "a schema change or bad upstream encoding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours_back": {
                    "type": "integer",
                    "description": "How many hours back to look. Default 1.",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_prediction_distribution",
        "description": (
            "Return summary statistics and a histogram of model output probabilities. "
            "Queries predictions within hours_back hours of now and computes mean, std, "
            "percentiles (p5/p25/p50/p75/p95), row count, and a 10-bucket histogram "
            "over [0, 1]. Use to detect whether the model score distribution has shifted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours_back": {
                    "type": "integer",
                    "description": "How many hours back to look. Default 1.",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_recent_deploys",
        "description": (
            "Return the model version history from the MLflow registry over the last N days. "
            "Each entry has version, registered_at (ISO string), and is_production. "
            "Use to check whether a model change coincides with the start of an incident."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to look. Default 7.",
                    "default": 7,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_incident_history",
        "description": (
            "Return past closed incidents ordered most recent first. "
            "Returns at most limit rows from within days_back days. Each dict has id, "
            "opened_at, closed_at, trigger_metric, agent_diagnosis, agent_action, and "
            "human_decision. Use to check whether the current incident resembles a "
            "pattern the agent has already diagnosed, or to detect seasonal patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to look. Default 30.",
                    "default": 30,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return. Default 20.",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_MAP = {
    "get_incident_details": _tools.get_incident_details,
    "get_recent_metrics": _tools.get_recent_metrics,
    "compare_feature_distributions": _tools.compare_feature_distributions,
    "get_null_rates": _tools.get_null_rates,
    "get_unseen_categories": _tools.get_unseen_categories,
    "get_prediction_distribution": _tools.get_prediction_distribution,
    "get_recent_deploys": _tools.get_recent_deploys,
    "get_incident_history": _tools.get_incident_history,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("Tool %s raised: %s", name, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first complete JSON object in text using brace counting.

    Locates the first '{', then counts opening and closing braces to find the
    matching '}'. Parses only that substring. Returns None if no valid JSON
    object is found, allowing callers to handle prose-wrapped or malformed
    responses without raising.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def investigate_incident(incident_id: int, max_iterations: int = 10) -> AgentResult:
    """Run the tool-use agent loop to investigate an open incident.

    Sends a conversation to Claude with the system prompt and tool schemas.
    On each iteration, executes any tool calls the model requests and appends
    results. Continues until the model produces a final text-only response
    containing a JSON diagnosis, or until max_iterations is reached.
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Investigate incident {incident_id} and produce your final diagnosis.",
        }
    ]

    tools_called: list[str] = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tools_called.append(block.name)
                result_str = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # stop_reason == "end_turn": parse the final JSON answer.
        final_text = ""
        for block in response.content:
            if block.type == "text":
                final_text = block.text.strip()
                break

        payload = _extract_json_object(final_text)
        if payload is not None:
            try:
                return AgentResult(
                    diagnosis=payload["diagnosis"],
                    action=payload["action"],
                    confidence=float(payload["confidence"]),
                    reasoning_trace=payload["reasoning_trace"],
                    tools_called=tools_called,
                    iterations=iteration,
                )
            except (KeyError, ValueError) as exc:
                logger.error("JSON found but missing required fields: %s", exc)

        logger.error("Failed to extract JSON from model response. Raw response: %s", final_text)
        return AgentResult(
            diagnosis="no_action_needed",
            action="alert_human",
            confidence=0.0,
            reasoning_trace=f"Failed to parse model response: {final_text[:500]}",
            tools_called=tools_called,
            iterations=iteration,
        )

    return AgentResult(
        diagnosis="no_action_needed",
        action="alert_human",
        confidence=0.0,
        reasoning_trace=(
            f"Agent reached the maximum iteration limit ({max_iterations}) "
            "without producing a final diagnosis."
        ),
        tools_called=tools_called,
        iterations=iteration,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.agent.loop INCIDENT_ID")
        sys.exit(1)

    try:
        incident_id = int(sys.argv[1])
    except ValueError:
        print(f"INCIDENT_ID must be an integer, got: {sys.argv[1]}")
        sys.exit(1)

    result = investigate_incident(incident_id)
    print(json.dumps(result.model_dump(), indent=2))
