# ML Incident Response Agent

## Project Overview

This project is an autonomous ML incident response system. A LightGBM fraud detection model is deployed behind a FastAPI service. A monitoring job watches the prediction stream for distribution shifts, null spikes, schema violations, and prediction drift. When something looks wrong, an agent investigates using a set of tools (querying prediction history, comparing feature distributions across windows, checking pipeline health, reviewing deploy history) and produces a structured diagnosis: genuine drift, pipeline bug, schema change, seasonal pattern, or no action needed. It recommends one of retrain, rollback, alert human, or suppress. A human approves the action through a Streamlit UI before it executes.

The dataset is the Bank Account Fraud (BAF) dataset from NeurIPS 2022 (Feedzai), which has temporal structure across 8 months with characterized natural drift. The training set is month 1. Months 2 through 8 are replayed as simulated production traffic, with optional synthetic failure modes injected on top (null injection, unseen categorical values, seasonal multipliers) so the agent has multiple distinguishable failure modes to diagnose.

The eval harness is the most important deliverable. It runs the full pipeline against scripted scenarios with known ground-truth diagnoses and measures agent accuracy per failure mode. This is what makes the project resume-worthy.

## Architecture

Components and data flow:

The FastAPI serving layer (`src/serving/`) loads the current production model from MLflow and exposes `/predict`. Every prediction is logged to the `predictions` table in Postgres with inputs, output, model version, and timestamp.

The traffic simulator (`scripts/simulate_traffic.py`) replays BAF months 2 through 8 against the serving endpoint at configurable speed. CLI flags inject specific failure modes: `--inject-nulls FEATURE RATE`, `--inject-unseen-category COLUMN VALUE`, `--inject-seasonal MULTIPLIER`.

The monitoring job (`src/monitoring/`) runs on a schedule, reads recent predictions from Postgres, computes PSI per feature against the training distribution, prediction distribution drift, null rates, and unseen category counts, and writes results to the `monitoring_metrics` table. When thresholds are exceeded, it creates an `incident` row with status `open`.

The agent (`src/agent/`) is triggered when a new open incident appears. It uses Anthropic's tool-use API directly (no LangChain). Its tools query Postgres and MLflow. It produces a structured diagnosis written to the `incidents` table and proposes an action. The action does not execute automatically.

The retraining pipeline (`src/training/`) is a script that takes a date window, pulls predictions from that window, retrains LightGBM, and registers the new model in MLflow with metrics. It runs as a separate process triggered by the orchestrator after human approval.

The deployment logic (`src/deployment/`) implements a blue-green check: candidate model evaluated on a holdout, must beat current production AUC by a configurable margin, otherwise rejected. If it passes, the MLflow model stage tag flips to `Production` and the serving layer picks up the new version on next model refresh.

The Streamlit UI (`src/ui/`) shows current monitoring metrics, open incidents with the agent's diagnosis and reasoning trace, and approve/reject buttons. Approvals enqueue the proposed action.

The eval harness (`src/eval/`) defines scenarios as YAML configs (injection type, time window, expected diagnosis). The runner resets state, replays traffic with the injection, lets monitoring detect, invokes the agent, and records whether the diagnosis matched. Output is a CSV and a markdown summary with per-failure-mode accuracy, false retrain rate, and average tools called per diagnosis.

MLflow runs locally with file-backed artifact store and Postgres backend store. All services share one Postgres instance (separate databases or schemas: `app` for predictions/metrics/incidents, `mlflow` for the registry).

## Storage

Postgres is the system of record for predictions, monitoring metrics, and incidents. SQLAlchemy is the ORM. Connection string is read from `DATABASE_URL` env var. Use Alembic for migrations from day one, even though the schema is small, because schema evolution is part of the story.

Schemas live in `src/common/db.py`:
- `predictions`: id, timestamp, model_version, input_features (JSONB), prediction, probability
- `monitoring_metrics`: id, timestamp, metric_name, feature_name (nullable), value, window_start, window_end
- `incidents`: id, opened_at, closed_at, status, trigger_metric, agent_diagnosis (JSONB), agent_action, human_decision, reasoning_trace (text)

MLflow registry holds models. The currently deployed model is the one with stage tag `Production`. Rollback means re-tagging a prior version.

## Code Conventions

Type hints on every function signature. Pydantic v2 for any structured config or API schema. Pure functions where possible, side effects pushed to the edges. Prefer explicit imports over star imports. Module-level constants in UPPER_CASE.

Docstrings in plain prose, not Google or NumPy format. One paragraph that says what the function does and why it exists. Skip docstrings for obvious helpers.

No emoji anywhere, including in print statements, logs, or generated text. No em dashes anywhere, including in docstrings, README content, log messages, or any string the user might see. Use commas or periods.

Logging via the standard `logging` module, not print, except in scripts where print is fine for progress output.

Tests live in `tests/`, mirror the `src/` layout, use pytest. Test only critical paths: the deployment validation gate logic, the drift math, the agent tool implementations, the eval scenario runner. Do not aim for coverage. Aim for tests that would catch a real bug.

Configuration through environment variables loaded into a Pydantic settings object in `src/common/config.py`. No hardcoded paths, ports, or thresholds outside that file.

## Commands

```
# Postgres and MLflow (docker compose)
docker compose up -d postgres mlflow

# Initial schema
alembic upgrade head

# Train the first production model
python scripts/train_v1.py

# Start the serving layer
uvicorn src.serving.app:app --reload --port 8000

# Run the monitoring loop
python -m src.monitoring.run

# Replay traffic
python scripts/simulate_traffic.py --speed 60 --month 2

# With injected failure mode
python scripts/simulate_traffic.py --speed 60 --month 3 --inject-nulls income 0.5

# Start the Streamlit UI
streamlit run src/ui/app.py

# Run the eval harness
python -m src.eval.run --scenarios src/eval/scenarios/

# Tests
pytest
```

## What Not to Do

Do not introduce Kafka, Spark, Kubeflow, DVC, LangChain, Prometheus, Grafana, Redis, Celery, or any cloud service. The point of the project is depth in a small surface area. If a request seems to need one of these, the answer is almost always a simpler local equivalent.

Do not write defensive code, input validation, or error handling beyond what is needed for the happy path and the obvious failure modes. This is a portfolio project, not a hardened service.

Do not add a web frontend beyond Streamlit. Do not add authentication. Do not add a REST API beyond the `/predict` endpoint.

Do not add new dependencies without flagging them in your response. The current set is intentional.

Do not write tutorial-style comments explaining what standard library calls do. Assume the reader is a competent Python engineer.

Do not invent file paths or commands. If something is not specified here, ask before assuming.

## Working Style

When given a task, restate the goal in one sentence before writing code. If the task is ambiguous about file structure or interface, ask one clarifying question rather than guessing. Prefer small, reviewable diffs. After implementing a task, list any deviations from the request and why.

When tests pass, say so plainly. Do not narrate every step. Do not summarize the code you just wrote unless asked.

If you need to add a dependency, stop and flag it before installing.
