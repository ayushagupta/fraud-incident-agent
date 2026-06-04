# ML Fraud Incident Response Agent

An autonomous incident response system for a production fraud detection model. When the monitoring job detects distribution shift, null spikes, or schema violations in the prediction stream, an LLM-based agent investigates the evidence using tool calls against Postgres and MLflow, produces a structured diagnosis, and proposes an action. A human approves the action through a Streamlit UI before anything executes.

The eval harness runs 18 scripted scenarios with known ground-truth diagnoses across four failure modes and passes 18/18 at an average agent confidence of 0.71.

---

## Architecture

```
BAF dataset (months 0-8)
        |
        v
simulate_traffic.py  ----POST /predict---->  FastAPI serving layer
                                                    |
                                              predictions table (Postgres)
                                                    |
                                            monitoring job (PSI, null rates,
                                            unseen categories, prediction drift)
                                                    |
                                            monitoring_metrics table
                                                    |
                                           threshold exceeded?
                                                    |
                                              incidents table (open)
                                                    |
                                            agent (Claude + tool calls)
                                                    |
                                          structured diagnosis + action
                                                    |
                                            Streamlit UI (human review)
                                                    |
                                      approve --> retrain / rollback / suppress
                                      reject --> incident closed, no action
```

Everything runs locally. Postgres and MLflow run in Docker. The serving layer, monitoring job, and UI are plain Python processes.

---

## Components

### Serving layer (`src/serving/app.py`)

FastAPI application that loads the current production model from the MLflow registry on startup and exposes two endpoints:

- `POST /predict` -- accepts a feature payload, scores it, writes the full input and output to the `predictions` table, and returns `{prediction, probability, model_version}`.
- `POST /reload` -- hot-swaps the in-memory model to the current production alias without restarting.
- `GET /health` -- returns 200 if the model is loaded and Postgres is reachable.

The model is identified by the `production` alias in the MLflow registry. Changing which version holds that alias is how deployments and rollbacks work.

### Traffic simulator (`scripts/simulate_traffic.py`)

Replays rows from a chosen BAF month against the serving endpoint at a configurable rate. Supports three injectable failure modes that the agent is designed to distinguish:

- `--inject-nulls FEATURE RATE` -- sets `FEATURE` to null on `RATE` fraction of requests, simulating an upstream pipeline outage.
- `--inject-unseen-category COLUMN VALUE` -- replaces `COLUMN` with `VALUE` on every request, simulating a schema change in an upstream encoding system.
- `--inject-seasonal MULTIPLIER` -- scales `income` by `MULTIPLIER` on every request.

### Monitoring job (`src/monitoring/run.py`)

Runs on a schedule (or once with `--once`). On each cycle it:

1. Reads the most recent `MONITORING_WINDOW_ROWS` predictions from Postgres.
2. Computes Population Stability Index (PSI) per numeric feature against the training reference distribution.
3. Computes null rates per feature.
4. Counts unseen category values per categorical feature.
5. Computes PSI on model output probabilities (prediction drift).
6. Writes all metrics to `monitoring_metrics`.
7. Evaluates trigger conditions. If any fire and no incident is already open, inserts a row into `incidents` with `status = open`.

Trigger conditions (configurable via env vars):

| Trigger | Default threshold |
|---|---|
| PSI alarm on a single feature | > 0.5 |
| PSI moderate on 3 or more features simultaneously | > 0.2 each |
| Null rate on any feature | > 0.20 |
| Any unseen category value | >= 1 count |
| Prediction distribution drift | > 0.25 |

### Agent (`src/agent/loop.py`)

Triggered when an open incident exists. Runs a tool-use loop against Claude (model configurable via `ANTHROPIC_MODEL` env var, defaults to `claude-sonnet-4-6`). The agent calls tools sequentially to gather evidence, then emits a single JSON object as its final response.

**Diagnoses the agent produces:**

- `genuine_drift` -- gradual PSI increase across multiple semantically related features, normal null rates. Recommends `retrain`.
- `pipeline_bug` -- sudden null rate spike on one or a few features while all other metrics are normal. Recommends `alert_human`.
- `schema_change` -- non-zero unseen category counts on categorical features. Recommends `alert_human`.
- `seasonal_pattern` -- elevated metrics that match a prior suppressed incident fingerprint. Recommends `suppress`.
- `no_action_needed` -- trigger fired but all evidence is within noise. Recommends `suppress`.

**Agent tools (read-only queries against Postgres and MLflow):**

| Tool | Purpose |
|---|---|
| `get_incident_details` | Read the trigger metadata for the open incident |
| `get_recent_metrics` | Trend a metric over the last N hours |
| `compare_feature_distributions` | Compare mean/std/percentiles of a feature across two time windows |
| `get_null_rates` | Latest null rate per feature |
| `get_unseen_categories` | Latest unseen category count per feature |
| `get_prediction_distribution` | Score distribution histogram and percentiles |
| `get_recent_deploys` | MLflow model version history (to rule out a bad deploy) |
| `get_incident_history` | Past closed incidents (to detect seasonal patterns) |

The agent is bounded to `max_iterations=10` and falls back to `alert_human` if it cannot produce a parseable JSON response.

### Retraining pipeline (`src/training/retrain.py`)

Trains a new LightGBM classifier on a specified window of BAF months. The last 20% of rows from the highest-numbered month in the window are held out for validation. Logs parameters, metrics (AUC, PR-AUC), and the model artifact to MLflow. Registers the result as a new version but does not promote it to production.

### Deployment gate (`src/deployment/gate.py`)

Loads the current production model and a candidate version from the registry, scores both on a shared holdout set that neither was trained on, and decides pass or fail based on PR-AUC delta. The required improvement margin is `PROMOTION_MARGIN` (default 0.01). On pass, the `production` alias is atomically moved to the candidate. On fail, production is unchanged.

### Rollback (`src/deployment/rollback.py`)

Re-points the `production` alias to any prior version without evaluation. Used when a fast rollback is needed after a bad deployment.

### Streamlit UI (`src/ui/app.py`)

Three pages:

- **Overview** -- current production model version, prediction volume in the last 1h and 24h, open incident count, and a PSI trend chart for the top-3 drifting features.
- **Incidents** -- lists all incidents ordered by status then recency. For each open incident: a button to trigger agent investigation, the full reasoning trace, and approve/reject buttons once the agent has produced a diagnosis.
  - Approving `retrain` runs the retraining pipeline and the deployment gate in-process, then shows the result.
  - Approving `rollback` prompts for a target version before executing.
  - Approving `alert_human` and `suppress` closes the incident with the appropriate decision tag.
- **Deploys** -- table of all MLflow model versions with training months, holdout PR-AUC, and production status. Includes a manual rollback selector with a confirmation step.

### Eval harness (`src/eval/`)

Runs scripted scenarios end-to-end. Each scenario is a YAML file specifying a BAF month, row count, optional failure injection, and expected `{diagnosis, action}`. The runner truncates all Postgres tables, replays traffic, runs one monitoring cycle, optionally invokes the agent, and records pass or fail.

**Latest results (18 scenarios):**

| Category | Passed |
|---|---|
| genuine_drift | 6 / 6 |
| pipeline_bug | 4 / 4 |
| schema_change | 4 / 4 |
| no_incident | 4 / 4 |
| **Total** | **18 / 18** |

Average tools called per scenario: 6.1. Average confidence on passing diagnoses: 0.71.

---

## Dataset

The Bank Account Fraud (BAF) dataset from NeurIPS 2022 (Feedzai). It covers 8 months of bank account opening applications with a `fraud_bool` label. The temporal structure creates natural covariate shift across months.

- Month 0 is used for training the initial model.
- Months 1-7 are replayed as simulated production traffic.
- The dataset is not included in this repository. Download `Base.csv` and place it at `data/raw/Base.csv`.

---

## Database Schema

Three tables in the `app` Postgres database, managed by Alembic.

**`predictions`**

| Column | Type | Notes |
|---|---|---|
| id | integer | primary key |
| timestamp | timestamptz | indexed |
| model_version | varchar(64) | MLflow version string |
| input_features | jsonb | full feature payload |
| prediction | integer | 0 or 1 |
| probability | float | model output probability |

**`monitoring_metrics`**

| Column | Type | Notes |
|---|---|---|
| id | integer | primary key |
| timestamp | timestamptz | indexed |
| metric_name | varchar(128) | `psi`, `null_rate`, `unseen_category_count`, `prediction_drift` |
| feature_name | varchar(128) | null for model-level metrics |
| value | float | |
| window_start | timestamptz | start of the prediction window scored |
| window_end | timestamptz | end of the prediction window scored |

**`incidents`**

| Column | Type | Notes |
|---|---|---|
| id | integer | primary key |
| opened_at | timestamptz | indexed |
| closed_at | timestamptz | null while open |
| status | varchar(32) | `open` or `closed` |
| trigger_metric | varchar(128) | compact JSON summary of what fired |
| agent_diagnosis | jsonb | `{diagnosis, confidence, tools_called, iterations}` |
| agent_action | varchar(64) | `retrain`, `rollback`, `alert_human`, or `suppress` |
| human_decision | varchar(32) | `approved`, `rejected`, `suppressed`, `acknowledged` |
| reasoning_trace | text | full agent reasoning |

MLflow uses a separate `mlflow` database on the same Postgres instance, configured at startup via `MLFLOW_BACKEND_STORE_URI`. Artifacts are stored in `./mlruns`.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An Anthropic API key (only required for agent invocation and eval runs)
- The BAF dataset (`data/raw/Base.csv`)

### Install dependencies

```bash
pip install -e .
```

### Environment variables

Create a `.env` file at the project root:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/app
MLFLOW_TRACKING_URI=http://localhost:5001
ANTHROPIC_API_KEY=<your key>
```

Postgres is exposed on host port 5433 and MLflow on 5001 to avoid conflicts with common local services.

### Start infrastructure

```bash
docker compose up -d postgres mlflow
```

Wait for the health check to pass (a few seconds), then apply the schema:

```bash
alembic upgrade head
```

### Train the initial model

```bash
python scripts/train_v1.py
```

This trains on BAF month 0, registers the model in MLflow as `fraud-detector`, and promotes it to the `production` alias.

### Build the monitoring reference profile

```bash
python -m src.monitoring.reference
```

This scores all month-0 rows through the production model and writes quantile bucket boundaries and expected proportions to `data/reference_profile.pkl`. Re-run this whenever the production model is replaced.

---

## Running the system

Start each component in a separate terminal.

**Serving layer:**

```bash
uvicorn src.serving.app:app --reload --port 8000
```

**Monitoring job (continuous):**

```bash
python -m src.monitoring.run --loop
```

**Traffic simulator:**

```bash
# Clean traffic from month 2
python scripts/simulate_traffic.py --month 2 --speed 60

# With null injection on income at 50%
python scripts/simulate_traffic.py --month 3 --speed 60 --inject-nulls income 0.5

# With an unseen category on device_os
python scripts/simulate_traffic.py --month 3 --speed 60 --inject-unseen-category device_os WindowsPhone

# With seasonal income scaling
python scripts/simulate_traffic.py --month 4 --speed 60 --inject-seasonal 2.0
```

**Streamlit UI:**

```bash
streamlit run src/ui/app.py
```

Open the UI, navigate to the Incidents page, and use "Run agent investigation" once the monitoring job has opened an incident.

---

## Eval harness

Runs all scenarios in a directory end-to-end and writes a markdown report.

```bash
python -m src.eval.run --scenarios src/eval/scenarios/
```

Output goes to `eval/results/run_TIMESTAMP.md` by default. Each scenario resets Postgres state, replays traffic, runs one monitoring cycle, and invokes the agent. The run requires the serving layer to be up and `ANTHROPIC_API_KEY` to be set.

**Scenario file format:**

```yaml
name: pipeline_bug_income_50pct
description: 50% null injection on income should be diagnosed as a pipeline bug
setup:
  traffic:
    month: 0
    rows: 1000
    speed: 1000.0
  injection:
    type: nulls        # nulls | unseen_category | seasonal
    feature: income
    rate: 0.5
expected:
  diagnosis: pipeline_bug
  action: alert_human
```

`diagnosis` must be one of `genuine_drift`, `pipeline_bug`, `schema_change`, `seasonal_pattern`, `no_action_needed`, or `no_incident`. `no_incident` means the monitoring job should not open an incident at all.

---

## Manual operations

**Retrain on a month window:**

```bash
python -m src.training.retrain --months 0 1 2 3
```

Registers a new candidate version but does not promote it.

**Run the deployment gate against a candidate:**

```bash
python -m src.deployment.gate 3
```

Scores versions 3 and the current production version against a shared holdout and promotes if the PR-AUC delta meets the margin.

**Manual rollback:**

```bash
python -m src.deployment.rollback 1
```

Points the `production` alias to version 1 immediately, no evaluation.

**Run the agent against a specific incident:**

```bash
python -m src.agent.loop 7
```

Prints the structured JSON result to stdout.

**Single monitoring cycle (for testing):**

```bash
python -m src.monitoring.run --once
```

---

## Configuration

All thresholds and connection strings are read from environment variables via `src/common/config.py`. The full set with defaults:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/app` | SQLAlchemy connection string |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `MLFLOW_MODEL_NAME` | `fraud-detector` | Registry model name |
| `MLFLOW_PRODUCTION_ALIAS` | `production` | Alias that identifies the live version |
| `ANTHROPIC_API_KEY` | (required for agent) | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model used for agent calls |
| `PSI_ALARM_THRESHOLD` | `0.5` | Single-feature PSI alarm trigger |
| `PSI_MODERATE_THRESHOLD` | `0.2` | PSI threshold for moderate-cluster trigger |
| `PSI_MODERATE_COUNT` | `3` | Min features above moderate threshold to trigger |
| `NULL_RATE_THRESHOLD` | `0.2` | Per-feature null rate alarm trigger |
| `PREDICTION_DRIFT_THRESHOLD` | `0.25` | Prediction distribution PSI trigger |
| `MONITORING_WINDOW_ROWS` | `1000` | Predictions per monitoring cycle |
| `MONITORING_INTERVAL_SECONDS` | `30` | Sleep between monitoring cycles in loop mode |
| `PROMOTION_MARGIN` | `0.01` | Minimum PR-AUC improvement required for promotion |
| `REFERENCE_PROFILE_PATH` | `data/reference_profile.pkl` | Path to the pickled reference profile |
| `RAW_DATA_PATH` | `data/raw/Base.csv` | Path to the BAF dataset |

`.env` at the project root is loaded automatically.

---

## Tests

```bash
pytest
```

Tests cover: PSI and null-rate math (`test_metrics.py`), agent tool queries (`test_agent_tools.py`), the agent loop's JSON parsing and iteration logic (`test_agent_loop.py`), the deployment gate pass/fail logic (`test_gate.py`), and the eval scenario runner (`test_eval_runner.py`). Integration tests that require Postgres or the Anthropic API are marked `@pytest.mark.integration` and are skipped by default.

---

## Project layout

```
src/
  agent/
    loop.py         -- tool-use agent loop and AgentResult model
    tools.py        -- read-only Postgres and MLflow query functions
  common/
    config.py       -- Pydantic settings, reads from env / .env
    db.py           -- SQLAlchemy ORM models and session factory
    mlflow_helpers.py
  deployment/
    gate.py         -- candidate vs. production PR-AUC comparison
    rollback.py     -- alias flip without evaluation
  eval/
    run.py          -- CLI entry point
    runner.py       -- per-scenario state reset, traffic replay, agent invocation
    scenario.py     -- Pydantic schema for YAML scenario files
    report.py       -- markdown report writer
    scenarios/      -- 18 YAML scenario files
  monitoring/
    metrics.py      -- PSI, null rate, and unseen category math
    reference.py    -- build and load the training reference profile
    run.py          -- monitoring loop and trigger evaluation
  serving/
    app.py          -- FastAPI app
  training/
    retrain.py      -- LightGBM retraining and MLflow registration
  ui/
    app.py          -- Streamlit UI (Overview, Incidents, Deploys pages)
scripts/
  train_v1.py       -- train and promote the initial production model
  simulate_traffic.py -- replay BAF traffic with optional failure injection
tests/              -- pytest suite mirroring src/
data/
  raw/Base.csv      -- BAF dataset (not in repo, download separately)
  reference_profile.pkl -- generated by src/monitoring/reference.py
eval/results/       -- markdown reports from eval runs
alembic/            -- migration scripts
docker-compose.yml  -- Postgres and MLflow services
pyproject.toml      -- dependencies and build config
```
