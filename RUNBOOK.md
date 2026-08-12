# Runbook

## Purpose

This runbook describes how to configure, start, verify, test, inspect, troubleshoot, and reset the drift-triage platform.

It is intended for:

* engineers onboarding to the repository;
* contributors validating changes locally;
* reviewers reproducing the walking-skeleton workflow;
* future maintainers diagnosing local or CI failures.

The system currently consists of:

* Model Service
* Triage Agent
* Async Tool Worker
* Dashboard
* PostgreSQL
* Redis
* One-shot migration service

For design rationale and component ownership, see [`ARCH.md`](ARCH.md). For contract and design decisions, see [`DECISIONS.md`](DECISIONS.md). For intentional MVP exclusions, see [`MVP_SCOPE.md`](MVP_SCOPE.md).

---

## Prerequisites

### Required software

Install the following before running the project:

* Git
* Docker Engine or Docker Desktop
* Docker Compose v2
* Python 3.11 for local linting and unit tests
* Bash
* `curl`

Recommended local development tools:

* VS Code
* WSL 2 when developing on Windows
* PostgreSQL client tools, although the repository can use `psql` inside the Postgres container
* Redis CLI, although the repository can use `redis-cli` inside the Redis container

### Verify Docker

```bash
docker --version
docker compose version
```

The project expects the Compose v2 command:

```bash
docker compose
```

not the legacy standalone command:

```bash
docker-compose
```

### Windows and WSL

When developing on Windows, run Git, Python, Bash, and Docker Compose commands from the same environment whenever possible.

The recommended setup is:

* Docker Desktop with WSL integration enabled;
* repository cloned inside the WSL filesystem;
* commands executed from a WSL terminal.

Example repository location:

```text
~/projects/drift-triage-copilot
```

Avoid placing active repositories under `/mnt/c/...` when possible, because filesystem performance and permission behavior can differ from the native WSL filesystem.

### Verify required host ports

The default host mappings are:

| Service       | Host port | Container port |
| ------------- | --------: | -------------: |
| Model Service |    `8020` |         `8000` |
| Triage Agent  |    `8001` |         `8001` |
| Dashboard     |    `8520` |         `8501` |
| PostgreSQL    |    `5432` |         `5432` |
| Redis         |    `6379` |         `6379` |

Check whether any of these ports are already in use before starting the stack.

Linux or WSL:

```bash
ss -ltnp | grep -E ':8020|:8001|:8520|:5432|:6379'
```

Alternative:

```bash
sudo lsof -iTCP -sTCP:LISTEN -P | grep -E '8020|8001|8520|5432|6379'
```

---

## First-Time Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/drift-triage-copilot.git
cd drift-triage-copilot
```

### 2. Create the local environment file

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```env
DRIFT_WEBHOOK_SECRET=replace-with-a-long-random-development-secret
```

Generate a local development secret with Python:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Do not commit `.env`.

Verify that Git ignores it:

```bash
git check-ignore -v .env
```

### 3. Build and start the complete stack

```bash
docker compose up --build
```

Run in detached mode when logs do not need to remain attached to the terminal:

```bash
docker compose up --build -d
```

The expected startup sequence is:

```text
PostgreSQL becomes healthy
    ↓
Redis becomes healthy
    ↓
Migration service runs Alembic and exits successfully
    ↓
Agent validates configuration and initializes dependencies
    ↓
Agent becomes healthy
    ↓
Model Service starts
    ↓
Worker starts consuming the Redis Stream
    ↓
Dashboard starts
```

### 4. Confirm service state

```bash
docker compose ps -a
```

Expected state:

| Service         | Expected state       |
| --------------- | -------------------- |
| `postgres`      | Running and healthy  |
| `redis`         | Running and healthy  |
| `migrate`       | Exited with code `0` |
| `agent`         | Running and healthy  |
| `worker`        | Running              |
| `model_service` | Running              |
| `dashboard`     | Running              |

The migration container is intentionally expected to stop after completing successfully.

---

## Verifying the Stack Is Healthy

### Inspect Compose state

```bash
docker compose ps -a
```

A container being `running` does not always mean it is ready. For services that expose health checks, also verify the health endpoint.

### Agent health

```bash
curl --fail http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok"
}
```

The Agent is considered ready only after:

* Postgres connectivity succeeds;
* Redis responds to `PING`;
* LangGraph checkpoint storage is initialized;
* the investigation graph has been compiled.

### Model Service health

```bash
curl --fail http://localhost:8020/health
```

Expected shape:

```json
{
  "status": "ok",
  "timestamp": "..."
}
```

### Dashboard health

```bash
curl --fail http://localhost:8520/_stcore/health
```

Expected response:

```text
ok
```

Open the Dashboard in a browser:

```text
http://localhost:8520
```

### PostgreSQL health

```bash
docker compose exec -T postgres \
  pg_isready -U postgres -d drift_triage
```

Expected output:

```text
/var/run/postgresql:5432 - accepting connections
```

### Redis health

```bash
docker compose exec -T redis redis-cli ping
```

Expected output:

```text
PONG
```

### Migration status

```bash
docker compose ps -a migrate
```

The migration service must show exit code `0`.

Inspect its logs:

```bash
docker compose logs migrate
```

---

## Exercising the System

## Trigger the Walking-Skeleton Workflow

Send the deterministic drift event through the Model Service:

```bash
curl -i \
  -X POST \
  http://localhost:8020/debug/drift
```

For a new `report_id`, the Model Service should report successful delivery:

```json
{
  "status": "delivered",
  "report_id": "drift-report-customer-churn-model-v12-2026-07-22T12:00:00Z",
  "agent_status_code": 200
}
```

The end-to-end flow is:

```text
Model Service builds and signs a hardcoded drift event
    ↓
Agent verifies the HMAC over the raw request bytes
    ↓
Agent atomically records the webhook and investigation
    ↓
Agent persists a LangGraph checkpoint
    ↓
Agent dispatches a retrain job to Redis Streams
    ↓
Worker atomically claims the job in Postgres
    ↓
Worker performs stub execution
    ↓
Worker records completion
    ↓
Worker acknowledges the Redis message
    ↓
Dashboard reads the resulting state from Postgres
```

### Test duplicate delivery

Call the same endpoint again:

```bash
curl -i \
  -X POST \
  http://localhost:8020/debug/drift
```

The Agent should recognize the existing `report_id` as a duplicate.

The second call must not create:

* another webhook receipt;
* another investigation;
* another checkpoint for a new thread;
* another retrain job.

---

## Inspect the Dashboard

Open:

```text
http://localhost:8520
```

The Dashboard displays:

* investigation ID;
* model name and version;
* report ID;
* severity;
* investigation status;
* latest retrain job;
* latest job status;
* attempt count;
* resulting model version;
* updated timestamp in UTC.

Use the **Refresh** button after triggering a new event.

---

## Inspect PostgreSQL

Open an interactive PostgreSQL shell:

```bash
docker compose exec postgres \
  psql -U postgres -d drift_triage
```

Exit with:

```text
\q
```

### Investigations

```sql
SELECT
    investigation_id,
    thread_id,
    model_name,
    model_version,
    model_uri,
    triggering_report_id,
    current_report_id,
    status,
    current_severity,
    created_at,
    updated_at
FROM investigations
ORDER BY created_at DESC;
```

### Webhook receipts

```sql
SELECT
    report_id,
    processing_status,
    investigation_id,
    received_timestamp,
    processed_timestamp,
    failure_reason
FROM webhook_receipts
ORDER BY received_timestamp DESC;
```

### Retrain jobs

```sql
SELECT
    retrain_job_id,
    investigation_id,
    model_name,
    source_model_version,
    job_status,
    attempt_count,
    worker_claimed_at,
    started_at,
    completed_at,
    resulting_model_version,
    failure_details
FROM retrain_jobs
ORDER BY created_at DESC;
```

### Verify receipt-to-investigation linkage

```sql
SELECT
    wr.report_id,
    wr.processing_status,
    wr.investigation_id AS receipt_investigation_id,
    i.investigation_id,
    i.thread_id,
    i.status
FROM webhook_receipts AS wr
LEFT JOIN investigations AS i
    ON i.investigation_id = wr.investigation_id
ORDER BY wr.received_timestamp DESC;
```

### Inspect LangGraph checkpoints

```sql
SELECT
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id
FROM checkpoints
ORDER BY checkpoint_id DESC;
```

The checkpoint `thread_id` should match the `thread_id` stored in the corresponding investigation.

### Run a scalar query non-interactively

```bash
docker compose exec -T postgres \
  psql \
  -v ON_ERROR_STOP=1 \
  -U postgres \
  -d drift_triage \
  -tAc "SELECT count(*) FROM investigations;"
```

---

## Inspect Redis

### Inspect the retrain stream

```bash
docker compose exec -T redis \
  redis-cli XRANGE async-tools:retrain - +
```

### Inspect consumer groups

```bash
docker compose exec -T redis \
  redis-cli XINFO GROUPS async-tools:retrain
```

### Inspect pending messages

```bash
docker compose exec -T redis \
  redis-cli XPENDING async-tools:retrain retrain-workers
```

After a successful worker execution and acknowledgment, the pending count should be `0`.

### Inspect stream metadata

```bash
docker compose exec -T redis \
  redis-cli XINFO STREAM async-tools:retrain
```

---

## Running the Test Suite Locally

## Install Test Dependencies

Create or activate a Python 3.11 virtual environment, then install the compiled test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

Verify the tools:

```bash
ruff --version
pytest --version
```

## Run Ruff

Lint:

```bash
ruff check .
```

Verify formatting:

```bash
ruff format --check .
```

Apply safe automatic lint fixes:

```bash
ruff check . --fix
```

Format the codebase:

```bash
ruff format .
```

Run the checks again after applying changes:

```bash
ruff check .
ruff format --check .
```

## Run Unit Tests

```bash
pytest tests/unit -v
```

More diagnostic output:

```bash
pytest tests/unit \
  --verbose \
  --showlocals \
  --strict-markers
```

Run one module:

```bash
pytest tests/unit/test_agent.py -v
```

Run one test:

```bash
pytest \
  tests/unit/test_worker.py::test_parse_retrain_job_message_maps_all_fields \
  -v
```

## Validate the Smoke-Test Script Syntax

Before starting Docker:

```bash
bash -n tests/integration/smoke_test.sh
```

No output means the syntax is valid.

## Run the Integration Smoke Test

```bash
DRIFT_WEBHOOK_SECRET=local-smoke-test-secret \
  tests/integration/smoke_test.sh
```

The script:

* resets the Compose stack and volumes;
* starts Postgres and Redis;
* runs migrations;
* starts all application services;
* waits for readiness;
* triggers the drift workflow;
* verifies investigation creation;
* verifies receipt linkage;
* verifies checkpoint persistence;
* verifies worker completion;
* verifies Redis acknowledgment;
* verifies duplicate-delivery idempotency;
* verifies invalid HMAC rejection;
* verifies Dashboard availability;
* captures logs;
* tears down the stack and test volumes.

Artifacts are written to:

```text
test-artifacts/
```

---

## Common Operations

## View Logs

All services:

```bash
docker compose logs
```

Follow logs continuously:

```bash
docker compose logs -f
```

One service:

```bash
docker compose logs agent
docker compose logs worker
docker compose logs model_service
docker compose logs dashboard
docker compose logs migrate
docker compose logs postgres
docker compose logs redis
```

Follow one service:

```bash
docker compose logs -f worker
```

Show recent lines only:

```bash
docker compose logs --tail=100 agent
```

Include timestamps:

```bash
docker compose logs --timestamps worker
```

## Restart One Service

```bash
docker compose restart worker
```

Restart and follow logs:

```bash
docker compose restart agent
docker compose logs -f agent
```

## Rebuild One Service

```bash
docker compose up -d --build agent
```

Worker:

```bash
docker compose up -d --build worker
```

Dashboard:

```bash
docker compose up -d --build dashboard
```

## Stop the Stack Without Deleting Data

```bash
docker compose down
```

The named Postgres volume remains.

## Reset the Database and Redis State

This deletes all local Compose volumes:

```bash
docker compose down -v --remove-orphans
```

Restart from a clean database:

```bash
docker compose up --build
```

Use this when validating migrations from scratch or reproducing CI behavior.

## Re-run Migrations

Run the migration container:

```bash
docker compose run --rm migrate
```

Because the migration image has this command configured:

```text
alembic upgrade head
```

it applies all pending revisions.

Alternatively:

```bash
docker compose up --build --exit-code-from migrate migrate
```

Inspect the current Alembic revision:

```bash
docker compose run --rm migrate \
  alembic current
```

Inspect migration heads:

```bash
docker compose run --rm migrate \
  alembic heads
```

View migration history:

```bash
docker compose run --rm migrate \
  alembic history
```

## Rebuild All Images Without Cache

```bash
docker compose build --no-cache
```

Then start:

```bash
docker compose up
```

Use this when troubleshooting stale dependency layers or Docker cache issues.

## Remove Unused Docker Resources

Review first:

```bash
docker system df
```

Remove unused build cache:

```bash
docker builder prune
```

Remove unused images, containers, and networks:

```bash
docker system prune
```

Do not add `-a` or `--volumes` unless you understand the data-loss impact.

---

## Troubleshooting

## Port Is Already Allocated

### Symptoms

Docker reports an error similar to:

```text
Bind for 0.0.0.0:8020 failed: port is already allocated
```

or:

```text
address already in use
```

### Diagnosis

Check the port:

```bash
ss -ltnp | grep ':8020'
```

Check Docker containers:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

A different project may already be exposing the port. During development, unrelated projects such as a local backend or Streamlit application may occupy common ports like `8000`, `8001`, or `8501`.

### Resolution

Stop the conflicting process or container:

```bash
docker stop <container-name>
```

Or change only the host-side port mapping:

```yaml
ports:
  - "8030:8000"
```

The right side is the container port. The left side is the host port.

For example:

```yaml
- "8020:8000"
```

means:

```text
Host:      localhost:8020
Container: model_service:8000
```

Internal Compose communication should continue using container ports:

```text
http://agent:8001/webhooks/drift
```

not host-mapped ports.

---

## PostgreSQL or Redis Port Conflict

### Symptoms

Postgres or Redis fails to start because `5432` or `6379` is already allocated.

### Diagnosis

```bash
ss -ltnp | grep -E ':5432|:6379'
```

Check local services:

```bash
sudo service postgresql status
sudo service redis-server status
```

Check Docker:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

### Resolution

Either stop the local daemon:

```bash
sudo service postgresql stop
sudo service redis-server stop
```

or change the host mapping while preserving the internal Compose port.

Example:

```yaml
postgres:
  ports:
    - "55432:5432"
```

Other containers must still connect to:

```text
postgres:5432
```

not `localhost:55432`.

---

## `localhost` Does Not Work Between Containers

### Symptoms

A container reports:

```text
Connection refused
```

when attempting to connect to:

```text
localhost:5432
localhost:6379
localhost:8001
```

### Cause

Inside a container, `localhost` refers to that same container.

### Resolution

Use Compose service names:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/drift_triage
REDIS_URL=redis://redis:6379/0
AGENT_WEBHOOK_URL=http://agent:8001/webhooks/drift
```

Compose provides DNS resolution for service names automatically.

---

## Psycopg 2 vs. Psycopg 3 Connection Strings

### Symptoms

SQLAlchemy reports that it cannot import `psycopg2`, even though `psycopg` is installed.

Examples:

```text
ModuleNotFoundError: No module named 'psycopg2'
```

or driver-loading errors during engine creation.

### Cause

This project uses psycopg 3.

SQLAlchemy must be told explicitly to use the psycopg 3 dialect:

```text
postgresql+psycopg://
```

A plain SQLAlchemy URL:

```text
postgresql://
```

may default to the psycopg2 driver depending on configuration and installed packages.

### Correct URLs

SQLAlchemy engine:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/drift_triage
```

LangGraph `PostgresSaver.from_conn_string()` uses the native psycopg connection string:

```env
LANGGRAPH_DATABASE_URL=postgresql://postgres:postgres@postgres:5432/drift_triage
```

Do not blindly use the same URL format for both libraries. They consume different connection abstractions.

---

## Environment Variables Are Missing in a New Terminal

### Symptoms

A command works in one terminal but fails in another with:

```text
DRIFT_WEBHOOK_SECRET is not configured
```

or Compose reports:

```text
Please set DRIFT_WEBHOOK_SECRET in .env
```

### Cause

Shell exports exist only in the process and its children. They do not persist across unrelated terminal sessions.

For example:

```bash
export DRIFT_WEBHOOK_SECRET=some-value
```

does not make the variable available in a newly opened terminal.

### Resolution

Use the repository `.env` file for Compose:

```bash
cp .env.example .env
```

Then set:

```env
DRIFT_WEBHOOK_SECRET=your-local-secret
```

For a one-off command:

```bash
DRIFT_WEBHOOK_SECRET=local-test-secret \
  tests/integration/smoke_test.sh
```

Verify availability:

```bash
printenv DRIFT_WEBHOOK_SECRET
```

Do not commit `.env`.

---

## Docker Compose Secret Interpolation Fails

### Symptoms

```text
required variable DRIFT_WEBHOOK_SECRET is missing a value
```

### Cause

The Compose file intentionally uses mandatory interpolation:

```yaml
${DRIFT_WEBHOOK_SECRET:?Please set DRIFT_WEBHOOK_SECRET in .env}
```

### Resolution

Create `.env` or export the variable before invoking Compose:

```bash
export DRIFT_WEBHOOK_SECRET=local-development-secret
docker compose config --quiet
docker compose up
```

The mandatory check is intentional. The stack should not silently start with a known shared-secret default.

---

## `docker compose wait` Does Not Mean “Wait Until Healthy”

### Symptoms

A terminal appears to hang when running:

```bash
docker compose wait postgres redis
```

while both services remain healthy and running.

### Cause

`docker compose wait` waits for containers to stop and returns their exit codes. It is not a general readiness command for long-running services.

Long-running services such as Postgres, Redis, the Agent, and the Worker are expected to remain running, so waiting for them to exit is incorrect.

### Correct approaches

Use Compose health dependencies:

```yaml
depends_on:
  postgres:
    condition: service_healthy
```

Inspect health:

```bash
docker compose ps
```

Poll an HTTP endpoint:

```bash
until curl -sf http://localhost:8001/health > /dev/null; do
    sleep 2
done
```

Use `docker compose wait` only when waiting for a one-shot container such as migrations to terminate.

For migrations:

```bash
docker compose up --build --exit-code-from migrate migrate
```

---

## Agent Starts but Is Not Ready

### Symptoms

The Agent container is running, but requests fail immediately after startup.

### Cause

Process startup and application readiness are not the same.

The Agent lifespan must first:

* validate configuration;
* connect to Postgres;
* connect to Redis;
* initialize the LangGraph checkpointer;
* compile the graph.

### Resolution

Check health:

```bash
curl -i http://localhost:8001/health
```

Inspect logs:

```bash
docker compose logs agent
```

The Model Service is configured to wait for the Agent’s health check rather than only waiting for the container process to start.

---

## Redis Worker Logs a Timeout Every Five Seconds

### Symptoms

Worker logs contain repeated errors similar to:

```text
redis.exceptions.TimeoutError
```

at approximately the same interval as `REDIS_BLOCK_MS`.

### Cause

`XREADGROUP BLOCK 5000` asks Redis to hold the connection open for up to five seconds.

If the redis-py socket timeout is also five seconds or shorter, the client may time out before Redis returns normally from the blocking read.

### Resolution

The Redis client socket timeout must exceed the blocking window:

```python
socket_timeout = (REDIS_BLOCK_MS / 1000) + 5
```

The extra margin ensures the client outlives the server-side blocking period.

Do not “fix” this by removing the blocking read. Blocking consumption is intentional and avoids busy polling.

---

## Worker Message Is Pending but Job Is Completed

### Symptoms

Postgres shows:

```text
job_status = completed
```

but Redis still shows a pending stream message.

### Cause

The Worker persists completion before acknowledging Redis.

If the Worker stops or the `XACK` call fails after the database update, Redis retains the message in the pending list.

This ordering is intentional:

```text
Persist durable completion
    ↓
Acknowledge delivery
```

The reverse order could lose work if the Worker acknowledged before the database update completed.

### Impact

A future redelivery is safe because the Worker checks the `retrain_job_id` primary key and sees that the durable job already exists. It skips duplicate execution.

### Resolution

For the MVP, inspect:

```bash
docker compose exec -T redis \
  redis-cli XPENDING async-tools:retrain retrain-workers
```

The future retry/recovery pass will use `XAUTOCLAIM` to reclaim abandoned messages.

Do not delete pending messages manually unless intentionally resetting local state.

---

## Worker Leaves a Failed Job Pending

### Symptoms

Postgres shows:

```text
job_status = failed
```

and the Redis message remains pending.

### Cause

This is deliberate.

The MVP does not yet implement:

* retry ownership;
* exponential backoff;
* maximum attempts;
* dead-letter movement;
* abandoned-message reclamation.

Automatically acknowledging a failed job would destroy the delivery record before recovery behavior exists.

### Resolution

Treat the message as retained for a future recovery mechanism. For a local reset:

```bash
docker compose down -v
docker compose up --build
```

---

## Model Service Returns `502 dispatch_failed`

### Symptoms

Calling:

```bash
curl -X POST http://localhost:8020/debug/drift
```

returns a structured `502`.

### Causes

* Agent is not running.
* Agent is not healthy.
* `AGENT_WEBHOOK_URL` is incorrect.
* Docker service DNS is incorrect.
* Agent rejected the payload.
* Agent returned an internal error.

### Diagnosis

```bash
docker compose ps
docker compose logs model_service
docker compose logs agent
curl -i http://localhost:8001/health
```

Verify:

```env
AGENT_WEBHOOK_URL=http://agent:8001/webhooks/drift
```

From the host, the Agent is available at `localhost:8001`. From the Model Service container, it is available at `agent:8001`.

---

## Agent Rejects the Webhook With `401`

### Symptoms

Agent response:

```json
{
  "status": "unauthorized",
  "error": "Invalid webhook signature"
}
```

### Causes

* Model Service and Agent use different secrets.
* The request body was modified after signing.
* The sender signed one JSON serialization and sent another.
* The signature header is missing or malformed.

### Resolution

Both services must use exactly the same secret:

```env
DRIFT_WEBHOOK_SECRET=...
```

The Model Service signs the exact bytes it sends:

```text
serialize payload
    ↓
calculate HMAC
    ↓
send those same bytes
```

The Agent verifies the HMAC against the raw request body before parsing JSON.

Inspect both containers:

```bash
docker compose exec model_service \
  printenv DRIFT_WEBHOOK_SECRET

docker compose exec agent \
  printenv DRIFT_WEBHOOK_SECRET
```

Do not print real production secrets in shared logs. This check is only appropriate for local development.

---

## Agent Returns `duplicate` on the First Test Call

### Symptoms

The first call during a new coding pass is treated as a duplicate.

### Cause

The debug payload intentionally uses a deterministic `report_id`, and the Postgres named volume preserves prior data across container restarts.

Restarting containers does not delete the volume.

### Resolution

Inspect existing state:

```bash
docker compose exec -T postgres \
  psql -U postgres -d drift_triage \
  -c "
  SELECT report_id, investigation_id, processing_status
  FROM webhook_receipts;
  "
```

For a clean test:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

---

## Deleting Only the Receipt Causes Investigation Insert Failure

### Symptoms

After deleting a webhook receipt manually, the next request fails with a unique-constraint violation on `investigations.triggering_report_id`.

### Cause

The investigation row still exists, and `triggering_report_id` is unique.

Deleting only the receipt creates an inconsistent local test state.

### Resolution

Delete dependent test data carefully, or reset the entire local database volume.

Preferred:

```bash
docker compose down -v
docker compose up --build
```

For manual deletion, account for foreign keys and delete related rows in the correct order.

---

## Checkpoint Persistence Fails

### Symptoms

Investigation status becomes:

```text
checkpoint_failed
```

### Cause

The investigation transaction succeeded, but LangGraph checkpoint persistence failed afterward. These operations use separate transaction systems and cannot be committed atomically together.

### Diagnosis

```bash
docker compose logs agent
```

Inspect the investigation:

```bash
docker compose exec -T postgres \
  psql -U postgres -d drift_triage \
  -c "
  SELECT investigation_id, thread_id, status
  FROM investigations
  WHERE status = 'checkpoint_failed';
  "
```

Verify checkpoint tables exist:

```bash
docker compose exec -T postgres \
  psql -U postgres -d drift_triage \
  -c "\dt checkpoint*"
```

### Resolution

Check:

* `LANGGRAPH_DATABASE_URL`;
* Postgres connectivity;
* psycopg dependency installation;
* LangGraph checkpoint table setup;
* Agent startup logs.

The investigation remains durable and recoverable. It is not silently deleted.

---

## Redis Dispatch Fails

### Symptoms

Investigation status becomes:

```text
dispatch_failed
```

### Cause

The investigation and checkpoint were created, but Redis `XADD` failed.

### Diagnosis

```bash
docker compose logs agent
docker compose logs redis
docker compose exec -T redis redis-cli ping
```

Verify:

```env
REDIS_URL=redis://redis:6379/0
RETRAIN_JOB_STREAM=async-tools:retrain
```

### Resolution

Restore Redis connectivity, then use a future recovery mechanism to re-dispatch the investigation. The MVP intentionally preserves the failed investigation instead of rolling back already committed state.

---

## Migration Container Fails

### Symptoms

* `migrate` exits nonzero;
* Agent and Worker do not start;
* tables do not exist.

### Diagnosis

```bash
docker compose logs migrate
docker compose ps -a migrate
```

Validate Compose interpolation:

```bash
docker compose config --quiet
```

Run migrations directly:

```bash
docker compose run --rm migrate
```

### Common causes

* incorrect `DATABASE_URL`;
* Postgres not healthy;
* missing Alembic migration files in the image;
* invalid migration syntax;
* migration order or foreign-key dependency error;
* driver mismatch.

### Resolution

Fix the migration failure before starting application services. Do not bypass the migration dependency by manually starting the Agent.

---

## Dockerfile Reports `unknown instruction`

### Symptoms

Docker build fails with an error such as:

```text
unknown instruction: "uvicorn",
```

or:

```text
unknown instruction: "streamlit",
```

### Cause

A JSON-array `CMD` was split across multiple Dockerfile lines without valid Docker continuation syntax.

### Correct form

Keep the command on one line:

```dockerfile
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Dashboard:

```dockerfile
CMD ["streamlit", "run", "dashboard/main.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
```

---

## Docker Build Cannot Find Service Files

### Symptoms

```text
COPY failed: file not found
```

### Cause

The Dockerfile expects the repository root as its build context.

For example:

```dockerfile
COPY agent ./agent
```

cannot work if the build context is `agent/`.

### Correct Compose configuration

```yaml
build:
  context: .
  dockerfile: agent/Dockerfile
```

Build manually from the repository root:

```bash
docker build \
  -f agent/Dockerfile \
  -t drift-agent:dev \
  .
```

---

## Cross-Service Dependency Conflicts in a Shared Environment

### Symptoms

Installing all service lock files into one virtual environment causes resolver conflicts or package downgrades.

### Cause

Each service has independently pinned runtime dependencies. Combining multiple fully pinned service environments into one shared environment can create incompatible transitive requirements.

This does not necessarily mean the containers are broken. Each container installs only its own dependency set.

### Resolution

For runtime:

* keep separate requirements files per service;
* build separate images;
* do not merge runtime dependency environments.

For unit tests:

* maintain a dedicated `requirements-test.txt`;
* compile it as a deliberate combined test environment;
* include test tooling such as `pytest` and `ruff` explicitly.

Do not install four independent pinned runtime files into one shared CI environment unless they have been intentionally resolved together.

---

## Ruff or Pytest Cannot Import Project Modules

### Symptoms

```text
ModuleNotFoundError: No module named 'agent'
```

### Causes

* tests are being run outside the repository root;
* package directories are missing `__init__.py`;
* the current working directory is incorrect;
* the virtual environment lacks service dependencies.

### Resolution

Run from the repository root:

```bash
pwd
pytest tests/unit -v
```

Verify package markers exist:

```text
agent/__init__.py
model_service/__init__.py
worker/__init__.py
dashboard/__init__.py
```

Install the test environment:

```bash
python -m pip install -r requirements-test.txt
```

---

## Streamlit Dashboard Does Not Show New Data

### Symptoms

The workflow completes, but the Dashboard still shows old state.

### Cause

The Dashboard intentionally uses manual refresh rather than automatic polling.

### Resolution

Click **Refresh** or reload the page.

Verify the underlying database first:

```bash
docker compose exec -T postgres \
  psql -U postgres -d drift_triage \
  -c "SELECT * FROM investigations ORDER BY created_at DESC;"
```

If Postgres has current data but Streamlit does not, inspect:

```bash
docker compose logs dashboard
```

---

## GitHub Actions Passes Locally but Fails in CI

### Common differences

* fresh runner with no existing Docker cache;
* different Docker Compose version;
* lower available CPU or memory;
* slower service startup;
* Linux filesystem behavior;
* no local `.env`;
* stricter clean-checkout behavior;
* missing executable bit on shell scripts.

### Diagnosis

Download the `smoke-test-artifacts` artifact from the failed workflow run.

It contains:

```text
test-artifacts/compose-logs.txt
```

Also inspect the failing workflow step.

### Checks

Ensure the script is executable in Git:

```bash
git ls-files --stage tests/integration/smoke_test.sh
```

Expected mode begins with:

```text
100755
```

Set it if needed:

```bash
chmod +x tests/integration/smoke_test.sh
git add tests/integration/smoke_test.sh
git commit -m "Make smoke test executable"
```

Validate workflow-required configuration:

```bash
DRIFT_WEBHOOK_SECRET=ci-only-shared-secret \
  docker compose config --quiet
```

---

## Known Limitations

The walking skeleton intentionally excludes several production capabilities, including real drift computation, real model training, complete supervisor orchestration, retries, dead-letter processing, promotion workflows, advanced authentication, and production observability.

The authoritative list is maintained in:

```text
MVP_SCOPE.md
```

See the **Explicitly Deferred** section there rather than duplicating the list in this runbook.

Operationally significant limitations include:

* the drift payload is deterministic and hardcoded;
* the Worker performs stub execution rather than model training;
* failed or abandoned Redis messages are not reclaimed automatically;
* retry backoff and DLQ movement are not implemented;
* the Dashboard is read-only and manually refreshed;
* HMAC uses one shared secret with no key rotation;
* checkpoint execution is suitable for MVP traffic but not yet load-tested;
* service containers still run as root;
* production-grade metrics, tracing, and alerting are deferred.

---

## Quick Reference

Start the stack:

```bash
docker compose up --build
```

Start in the background:

```bash
docker compose up --build -d
```

Check state:

```bash
docker compose ps -a
```

Trigger drift:

```bash
curl -X POST http://localhost:8020/debug/drift
```

Open Dashboard:

```text
http://localhost:8520
```

Follow logs:

```bash
docker compose logs -f
```

Run unit tests:

```bash
pytest tests/unit -v
```

Run lint:

```bash
ruff check .
ruff format --check .
```

Run smoke test:

```bash
DRIFT_WEBHOOK_SECRET=local-smoke-test-secret \
  tests/integration/smoke_test.sh
```

Reset everything:

```bash
docker compose down -v --remove-orphans
```
