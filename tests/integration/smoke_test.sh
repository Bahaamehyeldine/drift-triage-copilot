#!/usr/bin/env bash

set -euo pipefail


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

export DRIFT_WEBHOOK_SECRET="${DRIFT_WEBHOOK_SECRET:-ci-only-shared-secret}"

POSTGRES_SERVICE="postgres"
POSTGRES_USER="postgres"
POSTGRES_DATABASE="drift_triage"

AGENT_HEALTH_URL="http://localhost:8001/health"
AGENT_WEBHOOK_URL="http://localhost:8001/webhooks/drift"
MODEL_SERVICE_HEALTH_URL="http://localhost:8020/health"
MODEL_SERVICE_DRIFT_URL="http://localhost:8020/debug/drift"
DASHBOARD_HEALTH_URL="http://localhost:8520/_stcore/health"

REPORT_ID="drift-report-customer-churn-model-v12-2026-07-22T12:00:00Z"

READINESS_TIMEOUT_SECONDS=120
JOB_COMPLETION_TIMEOUT_SECONDS=60
POLL_INTERVAL_SECONDS=2

ARTIFACT_DIR="${ARTIFACT_DIR:-test-artifacts}"
COMPOSE_LOG_FILE="${ARTIFACT_DIR}/compose-logs.txt"


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

log() {
    printf '[smoke-test] %s\n' "$*"
}


fail() {
    printf '[smoke-test] ERROR: %s\n' "$*" >&2
    exit 1
}


# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

cleanup() {
    local exit_code=$?

    # Prevent recursive trap execution when this function exits.
    trap - EXIT

    mkdir -p "${ARTIFACT_DIR}"

    log "Capturing Docker Compose logs to ${COMPOSE_LOG_FILE}"

    docker compose logs \
        --no-color \
        --timestamps \
        > "${COMPOSE_LOG_FILE}" 2>&1 || true

    log "Stopping containers and removing test volumes"

    docker compose down \
        -v \
        --remove-orphans || true

    if [[ ${exit_code} -eq 0 ]]; then
        log "Smoke test completed successfully"
    else
        log "Smoke test failed with exit code ${exit_code}"
    fi

    exit "${exit_code}"
}


trap cleanup EXIT


# -----------------------------------------------------------------------------
# Readiness and assertion helpers
# -----------------------------------------------------------------------------

wait_for_http() {
    local name=$1
    local url=$2
    local timeout_seconds=${3:-${READINESS_TIMEOUT_SECONDS}}
    local elapsed=0

    log "Waiting for ${name}: ${url}"

    until curl \
        --silent \
        --show-error \
        --fail \
        --max-time 3 \
        "${url}" \
        > /dev/null 2>&1
    do
        sleep "${POLL_INTERVAL_SECONDS}"
        elapsed=$((elapsed + POLL_INTERVAL_SECONDS))

        if [[ ${elapsed} -ge ${timeout_seconds} ]]; then
            fail "Timed out after ${timeout_seconds}s waiting for ${name}"
        fi
    done

    log "${name} is ready"
}

wait_for_service_healthy() {
    local service=$1
    local timeout_seconds=${2:-${READINESS_TIMEOUT_SECONDS}}
    local elapsed=0
    local health=""

    log "Waiting for ${service} to become healthy"

    while true; do
        health="$(docker compose ps --format json "${service}" | python -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("missing")
else:
    print(json.loads(raw.splitlines()[0]).get("Health", "unknown"))
')"

        if [[ "${health}" == "healthy" ]]; then
            log "${service} is healthy"
            return 0
        fi

        sleep "${POLL_INTERVAL_SECONDS}"
        elapsed=$((elapsed + POLL_INTERVAL_SECONDS))

        if [[ ${elapsed} -ge ${timeout_seconds} ]]; then
            fail "Timed out after ${timeout_seconds}s waiting for ${service} to become healthy (last status: ${health})"
        fi
    done
}

postgres_scalar() {
    local query=$1

    docker compose exec -T "${POSTGRES_SERVICE}" \
        psql \
        -v ON_ERROR_STOP=1 \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DATABASE}" \
        -tAc "${query}" \
        | tr -d '[:space:]'
}


assert_equals() {
    local expected=$1
    local actual=$2
    local description=$3

    if [[ "${actual}" != "${expected}" ]]; then
        fail "${description}: expected '${expected}', got '${actual}'"
    fi

    log "Assertion passed: ${description} = ${expected}"
}


wait_for_postgres_value() {
    local description=$1
    local query=$2
    local expected=$3
    local timeout_seconds=${4:-${JOB_COMPLETION_TIMEOUT_SECONDS}}
    local elapsed=0
    local actual=""

    log "Waiting for ${description}"

    while true; do
        actual="$(postgres_scalar "${query}")"

        if [[ "${actual}" == "${expected}" ]]; then
            log "Observed ${description}: ${actual}"
            return 0
        fi

        sleep "${POLL_INTERVAL_SECONDS}"
        elapsed=$((elapsed + POLL_INTERVAL_SECONDS))

        if [[ ${elapsed} -ge ${timeout_seconds} ]]; then
            fail "Timed out after ${timeout_seconds}s waiting for ${description}; expected '${expected}', last value '${actual}'"
        fi
    done
}



# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

trigger_drift_event() {
    local response_file=$1
    local status_code

    status_code="$(
        curl \
            --silent \
            --show-error \
            --output "${response_file}" \
            --write-out '%{http_code}' \
            --request POST \
            "${MODEL_SERVICE_DRIFT_URL}"
    )"

    if [[ "${status_code}" != "200" ]]; then
        cat "${response_file}" >&2 || true
        fail "Drift trigger returned HTTP ${status_code}; expected 200"
    fi

    log "Drift trigger returned HTTP 200"
}


assert_invalid_signature_rejected() {
    local response_file="${ARTIFACT_DIR}/invalid-signature-response.json"
    local invalid_payload
    local status_code

    invalid_payload="$(
        cat <<'JSON'
{"schema_version":"1.0","event_type":"drift.severity.increased","report_id":"invalid-signature-test","timestamp":"2026-08-05T09:00:00Z","model":{"name":"customer-churn-model","version":"12"},"overall_severity":{"previous":"low","current":"high"},"signals":[],"summary":"Invalid signature test payload."}
JSON
    )"

    status_code="$(
        curl \
            --silent \
            --show-error \
            --output "${response_file}" \
            --write-out '%{http_code}' \
            --request POST \
            --header "Content-Type: application/json" \
            --header "X-Webhook-Signature: sha256=invalid-signature" \
            --data "${invalid_payload}" \
            "${AGENT_WEBHOOK_URL}"
    )"

    if [[ "${status_code}" != "401" ]]; then
        cat "${response_file}" >&2 || true
        fail "Invalid-signature webhook returned HTTP ${status_code}; expected 401"
    fi

    log "Invalid HMAC signature correctly rejected with HTTP 401"
}


# -----------------------------------------------------------------------------
# Test setup
# -----------------------------------------------------------------------------

mkdir -p "${ARTIFACT_DIR}"

log "Resetting any previous Docker Compose state"

docker compose down \
    -v \
    --remove-orphans || true


# -----------------------------------------------------------------------------
# Stage 1: Infrastructure
# -----------------------------------------------------------------------------

log "Starting Postgres and Redis"

docker compose up \
    --detach \
    --build \
    postgres \
    redis

wait_for_service_healthy postgres
wait_for_service_healthy redis


# -----------------------------------------------------------------------------
# Stage 2: Database migrations
# -----------------------------------------------------------------------------

log "Building and running database migrations"

docker compose up \
    --build \
    --exit-code-from migrate \
    migrate

migration_status="$(
    docker compose ps \
        --all \
        --format json \
        migrate \
        | python -c '
import json
import sys

raw = sys.stdin.read().strip()

if not raw:
    raise SystemExit("migrate service was not found")

documents = [
    json.loads(line)
    for line in raw.splitlines()
    if line.strip()
]

service = documents[0]
exit_code = service.get("ExitCode")

if exit_code is None:
    state = service.get("State", "")
    raise SystemExit(
        f"migrate service has no exit code; state={state}"
    )

print(exit_code)
'
)"

assert_equals \
    "0" \
    "${migration_status}" \
    "migration service exit code"


# -----------------------------------------------------------------------------
# Stage 3: Application services
# -----------------------------------------------------------------------------

log "Starting Agent, Worker, Model Service, and Dashboard"

docker compose up \
    --detach \
    --build \
    agent \
    worker \
    model_service \
    dashboard

wait_for_http \
    "Agent" \
    "${AGENT_HEALTH_URL}"

wait_for_http \
    "Model Service" \
    "${MODEL_SERVICE_HEALTH_URL}"

wait_for_http \
    "Dashboard" \
    "${DASHBOARD_HEALTH_URL}"


# -----------------------------------------------------------------------------
# Stage 4: First webhook delivery
# -----------------------------------------------------------------------------

log "Triggering deterministic drift event"

trigger_drift_event \
    "${ARTIFACT_DIR}/first-drift-response.json"


# -----------------------------------------------------------------------------
# Verify investigation creation
# -----------------------------------------------------------------------------

investigation_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM investigations
        WHERE triggering_report_id = '${REPORT_ID}';
    "
)"

assert_equals \
    "1" \
    "${investigation_count}" \
    "investigation count for deterministic report"


investigation_id="$(
    postgres_scalar "
        SELECT investigation_id
        FROM investigations
        WHERE triggering_report_id = '${REPORT_ID}';
    "
)"

if [[ -z "${investigation_id}" ]]; then
    fail "Investigation ID was not persisted"
fi

log "Persisted investigation_id=${investigation_id}"


thread_id="$(
    postgres_scalar "
        SELECT thread_id
        FROM investigations
        WHERE investigation_id = '${investigation_id}';
    "
)"

if [[ -z "${thread_id}" ]]; then
    fail "LangGraph thread ID was not persisted on the investigation"
fi

log "Persisted thread_id=${thread_id}"


# -----------------------------------------------------------------------------
# Verify webhook receipt linkage
# -----------------------------------------------------------------------------

receipt_investigation_id="$(
    postgres_scalar "
        SELECT investigation_id
        FROM webhook_receipts
        WHERE report_id = '${REPORT_ID}';
    "
)"

assert_equals \
    "${investigation_id}" \
    "${receipt_investigation_id}" \
    "webhook receipt investigation link"


receipt_status="$(
    postgres_scalar "
        SELECT processing_status
        FROM webhook_receipts
        WHERE report_id = '${REPORT_ID}';
    "
)"

assert_equals \
    "processed" \
    "${receipt_status}" \
    "webhook receipt processing status"


# -----------------------------------------------------------------------------
# Verify LangGraph checkpoint persistence
# -----------------------------------------------------------------------------

checkpoint_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM checkpoints
        WHERE thread_id = '${thread_id}';
    "
)"

if [[ "${checkpoint_count}" -lt 1 ]]; then
    fail "Expected at least one LangGraph checkpoint for thread_id=${thread_id}, got ${checkpoint_count}"
fi

log "Checkpoint persistence verified: ${checkpoint_count} checkpoint row(s)"


# -----------------------------------------------------------------------------
# Verify Worker execution
# -----------------------------------------------------------------------------

wait_for_postgres_value \
    "retrain job status" \
    "
        SELECT COALESCE(
            (
                SELECT job_status
                FROM retrain_jobs
                WHERE investigation_id = '${investigation_id}'
                ORDER BY created_at DESC, retrain_job_id DESC
                LIMIT 1
            ),
            'missing'
        );
    " \
    "completed"


retrain_job_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM retrain_jobs
        WHERE investigation_id = '${investigation_id}';
    "
)"

assert_equals \
    "1" \
    "${retrain_job_count}" \
    "retrain job count after first delivery"


attempt_count="$(
    postgres_scalar "
        SELECT attempt_count
        FROM retrain_jobs
        WHERE investigation_id = '${investigation_id}'
        ORDER BY created_at DESC, retrain_job_id DESC
        LIMIT 1;
    "
)"

assert_equals \
    "1" \
    "${attempt_count}" \
    "retrain job attempt count"


resulting_model_version="$(
    postgres_scalar "
        SELECT resulting_model_version
        FROM retrain_jobs
        WHERE investigation_id = '${investigation_id}'
        ORDER BY created_at DESC, retrain_job_id DESC
        LIMIT 1;
    "
)"

if [[ -z "${resulting_model_version}" ]]; then
    fail "Worker did not persist a resulting model version"
fi

log "Worker result verified: resulting_model_version=${resulting_model_version}"


# -----------------------------------------------------------------------------
# Verify Redis acknowledgment
# -----------------------------------------------------------------------------

pending_count="$(
    docker compose exec -T redis \
        redis-cli \
        XPENDING \
        async-tools:retrain \
        retrain-workers \
        | awk 'NR == 1 { print $1 }'
)"

assert_equals \
    "0" \
    "${pending_count}" \
    "Redis pending message count"


# -----------------------------------------------------------------------------
# Stage 5: Duplicate delivery and idempotency
# -----------------------------------------------------------------------------

log "Triggering the same deterministic drift event again"

trigger_drift_event \
    "${ARTIFACT_DIR}/duplicate-drift-response.json"


duplicate_investigation_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM investigations
        WHERE triggering_report_id = '${REPORT_ID}';
    "
)"

assert_equals \
    "1" \
    "${duplicate_investigation_count}" \
    "investigation count after duplicate delivery"


duplicate_receipt_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM webhook_receipts
        WHERE report_id = '${REPORT_ID}';
    "
)"

assert_equals \
    "1" \
    "${duplicate_receipt_count}" \
    "webhook receipt count after duplicate delivery"


duplicate_job_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM retrain_jobs
        WHERE investigation_id = '${investigation_id}';
    "
)"

assert_equals \
    "1" \
    "${duplicate_job_count}" \
    "retrain job count after duplicate delivery"


duplicate_checkpoint_count="$(
    postgres_scalar "
        SELECT count(*)
        FROM checkpoints
        WHERE thread_id = '${thread_id}';
    "
)"

assert_equals \
    "${checkpoint_count}" \
    "${duplicate_checkpoint_count}" \
    "checkpoint count after duplicate delivery"


# -----------------------------------------------------------------------------
# Stage 6: Invalid HMAC rejection
# -----------------------------------------------------------------------------

log "Verifying invalid HMAC rejection"

assert_invalid_signature_rejected


# -----------------------------------------------------------------------------
# Stage 7: Dashboard availability
# -----------------------------------------------------------------------------

log "Verifying Dashboard remains available after workflow completion"

wait_for_http \
    "Dashboard after workflow completion" \
    "${DASHBOARD_HEALTH_URL}" \
    30


# -----------------------------------------------------------------------------
# Final diagnostics
# -----------------------------------------------------------------------------

log "Final investigation state"

docker compose exec -T postgres \
    psql \
    -v ON_ERROR_STOP=1 \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DATABASE}" \
    -c "
        SELECT
            i.investigation_id,
            i.thread_id,
            i.model_name,
            i.model_version,
            i.status AS investigation_status,
            i.current_severity,
            rj.retrain_job_id,
            rj.job_status,
            rj.attempt_count,
            rj.resulting_model_version
        FROM investigations AS i
        LEFT JOIN retrain_jobs AS rj
            ON rj.investigation_id = i.investigation_id
        WHERE i.triggering_report_id = '${REPORT_ID}';
    "

log "All smoke-test invariants passed"

