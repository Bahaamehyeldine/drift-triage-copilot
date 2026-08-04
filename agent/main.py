# agent/main.py

import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import redis
import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from redis import Redis
from sqlalchemy.dialects.postgresql import insert as pg_insert


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

LANGGRAPH_DATABASE_URL = os.getenv(
    "LANGGRAPH_DATABASE_URL",
    DATABASE_URL,
)

RETRAIN_JOB_STREAM = os.getenv(
    "RETRAIN_JOB_STREAM",
    "async-tools:retrain",
)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class InvestigationGraphState(TypedDict):
    """
    Minimal workflow state persisted for one investigation.

    Postgres investigation records remain authoritative for business-facing
    lifecycle state. The checkpoint stores only workflow context required
    to resume LangGraph execution.
    """

    investigation_id: str
    report_id: str
    model_name: str
    model_version: str
    workflow_status: str


def initialize_investigation(
    state: InvestigationGraphState,
) -> dict[str, str]:
    """
    Minimal deterministic graph node for the walking skeleton.

    This proves that an investigation can enter a persisted LangGraph
    workflow. Real triage, action, and communication logic is deferred.
    """
    logger.info(
        "Initializing investigation workflow: "
        "investigation_id=%s report_id=%s",
        state["investigation_id"],
        state["report_id"],
    )

    return {
        "workflow_status": "initialized",
    }


def build_investigation_graph(
    checkpointer: PostgresSaver,
) -> Any:
    """
    Build the minimal persisted investigation graph.
    """
    builder = StateGraph(InvestigationGraphState)

    builder.add_node(
        "initialize_investigation",
        initialize_investigation,
    )

    builder.add_edge(
        START,
        "initialize_investigation",
    )

    builder.add_edge(
        "initialize_investigation",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )


# ---------------------------------------------------------------------------
# Database metadata
# ---------------------------------------------------------------------------

metadata = sa.MetaData()


investigations = sa.Table(
    "investigations",
    metadata,
    sa.Column(
        "investigation_id",
        sa.String(length=64),
        primary_key=True,
    ),
    sa.Column(
        "thread_id",
        sa.String(length=64),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "model_name",
        sa.String(length=255),
        nullable=False,
    ),
    sa.Column(
        "model_version",
        sa.String(length=128),
        nullable=False,
    ),
    sa.Column(
        "model_uri",
        sa.String(length=1024),
        nullable=False,
    ),
    sa.Column(
        "triggering_report_id",
        sa.String(length=128),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "current_report_id",
        sa.String(length=128),
        nullable=False,
    ),
    sa.Column(
        "status",
        sa.String(length=32),
        nullable=False,
        server_default="open",
    ),
    sa.Column(
        "current_severity",
        sa.String(length=16),
        nullable=False,
    ),
    sa.Column(
        "recommended_action",
        sa.String(length=64),
        nullable=True,
    ),
    sa.Column(
        "resolution",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "stale_reason",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "invalidation_reason",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "resolved_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
)


webhook_receipts = sa.Table(
    "webhook_receipts",
    metadata,
    sa.Column(
        "report_id",
        sa.String(length=128),
        primary_key=True,
    ),
    sa.Column(
        "received_timestamp",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "processing_status",
        sa.String(length=32),
        nullable=False,
        server_default="received",
    ),
    sa.Column(
        "investigation_id",
        sa.String(length=64),
        sa.ForeignKey(
            "investigations.investigation_id",
            ondelete="SET NULL",
        ),
        nullable=True,
    ),
    sa.Column(
        "failure_reason",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "processed_timestamp",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
)


# The configuration is validated during application startup. Keeping engine
# construction here allows the persistence functions to remain small and
# independent from FastAPI's Request object.
engine = (
    sa.create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )
    if DATABASE_URL
    else None
)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fail fast when required configuration is missing, then initialize
    long-lived infrastructure clients once per application process.
    """
    app.state.investigation_graph = None
    app.state.redis_client = None

    required_config = {
        "DRIFT_WEBHOOK_SECRET": DRIFT_WEBHOOK_SECRET,
        "DATABASE_URL": DATABASE_URL,
        "LANGGRAPH_DATABASE_URL": LANGGRAPH_DATABASE_URL,
        "REDIS_URL": REDIS_URL,
    }

    missing_config = [
        name
        for name, value in required_config.items()
        if not value
    ]

    if missing_config:
        raise RuntimeError(
            "Missing required Agent configuration: "
            + ", ".join(missing_config)
        )

    if engine is None:
        raise RuntimeError(
            "SQLAlchemy engine was not initialized"
        )

    redis_client: Redis | None = None

    try:
        # Confirm the application database is reachable before accepting
        # webhook traffic.
        def check_database_connection() -> None:
            with engine.connect() as connection:
                connection.execute(sa.text("SELECT 1"))

        await run_in_threadpool(
            check_database_connection,
        )

        logger.info(
            "Application database connection initialized"
        )

        # Create one Redis client for the application process.
        redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
        )

        await run_in_threadpool(
            redis_client.ping,
        )

        app.state.redis_client = redis_client

        logger.info(
            "Redis connection initialized: stream=%s",
            RETRAIN_JOB_STREAM,
        )

        # PostgresSaver manages the checkpoint connection lifecycle inside
        # this context manager.
        with PostgresSaver.from_conn_string(
            LANGGRAPH_DATABASE_URL
        ) as checkpointer:
            checkpointer.setup()

            app.state.investigation_graph = (
                build_investigation_graph(checkpointer)
            )

            logger.info(
                "LangGraph Postgres checkpointer initialized"
            )

            yield

    except Exception:
        logger.exception(
            "Failed to initialize Agent infrastructure"
        )
        raise

    finally:
        app.state.investigation_graph = None
        app.state.redis_client = None

        if redis_client is not None:
            try:
                await run_in_threadpool(
                    redis_client.close,
                )

                logger.info(
                    "Redis connection closed"
                )

            except Exception:
                logger.exception(
                    "Failed to close Redis client cleanly"
                )

        if engine is not None:
            try:
                await run_in_threadpool(
                    engine.dispose,
                )

                logger.info(
                    "Application database engine disposed"
                )

            except Exception:
                logger.exception(
                    "Failed to dispose database engine cleanly"
                )


app = FastAPI(
    title="Triage Agent",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class WebhookResponse(BaseModel):
    status: str
    report_id: str
    investigation_id: str | None = None
    retrain_job_id: str | None = None


class ErrorResponse(BaseModel):
    status: str
    error: str


class InvestigationInput(BaseModel):
    report_id: str
    model_name: str
    model_version: str
    current_severity: str


class InvestigationCreationResult(BaseModel):
    created: bool
    report_id: str
    investigation_id: str | None = None
    thread_id: str | None = None


# ---------------------------------------------------------------------------
# HMAC
# ---------------------------------------------------------------------------

def create_signature(
    payload_bytes: bytes,
    secret: str,
) -> str:
    """
    Recompute the HMAC-SHA256 signature over the exact request body.
    """
    digest = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def error_response(
    *,
    status_code: int,
    status: str,
    error: str,
) -> JSONResponse:
    """
    Return a structured top-level error body.

    JSONResponse avoids FastAPI's default HTTPException "detail" wrapper,
    keeping runtime responses aligned with the documented response model.
    """
    body = ErrorResponse(
        status=status,
        error=error,
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------

def extract_investigation_input(
    payload: dict[str, Any],
) -> InvestigationInput:
    """
    Validate only the webhook fields required by the walking skeleton.
    """
    report_id = payload.get("report_id")
    model = payload.get("model")
    overall_severity = payload.get("overall_severity")

    if not isinstance(report_id, str) or not report_id.strip():
        raise ValueError(
            "Missing or invalid report_id"
        )

    if not isinstance(model, dict):
        raise ValueError(
            "Missing or invalid model"
        )

    model_name = model.get("name")
    model_version = model.get("version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Missing or invalid model.name"
        )

    if (
        not isinstance(model_version, str)
        or not model_version.strip()
    ):
        raise ValueError(
            "Missing or invalid model.version"
        )

    if not isinstance(overall_severity, dict):
        raise ValueError(
            "Missing or invalid overall_severity"
        )

    current_severity = overall_severity.get("current")

    if (
        not isinstance(current_severity, str)
        or not current_severity.strip()
    ):
        raise ValueError(
            "Missing or invalid overall_severity.current"
        )

    return InvestigationInput(
        report_id=report_id,
        model_name=model_name,
        model_version=model_version,
        current_severity=current_severity,
    )


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

def build_model_uri(
    model_name: str,
    model_version: str,
) -> str:
    """
    Construct the canonical MLflow model URI from the webhook identity.
    """
    return f"models:/{model_name}/{model_version}"


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def generate_investigation_id() -> str:
    return f"inv_{uuid.uuid4().hex}"


def generate_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex}"


def generate_retrain_job_id() -> str:
    return f"retrain_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Investigation persistence
# ---------------------------------------------------------------------------

def create_investigation_for_webhook(
    investigation_input: InvestigationInput,
) -> InvestigationCreationResult:
    """
    Atomically claim the webhook, create its investigation, and link the
    webhook receipt to the investigation.

    Checkpoint persistence and Redis dispatch happen afterward because
    they belong to separate transactional systems.
    """
    if engine is None:
        raise RuntimeError(
            "Database engine is unavailable"
        )

    report_id = investigation_input.report_id
    investigation_id = generate_investigation_id()
    thread_id = generate_thread_id()

    model_uri = build_model_uri(
        investigation_input.model_name,
        investigation_input.model_version,
    )

    receipt_insert = (
        pg_insert(webhook_receipts)
        .values(
            report_id=report_id,
        )
        .on_conflict_do_nothing(
            index_elements=[
                webhook_receipts.c.report_id,
            ],
        )
        .returning(
            webhook_receipts.c.report_id,
        )
    )

    investigation_insert = (
        sa.insert(investigations)
        .values(
            investigation_id=investigation_id,
            thread_id=thread_id,
            model_name=investigation_input.model_name,
            model_version=investigation_input.model_version,
            model_uri=model_uri,
            triggering_report_id=report_id,
            current_report_id=report_id,
            status="open",
            current_severity=(
                investigation_input.current_severity
            ),
        )
    )

    receipt_update = (
        sa.update(webhook_receipts)
        .where(
            webhook_receipts.c.report_id == report_id
        )
        .values(
            investigation_id=investigation_id,
            processing_status="processed",
            processed_timestamp=sa.func.now(),
        )
    )

    with engine.begin() as connection:
        inserted_report_id = connection.execute(
            receipt_insert
        ).scalar_one_or_none()

        if inserted_report_id is None:
            logger.info(
                "Duplicate drift webhook detected: report_id=%s",
                report_id,
            )

            return InvestigationCreationResult(
                created=False,
                report_id=report_id,
            )

        connection.execute(
            investigation_insert
        )

        receipt_update_result = connection.execute(
            receipt_update
        )

        if receipt_update_result.rowcount != 1:
            raise RuntimeError(
                "Webhook receipt linking affected "
                f"{receipt_update_result.rowcount} rows for "
                f"report_id={report_id}"
            )

    return InvestigationCreationResult(
        created=True,
        report_id=report_id,
        investigation_id=investigation_id,
        thread_id=thread_id,
    )


def update_investigation_status(
    investigation_id: str,
    status: str,
) -> None:
    """
    Update the Dashboard-facing investigation status.

    updated_at is explicitly refreshed because its server default applies
    only during INSERT.
    """
    if engine is None:
        raise RuntimeError(
            "Database engine is unavailable"
        )

    statement = (
        sa.update(investigations)
        .where(
            investigations.c.investigation_id
            == investigation_id
        )
        .values(
            status=status,
            updated_at=sa.func.now(),
        )
    )

    with engine.begin() as connection:
        result = connection.execute(
            statement
        )

        if result.rowcount != 1:
            raise RuntimeError(
                "Investigation status update affected "
                f"{result.rowcount} rows for "
                f"investigation_id={investigation_id}"
            )


# ---------------------------------------------------------------------------
# LangGraph checkpoint persistence
# ---------------------------------------------------------------------------

def persist_initial_checkpoint(
    *,
    graph: Any,
    investigation_id: str,
    thread_id: str,
    investigation_input: InvestigationInput,
) -> None:
    """
    Invoke the minimal graph under the investigation's persisted
    LangGraph thread ID.
    """
    initial_state: InvestigationGraphState = {
        "investigation_id": investigation_id,
        "report_id": investigation_input.report_id,
        "model_name": investigation_input.model_name,
        "model_version": investigation_input.model_version,
        "workflow_status": "initializing",
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    graph.invoke(
        initial_state,
        config=config,
    )


# ---------------------------------------------------------------------------
# Redis dispatch
# ---------------------------------------------------------------------------

def dispatch_retrain_job(
    *,
    redis_client: Redis,
    retrain_job_id: str,
    investigation_id: str,
    report_id: str,
    model_name: str,
    source_model_version: str,
) -> str:
    """
    Dispatch one retrain job to the configured Redis Stream.

    Redis assigns the stream-entry ID. retrain_job_id is the durable
    application-level idempotency key that the Worker will claim in
    Postgres before performing expensive work.
    """
    job_payload = {
        "job_type": "retrain",
        "retrain_job_id": retrain_job_id,
        "investigation_id": investigation_id,
        "report_id": report_id,
        "model_name": model_name,
        "source_model_version": source_model_version,
    }

    stream_entry_id = redis_client.xadd(
        RETRAIN_JOB_STREAM,
        job_payload,
    )

    if not stream_entry_id:
        raise RuntimeError(
            "Redis XADD did not return a stream entry ID"
        )

    return str(stream_entry_id)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    graph = getattr(
        request.app.state,
        "investigation_graph",
        None,
    )

    redis_client = getattr(
        request.app.state,
        "redis_client",
        None,
    )

    if graph is None or redis_client is None:
        return {
            "status": "not_ready",
        }

    return {
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Drift webhook endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/webhooks/drift",
    response_model=WebhookResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid webhook payload",
        },
        401: {
            "model": ErrorResponse,
            "description": "Invalid webhook signature",
        },
        500: {
            "model": ErrorResponse,
            "description": (
                "Persistence, checkpoint, or queue dispatch failure"
            ),
        },
    },
)
async def receive_drift_webhook(
    request: Request,
) -> WebhookResponse | JSONResponse:
    """
    Pass 5 request flow:

    1. Read and authenticate the exact raw webhook bytes.
    2. Parse and validate the authenticated JSON payload.
    3. Atomically create the webhook receipt and investigation.
    4. Return 200 for duplicate webhook deliveries.
    5. Persist the initial LangGraph checkpoint.
    6. Generate and dispatch a retrain job to Redis Streams.
    7. Keep the investigation open on complete success.
    8. Mark checkpoint_failed or dispatch_failed on partial failure.
    """

    graph = getattr(
        request.app.state,
        "investigation_graph",
        None,
    )

    redis_client: Redis | None = getattr(
        request.app.state,
        "redis_client",
        None,
    )

    # These should never be unavailable after successful startup, but the
    # defensive check avoids an unstructured exception during shutdown or
    # abnormal lifecycle states.
    if graph is None or redis_client is None:
        logger.error(
            "Agent infrastructure is unavailable"
        )

        return error_response(
            status_code=500,
            status="infrastructure_unavailable",
            error="Agent infrastructure is unavailable",
        )

    # ------------------------------------------------------------------
    # Authenticate exact raw bytes
    # ------------------------------------------------------------------

    payload_bytes = await request.body()

    received_signature = request.headers.get(
        "X-Webhook-Signature"
    )

    if not received_signature:
        logger.warning(
            "Rejected drift webhook with missing signature"
        )

        return error_response(
            status_code=401,
            status="unauthorized",
            error="Missing webhook signature",
        )

    # DRIFT_WEBHOOK_SECRET is guaranteed by lifespan validation.
    expected_signature = create_signature(
        payload_bytes,
        DRIFT_WEBHOOK_SECRET,  # type: ignore[arg-type]
    )

    if not hmac.compare_digest(
        expected_signature,
        received_signature,
    ):
        logger.warning(
            "Rejected drift webhook with invalid signature"
        )

        return error_response(
            status_code=401,
            status="unauthorized",
            error="Invalid webhook signature",
        )

    # ------------------------------------------------------------------
    # Parse only after authenticity is established
    # ------------------------------------------------------------------

    try:
        payload = json.loads(
            payload_bytes
        )

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        logger.warning(
            "Rejected authenticated webhook because its body "
            "was invalid JSON"
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error="Webhook body is not valid JSON",
        )

    if not isinstance(payload, dict):
        logger.warning(
            "Rejected authenticated webhook because its JSON "
            "body was not an object"
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error="Webhook body must be a JSON object",
        )

    try:
        investigation_input = extract_investigation_input(
            payload
        )

    except ValueError as exc:
        logger.warning(
            "Rejected authenticated webhook with invalid payload: %s",
            exc,
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error=str(exc),
        )

    # ------------------------------------------------------------------
    # Create receipt and investigation
    # ------------------------------------------------------------------

    try:
        result = await run_in_threadpool(
            create_investigation_for_webhook,
            investigation_input,
        )

    except Exception:
        logger.exception(
            "Failed to create investigation: report_id=%s",
            investigation_input.report_id,
        )

        return error_response(
            status_code=500,
            status="persistence_error",
            error="Could not create investigation",
        )

    # Duplicate webhook deliveries must not recreate checkpoints or jobs.
    if not result.created:
        return WebhookResponse(
            status="duplicate",
            report_id=result.report_id,
        )

    if (
        not result.investigation_id
        or not result.thread_id
    ):
        logger.error(
            "Investigation creation returned incomplete identity: "
            "report_id=%s",
            result.report_id,
        )

        return error_response(
            status_code=500,
            status="internal_error",
            error="Investigation identity was not returned",
        )

    # ------------------------------------------------------------------
    # Persist initial LangGraph checkpoint
    # ------------------------------------------------------------------

    try:
        await run_in_threadpool(
            persist_initial_checkpoint,
            graph=graph,
            investigation_id=result.investigation_id,
            thread_id=result.thread_id,
            investigation_input=investigation_input,
        )

    except Exception:
        logger.exception(
            "Failed to persist investigation checkpoint: "
            "investigation_id=%s thread_id=%s report_id=%s",
            result.investigation_id,
            result.thread_id,
            result.report_id,
        )

        try:
            await run_in_threadpool(
                update_investigation_status,
                result.investigation_id,
                "checkpoint_failed",
            )

        except Exception:
            logger.exception(
                "Failed to mark checkpoint failure: "
                "investigation_id=%s",
                result.investigation_id,
            )

        return error_response(
            status_code=500,
            status="checkpoint_error",
            error=(
                "Investigation was created, but checkpoint "
                "persistence failed"
            ),
        )

    # ------------------------------------------------------------------
    # Dispatch retrain job
    # ------------------------------------------------------------------

    retrain_job_id = generate_retrain_job_id()

    try:
        stream_entry_id = await run_in_threadpool(
            dispatch_retrain_job,
            redis_client=redis_client,
            retrain_job_id=retrain_job_id,
            investigation_id=result.investigation_id,
            report_id=result.report_id,
            model_name=investigation_input.model_name,
            source_model_version=(
                investigation_input.model_version
            ),
        )

    except Exception:
        logger.exception(
            "Failed to dispatch retrain job: "
            "retrain_job_id=%s investigation_id=%s report_id=%s",
            retrain_job_id,
            result.investigation_id,
            result.report_id,
        )

        try:
            await run_in_threadpool(
                update_investigation_status,
                result.investigation_id,
                "dispatch_failed",
            )

        except Exception:
            logger.exception(
                "Failed to mark dispatch failure: "
                "investigation_id=%s",
                result.investigation_id,
            )

        return error_response(
            status_code=500,
            status="dispatch_error",
            error=(
                "Investigation and checkpoint were created, "
                "but retrain job dispatch failed"
            ),
        )

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    logger.info(
        "Investigation initialized and retrain job dispatched: "
        "investigation_id=%s thread_id=%s report_id=%s "
        "retrain_job_id=%s stream_entry_id=%s",
        result.investigation_id,
        result.thread_id,
        result.report_id,
        retrain_job_id,
        stream_entry_id,
    )

    return WebhookResponse(
        status="accepted",
        report_id=result.report_id,
        investigation_id=result.investigation_id,
        retrain_job_id=retrain_job_id,
    )