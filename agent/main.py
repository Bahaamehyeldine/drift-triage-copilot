# agent/main.py

import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
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

# Allows a separate connection string later if checkpoint persistence is
# moved to a dedicated database. For the MVP, it defaults to Postgres.
LANGGRAPH_DATABASE_URL = os.getenv(
    "LANGGRAPH_DATABASE_URL",
    DATABASE_URL,
)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class InvestigationGraphState(TypedDict):
    """
    Minimal workflow state persisted for an investigation.

    The investigations table remains authoritative for human-facing
    lifecycle status. This state contains only the context required to
    resume LangGraph execution.
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
    Minimal Pass 4 graph node.

    This node performs no triage or agent reasoning. It proves that the
    investigation can enter a LangGraph workflow and persist its state
    under the investigation's existing thread_id.
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
):
    """
    Build the minimal persisted investigation graph.

    The full triage/action/comms supervisor is intentionally deferred.
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
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the LangGraph Postgres checkpointer once for the service
    lifecycle rather than recreating it for every webhook request.
    """
    app.state.investigation_graph = None

    if not LANGGRAPH_DATABASE_URL:
        logger.error(
            "LANGGRAPH_DATABASE_URL and DATABASE_URL are not configured; "
            "checkpoint persistence is unavailable"
        )
        yield
        return

    try:
        # PostgresSaver manages its own Postgres connection lifecycle.
        with PostgresSaver.from_conn_string(
            LANGGRAPH_DATABASE_URL
        ) as checkpointer:
            # Creates or upgrades LangGraph's checkpoint tables.
            # For the MVP this is performed at startup.
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
            "Failed to initialize LangGraph checkpoint persistence"
        )
        raise


app = FastAPI(
    title="Triage Agent",
    version="0.1.0",
    lifespan=lifespan,
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


engine = (
    sa.create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )
    if DATABASE_URL
    else None
)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class WebhookResponse(BaseModel):
    status: str
    report_id: str
    investigation_id: str | None = None


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
    Recompute the HMAC-SHA256 signature over the exact request bytes.
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
    Return a structured top-level error response.
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
    Validate the webhook fields required by Pass 4.
    """
    report_id = payload.get("report_id")
    model = payload.get("model")
    overall_severity = payload.get("overall_severity")

    if not isinstance(report_id, str) or not report_id.strip():
        raise ValueError("Missing or invalid report_id")

    if not isinstance(model, dict):
        raise ValueError("Missing or invalid model")

    model_name = model.get("name")
    model_version = model.get("version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Missing or invalid model.name")

    if (
        not isinstance(model_version, str)
        or not model_version.strip()
    ):
        raise ValueError("Missing or invalid model.version")

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
# IDs
# ---------------------------------------------------------------------------

def generate_investigation_id() -> str:
    return f"inv_{uuid.uuid4().hex}"


def generate_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Investigation persistence
# ---------------------------------------------------------------------------

def create_investigation_for_webhook(
    investigation_input: InvestigationInput,
) -> InvestigationCreationResult:
    """
    Atomically claim the webhook, create its investigation, and link the
    receipt to the new investigation.

    LangGraph checkpoint persistence intentionally happens afterward,
    outside this transaction.
    """
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")

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
            index_elements=[webhook_receipts.c.report_id],
        )
        .returning(webhook_receipts.c.report_id)
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

        connection.execute(
            receipt_update
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

    updated_at is explicitly refreshed because its database default only
    runs during INSERT, not on subsequent UPDATE operations.
    """
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")

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
        result = connection.execute(statement)

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
    Invoke the minimal graph using the investigation's persisted thread_id.

    LangGraph stores the resulting graph state as a checkpoint associated
    with this thread.
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
        }
    }

    graph.invoke(
        initial_state,
        config=config,
    )


# ---------------------------------------------------------------------------
# Endpoint
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
                "Agent configuration, persistence, or "
                "checkpoint error"
            ),
        },
    },
)
async def receive_drift_webhook(
    request: Request,
) -> WebhookResponse | JSONResponse:
    """
    Pass 4 request flow:

    1. Authenticate the raw webhook body.
    2. Parse and validate the trusted JSON payload.
    3. Atomically create the receipt and investigation.
    4. Return immediately for duplicate deliveries.
    5. Invoke the minimal LangGraph using the stored thread_id.
    6. Persist the resulting checkpoint in Postgres.
    7. Keep the investigation status as "open" on success.
    8. Mark it "checkpoint_failed" if graph persistence fails.

    Redis dispatch remains outside this pass.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    if not DRIFT_WEBHOOK_SECRET:
        logger.error(
            "DRIFT_WEBHOOK_SECRET is not configured"
        )

        return error_response(
            status_code=500,
            status="configuration_error",
            error="DRIFT_WEBHOOK_SECRET is not configured",
        )

    if engine is None:
        logger.error(
            "DATABASE_URL is not configured"
        )

        return error_response(
            status_code=500,
            status="configuration_error",
            error="DATABASE_URL is not configured",
        )

    graph = getattr(
        request.app.state,
        "investigation_graph",
        None,
    )

    if graph is None:
        logger.error(
            "LangGraph investigation graph is unavailable"
        )

        return error_response(
            status_code=500,
            status="configuration_error",
            error="Checkpoint persistence is unavailable",
        )

    # ------------------------------------------------------------------
    # Authenticate raw request bytes
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

    expected_signature = create_signature(
        payload_bytes,
        DRIFT_WEBHOOK_SECRET,
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
    # Parse only after authentication
    # ------------------------------------------------------------------

    try:
        payload = json.loads(payload_bytes)

    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(
            "Rejected authenticated webhook because body "
            "was invalid JSON"
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error="Webhook body is not valid JSON",
        )

    if not isinstance(payload, dict):
        logger.warning(
            "Rejected authenticated webhook because JSON "
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

    # ------------------------------------------------------------------
    # Duplicate webhook: do not invoke graph again
    # ------------------------------------------------------------------

    if not result.created:
        return WebhookResponse(
            status="duplicate",
            report_id=result.report_id,
        )

    if not result.investigation_id or not result.thread_id:
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
                "Failed to mark investigation checkpoint failure: "
                "investigation_id=%s",
                result.investigation_id,
            )

        return error_response(
            status_code=500,
            status="checkpoint_error",
            error="Investigation was created, but checkpoint persistence failed",
        )

    # Success intentionally leaves investigations.status = "open".
    logger.info(
        "Created investigation and persisted checkpoint: "
        "investigation_id=%s thread_id=%s report_id=%s",
        result.investigation_id,
        result.thread_id,
        result.report_id,
    )

    return WebhookResponse(
        status="accepted",
        report_id=result.report_id,
        investigation_id=result.investigation_id,
    )