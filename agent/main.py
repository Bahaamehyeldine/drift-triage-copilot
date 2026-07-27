
# agent/main.py

import hashlib
import hmac
import json
import logging
import os
import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Triage Agent",
    version="0.1.0",
)


DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL")


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
# API response models
# ---------------------------------------------------------------------------

class WebhookResponse(BaseModel):
    status: str
    report_id: str
    investigation_id: str | None = None


class ErrorResponse(BaseModel):
    status: str
    error: str


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------

class InvestigationInput(BaseModel):
    report_id: str
    model_name: str
    model_version: str
    current_severity: str


class InvestigationCreationResult(BaseModel):
    created: bool
    report_id: str
    investigation_id: str | None = None


# ---------------------------------------------------------------------------
# HMAC
# ---------------------------------------------------------------------------

def create_signature(
    payload_bytes: bytes,
    secret: str,
) -> str:
    """
    Recompute the HMAC-SHA256 signature over the exact bytes received
    from the Model Service.
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
    Return the structured error shape documented by this API.
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
# Webhook payload validation
# ---------------------------------------------------------------------------

def extract_investigation_input(
    payload: dict[str, Any],
) -> InvestigationInput:
    """
    Extract and validate the fields required to create an investigation.

    Full webhook schema validation can be formalized later. Pass 3 validates
    the contract fields that are required for persistence.
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

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Missing or invalid model.version")

    if not isinstance(overall_severity, dict):
        raise ValueError("Missing or invalid overall_severity")

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
    Construct the canonical MLflow model URI from the model identity
    already present in the webhook contract.

    This keeps URI construction in one place and avoids adding redundant
    model_uri data to the webhook payload.
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
# Persistence
# ---------------------------------------------------------------------------

def create_investigation_for_webhook(
    investigation_input: InvestigationInput,
) -> InvestigationCreationResult:
    """
    Atomically:

    1. Claim the report_id by inserting webhook_receipts.
    2. Detect duplicate deliveries through the report_id primary key.
    3. Create the investigation.
    4. Link the receipt to the investigation.
    5. Mark the receipt as processed.

    The entire operation runs in one transaction.

    If investigation creation or receipt linking fails, the initial receipt
    insert is rolled back as well. A later webhook retry can therefore safely
    attempt the operation again.
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

    investigation_insert = sa.insert(
        investigations
    ).values(
        investigation_id=investigation_id,
        thread_id=thread_id,
        model_name=investigation_input.model_name,
        model_version=investigation_input.model_version,
        model_uri=model_uri,
        triggering_report_id=report_id,
        current_report_id=report_id,
        status="open",
        current_severity=investigation_input.current_severity,
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

        # ---------------------------------------------------------------
        # Duplicate delivery
        # ---------------------------------------------------------------

        if inserted_report_id is None:
            logger.info(
                "Duplicate drift webhook detected: report_id=%s",
                report_id,
            )

            return InvestigationCreationResult(
                created=False,
                report_id=report_id,
            )

        # ---------------------------------------------------------------
        # New delivery
        # ---------------------------------------------------------------

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
                "Agent configuration or persistence error"
            ),
        },
    },
)
async def receive_drift_webhook(
    request: Request,
) -> WebhookResponse | JSONResponse:
    """
    Pass 3 workflow:

        Model Service
             |
             v
        raw webhook
             |
             v
        verify HMAC
             |
             v
        parse trusted JSON
             |
             v
        validate investigation fields
             |
             v
        atomic database transaction
             |
             +--> insert webhook_receipt
             |
             +--> duplicate?
             |       |
             |       +--> yes -> return 200 duplicate
             |
             +--> create investigation
             |
             +--> link receipt
             |
             +--> mark receipt processed
             |
             v
        return 200 accepted

    LangGraph checkpoint creation and Redis dispatch remain intentionally
    outside Pass 3.
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

    # ------------------------------------------------------------------
    # Read exact signed bytes
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

    # ------------------------------------------------------------------
    # Authenticate before processing content
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Extract fields needed for investigation creation
    # ------------------------------------------------------------------

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
    # Dedup + investigation creation
    #
    # Synchronous SQLAlchemy Core is intentionally executed in FastAPI's
    # thread pool so the async event loop is not blocked by database I/O.
    # ------------------------------------------------------------------

    try:
        result = await run_in_threadpool(
            create_investigation_for_webhook,
            investigation_input,
        )

    except Exception:
        logger.exception(
            "Failed to process drift webhook: report_id=%s",
            investigation_input.report_id,
        )

        return error_response(
            status_code=500,
            status="persistence_error",
            error="Could not create investigation",
        )

    # ------------------------------------------------------------------
    # Duplicate delivery
    # ------------------------------------------------------------------

    if not result.created:
        return WebhookResponse(
            status="duplicate",
            report_id=result.report_id,
        )

    # ------------------------------------------------------------------
    # Successfully created investigation
    # ------------------------------------------------------------------

    logger.info(
        "Created investigation: "
        "investigation_id=%s report_id=%s model=%s version=%s",
        result.investigation_id,
        result.report_id,
        investigation_input.model_name,
        investigation_input.model_version,
    )

    return WebhookResponse(
        status="accepted",
        report_id=result.report_id,
        investigation_id=result.investigation_id,
    )
