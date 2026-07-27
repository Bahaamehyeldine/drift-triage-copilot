
# agent/main.py

import hashlib
import hmac
import json
import logging
import os
from typing import Any

import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Triage Agent",
    version="0.1.0",
)


DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

metadata = sa.MetaData()

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
# Response models
# ---------------------------------------------------------------------------

class WebhookResponse(BaseModel):
    status: str
    report_id: str


class ErrorResponse(BaseModel):
    status: str
    error: str


# ---------------------------------------------------------------------------
# HMAC
# ---------------------------------------------------------------------------

def create_signature(
    payload_bytes: bytes,
    secret: str,
) -> str:
    """
    Recompute the HMAC-SHA256 signature over the exact raw request bytes.
    """
    digest = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def error_response(
    *,
    status_code: int,
    status: str,
    error: str,
) -> JSONResponse:
    """
    Return a structured error response without FastAPI's default
    top-level "detail" wrapper.
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
# Persistence
# ---------------------------------------------------------------------------

def persist_webhook_receipt(report_id: str) -> bool:
    """
    Atomically insert a webhook receipt.

    Returns:
        True:
            The receipt was inserted and this is the first delivery.

        False:
            The report_id already existed and this is a duplicate delivery.

    PostgreSQL performs the uniqueness check and insert atomically, avoiding
    the race condition that would exist with a separate SELECT followed by
    INSERT.
    """
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")

    statement = (
        pg_insert(webhook_receipts)
        .values(
            report_id=report_id,
        )
        .on_conflict_do_nothing(
            index_elements=[webhook_receipts.c.report_id],
        )
        .returning(webhook_receipts.c.report_id)
    )

    with engine.begin() as connection:
        inserted_report_id = connection.execute(
            statement
        ).scalar_one_or_none()

    return inserted_report_id is not None


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
            "description": "Agent configuration or persistence error",
        },
    },
)
async def receive_drift_webhook(
    request: Request,
) -> WebhookResponse | JSONResponse:
    """
    Pass 2:

    1. Read the raw request bytes.
    2. Verify the HMAC signature.
    3. Parse JSON only after authenticity has been verified.
    4. Extract report_id.
    5. Atomically deduplicate through webhook_receipts.
    6. Return 200 for both new and duplicate deliveries.

    Investigation creation, LangGraph checkpointing, and Redis dispatch
    intentionally remain outside this pass.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    if not DRIFT_WEBHOOK_SECRET:
        logger.error("DRIFT_WEBHOOK_SECRET is not configured")

        return error_response(
            status_code=500,
            status="configuration_error",
            error="DRIFT_WEBHOOK_SECRET is not configured",
        )

    if engine is None:
        logger.error("DATABASE_URL is not configured")

        return error_response(
            status_code=500,
            status="configuration_error",
            error="DATABASE_URL is not configured",
        )

    # ------------------------------------------------------------------
    # Read the exact bytes signed by Model Service
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
    # Only parse JSON after authenticity has been verified
    # ------------------------------------------------------------------

    try:
        payload: dict[str, Any] = json.loads(payload_bytes)

    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(
            "Rejected authenticated webhook because body was invalid JSON"
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error="Webhook body is not valid JSON",
        )

    report_id = payload.get("report_id")

    if not isinstance(report_id, str) or not report_id.strip():
        logger.warning(
            "Rejected authenticated webhook with missing or invalid report_id"
        )

        return error_response(
            status_code=400,
            status="invalid_payload",
            error="Webhook payload is missing a valid report_id",
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    try:
        inserted = await run_in_threadpool(
            persist_webhook_receipt,
            report_id,
        )

    except Exception:
        logger.exception(
            "Could not persist webhook receipt: report_id=%s",
            report_id,
        )

        return error_response(
            status_code=500,
            status="persistence_error",
            error="Could not persist webhook receipt",
        )

    if not inserted:
        logger.info(
            "Duplicate drift webhook received: report_id=%s",
            report_id,
        )

        return WebhookResponse(
            status="duplicate",
            report_id=report_id,
        )

    logger.info(
        "Accepted new drift webhook: report_id=%s",
        report_id,
    )

    return WebhookResponse(
        status="accepted",
        report_id=report_id,
    )

