# agent/main.py

import hashlib
import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI(
    title="Triage Agent",
    version="0.1.0",
)


DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")


class WebhookAcceptedResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    status: str
    error: str


def create_signature(payload_bytes: bytes, secret: str) -> str:
    """
    Recompute the HMAC-SHA256 signature over the exact raw request bytes.
    """
    digest = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return f"sha256={digest}"


def error_response(
    *,
    status_code: int,
    status: str,
    error: str,
) -> JSONResponse:
    body = ErrorResponse(
        status=status,
        error=error,
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


@app.post(
    "/webhooks/drift",
    response_model=WebhookAcceptedResponse,
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Invalid webhook signature",
        },
        500: {
            "model": ErrorResponse,
            "description": "Agent configuration error",
        },
    },
)
async def receive_drift_webhook(
    request: Request,
) -> WebhookAcceptedResponse | JSONResponse:
    """
    Verify the drift webhook HMAC signature.

    Pass 1 intentionally performs no parsing, persistence, deduplication,
    checkpointing, or Redis dispatch.
    """

    if not DRIFT_WEBHOOK_SECRET:
        return error_response(
            status_code=500,
            status="configuration_error",
            error="DRIFT_WEBHOOK_SECRET is not configured",
        )

    payload_bytes = await request.body()

    received_signature = request.headers.get(
        "X-Webhook-Signature"
    )

    if not received_signature:
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
        return error_response(
            status_code=401,
            status="unauthorized",
            error="Invalid webhook signature",
        )

    return WebhookAcceptedResponse(
        status="accepted",
    )