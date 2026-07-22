
# model_service/main.py

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Service",
    version="0.1.0",
)


AGENT_WEBHOOK_URL = os.getenv(
    "AGENT_WEBHOOK_URL",
    "http://agent:8001/webhooks/drift",
)

DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")

AGENT_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "5")
)


class DriftDispatchResponse(BaseModel):
    status: str
    report_id: str
    agent_status_code: int


class ErrorResponse(BaseModel):
    status: str
    report_id: str
    error: str
    details: str | None = None


def build_drift_payload() -> dict[str, Any]:
    """
    Build the deterministic drift webhook documented in DECISIONS.md.

    The fixed report_id allows repeated requests to test webhook
    deduplication in the Agent.
    """
    return {
        "schema_version": "1.0",
        "event_type": "drift.severity.increased",
        "report_id": (
            "drift-report-customer-churn-model-v12-"
            "2026-07-22T12:00:00Z"
        ),
        "timestamp": "2026-07-22T12:00:00Z",
        "model": {
            "name": "customer-churn-model",
            "version": "12",
        },
        "overall_severity": {
            "previous": "low",
            "current": "high",
        },
        "signals": [
            {
                "feature": "monthly_charges",
                "test": "psi",
                "value": 0.31,
                "severity": "high",
            },
            {
                "feature": "country",
                "test": "chi2",
                "value": 0.018,
                "severity": "medium",
            },
        ],
        "summary": (
            "Drift severity increased from low to high. "
            "High PSI drift was detected for monthly_charges, and medium "
            "categorical drift was detected for country."
        ),
    }


def serialize_payload(payload: dict[str, Any]) -> bytes:
    """
    Serialize the payload deterministically.

    The exact serialized bytes are both signed and sent to the Agent,
    preventing an HMAC mismatch caused by different JSON formatting.
    """
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def create_signature(payload_bytes: bytes, secret: str) -> str:
    """
    Create an HMAC-SHA256 signature for the serialized request body.
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
    report_id: str,
    error: str,
    details: str | None = None,
) -> JSONResponse:
    """
    Return a structured error body matching the documented OpenAPI model.

    JSONResponse is used instead of HTTPException so FastAPI does not wrap
    the response body inside a top-level "detail" field.
    """
    body = ErrorResponse(
        status=status,
        report_id=report_id,
        error=error,
        details=details,
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/debug/drift",
    response_model=DriftDispatchResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Model Service configuration error",
        },
        502: {
            "model": ErrorResponse,
            "description": "Drift webhook delivery failed",
        },
    },
)
async def trigger_debug_drift() -> DriftDispatchResponse | JSONResponse:
    payload = build_drift_payload()
    report_id = payload["report_id"]

    if not DRIFT_WEBHOOK_SECRET:
        logger.error("DRIFT_WEBHOOK_SECRET is not configured")

        return error_response(
            status_code=500,
            status="configuration_error",
            report_id=report_id,
            error="DRIFT_WEBHOOK_SECRET is not configured",
        )

    payload_bytes = serialize_payload(payload)

    signature = create_signature(
        payload_bytes,
        DRIFT_WEBHOOK_SECRET,
    )

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": signature,
    }

    try:
        async with httpx.AsyncClient(
            timeout=AGENT_REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                AGENT_WEBHOOK_URL,
                content=payload_bytes,
                headers=headers,
            )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Agent rejected drift webhook: "
            "report_id=%s status_code=%s",
            report_id,
            exc.response.status_code,
        )

        return error_response(
            status_code=502,
            status="dispatch_failed",
            report_id=report_id,
            error="Agent returned an unsuccessful response",
            details=(
                f"Agent responded with HTTP "
                f"{exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ),
        )

    except httpx.RequestError as exc:
        logger.exception(
            "Could not reach Agent: report_id=%s url=%s",
            report_id,
            AGENT_WEBHOOK_URL,
        )

        return error_response(
            status_code=502,
            status="dispatch_failed",
            report_id=report_id,
            error="Could not deliver drift webhook to Agent",
            details=str(exc),
        )

    logger.info(
        "Drift webhook delivered: "
        "report_id=%s agent_status_code=%s",
        report_id,
        response.status_code,
    )

    return DriftDispatchResponse(
        status="delivered",
        report_id=report_id,
        agent_status_code=response.status_code,
    )

