# model_service/main.py

import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

import httpx
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

import mlflow
from mlflow import MlflowClient
from model_service.registry import ResolvedModelVersion, resolve_model_version
from shared.inference import get_positive_class_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# MLflow deployment configuration
# -----------------------------------------------------------------------------

MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
MLFLOW_REGISTERED_MODEL_NAME_ENV: Final[str] = "MLFLOW_REGISTERED_MODEL_NAME"
MODEL_VERSION_ENV: Final[str] = "MODEL_VERSION"

DEFAULT_MODEL_VERSION: Final[str] = "latest"

OPERATING_THRESHOLD_TAG: Final[str] = "operating_threshold"

EXPECTED_PIPELINE_STEPS: Final[frozenset[str]] = frozenset(
    {
        "preprocessor",
        "classifier",
    }
)

EXPECTED_PREPROCESSOR_STEPS: Final[frozenset[str]] = frozenset(
    {
        "domain_features",
        "column_preprocessing",
    }
)

EXPECTED_DOMAIN_TRANSFORMER_MODULE: Final[str] = "shared.preprocessing"


# -----------------------------------------------------------------------------
# Deployment state
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedRegisteredModel:
    """
    Immutable representation of the ML artifact deployed by Model Service.

    The registry is authoritative for model identity and operating threshold.
    The loaded sklearn Pipeline is retained in memory for the lifetime of the
    service and reused by /predict on every request. positive_class_index is
    resolved once here rather than on every prediction request.
    """

    pipeline: Pipeline
    positive_class_index: int

    name: str
    version: str
    run_id: str

    operating_threshold: float
    model_uri: str


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def require_environment_variable(name: str) -> str:
    """
    Return a required environment variable after trimming whitespace.

    Model Service fails startup rather than entering a partially configured
    state when required deployment configuration is absent.
    """
    value = os.environ.get(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} is not configured")

    return value.strip()


def get_model_version_request() -> str:
    """
    Resolve the requested deployment version.

    MODEL_VERSION defaults to 'latest' for local/development deployments but
    may be pinned to a concrete numeric registry version in production.
    """
    value = os.environ.get(MODEL_VERSION_ENV, DEFAULT_MODEL_VERSION)
    value = value.strip()

    if not value:
        raise RuntimeError(f"{MODEL_VERSION_ENV} must not be empty")

    return value


# -----------------------------------------------------------------------------
# Registered artifact validation
# -----------------------------------------------------------------------------


def validate_loaded_pipeline(pipeline: object) -> Pipeline:
    """
    Validate the structural contract expected from the registered artifact.

    The service deliberately validates the artifact after deserialization
    rather than assuming every version registered under the model name has
    the expected shape.
    """
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError(
            "Registered artifact is not an sklearn Pipeline: "
            f"type={type(pipeline).__module__}.{type(pipeline).__qualname__}"
        )

    pipeline_steps = set(pipeline.named_steps)
    missing_pipeline_steps = EXPECTED_PIPELINE_STEPS - pipeline_steps

    if missing_pipeline_steps:
        raise RuntimeError(
            "Registered pipeline is missing required steps: "
            f"{sorted(missing_pipeline_steps)}"
        )

    preprocessor = pipeline.named_steps["preprocessor"]

    if not isinstance(preprocessor, Pipeline):
        raise RuntimeError(
            "Registered pipeline's 'preprocessor' step is not an sklearn Pipeline"
        )

    preprocessor_steps = set(preprocessor.named_steps)
    missing_preprocessor_steps = EXPECTED_PREPROCESSOR_STEPS - preprocessor_steps

    if missing_preprocessor_steps:
        raise RuntimeError(
            "Registered preprocessing pipeline is missing required steps: "
            f"{sorted(missing_preprocessor_steps)}"
        )

    domain_transformer = preprocessor.named_steps["domain_features"]
    transformer_type = type(domain_transformer)
    transformer_module = transformer_type.__module__

    if transformer_module != EXPECTED_DOMAIN_TRANSFORMER_MODULE:
        raise RuntimeError(
            "Registered model uses an incompatible domain transformer module: "
            f"expected={EXPECTED_DOMAIN_TRANSFORMER_MODULE!r}, "
            f"actual={transformer_module!r}, "
            f"class={transformer_type.__qualname__!r}"
        )

    logger.info(
        "Validated registered pipeline structure: domain_transformer=%s.%s",
        transformer_module,
        transformer_type.__qualname__,
    )

    return pipeline


def resolve_operating_threshold(
    *,
    client: MlflowClient,
    resolved_model: ResolvedModelVersion,
) -> float:
    """
    Resolve the operating threshold stored on the registered model version.

    Model Service does not hardcode the threshold because the threshold is
    part of the trained model's deployment metadata.
    """
    model_version = client.get_model_version(
        name=resolved_model.name,
        version=resolved_model.version,
    )

    threshold_value = model_version.tags.get(OPERATING_THRESHOLD_TAG)

    if threshold_value is None:
        raise RuntimeError(
            "Registered model version is missing required "
            f"{OPERATING_THRESHOLD_TAG!r} tag: "
            f"name={resolved_model.name!r} version={resolved_model.version!r}"
        )

    try:
        threshold = float(threshold_value)
    except ValueError as exc:
        raise RuntimeError(
            "Registered model operating threshold is not numeric: "
            f"value={threshold_value!r}"
        ) from exc

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"Registered model operating threshold is outside [0, 1]: {threshold}"
        )

    return threshold


# -----------------------------------------------------------------------------
# Registered model loading
# -----------------------------------------------------------------------------


def load_registered_model() -> LoadedRegisteredModel:
    """
    Resolve, download, deserialize, and validate the model deployed by this
    Model Service instance.

    Any failure propagates out of FastAPI lifespan startup so the service
    never reports readiness with an invalid or unavailable model.
    """
    tracking_uri = require_environment_variable(MLFLOW_TRACKING_URI_ENV)
    registered_model_name = require_environment_variable(
        MLFLOW_REGISTERED_MODEL_NAME_ENV
    )
    requested_version = get_model_version_request()

    logger.info("Connecting to MLflow: tracking_uri=%s", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info(
        "Resolving registered model: name=%s requested_version=%s",
        registered_model_name,
        requested_version,
    )

    resolved_model = resolve_model_version(
        client=client,
        registered_model_name=registered_model_name,
        requested_version=requested_version,
    )

    model_uri = f"models:/{resolved_model.name}/{resolved_model.version}"

    logger.info("Loading registered model artifact: %s", model_uri)

    loaded_artifact = mlflow.sklearn.load_model(model_uri)
    pipeline = validate_loaded_pipeline(loaded_artifact)
    positive_class_index = get_positive_class_index(pipeline)
    operating_threshold = resolve_operating_threshold(
        client=client, resolved_model=resolved_model
    )

    logger.info(
        "Loaded registered model successfully: name=%s version=%s run_id=%s",
        resolved_model.name,
        resolved_model.version,
        resolved_model.run_id,
    )

    logger.info("Operating threshold: %.12f", operating_threshold)

    return LoadedRegisteredModel(
        pipeline=pipeline,
        positive_class_index=positive_class_index,
        name=resolved_model.name,
        version=resolved_model.version,
        run_id=resolved_model.run_id,
        operating_threshold=operating_threshold,
        model_uri=model_uri,
    )


# -----------------------------------------------------------------------------
# Application lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the registered ML artifact before accepting HTTP traffic.

    Startup is intentionally fail-fast. If registry resolution,
    deserialization, structural validation, or threshold validation fails,
    the exception propagates and the service does not enter a ready state.
    """
    logger.info("Starting Model Service")

    deployed_model = load_registered_model()
    app.state.deployed_model = deployed_model

    logger.info(
        "Model Service ready: model=%s version=%s threshold=%.12f",
        deployed_model.name,
        deployed_model.version,
        deployed_model.operating_threshold,
    )

    try:
        yield
    finally:
        logger.info(
            "Shutting down Model Service: model=%s version=%s",
            deployed_model.name,
            deployed_model.version,
        )
        del app.state.deployed_model


app = FastAPI(
    title="Model Service",
    version="0.1.0",
    lifespan=lifespan,
)


AGENT_WEBHOOK_URL = os.getenv(
    "AGENT_WEBHOOK_URL",
    "http://agent:8001/webhooks/drift",
)

DRIFT_WEBHOOK_SECRET = os.getenv("DRIFT_WEBHOOK_SECRET")

AGENT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "5"))


class DriftDispatchResponse(BaseModel):
    status: str
    report_id: str
    agent_status_code: int


class ErrorResponse(BaseModel):
    status: str
    report_id: str
    error: str
    details: str | None = None


# -----------------------------------------------------------------------------
# Prediction request/response models
# -----------------------------------------------------------------------------

POSITIVE_PREDICTION_LABEL: Final[str] = "yes"
NEGATIVE_PREDICTION_LABEL: Final[str] = "no"


class PredictionRequest(BaseModel):
    """
    Raw feature schema matching the UCI Bank Marketing dataset exactly.

    The registered pipeline's domain transformer (shared/preprocessing.py)
    expects these exact raw columns, including duration and pdays, which it
    consumes internally before the classifier ever sees the data. This is a
    deliberate choice to preserve the exact schema the trained pipeline
    expects rather than inventing a slimmer serving-only schema.

    Categorical fields are constrained to the exact category values observed
    in the training data (verified directly against the raw CSV, not
    assumed) so an invalid category is rejected with a clear 422 instead of
    silently landing in scikit-learn's "unknown category" bucket.
    """

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=17, le=120, description="Client age in years.")
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ]
    default: Literal["no", "unknown", "yes"] = Field(
        description="Has credit in default?"
    )
    housing: Literal["no", "unknown", "yes"] = Field(description="Has a housing loan?")
    loan: Literal["no", "unknown", "yes"] = Field(description="Has a personal loan?")
    contact: Literal["cellular", "telephone"]
    month: Literal[
        "apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"
    ] = Field(description="Month of last contact.")
    day_of_week: Literal["fri", "mon", "thu", "tue", "wed"]
    duration: int = Field(
        ge=0,
        description=(
            "Duration in seconds of the last contact call. Required for "
            "schema compatibility with the trained pipeline, but dropped "
            "internally and never used in the prediction — it is a known "
            "leakage feature (duration is only known after a call ends). "
            "Any value is accepted."
        ),
    )
    campaign: int = Field(
        ge=1,
        description="Number of contacts performed for this client in this campaign.",
    )
    pdays: int = Field(
        ge=0,
        le=999,
        description=(
            "Days since last contact from a previous campaign. "
            "999 means never previously contacted."
        ),
    )
    previous: int = Field(ge=0, description="Number of contacts before this campaign.")
    poutcome: Literal["failure", "nonexistent", "success"] = Field(
        description="Outcome of the previous marketing campaign."
    )
    emp_var_rate: float = Field(
        alias="emp.var.rate",
        description="Employment variation rate (quarterly indicator).",
    )
    cons_price_idx: float = Field(
        alias="cons.price.idx",
        description="Consumer price index (monthly indicator).",
    )
    cons_conf_idx: float = Field(
        alias="cons.conf.idx",
        description="Consumer confidence index (monthly indicator).",
    )
    euribor3m: float = Field(description="Euribor 3-month rate (daily indicator).")
    nr_employed: float = Field(
        alias="nr.employed", description="Number of employees (quarterly indicator)."
    )


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    probability: float
    threshold: float
    prediction: int
    prediction_label: str


def decide_prediction(
    *,
    probability: float,
    threshold: float,
) -> tuple[int, str]:
    """
    Apply the registered model's frozen decision rule to one probability.

    A pure function so the boundary condition (probability exactly equal to
    the threshold counts as a positive prediction, matching the >= used
    throughout training/evaluate.py) is directly unit-testable without a
    live model or an HTTP request.
    """
    prediction = 1 if probability >= threshold else 0
    prediction_label = (
        POSITIVE_PREDICTION_LABEL if prediction == 1 else NEGATIVE_PREDICTION_LABEL
    )

    return prediction, prediction_label


def get_deployed_model(request: Request) -> LoadedRegisteredModel:
    """
    Return the model loaded during application lifespan startup.

    Reaching this failure would indicate an application lifecycle invariant
    has been broken, because successful startup always installs this state.
    """
    deployed_model = getattr(request.app.state, "deployed_model", None)

    if not isinstance(deployed_model, LoadedRegisteredModel):
        raise RuntimeError("Model Service deployment state is unavailable")

    return deployed_model


def build_drift_payload(
    *,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    """
    Build the deterministic drift webhook documented in DECISIONS.md.

    The fixed report_id allows repeated requests to test webhook
    deduplication in the Agent. Model identity is supplied by the registered
    artifact loaded during application startup rather than being hardcoded.
    """
    return {
        "schema_version": "1.0",
        "event_type": "drift.severity.increased",
        "report_id": ("drift-report-customer-churn-model-v12-2026-07-22T12:00:00Z"),
        "timestamp": "2026-07-22T12:00:00Z",
        "model": {
            "name": model_name,
            "version": model_version,
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
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: Request,
    payload: PredictionRequest,
) -> PredictionResponse:
    """
    Score one customer against the registered model's frozen decision rule.

    Inference only: this is a single predict_proba call over one row, not an
    evaluation. It never touches the sealed test set and never computes an
    aggregate metric — see training/evaluate.py for that.
    """
    deployed_model = get_deployed_model(request)

    row = pd.DataFrame([payload.model_dump(by_alias=True)])

    probability = float(
        deployed_model.pipeline.predict_proba(row)[
            :, deployed_model.positive_class_index
        ][0]
    )

    prediction, prediction_label = decide_prediction(
        probability=probability,
        threshold=deployed_model.operating_threshold,
    )

    logger.info(
        "Prediction served: model=%s version=%s probability=%.6f "
        "threshold=%.6f prediction=%s",
        deployed_model.name,
        deployed_model.version,
        probability,
        deployed_model.operating_threshold,
        prediction_label,
    )

    return PredictionResponse(
        model_name=deployed_model.name,
        model_version=deployed_model.version,
        probability=probability,
        threshold=deployed_model.operating_threshold,
        prediction=prediction,
        prediction_label=prediction_label,
    )


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
async def trigger_debug_drift(
    request: Request,
) -> DriftDispatchResponse | JSONResponse:
    deployed_model = get_deployed_model(request)

    payload = build_drift_payload(
        model_name=deployed_model.name,
        model_version=deployed_model.version,
    )
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
            "Agent rejected drift webhook: report_id=%s status_code=%s",
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
        "Drift webhook delivered: report_id=%s agent_status_code=%s",
        report_id,
        response.status_code,
    )

    return DriftDispatchResponse(
        status="delivered",
        report_id=report_id,
        agent_status_code=response.status_code,
    )
