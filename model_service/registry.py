from __future__ import annotations

import logging
from dataclasses import dataclass

from mlflow.entities.model_registry import ModelVersion

from mlflow import MlflowClient

logger = logging.getLogger(__name__)


LATEST_MODEL_VERSION = "latest"


@dataclass(frozen=True)
class ResolvedModelVersion:
    """
    Stable application representation of an MLflow registered model version.

    The Model Service should not need to pass MLflow entity objects throughout
    the rest of the application. Registry-specific details terminate here.
    """

    name: str
    version: str
    run_id: str


def resolve_model_version(
    *,
    client: MlflowClient,
    registered_model_name: str,
    requested_version: str,
) -> ResolvedModelVersion:
    """
    Resolve the registered model version that the service should deploy.

    Supported behavior:

    - requested_version == "latest":
        discover all registered versions and select the highest numeric
        version explicitly;

    - requested_version == "<integer>":
        fetch that exact registered version.

    MLflow stages are deliberately not used. Model stages and
    get_latest_versions() are deprecated concepts in newer MLflow workflows.

    Raises:
        ValueError:
            The configuration is empty or an explicit version is invalid.

        RuntimeError:
            No registered versions exist, or the resolved model version is
            missing required registry metadata.

        MLflow exceptions:
            Propagated unchanged when registry communication fails. Startup
            should fail rather than hide infrastructure errors.
    """
    model_name = registered_model_name.strip()
    version_request = requested_version.strip()

    if not model_name:
        raise ValueError("MLFLOW_REGISTERED_MODEL_NAME must not be empty")

    if not version_request:
        raise ValueError("MODEL_VERSION must not be empty")

    if version_request.lower() == LATEST_MODEL_VERSION:
        model_version = _resolve_highest_model_version(
            client=client,
            registered_model_name=model_name,
        )
    else:
        model_version = _resolve_explicit_model_version(
            client=client,
            registered_model_name=model_name,
            requested_version=version_request,
        )

    run_id = model_version.run_id

    if run_id is None or not run_id.strip():
        raise RuntimeError(
            "Resolved MLflow model version has no associated run_id: "
            f"name={model_name!r} "
            f"version={model_version.version!r}"
        )

    resolved = ResolvedModelVersion(
        name=model_name,
        version=str(model_version.version),
        run_id=run_id,
    )

    logger.info(
        "Resolved registered model: name=%s version=%s run_id=%s",
        resolved.name,
        resolved.version,
        resolved.run_id,
    )

    return resolved


def _resolve_highest_model_version(
    *,
    client: MlflowClient,
    registered_model_name: str,
) -> ModelVersion:
    """
    Return the highest numeric version registered under the given model name.

    The selection is explicit rather than relying on deprecated MLflow stage
    semantics.
    """
    filter_string = f"name = '{_escape_filter_value(registered_model_name)}'"

    model_versions = list(
        client.search_model_versions(
            filter_string=filter_string,
        )
    )

    if not model_versions:
        raise RuntimeError(
            f"No registered MLflow model versions found: name={registered_model_name!r}"
        )

    try:
        return max(
            model_versions,
            key=lambda model_version: int(model_version.version),
        )
    except (TypeError, ValueError) as exc:
        versions = [str(model_version.version) for model_version in model_versions]

        raise RuntimeError(
            "MLflow returned a non-numeric model version; "
            "cannot determine the highest registered version: "
            f"name={registered_model_name!r} "
            f"versions={versions!r}"
        ) from exc


def _resolve_explicit_model_version(
    *,
    client: MlflowClient,
    registered_model_name: str,
    requested_version: str,
) -> ModelVersion:
    """
    Resolve one explicitly pinned numeric model version.
    """
    try:
        numeric_version = int(requested_version)
    except ValueError as exc:
        raise ValueError(
            "MODEL_VERSION must be either 'latest' or a positive integer; "
            f"received {requested_version!r}"
        ) from exc

    if numeric_version <= 0:
        raise ValueError(
            f"MODEL_VERSION must be a positive integer; received {requested_version!r}"
        )

    return client.get_model_version(
        name=registered_model_name,
        version=str(numeric_version),
    )


def _escape_filter_value(
    value: str,
) -> str:
    """
    Escape single quotes before inserting a model name into the MLflow
    registry filter expression.

    Model names are deployment configuration rather than user-controlled
    request values, but escaping here keeps query construction explicit and
    defensive.
    """
    return value.replace(
        "'",
        "\\'",
    )
