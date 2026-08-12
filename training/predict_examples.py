from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

import mlflow.sklearn
import pandas as pd
from mlflow.entities.model_registry import ModelVersion

import mlflow
from mlflow import MlflowClient
from training.train import (
    DEFAULT_MLFLOW_REGISTERED_MODEL_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    NEGATIVE_TARGET_VALUE,
    OPERATING_THRESHOLD_TAG,
    POSITIVE_TARGET_VALUE,
    create_stratified_splits,
    get_positive_class_index,
    load_dataset,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_VERSION_ENV: Final[str] = "MODEL_VERSION"

# Unlike training/evaluate.py, "latest" is an accepted default here: this
# script produces individual predictions for demonstration, not a frozen
# aggregate metric, so pinning to an exact version is a convenience rather
# than a correctness requirement.
DEFAULT_MODEL_VERSION: Final[str] = "latest"

EXAMPLES_PER_CLASS: Final[int] = 3


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionExample:
    """
    One scored inference example: real features, real label, real output.

    This answers "what would the deployed model predict for this customer?",
    not "how good is the model overall?" — that second question belongs to
    evaluate.py, already answered once and frozen in MLflow. Nothing here
    computes or logs an aggregate metric.
    """

    row_index: int
    actual_label: str
    predicted_probability: float
    operating_threshold: float
    predicted_label: str


# -----------------------------------------------------------------------------
# MLflow configuration
# -----------------------------------------------------------------------------


def configure_mlflow() -> tuple[str, str]:
    """
    Resolve the tracking server, registered model name, and requested version.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )

    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME",
        DEFAULT_MLFLOW_REGISTERED_MODEL_NAME,
    )

    requested_version = os.environ.get(MODEL_VERSION_ENV, DEFAULT_MODEL_VERSION).strip()

    if not requested_version:
        raise RuntimeError(f"{MODEL_VERSION_ENV} must not be empty")

    mlflow.set_tracking_uri(tracking_uri)

    logger.info(
        "Configured inference demo: tracking_uri=%s model=%s requested_version=%s",
        tracking_uri,
        registered_model_name,
        requested_version,
    )

    return registered_model_name, requested_version


def resolve_version(
    *,
    client: MlflowClient,
    registered_model_name: str,
    requested_version: str,
) -> ModelVersion:
    """
    Resolve "latest" to the highest numeric registered version, or fetch an
    explicitly pinned version directly. Mirrors model_service/registry.py's
    resolution logic without importing across the service boundary.
    """
    if requested_version.lower() == "latest":
        model_versions = list(
            client.search_model_versions(f"name = '{registered_model_name}'")
        )

        if not model_versions:
            raise RuntimeError(
                f"No registered versions found for {registered_model_name!r}"
            )

        return max(model_versions, key=lambda model_version: int(model_version.version))

    return client.get_model_version(
        name=registered_model_name,
        version=requested_version,
    )


def resolve_operating_threshold(
    *,
    client: MlflowClient,
    model_name: str,
    model_version: str,
) -> float:
    """
    Read the frozen operating threshold stored on the registered version.
    """
    version_details = client.get_model_version(
        name=model_name,
        version=model_version,
    )

    threshold_value = version_details.tags.get(OPERATING_THRESHOLD_TAG)

    if threshold_value is None:
        raise RuntimeError(
            f"Registered model version is missing the {OPERATING_THRESHOLD_TAG!r} tag: "
            f"name={model_name!r} version={model_version!r}"
        )

    threshold = float(threshold_value)

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"Registered operating threshold is outside [0, 1]: {threshold}"
        )

    return threshold


# -----------------------------------------------------------------------------
# Example selection
# -----------------------------------------------------------------------------


def select_demo_rows(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    examples_per_class: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Deterministically select a small, class-balanced set of validation rows.

    Selection is by source-row index order, not randomized, so the same
    examples are shown on every run. Drawn from validation, never test: the
    test split's single-use guarantee is not spent by printing predictions.
    """
    positive_indices = list(y[y == 1].index[:examples_per_class])
    negative_indices = list(y[y == 0].index[:examples_per_class])

    if (
        len(positive_indices) < examples_per_class
        or len(negative_indices) < examples_per_class
    ):
        raise RuntimeError(
            "Validation split does not contain enough rows of each class: "
            f"found positive={len(positive_indices)} negative={len(negative_indices)}, "
            f"requested {examples_per_class} of each"
        )

    selected_indices = sorted(positive_indices + negative_indices)

    return X.loc[selected_indices], y.loc[selected_indices]


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def score_examples(
    *,
    pipeline,
    demo_X: pd.DataFrame,
    demo_y: pd.Series,
    operating_threshold: float,
) -> list[PredictionExample]:
    """
    Run the loaded pipeline on the selected rows and pair each prediction
    with its ground-truth label. This is a single predict_proba call over a
    handful of rows — not model fitting, not threshold search, not an
    aggregate metric.
    """
    positive_class_index = get_positive_class_index(pipeline)

    probabilities = pipeline.predict_proba(demo_X)[:, positive_class_index]

    examples: list[PredictionExample] = []

    for row_index, actual_value, probability in zip(
        demo_X.index,
        demo_y,
        probabilities,
        strict=True,
    ):
        predicted_label = (
            POSITIVE_TARGET_VALUE
            if probability >= operating_threshold
            else NEGATIVE_TARGET_VALUE
        )

        actual_label = (
            POSITIVE_TARGET_VALUE if actual_value == 1 else NEGATIVE_TARGET_VALUE
        )

        examples.append(
            PredictionExample(
                row_index=int(row_index),
                actual_label=actual_label,
                predicted_probability=float(probability),
                operating_threshold=operating_threshold,
                predicted_label=predicted_label,
            )
        )

    return examples


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def log_examples(
    *,
    model_name: str,
    model_version: str,
    examples: list[PredictionExample],
) -> None:
    logger.info("Model: %s version=%s", model_name, model_version)
    logger.info("Examples drawn from the validation split (never test)")

    for position, example in enumerate(examples, start=1):
        logger.info(
            "Example #%d (row=%d): actual=%-3s probability=%.4f "
            "threshold=%.6f predicted=%-3s %s",
            position,
            example.row_index,
            example.actual_label,
            example.predicted_probability,
            example.operating_threshold,
            example.predicted_label,
            "MATCH" if example.actual_label == example.predicted_label else "mismatch",
        )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    registered_model_name, requested_version = configure_mlflow()

    client = MlflowClient()

    resolved_model_version = resolve_version(
        client=client,
        registered_model_name=registered_model_name,
        requested_version=requested_version,
    )

    model_version = str(resolved_model_version.version)

    operating_threshold = resolve_operating_threshold(
        client=client,
        model_name=registered_model_name,
        model_version=model_version,
    )

    model_uri = f"models:/{registered_model_name}/{model_version}"

    logger.info("Loading registered model for inference: %s", model_uri)

    pipeline = mlflow.sklearn.load_model(model_uri)

    X, y = load_dataset()
    splits = create_stratified_splits(X, y)

    demo_X, demo_y = select_demo_rows(
        splits.X_validation,
        splits.y_validation,
        examples_per_class=EXAMPLES_PER_CLASS,
    )

    examples = score_examples(
        pipeline=pipeline,
        demo_X=demo_X,
        demo_y=demo_y,
        operating_threshold=operating_threshold,
    )

    log_examples(
        model_name=registered_model_name,
        model_version=model_version,
        examples=examples,
    )


if __name__ == "__main__":
    main()
