from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

import numpy as np
from mlflow.entities.model_registry import ModelVersion
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
from mlflow import MlflowClient
from training.train import (
    DATASET_PATH,
    DATASET_SHA256_KEY,
    DEFAULT_MLFLOW_REGISTERED_MODEL_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    GIT_COMMIT_SHA_KEY,
    GIT_WORKTREE_DIRTY_KEY,
    OPERATING_THRESHOLD_TAG,
    SPLIT_MEMBERSHIP_SHA256_KEY,
    SPLIT_SPEC_SHA256_KEY,
    DatasetSplits,
    compute_file_sha256,
    compute_split_membership_sha256,
    compute_split_spec_sha256,
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

MODEL_VERSION_ENV: Final[str] = "MLFLOW_MODEL_VERSION"

FINAL_TEST_EVALUATION_TAG: Final[str] = "final_test_evaluation"
FINAL_TEST_EVALUATION_COMPLETED: Final[str] = "completed"

TEST_ROC_AUC_METRIC: Final[str] = "test_roc_auc"
TEST_PRECISION_METRIC: Final[str] = "test_precision"
TEST_RECALL_METRIC: Final[str] = "test_recall"
TEST_F1_METRIC: Final[str] = "test_f1"

TEST_METRIC_KEYS: Final[tuple[str, ...]] = (
    TEST_ROC_AUC_METRIC,
    TEST_PRECISION_METRIC,
    TEST_RECALL_METRIC,
    TEST_F1_METRIC,
)

THRESHOLD_TOLERANCE: Final[float] = 1e-6


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredProvenance:
    """
    Training provenance recovered from MLflow.
    """

    dataset_sha256: str
    split_spec_sha256: str
    split_membership_sha256: str
    git_commit_sha: str
    git_worktree_dirty: str


@dataclass(frozen=True)
class FinalTestMetrics:
    """
    Immutable result of the final test evaluation.
    """

    model_name: str
    model_version: str
    run_id: str

    operating_threshold: float

    roc_auc: float
    precision: float
    recall: float
    f1: float

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


# -----------------------------------------------------------------------------
# MLflow configuration
# -----------------------------------------------------------------------------


def configure_mlflow() -> tuple[str, str]:
    """
    Resolve the exact registered model version to evaluate.

    The evaluator intentionally requires an explicit model version rather
    than resolving "latest", so final evaluation is deterministic.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )

    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME",
        DEFAULT_MLFLOW_REGISTERED_MODEL_NAME,
    )

    model_version = os.environ.get(MODEL_VERSION_ENV)

    if model_version is None or not model_version.strip():
        raise RuntimeError(
            f"{MODEL_VERSION_ENV} is required. "
            "Final test evaluation must target an explicit "
            "registered model version."
        )

    mlflow.set_tracking_uri(tracking_uri)

    model_version = model_version.strip()

    logger.info(
        "Configured final evaluation: tracking_uri=%s model=%s version=%s",
        tracking_uri,
        registered_model_name,
        model_version,
    )

    return (
        registered_model_name,
        model_version,
    )


# -----------------------------------------------------------------------------
# Registry resolution
# -----------------------------------------------------------------------------


def resolve_registered_model(
    *,
    client: MlflowClient,
    model_name: str,
    model_version: str,
) -> ModelVersion:
    """
    Fetch and validate the exact registered model version.
    """
    model_version_details = client.get_model_version(
        name=model_name,
        version=model_version,
    )

    if model_version_details.run_id is None:
        raise RuntimeError(
            "Registered model version is not associated with an MLflow training run"
        )

    logger.info(
        "Resolved registered model: name=%s version=%s run_id=%s status=%s",
        model_name,
        model_version,
        model_version_details.run_id,
        model_version_details.status,
    )

    return model_version_details


# -----------------------------------------------------------------------------
# One-time evaluation guard
# -----------------------------------------------------------------------------


def assert_test_has_not_been_consumed(
    *,
    client: MlflowClient,
    model_version_details: ModelVersion,
) -> None:
    """
    Refuse repeated or partially repeated final test evaluation.
    """
    run_id = model_version_details.run_id

    if run_id is None:
        raise RuntimeError("Model version has no run_id")

    run = client.get_run(run_id)

    model_status = model_version_details.tags.get(FINAL_TEST_EVALUATION_TAG)

    run_status = run.data.tags.get(FINAL_TEST_EVALUATION_TAG)

    existing_test_metrics = [
        metric_name
        for metric_name in TEST_METRIC_KEYS
        if metric_name in run.data.metrics
    ]

    if (
        model_status == FINAL_TEST_EVALUATION_COMPLETED
        or run_status == FINAL_TEST_EVALUATION_COMPLETED
        or existing_test_metrics
    ):
        raise RuntimeError(
            "Final test evaluation has already been consumed "
            "or partially persisted for "
            f"{model_version_details.name} "
            f"version={model_version_details.version}. "
            f"Existing test metrics={existing_test_metrics}. "
            "Refusing to reuse the sealed test partition."
        )


# -----------------------------------------------------------------------------
# Threshold verification
# -----------------------------------------------------------------------------


def resolve_operating_threshold(
    *,
    client: MlflowClient,
    model_version_details: ModelVersion,
) -> float:
    """
    Recover the validation-selected threshold and verify its registry copy.
    """
    run_id = model_version_details.run_id

    if run_id is None:
        raise RuntimeError("Model version has no run_id")

    run = client.get_run(run_id)

    threshold_metric = run.data.metrics.get("operating_threshold")

    if threshold_metric is None:
        raise RuntimeError("MLflow training run does not contain operating_threshold")

    threshold = float(threshold_metric)

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Stored operating threshold is outside [0, 1]: {threshold}")

    threshold_tag = model_version_details.tags.get(OPERATING_THRESHOLD_TAG)

    if threshold_tag is None:
        raise RuntimeError(
            f"Registered model version is missing {OPERATING_THRESHOLD_TAG!r}"
        )

    try:
        tagged_threshold = float(threshold_tag)
    except ValueError as exc:
        raise RuntimeError(
            f"Registered operating threshold is not numeric: {threshold_tag!r}"
        ) from exc

    if not np.isclose(
        threshold,
        tagged_threshold,
        rtol=0.0,
        atol=THRESHOLD_TOLERANCE,
    ):
        raise RuntimeError(
            "Training-run threshold and registered-model "
            "threshold disagree: "
            f"run={threshold:.12f}, "
            f"registry={tagged_threshold:.12f}"
        )

    logger.info(
        "Frozen operating threshold verified: %.12f",
        threshold,
    )

    return threshold


# -----------------------------------------------------------------------------
# Training provenance resolution
# -----------------------------------------------------------------------------


def _required_parameter(
    parameters: dict[str, str],
    key: str,
) -> str:
    """
    Retrieve required provenance without accepting missing metadata.
    """
    value = parameters.get(key)

    if value is None or not value:
        raise RuntimeError(
            f"Training run is missing required provenance parameter {key!r}"
        )

    return value


def resolve_training_provenance(
    *,
    client: MlflowClient,
    model_version_details: ModelVersion,
) -> StoredProvenance:
    """
    Recover provenance from the training run and verify registry copies.
    """
    run_id = model_version_details.run_id

    if run_id is None:
        raise RuntimeError("Model version has no run_id")

    run = client.get_run(run_id)

    params = run.data.params

    provenance = StoredProvenance(
        dataset_sha256=_required_parameter(
            params,
            DATASET_SHA256_KEY,
        ),
        split_spec_sha256=_required_parameter(
            params,
            SPLIT_SPEC_SHA256_KEY,
        ),
        split_membership_sha256=_required_parameter(
            params,
            SPLIT_MEMBERSHIP_SHA256_KEY,
        ),
        git_commit_sha=_required_parameter(
            params,
            GIT_COMMIT_SHA_KEY,
        ),
        git_worktree_dirty=_required_parameter(
            params,
            GIT_WORKTREE_DIRTY_KEY,
        ),
    )

    registry_expected = {
        DATASET_SHA256_KEY: provenance.dataset_sha256,
        SPLIT_SPEC_SHA256_KEY: provenance.split_spec_sha256,
        SPLIT_MEMBERSHIP_SHA256_KEY: provenance.split_membership_sha256,
        GIT_COMMIT_SHA_KEY: provenance.git_commit_sha,
        GIT_WORKTREE_DIRTY_KEY: provenance.git_worktree_dirty,
    }

    for key, expected_value in registry_expected.items():
        actual_value = model_version_details.tags.get(key)

        if actual_value is None:
            raise RuntimeError(
                f"Registered model version is missing provenance tag {key!r}"
            )

        if actual_value != expected_value:
            raise RuntimeError(
                "Training-run provenance and registry metadata disagree "
                f"for {key!r}: "
                f"run={expected_value!r}, "
                f"registry={actual_value!r}"
            )

    logger.info(
        "Resolved training provenance: "
        "dataset_sha256=%s "
        "split_spec_sha256=%s "
        "split_membership_sha256=%s "
        "git_commit=%s dirty=%s",
        provenance.dataset_sha256,
        provenance.split_spec_sha256,
        provenance.split_membership_sha256,
        provenance.git_commit_sha,
        provenance.git_worktree_dirty,
    )

    return provenance


# -----------------------------------------------------------------------------
# Dataset and split verification
# -----------------------------------------------------------------------------


def verify_dataset_provenance(
    *,
    provenance: StoredProvenance,
) -> None:
    """
    Verify the exact raw dataset bytes before test reconstruction.
    """
    current_dataset_sha256 = compute_file_sha256(DATASET_PATH)

    if current_dataset_sha256 != provenance.dataset_sha256:
        raise RuntimeError(
            "Dataset fingerprint mismatch. "
            "The current CSV is not the dataset used to train "
            "the registered model: "
            f"expected={provenance.dataset_sha256}, "
            f"actual={current_dataset_sha256}"
        )

    logger.info(
        "Dataset fingerprint verified: %s",
        current_dataset_sha256,
    )


def verify_split_spec_provenance(
    *,
    provenance: StoredProvenance,
) -> None:
    """
    Verify the split algorithm/configuration fingerprint.
    """
    current_split_spec_sha256 = compute_split_spec_sha256()

    if current_split_spec_sha256 != provenance.split_spec_sha256:
        raise RuntimeError(
            "Split specification fingerprint mismatch. "
            "Current split configuration differs from training: "
            f"expected={provenance.split_spec_sha256}, "
            f"actual={current_split_spec_sha256}"
        )

    logger.info(
        "Split specification fingerprint verified: %s",
        current_split_spec_sha256,
    )


def reconstruct_and_verify_splits(
    *,
    provenance: StoredProvenance,
) -> DatasetSplits:
    """
    Reconstruct deterministic partitions and verify exact row membership.
    """
    X, y = load_dataset()

    splits = create_stratified_splits(
        X,
        y,
    )

    current_membership_sha256 = compute_split_membership_sha256(splits)

    if current_membership_sha256 != provenance.split_membership_sha256:
        raise RuntimeError(
            "Split membership fingerprint mismatch. "
            "The reconstructed train/validation/test partitions "
            "do not match those used during training: "
            f"expected={provenance.split_membership_sha256}, "
            f"actual={current_membership_sha256}"
        )

    logger.info(
        "Exact split membership verified: %s",
        current_membership_sha256,
    )

    return splits


# -----------------------------------------------------------------------------
# Final test evaluation
# -----------------------------------------------------------------------------


def evaluate_registered_model_on_test(
    *,
    model_name: str,
    model_version: str,
    run_id: str,
    operating_threshold: float,
    splits: DatasetSplits,
) -> FinalTestMetrics:
    """
    Evaluate the exact registered artifact using the frozen threshold.

    No model fitting, threshold search, candidate selection, or tuning is
    performed here.
    """
    model_uri = f"models:/{model_name}/{model_version}"

    logger.info(
        "Loading registered artifact: %s",
        model_uri,
    )

    registered_pipeline = mlflow.sklearn.load_model(model_uri)

    positive_class_index = get_positive_class_index(registered_pipeline)

    test_probabilities = registered_pipeline.predict_proba(splits.X_test)[
        :, positive_class_index
    ]

    if len(test_probabilities) != len(splits.y_test):
        raise RuntimeError("Test probability count does not match test target count")

    if not np.isfinite(test_probabilities).all():
        raise RuntimeError("Test probabilities contain non-finite values")

    if (test_probabilities < 0.0).any() or (test_probabilities > 1.0).any():
        raise RuntimeError("Test probabilities must be between 0 and 1")

    test_predictions = (test_probabilities >= operating_threshold).astype("int8")

    roc_auc = float(
        roc_auc_score(
            splits.y_test,
            test_probabilities,
        )
    )

    precision = float(
        precision_score(
            splits.y_test,
            test_predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            splits.y_test,
            test_predictions,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            splits.y_test,
            test_predictions,
            zero_division=0,
        )
    )

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_positives,
    ) = confusion_matrix(
        splits.y_test,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    return FinalTestMetrics(
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        operating_threshold=operating_threshold,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def persist_final_test_metrics(
    *,
    client: MlflowClient,
    metrics: FinalTestMetrics,
) -> None:
    """
    Persist final metrics onto the original training run and model version.
    """
    metric_values = {
        TEST_ROC_AUC_METRIC: metrics.roc_auc,
        TEST_PRECISION_METRIC: metrics.precision,
        TEST_RECALL_METRIC: metrics.recall,
        TEST_F1_METRIC: metrics.f1,
    }

    for key, value in metric_values.items():
        client.log_metric(
            run_id=metrics.run_id,
            key=key,
            value=value,
        )

    client.set_tag(
        run_id=metrics.run_id,
        key=FINAL_TEST_EVALUATION_TAG,
        value=FINAL_TEST_EVALUATION_COMPLETED,
    )

    client.set_model_version_tag(
        name=metrics.model_name,
        version=metrics.model_version,
        key=FINAL_TEST_EVALUATION_TAG,
        value=FINAL_TEST_EVALUATION_COMPLETED,
    )

    logger.info(
        "Persisted final test metrics: run_id=%s model=%s version=%s",
        metrics.run_id,
        metrics.model_name,
        metrics.model_version,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def log_final_results(
    metrics: FinalTestMetrics,
) -> None:
    """
    Log the final frozen evaluation in submission-ready form.
    """
    logger.info(
        "FINAL MODEL: %s version=%s",
        metrics.model_name,
        metrics.model_version,
    )

    logger.info(
        "FINAL OPERATING THRESHOLD: %.12f",
        metrics.operating_threshold,
    )

    logger.info(
        "FINAL TEST ROC AUC: %.6f",
        metrics.roc_auc,
    )

    logger.info(
        "FINAL TEST PRECISION: %.6f",
        metrics.precision,
    )

    logger.info(
        "FINAL TEST RECALL: %.6f",
        metrics.recall,
    )

    logger.info(
        "FINAL TEST F1: %.6f",
        metrics.f1,
    )

    logger.info(
        "FINAL TEST CONFUSION MATRIX: tn=%d fp=%d fn=%d tp=%d",
        metrics.true_negatives,
        metrics.false_positives,
        metrics.false_negatives,
        metrics.true_positives,
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    model_name, model_version = configure_mlflow()

    client = MlflowClient()

    model_version_details = resolve_registered_model(
        client=client,
        model_name=model_name,
        model_version=model_version,
    )

    # ------------------------------------------------------------------
    # Prevent repeated or partial test evaluation
    # ------------------------------------------------------------------

    assert_test_has_not_been_consumed(
        client=client,
        model_version_details=model_version_details,
    )

    # ------------------------------------------------------------------
    # Recover frozen model metadata
    # ------------------------------------------------------------------

    operating_threshold = resolve_operating_threshold(
        client=client,
        model_version_details=model_version_details,
    )

    provenance = resolve_training_provenance(
        client=client,
        model_version_details=model_version_details,
    )

    # ------------------------------------------------------------------
    # Verify provenance before any test inference
    # ------------------------------------------------------------------

    verify_dataset_provenance(provenance=provenance)

    verify_split_spec_provenance(provenance=provenance)

    splits = reconstruct_and_verify_splits(provenance=provenance)

    run_id = model_version_details.run_id

    if run_id is None:
        raise RuntimeError("Registered model version lost its run_id")

    logger.info(
        "Dataset, split specification, and exact split membership "
        "all match training provenance."
    )

    logger.info(
        "Model configuration and operating threshold are frozen. "
        "Beginning one-time final evaluation of the sealed test partition."
    )

    # ------------------------------------------------------------------
    # First model-quality use of X_test
    # ------------------------------------------------------------------

    metrics = evaluate_registered_model_on_test(
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        operating_threshold=operating_threshold,
        splits=splits,
    )

    log_final_results(metrics)

    persist_final_test_metrics(
        client=client,
        metrics=metrics,
    )

    logger.info(
        "Final test evaluation completed successfully. "
        "Do not use test results for additional model, feature, "
        "hyperparameter, or threshold tuning."
    )


if __name__ == "__main__":
    main()
