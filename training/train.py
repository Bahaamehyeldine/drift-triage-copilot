from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import mlflow
from mlflow import MlflowClient
from training.preprocess import build_preprocessor

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

DATASET_PATH: Final[Path] = Path("data/bank-additional-full.csv")

TARGET_COLUMN: Final[str] = "y"

POSITIVE_TARGET_VALUE: Final[str] = "yes"
NEGATIVE_TARGET_VALUE: Final[str] = "no"

RANDOM_STATE: Final[int] = 42

TRAIN_SIZE: Final[float] = 0.60
VALIDATION_SIZE: Final[float] = 0.20
TEST_SIZE: Final[float] = 0.20

MINIMUM_VALIDATION_RECALL: Final[float] = 0.75

EXPECTED_RAW_COLUMN_COUNT: Final[int] = 21
EXPECTED_FEATURE_COUNT: Final[int] = 20

MODEL_C: Final[float] = 1.0
MODEL_MAX_ITERATIONS: Final[int] = 2_000

# training/train.py runs on the host, not inside the docker-compose network,
# so it must reach MLflow through the published port rather than the
# ${MLFLOW_TRACKING_URI} service hostname used by in-network containers.
DEFAULT_MLFLOW_TRACKING_URI: Final[str] = "http://localhost:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME: Final[str] = "bank-marketing"
DEFAULT_MLFLOW_REGISTERED_MODEL_NAME: Final[str] = "bank-marketing-classifier"

# mlflow's default skops serialization refuses to deserialize types outside
# a known-safe allowlist. build_preprocessor() returns a pipeline containing
# our own BankMarketingFeatureTransformer, so it must be explicitly trusted;
# numpy.dtype is pulled in as part of that transformer's fitted state.
MLFLOW_TRUSTED_MODEL_TYPES: Final[tuple[str, ...]] = (
    "training.preprocess.BankMarketingFeatureTransformer",
    "numpy.dtype",
)

OPERATING_THRESHOLD_TAG: Final[str] = "operating_threshold"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplits:
    """
    Raw train, validation, and test partitions.

    All learned preprocessing is fitted only after these partitions exist.
    """

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ValidationMetrics:
    """
    Validation metrics at the selected operating threshold.
    """

    threshold: float
    roc_auc: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ModelCandidate:
    """
    One explicitly defined logistic-regression experiment.
    """

    name: str
    class_weight: str | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """
    Validation result for one trained candidate.
    """

    candidate: ModelCandidate
    metrics: ValidationMetrics
    pipeline: Pipeline


@dataclass(frozen=True)
class ModelRegistration:
    """
    Identifiers for one registered MLflow model version.
    """

    run_id: str
    model_name: str
    model_version: str


# -----------------------------------------------------------------------------
# Candidate definitions
# -----------------------------------------------------------------------------

MODEL_CANDIDATES: Final[tuple[ModelCandidate, ...]] = (
    ModelCandidate(
        name="logistic_regression_unweighted",
        class_weight=None,
    ),
    ModelCandidate(
        name="logistic_regression_balanced",
        class_weight="balanced",
    ),
)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def load_dataset(
    dataset_path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the raw UCI Bank Marketing dataset.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    dataframe = pd.read_csv(
        dataset_path,
        sep=";",
    )

    if dataframe.shape[1] != EXPECTED_RAW_COLUMN_COUNT:
        raise ValueError(
            "Unexpected dataset schema: "
            f"expected {EXPECTED_RAW_COLUMN_COUNT} columns, "
            f"found {dataframe.shape[1]}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} is missing")

    observed_target_values = set(dataframe[TARGET_COLUMN].dropna().unique().tolist())

    expected_target_values = {
        NEGATIVE_TARGET_VALUE,
        POSITIVE_TARGET_VALUE,
    }

    if observed_target_values != expected_target_values:
        raise ValueError(
            "Unexpected target values: "
            f"expected {sorted(expected_target_values)}, "
            f"found {sorted(observed_target_values)}"
        )

    X = dataframe.drop(columns=[TARGET_COLUMN])

    y = (
        dataframe[TARGET_COLUMN]
        .map(
            {
                NEGATIVE_TARGET_VALUE: 0,
                POSITIVE_TARGET_VALUE: 1,
            }
        )
        .astype("int8")
    )

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature count: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {X.shape[1]}"
        )

    if y.isna().any():
        raise ValueError("Target encoding produced missing values")

    logger.info(
        "Loaded dataset: rows=%d raw_features=%d positive_rate=%.4f",
        len(X),
        X.shape[1],
        float(y.mean()),
    )

    return X, y


# -----------------------------------------------------------------------------
# Stratified splitting
# -----------------------------------------------------------------------------


def create_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
) -> DatasetSplits:
    """
    Create deterministic stratified 60/20/20 train/validation/test splits.
    """
    total_fraction = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE

    if not np.isclose(
        total_fraction,
        1.0,
    ):
        raise RuntimeError("Train, validation, and test fractions must sum to 1.0")

    holdout_size = VALIDATION_SIZE + TEST_SIZE

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    relative_test_size = TEST_SIZE / holdout_size

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_holdout,
        y_holdout,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    splits = DatasetSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )

    validate_splits(
        splits,
        total_row_count=len(X),
    )

    return splits


def validate_splits(
    splits: DatasetSplits,
    *,
    total_row_count: int,
) -> None:
    """
    Verify split integrity and prevent accidental overlap.
    """
    split_pairs = {
        "train": (
            splits.X_train,
            splits.y_train,
        ),
        "validation": (
            splits.X_validation,
            splits.y_validation,
        ),
        "test": (
            splits.X_test,
            splits.y_test,
        ),
    }

    for split_name, (features, target) in split_pairs.items():
        if len(features) != len(target):
            raise RuntimeError(f"{split_name} feature and target lengths differ")

        if not features.index.equals(target.index):
            raise RuntimeError(
                f"{split_name} feature and target indexes are misaligned"
            )

    train_indexes = set(splits.X_train.index)
    validation_indexes = set(splits.X_validation.index)
    test_indexes = set(splits.X_test.index)

    if train_indexes & validation_indexes:
        raise RuntimeError("Training and validation partitions overlap")

    if train_indexes & test_indexes:
        raise RuntimeError("Training and test partitions overlap")

    if validation_indexes & test_indexes:
        raise RuntimeError("Validation and test partitions overlap")

    combined_indexes = train_indexes | validation_indexes | test_indexes

    if len(combined_indexes) != total_row_count:
        raise RuntimeError("The dataset split lost or duplicated rows")


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def build_model_pipeline(
    candidate: ModelCandidate,
) -> Pipeline:
    """
    Build one logistic-regression candidate.

    Every candidate uses the same preprocessing and hyperparameters except
    for the explicitly varied class_weight setting.
    """
    classifier = LogisticRegression(
        C=MODEL_C,
        l1_ratio=0.0,
        solver="liblinear",
        class_weight=candidate.class_weight,
        max_iter=MODEL_MAX_ITERATIONS,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Probability handling
# -----------------------------------------------------------------------------


def get_positive_class_index(
    model_pipeline: Pipeline,
) -> int:
    """
    Return the predict_proba column corresponding to positive class 1.
    """
    classifier = model_pipeline.named_steps["classifier"]

    matching_indices = np.flatnonzero(classifier.classes_ == 1)

    if len(matching_indices) != 1:
        raise RuntimeError(
            "Expected exactly one positive class labeled 1; "
            f"found classes={classifier.classes_.tolist()}"
        )

    return int(matching_indices[0])


# -----------------------------------------------------------------------------
# Validation threshold selection
# -----------------------------------------------------------------------------


def evaluate_validation_threshold(
    *,
    model_pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    minimum_recall: float,
) -> ValidationMetrics:
    """
    Select the highest threshold satisfying the validation recall constraint.

    Selection rule:

        maximize threshold
        subject to recall >= minimum_recall
    """
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0.0 and 1.0")

    if len(X_validation) != len(y_validation):
        raise ValueError("Validation features and target lengths differ")

    if len(y_validation) == 0:
        raise ValueError("Validation split cannot be empty")

    positive_class_index = get_positive_class_index(model_pipeline)

    validation_probabilities = model_pipeline.predict_proba(X_validation)[
        :, positive_class_index
    ]

    if not np.isfinite(validation_probabilities).all():
        raise RuntimeError("Validation probabilities contain non-finite values")

    if (validation_probabilities < 0.0).any() or (validation_probabilities > 1.0).any():
        raise RuntimeError("Validation probabilities must be between 0 and 1")

    (
        curve_precision,
        curve_recall,
        thresholds,
    ) = precision_recall_curve(
        y_validation,
        validation_probabilities,
    )

    if thresholds.size == 0:
        raise RuntimeError("precision_recall_curve produced no candidate thresholds")

    threshold_precision = curve_precision[:-1]
    threshold_recall = curve_recall[:-1]

    if not (len(thresholds) == len(threshold_precision) == len(threshold_recall)):
        raise RuntimeError("Precision-recall arrays are unexpectedly misaligned")

    eligible_indices = np.flatnonzero(threshold_recall >= minimum_recall)

    if len(eligible_indices) == 0:
        raise RuntimeError(
            f"No validation threshold satisfies recall >= {minimum_recall:.2f}"
        )

    selected_index = int(eligible_indices[-1])

    selected_threshold = float(thresholds[selected_index])

    validation_predictions = (validation_probabilities >= selected_threshold).astype(
        "int8"
    )

    validation_auc = float(
        roc_auc_score(
            y_validation,
            validation_probabilities,
        )
    )

    validation_precision = float(
        precision_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
    )

    validation_recall = float(
        recall_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
    )

    validation_f1 = float(
        f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
    )

    if validation_recall + 1e-12 < minimum_recall:
        raise RuntimeError(
            "Selected threshold violated the recall constraint: "
            f"recall={validation_recall:.6f}, "
            f"required={minimum_recall:.6f}"
        )

    logger.debug(
        "Selected threshold point: "
        "threshold=%.6f curve_precision=%.6f curve_recall=%.6f",
        selected_threshold,
        float(threshold_precision[selected_index]),
        float(threshold_recall[selected_index]),
    )

    return ValidationMetrics(
        threshold=selected_threshold,
        roc_auc=validation_auc,
        precision=validation_precision,
        recall=validation_recall,
        f1=validation_f1,
    )


# -----------------------------------------------------------------------------
# Candidate evaluation
# -----------------------------------------------------------------------------


def evaluate_candidate(
    *,
    candidate: ModelCandidate,
    splits: DatasetSplits,
) -> CandidateEvaluation:
    """
    Train one candidate on training data and evaluate it on validation only.
    """
    logger.info(
        "Training candidate: %s class_weight=%s",
        candidate.name,
        candidate.class_weight,
    )

    model_pipeline = build_model_pipeline(candidate)

    model_pipeline.fit(
        splits.X_train,
        splits.y_train,
    )

    log_fitted_pipeline_summary(
        model_pipeline,
        splits,
        candidate_name=candidate.name,
    )

    metrics = evaluate_validation_threshold(
        model_pipeline=model_pipeline,
        X_validation=splits.X_validation,
        y_validation=splits.y_validation,
        minimum_recall=MINIMUM_VALIDATION_RECALL,
    )

    logger.info(
        "Candidate validation result: "
        "name=%s threshold=%.6f auc=%.6f "
        "precision=%.6f recall=%.6f f1=%.6f",
        candidate.name,
        metrics.threshold,
        metrics.roc_auc,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    )

    return CandidateEvaluation(
        candidate=candidate,
        metrics=metrics,
        pipeline=model_pipeline,
    )


def select_best_candidate(
    evaluations: list[CandidateEvaluation],
) -> CandidateEvaluation:
    """
    Select the final candidate using validation data only.

    Selection policy:
        1. all candidates must satisfy the recall constraint;
        2. prefer higher F1 at the selected operating threshold;
        3. use precision as the first tie-breaker;
        4. use ROC AUC as the second tie-breaker.

    The test set is not consulted.
    """
    if not evaluations:
        raise ValueError("No candidate evaluations were provided")

    for evaluation in evaluations:
        if evaluation.metrics.recall + 1e-12 < MINIMUM_VALIDATION_RECALL:
            raise RuntimeError(
                "Candidate unexpectedly violates recall constraint: "
                f"{evaluation.candidate.name}"
            )

    return max(
        evaluations,
        key=lambda evaluation: (
            evaluation.metrics.f1,
            evaluation.metrics.precision,
            evaluation.metrics.roc_auc,
        ),
    )


# -----------------------------------------------------------------------------
# MLflow registration
# -----------------------------------------------------------------------------


def configure_mlflow() -> tuple[str, str]:
    """
    Point the MLflow client at the tracking server and target experiment.

    Returns the resolved experiment name and registered model name so the
    caller logs exactly what configuration was used.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_MLFLOW_EXPERIMENT_NAME,
    )
    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME",
        DEFAULT_MLFLOW_REGISTERED_MODEL_NAME,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "Configured MLflow: tracking_uri=%s experiment=%s registered_model=%s",
        tracking_uri,
        experiment_name,
        registered_model_name,
    )

    return experiment_name, registered_model_name


def register_selected_model(
    *,
    selected: CandidateEvaluation,
    splits: DatasetSplits,
    registered_model_name: str,
) -> ModelRegistration:
    """
    Log the selected candidate's run and register it as a new model version.

    Records parameters, validation metrics, the operating threshold, and the
    fitted preprocessing + classifier pipeline as a single MLflow model.
    """
    classifier = selected.pipeline.named_steps["classifier"]

    signature = infer_signature(
        splits.X_validation,
        selected.pipeline.predict(splits.X_validation),
    )

    with mlflow.start_run(
        run_name=selected.candidate.name,
    ) as run:
        mlflow.log_params(
            {
                "candidate_name": selected.candidate.name,
                "class_weight": (
                    selected.candidate.class_weight
                    if selected.candidate.class_weight is not None
                    else "None"
                ),
                "C": classifier.C,
                "l1_ratio": classifier.l1_ratio,
                "solver": classifier.solver,
                "max_iter": classifier.max_iter,
                "random_state": classifier.random_state,
                "minimum_validation_recall": MINIMUM_VALIDATION_RECALL,
            }
        )

        mlflow.log_metrics(
            {
                "validation_roc_auc": selected.metrics.roc_auc,
                "validation_precision": selected.metrics.precision,
                "validation_recall": selected.metrics.recall,
                "validation_f1": selected.metrics.f1,
                "operating_threshold": selected.metrics.threshold,
            }
        )

        mlflow.set_tag(
            OPERATING_THRESHOLD_TAG,
            f"{selected.metrics.threshold:.6f}",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=selected.pipeline,
            name="model",
            signature=signature,
            registered_model_name=registered_model_name,
            skops_trusted_types=list(MLFLOW_TRUSTED_MODEL_TYPES),
        )

        run_id = run.info.run_id

    if model_info.registered_model_version is None:
        raise RuntimeError("Model logging did not return a registered model version")

    model_version = str(model_info.registered_model_version)

    client = MlflowClient()

    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version,
        key=OPERATING_THRESHOLD_TAG,
        value=f"{selected.metrics.threshold:.6f}",
    )

    logger.info(
        "Registered model: name=%s version=%s run_id=%s",
        registered_model_name,
        model_version,
        run_id,
    )

    return ModelRegistration(
        run_id=run_id,
        model_name=registered_model_name,
        model_version=model_version,
    )


def verify_registered_model(
    *,
    registration: ModelRegistration,
    selected: CandidateEvaluation,
    splits: DatasetSplits,
) -> None:
    """
    Load the registered model version back from MLflow and confirm it
    reproduces the predictions and probabilities of the pipeline that was
    actually evaluated on validation data.

    Registration is worthless if the artifact that comes back out cannot be
    loaded or silently disagrees with the model that was selected.
    """
    client = MlflowClient()

    model_version_details = client.get_model_version(
        name=registration.model_name,
        version=registration.model_version,
    )

    if model_version_details.run_id != registration.run_id:
        raise RuntimeError(
            "Registered model version is not linked to the expected run: "
            f"expected run_id={registration.run_id}, "
            f"found run_id={model_version_details.run_id}"
        )

    model_uri = f"models:/{registration.model_name}/{registration.model_version}"

    loaded_pipeline = mlflow.sklearn.load_model(model_uri)

    expected_predictions = selected.pipeline.predict(splits.X_validation)

    loaded_predictions = loaded_pipeline.predict(splits.X_validation)

    if not np.array_equal(
        expected_predictions,
        loaded_predictions,
    ):
        raise RuntimeError("Loaded model predictions do not match the trained pipeline")

    expected_positive_class_index = get_positive_class_index(selected.pipeline)

    loaded_positive_class_index = get_positive_class_index(loaded_pipeline)

    expected_probabilities = selected.pipeline.predict_proba(splits.X_validation)[
        :, expected_positive_class_index
    ]

    loaded_probabilities = loaded_pipeline.predict_proba(splits.X_validation)[
        :, loaded_positive_class_index
    ]

    if not np.allclose(
        expected_probabilities,
        loaded_probabilities,
    ):
        raise RuntimeError(
            "Loaded model probabilities do not match the trained pipeline"
        )

    logger.info(
        "Verified registered model: name=%s version=%s status=%s "
        "loaded pipeline reproduces training-time predictions",
        registration.model_name,
        registration.model_version,
        model_version_details.status,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def log_split_summary(
    splits: DatasetSplits,
) -> None:
    split_data = {
        "train": (
            splits.X_train,
            splits.y_train,
        ),
        "validation": (
            splits.X_validation,
            splits.y_validation,
        ),
        "test": (
            splits.X_test,
            splits.y_test,
        ),
    }

    total_rows = sum(len(features) for features, _ in split_data.values())

    for split_name, (features, target) in split_data.items():
        logger.info(
            "%s split: rows=%d proportion=%.4f positive_rate=%.4f",
            split_name,
            len(features),
            len(features) / total_rows,
            float(target.mean()),
        )


def log_fitted_pipeline_summary(
    model_pipeline: Pipeline,
    splits: DatasetSplits,
    *,
    candidate_name: str,
) -> None:
    """
    Verify feature-schema consistency using preprocessing fitted on training.
    """
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]

    transformed_train = fitted_preprocessor.transform(splits.X_train)

    transformed_validation = fitted_preprocessor.transform(splits.X_validation)

    transformed_test = fitted_preprocessor.transform(splits.X_test)

    logger.info(
        "%s transformed shapes: train=%s validation=%s test=%s",
        candidate_name,
        transformed_train.shape,
        transformed_validation.shape,
        transformed_test.shape,
    )

    transformed_feature_counts = {
        transformed_train.shape[1],
        transformed_validation.shape[1],
        transformed_test.shape[1],
    }

    if len(transformed_feature_counts) != 1:
        raise RuntimeError(
            "Preprocessing produced inconsistent feature counts across partitions"
        )

    feature_names = fitted_preprocessor.named_steps[
        "column_preprocessing"
    ].get_feature_names_out()

    if len(feature_names) != transformed_train.shape[1]:
        raise RuntimeError(
            "Transformed matrix width does not match generated feature-name count"
        )

    classifier = model_pipeline.named_steps["classifier"]

    logger.info(
        "%s fitted successfully: features=%d iterations=%s classes=%s",
        candidate_name,
        transformed_train.shape[1],
        classifier.n_iter_.tolist(),
        classifier.classes_.tolist(),
    )


def log_comparison_table(
    evaluations: list[CandidateEvaluation],
) -> None:
    """
    Print a concise side-by-side validation comparison.
    """
    comparison = pd.DataFrame(
        [
            {
                "candidate": evaluation.candidate.name,
                "class_weight": (
                    evaluation.candidate.class_weight
                    if evaluation.candidate.class_weight is not None
                    else "None"
                ),
                "threshold": evaluation.metrics.threshold,
                "roc_auc": evaluation.metrics.roc_auc,
                "precision": evaluation.metrics.precision,
                "recall": evaluation.metrics.recall,
                "f1": evaluation.metrics.f1,
            }
            for evaluation in evaluations
        ]
    )

    logger.info(
        "Validation candidate comparison:\n%s",
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Load and split once
    # ------------------------------------------------------------------

    X, y = load_dataset()

    splits = create_stratified_splits(
        X,
        y,
    )

    log_split_summary(splits)

    # ------------------------------------------------------------------
    # Evaluate controlled candidate set
    # ------------------------------------------------------------------

    evaluations: list[CandidateEvaluation] = []

    for candidate in MODEL_CANDIDATES:
        evaluation = evaluate_candidate(
            candidate=candidate,
            splits=splits,
        )

        evaluations.append(evaluation)

    # ------------------------------------------------------------------
    # Compare validation performance
    # ------------------------------------------------------------------

    log_comparison_table(evaluations)

    selected = select_best_candidate(evaluations)

    logger.info(
        "Selected candidate: %s",
        selected.candidate.name,
    )

    logger.info(
        "Selected class_weight: %s",
        selected.candidate.class_weight,
    )

    logger.info(
        "Selected operating threshold: %.6f",
        selected.metrics.threshold,
    )

    logger.info(
        "Selected validation ROC AUC: %.6f",
        selected.metrics.roc_auc,
    )

    logger.info(
        "Selected validation precision: %.6f",
        selected.metrics.precision,
    )

    logger.info(
        "Selected validation recall: %.6f",
        selected.metrics.recall,
    )

    logger.info(
        "Selected validation F1: %.6f",
        selected.metrics.f1,
    )

    # ------------------------------------------------------------------
    # Register the selected model in MLflow
    # ------------------------------------------------------------------

    _, registered_model_name = configure_mlflow()

    registration = register_selected_model(
        selected=selected,
        splits=splits,
        registered_model_name=registered_model_name,
    )

    verify_registered_model(
        registration=registration,
        selected=selected,
        splits=splits,
    )

    # ------------------------------------------------------------------
    # Explicit guardrail
    # ------------------------------------------------------------------

    logger.info("Model selection completed using training and validation data only")

    logger.info(
        "Test split remains untouched: no test predictions or "
        "test metrics have been computed"
    )


if __name__ == "__main__":
    main()
