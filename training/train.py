# training/train.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
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

from training.preprocess import build_preprocessor


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FORMAT: Final[str] = (
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_PATH: Final[Path] = Path(
    "data/bank-additional-full.csv"
)

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


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSplits:
    """
    Raw train, validation, and test partitions.

    No learned preprocessing is performed before splitting. This prevents
    validation/test statistics from leaking into the training pipeline.
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
    Validation metrics for the selected operating threshold.

    The threshold is selected exclusively from validation data and must satisfy
    the minimum-recall operating constraint.
    """

    threshold: float
    roc_auc: float
    precision: float
    recall: float
    f1: float


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_dataset(
    dataset_path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the raw UCI Bank Marketing dataset.

    Returns:
        X:
            Raw model features. Domain-specific transformations such as
            dropping duration and transforming pdays remain inside the
            sklearn preprocessing pipeline.

        y:
            Binary target where:
                no  -> 0
                yes -> 1
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset was not found: {dataset_path}"
        )

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
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} is missing"
        )

    observed_target_values = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

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

    X = dataframe.drop(
        columns=[TARGET_COLUMN]
    )

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
        raise ValueError(
            "Target encoding produced missing values"
        )

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
    Split the raw dataset into stratified 60/20/20 partitions.

    Split order:

        raw dataset
            ↓
        60% train + 40% temporary holdout
            ↓
        holdout split equally
            ↓
        20% validation + 20% test

    Preprocessing is intentionally not fitted before this operation.
    """
    split_fraction_sum = (
        TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if not np.isclose(
        split_fraction_sum,
        1.0,
    ):
        raise RuntimeError(
            "Train, validation, and test fractions must sum to 1.0"
        )

    holdout_size = (
        VALIDATION_SIZE
        + TEST_SIZE
    )

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

    relative_test_size = (
        TEST_SIZE
        / holdout_size
    )

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
    Validate partition integrity.

    Checks:
        - feature and target lengths match;
        - indexes remain aligned;
        - train/validation/test partitions do not overlap;
        - all original rows are accounted for exactly once.
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
            raise RuntimeError(
                f"{split_name} feature and target lengths differ"
            )

        if not features.index.equals(target.index):
            raise RuntimeError(
                f"{split_name} feature and target indexes are misaligned"
            )

    train_indexes = set(
        splits.X_train.index
    )
    validation_indexes = set(
        splits.X_validation.index
    )
    test_indexes = set(
        splits.X_test.index
    )

    if train_indexes & validation_indexes:
        raise RuntimeError(
            "Training and validation partitions overlap"
        )

    if train_indexes & test_indexes:
        raise RuntimeError(
            "Training and test partitions overlap"
        )

    if validation_indexes & test_indexes:
        raise RuntimeError(
            "Validation and test partitions overlap"
        )

    combined_indexes = (
        train_indexes
        | validation_indexes
        | test_indexes
    )

    if len(combined_indexes) != total_row_count:
        raise RuntimeError(
            "The dataset split lost or duplicated rows"
        )


# -----------------------------------------------------------------------------
# Model pipeline
# -----------------------------------------------------------------------------

def build_model_pipeline() -> Pipeline:
    """
    Build the baseline model pipeline.

    Preprocessing and classification are kept in one sklearn Pipeline so all
    learned preprocessing state is fitted exclusively from the training split.

    l1_ratio=0.0 represents pure L2 regularization.
    """
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="liblinear",
        max_iter=2_000,
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
# Validation threshold selection
# -----------------------------------------------------------------------------

def get_positive_class_index(
    model_pipeline: Pipeline,
) -> int:
    """
    Resolve the predict_proba column corresponding to positive class 1.

    Avoids assuming class 1 is always the second probability column.
    """
    classifier = model_pipeline.named_steps[
        "classifier"
    ]

    matching_indices = np.flatnonzero(
        classifier.classes_ == 1
    )

    if len(matching_indices) != 1:
        raise RuntimeError(
            "Expected exactly one positive class labeled 1; "
            f"found classes={classifier.classes_.tolist()}"
        )

    return int(
        matching_indices[0]
    )


def evaluate_validation_threshold(
    *,
    model_pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    minimum_recall: float,
) -> ValidationMetrics:
    """
    Select and evaluate the operating threshold on validation data only.

    Selection rule:

        maximize threshold
        subject to recall >= minimum_recall

    The test split must never be passed to this function.

    The selected threshold is applied directly to validation probabilities and
    recall is independently recomputed from the resulting binary predictions.
    This verifies the actual decision rule that will later be used in serving.
    """
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError(
            "minimum_recall must be between 0.0 and 1.0"
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Validation features and target lengths differ"
        )

    if len(y_validation) == 0:
        raise ValueError(
            "Validation split cannot be empty"
        )

    positive_class_index = get_positive_class_index(
        model_pipeline
    )

    validation_probabilities = (
        model_pipeline.predict_proba(
            X_validation
        )[:, positive_class_index]
    )

    if not np.isfinite(
        validation_probabilities
    ).all():
        raise RuntimeError(
            "Validation probabilities contain non-finite values"
        )

    if (
        (validation_probabilities < 0.0).any()
        or (validation_probabilities > 1.0).any()
    ):
        raise RuntimeError(
            "Validation probabilities must be between 0 and 1"
        )

    (
        curve_precision,
        curve_recall,
        thresholds,
    ) = precision_recall_curve(
        y_validation,
        validation_probabilities,
    )

    if thresholds.size == 0:
        raise RuntimeError(
            "precision_recall_curve produced no candidate thresholds"
        )

    # sklearn returns one additional precision/recall pair that has no matching
    # threshold. Restrict both arrays to the threshold-aligned points.
    threshold_precision = curve_precision[:-1]
    threshold_recall = curve_recall[:-1]

    if not (
        len(thresholds)
        == len(threshold_precision)
        == len(threshold_recall)
    ):
        raise RuntimeError(
            "Precision-recall curve arrays are unexpectedly misaligned"
        )

    eligible_indices = np.flatnonzero(
        threshold_recall
        >= minimum_recall
    )

    if len(eligible_indices) == 0:
        raise RuntimeError(
            "No validation threshold satisfies "
            f"recall >= {minimum_recall:.2f}"
        )

    # sklearn returns thresholds in increasing order. The last eligible index
    # is therefore the highest threshold that still satisfies the recall
    # constraint.
    selected_index = int(
        eligible_indices[-1]
    )

    selected_threshold = float(
        thresholds[selected_index]
    )

    validation_predictions = (
        validation_probabilities
        >= selected_threshold
    ).astype("int8")

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

    # Verify the actual deployed decision rule rather than trusting only the
    # precision-recall curve's threshold lookup.
    if validation_recall + 1e-12 < minimum_recall:
        raise RuntimeError(
            "Selected threshold violated the recall constraint: "
            f"recall={validation_recall:.6f}, "
            f"required={minimum_recall:.6f}"
        )

    logger.debug(
        "Selected PR-curve point: "
        "threshold=%.6f curve_precision=%.6f curve_recall=%.6f",
        selected_threshold,
        float(
            threshold_precision[selected_index]
        ),
        float(
            threshold_recall[selected_index]
        ),
    )

    return ValidationMetrics(
        threshold=selected_threshold,
        roc_auc=validation_auc,
        precision=validation_precision,
        recall=validation_recall,
        f1=validation_f1,
    )


# -----------------------------------------------------------------------------
# Reporting and fit verification
# -----------------------------------------------------------------------------

def log_split_summary(
    splits: DatasetSplits,
) -> None:
    """
    Log split sizes and positive-class rates.
    """
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

    total_rows = sum(
        len(features)
        for features, _ in split_data.values()
    )

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
) -> None:
    """
    Verify consistent feature transformation across partitions.

    Validation and test partitions are transformed only with preprocessing
    state already learned from the training split.
    """
    fitted_preprocessor = model_pipeline.named_steps[
        "preprocessor"
    ]

    transformed_train = (
        fitted_preprocessor.transform(
            splits.X_train
        )
    )

    transformed_validation = (
        fitted_preprocessor.transform(
            splits.X_validation
        )
    )

    transformed_test = (
        fitted_preprocessor.transform(
            splits.X_test
        )
    )

    logger.info(
        "Transformed shapes: train=%s validation=%s test=%s",
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
            "Preprocessing produced inconsistent feature counts "
            "across partitions"
        )

    feature_names = (
        fitted_preprocessor
        .named_steps["column_preprocessing"]
        .get_feature_names_out()
    )

    if len(feature_names) != transformed_train.shape[1]:
        raise RuntimeError(
            "Transformed matrix width does not match generated "
            "feature-name count"
        )

    classifier = model_pipeline.named_steps[
        "classifier"
    ]

    logger.info(
        "Logistic regression fitted successfully: "
        "features=%d iterations=%s classes=%s",
        transformed_train.shape[1],
        classifier.n_iter_.tolist(),
        classifier.classes_.tolist(),
    )


def log_validation_metrics(
    metrics: ValidationMetrics,
) -> None:
    """
    Log the selected validation operating point.
    """
    logger.info(
        "Selected operating threshold: %.6f",
        metrics.threshold,
    )

    logger.info(
        "Validation ROC AUC: %.6f",
        metrics.roc_auc,
    )

    logger.info(
        "Validation precision: %.6f",
        metrics.precision,
    )

    logger.info(
        "Validation recall: %.6f",
        metrics.recall,
    )

    logger.info(
        "Validation F1: %.6f",
        metrics.f1,
    )

    logger.info(
        "Operating rule satisfied: validation recall >= %.2f",
        MINIMUM_VALIDATION_RECALL,
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------

    X, y = load_dataset()

    # ------------------------------------------------------------------
    # Split before fitting any learned preprocessing
    # ------------------------------------------------------------------

    splits = create_stratified_splits(
        X,
        y,
    )

    log_split_summary(
        splits
    )

    # ------------------------------------------------------------------
    # Build and fit baseline
    # ------------------------------------------------------------------

    model_pipeline = build_model_pipeline()

    logger.info(
        "Fitting preprocessing and logistic regression "
        "on the training split only"
    )

    model_pipeline.fit(
        splits.X_train,
        splits.y_train,
    )

    log_fitted_pipeline_summary(
        model_pipeline,
        splits,
    )

    # ------------------------------------------------------------------
    # Validation-only operating-threshold selection
    # ------------------------------------------------------------------

    logger.info(
        "Evaluating candidate model on validation split only"
    )

    validation_metrics = (
        evaluate_validation_threshold(
            model_pipeline=model_pipeline,
            X_validation=splits.X_validation,
            y_validation=splits.y_validation,
            minimum_recall=MINIMUM_VALIDATION_RECALL,
        )
    )

    log_validation_metrics(
        validation_metrics
    )

    # ------------------------------------------------------------------
    # Explicit test-set guardrail
    # ------------------------------------------------------------------

    logger.info(
        "Validation selection completed successfully"
    )

    logger.info(
        "Test split remains untouched: no test predictions or "
        "test metrics have been computed"
    )


if __name__ == "__main__":
    main()