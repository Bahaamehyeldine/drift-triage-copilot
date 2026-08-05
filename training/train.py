from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.linear_model import LogisticRegression
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
TEST_SIZE: Final[float] = 0.20
VALIDATION_SIZE: Final[float] = 0.20
TRAIN_SIZE: Final[float] = 0.60

EXPECTED_RAW_COLUMN_COUNT: Final[int] = 21
EXPECTED_FEATURE_COUNT: Final[int] = 20


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSplits:
    """
    Raw train, validation, and test partitions.

    The splits intentionally contain unprocessed features. The fitted sklearn
    pipeline owns all learned preprocessing state.
    """

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


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
            Raw feature DataFrame containing all 20 source features. Leakage
            removal and domain feature engineering happen inside the sklearn
            preprocessing pipeline.

        y:
            Binary integer target where 1 represents subscription and 0
            represents no subscription.
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

    target_values = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    expected_target_values = {
        NEGATIVE_TARGET_VALUE,
        POSITIVE_TARGET_VALUE,
    }

    if target_values != expected_target_values:
        raise ValueError(
            "Unexpected target values: "
            f"expected {sorted(expected_target_values)}, "
            f"found {sorted(target_values)}"
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
    Split raw data into stratified 60/20/20 partitions.

    The first split reserves 40% for validation and testing. The second split
    divides that temporary partition equally, producing:

        training   = 60%
        validation = 20%
        test       = 20%

    No preprocessing is fitted before these splits are created.
    """
    holdout_size = VALIDATION_SIZE + TEST_SIZE

    if not abs(
        TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0
    ) < 1e-12:
        raise RuntimeError(
            "Train, validation, and test fractions must sum to 1.0"
        )

    X_train, X_holdout, y_train, y_holdout = train_test_split(
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
    Validate partition sizes, index isolation, and target alignment.
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

    train_indexes = set(splits.X_train.index)
    validation_indexes = set(splits.X_validation.index)
    test_indexes = set(splits.X_test.index)

    if train_indexes.intersection(validation_indexes):
        raise RuntimeError(
            "Training and validation partitions overlap"
        )

    if train_indexes.intersection(test_indexes):
        raise RuntimeError(
            "Training and test partitions overlap"
        )

    if validation_indexes.intersection(test_indexes):
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
    Build the logistic-regression baseline.

    All learned preprocessing and classifier state is contained within one
    sklearn Pipeline. Calling fit on this object with only the training split
    prevents validation/test leakage.
    """
    classifier = LogisticRegression(
        penalty="l2",
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
    Confirm that the fitted training-only preprocessor can transform all
    partitions using the same learned state.
    """
    fitted_preprocessor = model_pipeline.named_steps[
        "preprocessor"
    ]

    transformed_train = fitted_preprocessor.transform(
        splits.X_train
    )

    transformed_validation = fitted_preprocessor.transform(
        splits.X_validation
    )

    transformed_test = fitted_preprocessor.transform(
        splits.X_test
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


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    X, y = load_dataset()

    splits = create_stratified_splits(
        X,
        y,
    )

    log_split_summary(
        splits
    )

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

    logger.info(
        "Baseline training fit completed successfully. "
        "Validation and test partitions have not been used for fitting."
    )


if __name__ == "__main__":
    main()

