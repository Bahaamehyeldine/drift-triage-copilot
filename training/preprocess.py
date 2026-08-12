from __future__ import annotations

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.preprocessing import BankMarketingFeatureTransformer

CATEGORICAL_FEATURES: Final[list[str]] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

CONTINUOUS_NUMERIC_FEATURES: Final[list[str]] = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "pdays_since_previous_contact",
]

BINARY_FEATURES: Final[list[str]] = [
    "was_previously_contacted",
]


def build_preprocessor() -> Pipeline:
    """
    Build the preprocessing pipeline for the logistic-regression model.

    Pipeline stages:
    1. apply deterministic domain feature engineering;
    2. impute and scale continuous numeric features;
    3. preserve the binary contact indicator;
    4. one-hot encode nominal categorical features.

    The resulting object can be embedded directly inside the final sklearn
    model Pipeline.
    """
    continuous_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "continuous_numeric",
                continuous_numeric_pipeline,
                CONTINUOUS_NUMERIC_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "domain_features",
                BankMarketingFeatureTransformer(),
            ),
            (
                "column_preprocessing",
                column_transformer,
            ),
        ]
    )
