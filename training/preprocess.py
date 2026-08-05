from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PDAYS_SENTINEL: Final[int] = 999
LEAKAGE_FEATURES: Final[list[str]] = ["duration"]

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


class BankMarketingFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Apply domain-specific feature transformations to the raw dataset.

    Responsibilities:
    - remove the post-call duration leakage feature;
    - decompose the pdays sentinel representation into:
      - a previous-contact indicator;
      - a nullable numeric recency feature;
    - preserve all other columns unchanged.

    This transformer performs deterministic feature engineering only.
    Statistical preprocessing such as imputation, scaling, and encoding
    remains inside the downstream ColumnTransformer.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> BankMarketingFeatureTransformer:
        self._validate_input(X)
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_input(X)

        transformed = X.copy()

        transformed["was_previously_contacted"] = (
            transformed["pdays"] != PDAYS_SENTINEL
        ).astype("int8")

        transformed["pdays_since_previous_contact"] = (
            transformed["pdays"]
            .replace(PDAYS_SENTINEL, np.nan)
            .astype("float64")
        )

        transformed = transformed.drop(
            columns=[
                "pdays",
                *LEAKAGE_FEATURES,
            ]
        )

        return transformed

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BankMarketingFeatureTransformer expects a pandas DataFrame"
            )

        required_columns = {
            "pdays",
            *LEAKAGE_FEATURES,
        }

        missing_columns = sorted(
            required_columns.difference(X.columns)
        )

        if missing_columns:
            raise ValueError(
                "Input data is missing required columns: "
                + ", ".join(missing_columns)
            )


def build_preprocessor() -> Pipeline:
    """
    Build the complete preprocessing pipeline for the logistic-regression
    baseline.

    Output:
        A fitted sklearn Pipeline that:
        1. performs domain-specific feature engineering;
        2. imputes and scales continuous numeric features;
        3. passes the binary contact indicator through unchanged;
        4. one-hot encodes categorical features.
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