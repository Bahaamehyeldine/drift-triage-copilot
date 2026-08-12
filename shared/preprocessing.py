from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PDAYS_SENTINEL: Final[int] = 999
LEAKAGE_FEATURES: Final[list[str]] = ["duration"]


class BankMarketingFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Apply deterministic, domain-specific feature transformations to the raw
    UCI Bank Marketing feature set.

    Responsibilities:
    - remove the post-call duration leakage feature;
    - decompose the pdays sentinel encoding into:
      - was_previously_contacted
      - pdays_since_previous_contact
    - preserve all other source features unchanged.

    Statistical preprocessing such as imputation, scaling, and categorical
    encoding remains outside this class.
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
            .replace(
                PDAYS_SENTINEL,
                np.nan,
            )
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
    def _validate_input(
        X: pd.DataFrame,
    ) -> None:
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "BankMarketingFeatureTransformer expects a pandas DataFrame"
            )

        required_columns = {
            "pdays",
            *LEAKAGE_FEATURES,
        }

        missing_columns = sorted(required_columns.difference(X.columns))

        if missing_columns:
            raise ValueError(
                "Input data is missing required columns: " + ", ".join(missing_columns)
            )
