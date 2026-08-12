from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def get_positive_class_index(
    model_pipeline: Pipeline,
) -> int:
    """
    Resolve the predict_proba column corresponding to positive class 1.

    Generic sklearn Pipeline introspection, not tied to how the model was
    trained — lives in shared/ so both training/ and model_service/ can use
    it without model_service depending on the training subsystem.
    """
    classifier = model_pipeline.named_steps["classifier"]

    matching_indices = np.flatnonzero(classifier.classes_ == 1)

    if len(matching_indices) != 1:
        raise RuntimeError(
            "Expected exactly one positive class labeled 1; "
            f"found classes={classifier.classes_.tolist()}"
        )

    return int(matching_indices[0])
