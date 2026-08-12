# tests/unit/test_model_service.py

from __future__ import annotations

import pytest
from pydantic import ValidationError

from model_service.main import (
    LoadedRegisteredModel,
    PredictionRequest,
    create_signature,
    decide_prediction,
    get_deployed_model,
)


def test_create_signature_matches_known_hmac_sha256_vector() -> None:
    """
    Verify signature generation against a known HMAC-SHA256 value.

    Test vector:
        secret:  key
        message: The quick brown fox jumps over the lazy dog
    """
    payload = b"The quick brown fox jumps over the lazy dog"
    secret = "key"

    signature = create_signature(
        payload_bytes=payload,
        secret=secret,
    )

    assert signature == (
        "sha256=f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    )


def test_create_signature_changes_when_payload_changes() -> None:
    """
    A modified request body must produce a different signature.
    """
    secret = "development-shared-secret"

    original_signature = create_signature(
        payload_bytes=b'{"status":"original"}',
        secret=secret,
    )

    modified_signature = create_signature(
        payload_bytes=b'{"status":"modified"}',
        secret=secret,
    )

    assert original_signature != modified_signature


def test_create_signature_changes_when_secret_changes() -> None:
    """
    The same payload signed with different secrets must not produce
    the same signature.
    """
    payload = b'{"report_id":"drift_report_001"}'

    first_signature = create_signature(
        payload_bytes=payload,
        secret="first-secret",
    )

    second_signature = create_signature(
        payload_bytes=payload,
        secret="second-secret",
    )

    assert first_signature != second_signature


def test_create_signature_uses_expected_prefix() -> None:
    """
    The wire contract requires signatures to use the sha256=<digest> form.
    """
    signature = create_signature(
        payload_bytes=b"payload",
        secret="secret",
    )

    assert signature.startswith("sha256=")
    assert len(signature) == len("sha256=") + 64


# -----------------------------------------------------------------------------
# PredictionRequest schema validation
# -----------------------------------------------------------------------------


def _valid_prediction_payload() -> dict[str, object]:
    """
    A raw payload matching the exact schema the registered pipeline expects,
    verified directly against the UCI Bank Marketing dataset's real category
    values and dtypes rather than assumed.
    """
    return {
        "age": 35,
        "job": "technician",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 250,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }


def test_prediction_request_accepts_a_valid_payload() -> None:
    request = PredictionRequest(**_valid_prediction_payload())

    assert request.age == 35
    assert request.job == "technician"
    assert request.pdays == 999


def test_prediction_request_dumps_dotted_aliases_for_the_pipeline() -> None:
    """
    The registered pipeline's ColumnTransformer expects raw column names
    containing dots (e.g. "emp.var.rate"), not the Python-safe attribute
    names. model_dump(by_alias=True) is what the endpoint actually calls
    before building the DataFrame handed to predict_proba.
    """
    request = PredictionRequest(**_valid_prediction_payload())

    dumped = request.model_dump(by_alias=True)

    assert "emp.var.rate" in dumped
    assert "cons.price.idx" in dumped
    assert "cons.conf.idx" in dumped
    assert "nr.employed" in dumped
    assert "emp_var_rate" not in dumped


def test_prediction_request_rejects_an_unknown_category() -> None:
    payload = _valid_prediction_payload()
    payload["job"] = "not_a_real_job"

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_request_rejects_a_missing_field() -> None:
    payload = _valid_prediction_payload()
    del payload["nr.employed"]

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_request_rejects_a_non_numeric_age() -> None:
    payload = _valid_prediction_payload()
    payload["age"] = "thirty-five"

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("pdays", 1000),
        ("pdays", -1),
        ("campaign", 0),
        ("previous", -1),
        ("age", 16),
    ],
)
def test_prediction_request_rejects_out_of_range_values(
    field: str,
    bad_value: int,
) -> None:
    payload = _valid_prediction_payload()
    payload[field] = bad_value

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_request_accepts_the_never_contacted_sentinel() -> None:
    """
    pdays=999 is a real sentinel meaning "never previously contacted" in the
    training data, not an out-of-range value to reject.
    """
    payload = _valid_prediction_payload()
    payload["pdays"] = 999

    request = PredictionRequest(**payload)

    assert request.pdays == 999


# -----------------------------------------------------------------------------
# decide_prediction
# -----------------------------------------------------------------------------


def test_decide_prediction_returns_positive_above_threshold() -> None:
    prediction, label = decide_prediction(probability=0.9, threshold=0.385777)

    assert prediction == 1
    assert label == "yes"


def test_decide_prediction_returns_negative_below_threshold() -> None:
    prediction, label = decide_prediction(probability=0.1, threshold=0.385777)

    assert prediction == 0
    assert label == "no"


def test_decide_prediction_treats_exact_threshold_as_positive() -> None:
    """
    Matches the >= semantics used throughout training/evaluate.py's
    threshold application, not a stricter > comparison.
    """
    prediction, label = decide_prediction(probability=0.385777, threshold=0.385777)

    assert prediction == 1
    assert label == "yes"


# -----------------------------------------------------------------------------
# get_deployed_model
# -----------------------------------------------------------------------------


class _FakeAppState:
    pass


class _FakeApp:
    def __init__(self) -> None:
        self.state = _FakeAppState()


class _FakeRequest:
    """
    get_deployed_model only ever accesses request.app.state, so a minimal
    stand-in avoids depending on Starlette's real Request machinery (and,
    critically, avoids triggering the FastAPI lifespan that would otherwise
    try to connect to a real MLflow server).
    """

    def __init__(self, app: _FakeApp) -> None:
        self.app = app


def _fake_deployed_model() -> LoadedRegisteredModel:
    return LoadedRegisteredModel(
        pipeline=object(),  # type: ignore[arg-type]
        positive_class_index=1,
        name="bank-marketing-classifier",
        version="3",
        run_id="fake-run-id",
        operating_threshold=0.385777,
        model_uri="models:/bank-marketing-classifier/3",
    )


def test_get_deployed_model_returns_state_when_present() -> None:
    app = _FakeApp()
    app.state.deployed_model = _fake_deployed_model()

    resolved = get_deployed_model(_FakeRequest(app))  # type: ignore[arg-type]

    assert resolved.name == "bank-marketing-classifier"
    assert resolved.version == "3"


def test_get_deployed_model_raises_when_state_is_absent() -> None:
    app = _FakeApp()

    with pytest.raises(RuntimeError):
        get_deployed_model(_FakeRequest(app))  # type: ignore[arg-type]


def test_get_deployed_model_raises_when_state_has_wrong_type() -> None:
    app = _FakeApp()
    app.state.deployed_model = "not-a-loaded-model"

    with pytest.raises(RuntimeError):
        get_deployed_model(_FakeRequest(app))  # type: ignore[arg-type]
