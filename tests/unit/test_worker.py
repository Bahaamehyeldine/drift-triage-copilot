# tests/unit/test_worker.py

from __future__ import annotations

import re
from typing import Any

import pytest

from worker.main import (
    RetrainJobMessage,
    parse_retrain_job_message,
    require_message_field,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def valid_retrain_message() -> dict[str, str]:
    """
    Return a complete Redis Stream payload for one retrain job.

    Each test receives a fresh dictionary, preventing mutations from leaking
    between parameterized cases.
    """
    return {
        "job_type": "retrain",
        "retrain_job_id": (
            "retrain_4e2e3f1f7e8d4d3e9a4e7f1c6b8a2d10"
        ),
        "investigation_id": (
            "inv_2a4f6d8c0b1e4f5a9c3d7e8f1a2b4c6d"
        ),
        "report_id": (
            "drift-report-customer-churn-model-v12-"
            "2026-07-22T12:00:00Z"
        ),
        "model_name": "customer-churn-model",
        "source_model_version": "12",
    }


# -----------------------------------------------------------------------------
# require_message_field
# -----------------------------------------------------------------------------

def test_require_message_field_returns_valid_value() -> None:
    fields = {
        "retrain_job_id": "retrain_123",
    }

    result = require_message_field(
        fields,
        "retrain_job_id",
    )

    assert result == "retrain_123"


@pytest.mark.parametrize(
    ("fields", "field_name"),
    [
        pytest.param(
            {},
            "retrain_job_id",
            id="missing-field",
        ),
        pytest.param(
            {"retrain_job_id": ""},
            "retrain_job_id",
            id="empty-field",
        ),
        pytest.param(
            {"retrain_job_id": "   "},
            "retrain_job_id",
            id="blank-field",
        ),
        pytest.param(
            {"retrain_job_id": None},
            "retrain_job_id",
            id="non-string-field",
        ),
    ],
)
def test_require_message_field_rejects_invalid_values(
    fields: dict[str, Any],
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Missing or invalid stream field: {field_name}"
        ),
    ):
        require_message_field(
            fields,
            field_name,
        )


# -----------------------------------------------------------------------------
# parse_retrain_job_message — success path
# -----------------------------------------------------------------------------

def test_parse_retrain_job_message_maps_all_fields(
    valid_retrain_message: dict[str, str],
) -> None:
    result = parse_retrain_job_message(
        valid_retrain_message
    )

    assert isinstance(
        result,
        RetrainJobMessage,
    )

    assert result.retrain_job_id == (
        "retrain_4e2e3f1f7e8d4d3e9a4e7f1c6b8a2d10"
    )
    assert result.investigation_id == (
        "inv_2a4f6d8c0b1e4f5a9c3d7e8f1a2b4c6d"
    )
    assert result.report_id == (
        "drift-report-customer-churn-model-v12-"
        "2026-07-22T12:00:00Z"
    )
    assert result.model_name == "customer-churn-model"
    assert result.source_model_version == "12"


def test_parse_retrain_job_message_ignores_unrelated_stream_fields(
    valid_retrain_message: dict[str, str],
) -> None:
    """
    Redis metadata added in future versions should not affect parsing of the
    fields required by the current retrain-job contract.
    """
    valid_retrain_message["trace_id"] = "trace_abc123"
    valid_retrain_message["dispatch_attempt"] = "1"

    result = parse_retrain_job_message(
        valid_retrain_message
    )

    assert result.model_name == "customer-churn-model"
    assert result.source_model_version == "12"


# -----------------------------------------------------------------------------
# parse_retrain_job_message — required field validation
# -----------------------------------------------------------------------------

REQUIRED_FIELD_CASES = [
    pytest.param(
        "job_type",
        None,
        "Missing or invalid stream field: job_type",
        id="missing-job-type",
    ),
    pytest.param(
        "job_type",
        "",
        "Missing or invalid stream field: job_type",
        id="empty-job-type",
    ),
    pytest.param(
        "job_type",
        "   ",
        "Missing or invalid stream field: job_type",
        id="blank-job-type",
    ),
    pytest.param(
        "retrain_job_id",
        None,
        "Missing or invalid stream field: retrain_job_id",
        id="missing-retrain-job-id",
    ),
    pytest.param(
        "retrain_job_id",
        "",
        "Missing or invalid stream field: retrain_job_id",
        id="empty-retrain-job-id",
    ),
    pytest.param(
        "retrain_job_id",
        "   ",
        "Missing or invalid stream field: retrain_job_id",
        id="blank-retrain-job-id",
    ),
    pytest.param(
        "investigation_id",
        None,
        "Missing or invalid stream field: investigation_id",
        id="missing-investigation-id",
    ),
    pytest.param(
        "investigation_id",
        "",
        "Missing or invalid stream field: investigation_id",
        id="empty-investigation-id",
    ),
    pytest.param(
        "investigation_id",
        "   ",
        "Missing or invalid stream field: investigation_id",
        id="blank-investigation-id",
    ),
    pytest.param(
        "report_id",
        None,
        "Missing or invalid stream field: report_id",
        id="missing-report-id",
    ),
    pytest.param(
        "report_id",
        "",
        "Missing or invalid stream field: report_id",
        id="empty-report-id",
    ),
    pytest.param(
        "report_id",
        "   ",
        "Missing or invalid stream field: report_id",
        id="blank-report-id",
    ),
    pytest.param(
        "model_name",
        None,
        "Missing or invalid stream field: model_name",
        id="missing-model-name",
    ),
    pytest.param(
        "model_name",
        "",
        "Missing or invalid stream field: model_name",
        id="empty-model-name",
    ),
    pytest.param(
        "model_name",
        "   ",
        "Missing or invalid stream field: model_name",
        id="blank-model-name",
    ),
    pytest.param(
        "source_model_version",
        None,
        "Missing or invalid stream field: source_model_version",
        id="missing-source-model-version",
    ),
    pytest.param(
        "source_model_version",
        "",
        "Missing or invalid stream field: source_model_version",
        id="empty-source-model-version",
    ),
    pytest.param(
        "source_model_version",
        "   ",
        "Missing or invalid stream field: source_model_version",
        id="blank-source-model-version",
    ),
]


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_error",
    ),
    REQUIRED_FIELD_CASES,
)
def test_parse_retrain_job_message_rejects_missing_or_empty_fields(
    valid_retrain_message: dict[str, str],
    field_name: str,
    invalid_value: str | None,
    expected_error: str,
) -> None:
    """
    Protect every required Redis message field independently.

    A None value is used to represent a missing or malformed decoded field.
    Deleting the key is preferable for the explicit missing-field cases.
    """
    mutated_message: dict[str, Any] = dict(
        valid_retrain_message
    )

    if invalid_value is None:
        mutated_message.pop(
            field_name,
            None,
        )
    else:
        mutated_message[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=re.escape(expected_error),
    ):
        parse_retrain_job_message(
            mutated_message
        )


# -----------------------------------------------------------------------------
# parse_retrain_job_message — job type contract
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unsupported_job_type",
    [
        pytest.param(
            "rollback",
            id="rollback-job",
        ),
        pytest.param(
            "replay",
            id="replay-job",
        ),
        pytest.param(
            "RETRAIN",
            id="incorrect-case",
        ),
        pytest.param(
            "retrain_v2",
            id="unknown-versioned-type",
        ),
    ],
)
def test_parse_retrain_job_message_rejects_unsupported_job_type(
    valid_retrain_message: dict[str, str],
    unsupported_job_type: str,
) -> None:
    valid_retrain_message[
        "job_type"
    ] = unsupported_job_type

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unsupported job_type: {unsupported_job_type}"
        ),
    ):
        parse_retrain_job_message(
            valid_retrain_message
        )
