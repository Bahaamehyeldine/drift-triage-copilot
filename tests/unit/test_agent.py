# tests/unit/test_agent.py

from __future__ import annotations

import re
from typing import Any

import pytest

from agent.main import (
    InvestigationInput,
    build_model_uri,
    extract_investigation_input,
    generate_investigation_id,
    generate_thread_id,
)

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def valid_webhook_payload() -> dict[str, Any]:
    """
    Return the minimum valid webhook payload required to create an
    investigation.

    Each validation test receives a fresh dictionary, preventing mutations in
    one test from leaking into another.
    """
    return {
        "schema_version": "1.0",
        "event_type": "drift.severity.increased",
        "report_id": ("drift-report-customer-churn-model-v12-2026-07-22T12:00:00Z"),
        "timestamp": "2026-07-22T12:00:00Z",
        "model": {
            "name": "customer-churn-model",
            "version": "12",
        },
        "overall_severity": {
            "previous": "low",
            "current": "high",
        },
        "signals": [],
        "summary": "Drift severity increased from low to high.",
    }


# -----------------------------------------------------------------------------
# Model URI
# -----------------------------------------------------------------------------


def test_build_model_uri_constructs_canonical_mlflow_uri() -> None:
    model_uri = build_model_uri(
        model_name="customer-churn-model",
        model_version="12",
    )

    assert model_uri == "models:/customer-churn-model/12"


def test_build_model_uri_preserves_supplied_model_identity() -> None:
    model_uri = build_model_uri(
        model_name="fraud_detection_v2",
        model_version="2026.08.05",
    )

    assert model_uri == "models:/fraud_detection_v2/2026.08.05"


# -----------------------------------------------------------------------------
# ID generation
# -----------------------------------------------------------------------------

INVESTIGATION_ID_PATTERN = re.compile(r"^inv_[0-9a-f]{32}$")

THREAD_ID_PATTERN = re.compile(r"^thread_[0-9a-f]{32}$")


def test_generate_investigation_id_uses_expected_format() -> None:
    investigation_id = generate_investigation_id()

    assert INVESTIGATION_ID_PATTERN.fullmatch(investigation_id)


def test_generate_investigation_id_returns_unique_values() -> None:
    generated_ids = {generate_investigation_id() for _ in range(100)}

    assert len(generated_ids) == 100


def test_generate_thread_id_uses_expected_format() -> None:
    thread_id = generate_thread_id()

    assert THREAD_ID_PATTERN.fullmatch(thread_id)


def test_generate_thread_id_returns_unique_values() -> None:
    generated_ids = {generate_thread_id() for _ in range(100)}

    assert len(generated_ids) == 100


def test_investigation_and_thread_ids_use_distinct_namespaces() -> None:
    investigation_id = generate_investigation_id()
    thread_id = generate_thread_id()

    assert investigation_id.startswith("inv_")
    assert thread_id.startswith("thread_")
    assert investigation_id != thread_id


# -----------------------------------------------------------------------------
# Webhook payload extraction — success path
# -----------------------------------------------------------------------------


def test_extract_investigation_input_returns_required_fields(
    valid_webhook_payload: dict[str, Any],
) -> None:
    result = extract_investigation_input(valid_webhook_payload)

    assert isinstance(
        result,
        InvestigationInput,
    )

    assert result.report_id == (
        "drift-report-customer-churn-model-v12-2026-07-22T12:00:00Z"
    )
    assert result.model_name == "customer-churn-model"
    assert result.model_version == "12"
    assert result.current_severity == "high"


def test_extract_investigation_input_ignores_unused_contract_fields(
    valid_webhook_payload: dict[str, Any],
) -> None:
    """
    Pass 3+ intentionally extracts only persistence-critical fields.

    Other valid webhook fields must not affect investigation input creation.
    """
    valid_webhook_payload["additional_metadata"] = {
        "source": "debug-endpoint",
    }

    result = extract_investigation_input(valid_webhook_payload)

    assert result.model_name == "customer-churn-model"
    assert result.current_severity == "high"


# -----------------------------------------------------------------------------
# Webhook payload extraction — validation branches
# -----------------------------------------------------------------------------

PayloadMutation = tuple[
    str,
    Any,
    str,
]


INVALID_PAYLOAD_CASES: list[PayloadMutation] = [
    pytest.param(
        "report_id",
        None,
        "Missing or invalid report_id",
        id="missing-report-id",
    ),
    pytest.param(
        "report_id",
        "",
        "Missing or invalid report_id",
        id="empty-report-id",
    ),
    pytest.param(
        "model",
        None,
        "Missing or invalid model",
        id="missing-model",
    ),
    pytest.param(
        "model",
        "customer-churn-model",
        "Missing or invalid model",
        id="model-not-object",
    ),
    pytest.param(
        "model.name",
        None,
        "Missing or invalid model.name",
        id="missing-model-name",
    ),
    pytest.param(
        "model.name",
        "   ",
        "Missing or invalid model.name",
        id="blank-model-name",
    ),
    pytest.param(
        "model.version",
        None,
        "Missing or invalid model.version",
        id="missing-model-version",
    ),
    pytest.param(
        "model.version",
        "",
        "Missing or invalid model.version",
        id="empty-model-version",
    ),
    pytest.param(
        "overall_severity",
        None,
        "Missing or invalid overall_severity",
        id="missing-overall-severity",
    ),
    pytest.param(
        "overall_severity",
        "high",
        "Missing or invalid overall_severity",
        id="overall-severity-not-object",
    ),
    pytest.param(
        "overall_severity.current",
        None,
        "Missing or invalid overall_severity.current",
        id="missing-current-severity",
    ),
    pytest.param(
        "overall_severity.current",
        "   ",
        "Missing or invalid overall_severity.current",
        id="blank-current-severity",
    ),
]


@pytest.mark.parametrize(
    (
        "field_path",
        "invalid_value",
        "expected_error",
    ),
    INVALID_PAYLOAD_CASES,
)
def test_extract_investigation_input_rejects_invalid_payload(
    valid_webhook_payload: dict[str, Any],
    field_path: str,
    invalid_value: Any,
    expected_error: str,
) -> None:
    """
    Validate every independent persistence-critical contract branch.

    Parameter IDs ensure pytest reports the exact rule that regressed.
    """
    set_nested_value(
        valid_webhook_payload,
        field_path,
        invalid_value,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(expected_error),
    ):
        extract_investigation_input(valid_webhook_payload)


def set_nested_value(
    payload: dict[str, Any],
    field_path: str,
    value: Any,
) -> None:
    """
    Set a top-level or one-level nested payload field for validation tests.

    Examples:
        report_id
        model.name
        overall_severity.current
    """
    path_parts = field_path.split(".")

    if len(path_parts) == 1:
        payload[path_parts[0]] = value
        return

    if len(path_parts) != 2:
        raise ValueError(f"Unsupported test field path: {field_path}")

    parent_key, child_key = path_parts
    parent = payload.get(parent_key)

    if not isinstance(parent, dict):
        raise AssertionError(
            f"Expected {parent_key!r} to be a dictionary "
            f"while preparing test case {field_path!r}"
        )

    parent[child_key] = value
