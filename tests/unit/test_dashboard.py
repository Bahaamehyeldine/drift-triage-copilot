# tests/unit/test_dashboard.py

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd

from dashboard.main import (
    IGNORED_DURATION_PLACEHOLDER,
    DashboardMetrics,
    build_prediction_payload,
    calculate_metrics,
    format_utc_timestamp,
)

# -----------------------------------------------------------------------------
# format_utc_timestamp
# -----------------------------------------------------------------------------


def test_format_utc_timestamp_returns_placeholder_for_none() -> None:
    result = format_utc_timestamp(None)

    assert result == "—"


def test_format_utc_timestamp_treats_naive_datetime_as_utc() -> None:
    """
    Naive timestamps are treated as UTC defensively.

    This preserves deterministic output even if a driver or manually-created
    value does not include timezone metadata.
    """
    value = datetime(
        2026,
        8,
        5,
        9,
        30,
        45,
    )

    result = format_utc_timestamp(value)

    assert result == "2026-08-05 09:30:45 UTC"


def test_format_utc_timestamp_converts_aware_datetime_to_utc() -> None:
    beirut_offset = timezone(timedelta(hours=3))

    value = datetime(
        2026,
        8,
        5,
        12,
        30,
        45,
        tzinfo=beirut_offset,
    )

    result = format_utc_timestamp(value)

    assert result == "2026-08-05 09:30:45 UTC"


def test_format_utc_timestamp_preserves_utc_datetime() -> None:
    value = datetime(
        2026,
        8,
        5,
        9,
        30,
        45,
        tzinfo=UTC,
    )

    result = format_utc_timestamp(value)

    assert result == "2026-08-05 09:30:45 UTC"


def test_format_utc_timestamp_stringifies_unexpected_value() -> None:
    """
    Unexpected non-datetime values are rendered rather than raising.

    The Dashboard remains resilient if a future query or driver returns an
    already-formatted value.
    """
    result = format_utc_timestamp("2026-08-05T09:30:45Z")

    assert result == "2026-08-05T09:30:45Z"


def test_format_utc_timestamp_stringifies_numeric_value() -> None:
    result = format_utc_timestamp(12345)

    assert result == "12345"


# -----------------------------------------------------------------------------
# calculate_metrics
# -----------------------------------------------------------------------------


def test_calculate_metrics_returns_zero_counts_for_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "Investigation Status",
            "Latest Job Status",
        ]
    )

    result = calculate_metrics(dataframe)

    assert result == DashboardMetrics(
        investigations=0,
        open_investigations=0,
        blocked_investigations=0,
        completed_jobs=0,
    )


def test_calculate_metrics_counts_realistic_status_mix() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "Investigation Status": "open",
                "Latest Job Status": "completed",
            },
            {
                "Investigation Status": "checkpoint_failed",
                "Latest Job Status": "—",
            },
            {
                "Investigation Status": "dispatch_failed",
                "Latest Job Status": "failed",
            },
            {
                "Investigation Status": "resolved",
                "Latest Job Status": "completed",
            },
            {
                "Investigation Status": "open",
                "Latest Job Status": "running",
            },
        ]
    )

    result = calculate_metrics(dataframe)

    assert result == DashboardMetrics(
        investigations=5,
        open_investigations=2,
        blocked_investigations=2,
        completed_jobs=2,
    )


def test_calculate_metrics_does_not_count_unrelated_failure_status_as_blocked() -> None:
    """
    Only investigation-level checkpoint and dispatch failures are classified
    as blocked. A failed async job alone does not change the investigation
    status count.
    """
    dataframe = pd.DataFrame(
        [
            {
                "Investigation Status": "open",
                "Latest Job Status": "failed",
            },
            {
                "Investigation Status": "resolved",
                "Latest Job Status": "completed",
            },
        ]
    )

    result = calculate_metrics(dataframe)

    assert result.investigations == 2
    assert result.open_investigations == 1
    assert result.blocked_investigations == 0
    assert result.completed_jobs == 1


def test_calculate_metrics_handles_missing_job_status_values() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "Investigation Status": "open",
                "Latest Job Status": None,
            },
            {
                "Investigation Status": "checkpoint_failed",
                "Latest Job Status": pd.NA,
            },
        ]
    )

    result = calculate_metrics(dataframe)

    assert result == DashboardMetrics(
        investigations=2,
        open_investigations=1,
        blocked_investigations=1,
        completed_jobs=0,
    )


def test_calculate_metrics_counts_only_exact_status_matches() -> None:
    """
    Status matching is intentionally case-sensitive and contract-driven.

    Unexpected casing should not be silently normalized by the presentation
    layer because that could hide inconsistent persisted state.
    """
    dataframe = pd.DataFrame(
        [
            {
                "Investigation Status": "OPEN",
                "Latest Job Status": "COMPLETED",
            },
            {
                "Investigation Status": "open",
                "Latest Job Status": "completed",
            },
        ]
    )

    result = calculate_metrics(dataframe)

    assert result == DashboardMetrics(
        investigations=2,
        open_investigations=1,
        blocked_investigations=0,
        completed_jobs=1,
    )


# -----------------------------------------------------------------------------
# build_prediction_payload
# -----------------------------------------------------------------------------


def _prediction_form_values() -> dict[str, object]:
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
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0,
    }


def test_build_prediction_payload_uses_dotted_keys_for_economic_indicators() -> None:
    """
    Model Service's PredictionRequest expects the raw dataset's dotted
    column names, not the Python-safe form field names.
    """
    payload = build_prediction_payload(**_prediction_form_values())

    assert payload["emp.var.rate"] == 1.1
    assert payload["cons.price.idx"] == 93.994
    assert payload["cons.conf.idx"] == -36.4
    assert payload["nr.employed"] == 5191.0
    assert "emp_var_rate" not in payload


def test_build_prediction_payload_never_sends_a_user_supplied_duration() -> None:
    """
    duration is not a form field: the payload always carries the fixed
    placeholder, since it is dropped internally by the registered pipeline
    and asking a user to guess it would be meaningless.
    """
    payload = build_prediction_payload(**_prediction_form_values())

    assert payload["duration"] == IGNORED_DURATION_PLACEHOLDER


def test_build_prediction_payload_preserves_the_never_contacted_sentinel() -> None:
    values = _prediction_form_values()
    values["pdays"] = 999

    payload = build_prediction_payload(**values)

    assert payload["pdays"] == 999


def test_build_prediction_payload_includes_every_field_the_pipeline_requires() -> None:
    payload = build_prediction_payload(**_prediction_form_values())

    expected_keys = {
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    }

    assert set(payload.keys()) == expected_keys
