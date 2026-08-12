# dashboard/main.py

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import httpx
import pandas as pd
import sqlalchemy as sa
import streamlit as st
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------

APP_TITLE: Final[str] = "Drift Triage Dashboard"
APP_ICON: Final[str] = "📊"

EMPTY_VALUE: Final[str] = "—"

INVESTIGATION_STATUS_OPEN: Final[str] = "open"
JOB_STATUS_COMPLETED: Final[str] = "completed"

DEFAULT_MODEL_SERVICE_URL: Final[str] = "http://model_service:8000"


@dataclass(frozen=True)
class Settings:
    """
    Runtime configuration loaded from environment variables.

    Postgres is a mandatory dependency: the Dashboard cannot render its core
    monitoring view without it. Model Service is not — the prediction form
    is an additional feature that degrades to a visible error if Model
    Service is unreachable, rather than taking down investigation
    monitoring, so MODEL_SERVICE_URL has a working default instead of being
    a hard startup requirement.
    """

    database_url: str
    model_service_url: str

    @classmethod
    def from_environment(cls) -> Settings:
        database_url = os.getenv("DATABASE_URL", "").strip()

        if not database_url:
            raise RuntimeError("DATABASE_URL is not configured")

        model_service_url = (
            os.getenv("MODEL_SERVICE_URL", DEFAULT_MODEL_SERVICE_URL).strip()
            or DEFAULT_MODEL_SERVICE_URL
        )

        return cls(
            database_url=database_url,
            model_service_url=model_service_url,
        )


# -----------------------------------------------------------------------------
# Database schema
# -----------------------------------------------------------------------------

metadata = sa.MetaData()


investigations = sa.Table(
    "investigations",
    metadata,
    sa.Column(
        "investigation_id",
        sa.String(length=64),
        primary_key=True,
    ),
    sa.Column(
        "thread_id",
        sa.String(length=64),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "model_name",
        sa.String(length=255),
        nullable=False,
    ),
    sa.Column(
        "model_version",
        sa.String(length=128),
        nullable=False,
    ),
    sa.Column(
        "model_uri",
        sa.String(length=1024),
        nullable=False,
    ),
    sa.Column(
        "triggering_report_id",
        sa.String(length=128),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "current_report_id",
        sa.String(length=128),
        nullable=False,
    ),
    sa.Column(
        "status",
        sa.String(length=32),
        nullable=False,
    ),
    sa.Column(
        "current_severity",
        sa.String(length=16),
        nullable=False,
    ),
    sa.Column(
        "recommended_action",
        sa.String(length=64),
        nullable=True,
    ),
    sa.Column(
        "resolution",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "stale_reason",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "invalidation_reason",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
    ),
    sa.Column(
        "resolved_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
)


retrain_jobs = sa.Table(
    "retrain_jobs",
    metadata,
    sa.Column(
        "retrain_job_id",
        sa.String(length=64),
        primary_key=True,
    ),
    sa.Column(
        "investigation_id",
        sa.String(length=64),
        sa.ForeignKey(
            "investigations.investigation_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    ),
    sa.Column(
        "model_name",
        sa.String(length=255),
        nullable=False,
    ),
    sa.Column(
        "source_model_version",
        sa.String(length=128),
        nullable=False,
    ),
    sa.Column(
        "job_status",
        sa.String(length=32),
        nullable=False,
    ),
    sa.Column(
        "worker_claimed_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
    sa.Column(
        "started_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
    sa.Column(
        "completed_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
    sa.Column(
        "attempt_count",
        sa.Integer(),
        nullable=False,
    ),
    sa.Column(
        "resulting_model_version",
        sa.String(length=128),
        nullable=True,
    ),
    sa.Column(
        "failure_details",
        sa.Text(),
        nullable=True,
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
    ),
)


# -----------------------------------------------------------------------------
# Database infrastructure
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def create_database_engine(
    database_url: str,
) -> Engine:
    """
    Create one SQLAlchemy connection pool per Streamlit process.

    Streamlit reruns the application script on interaction, so the engine
    is cached as a process-level resource rather than recreated on every
    refresh.
    """
    logger.info("Initializing Dashboard database engine")

    return sa.create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


def verify_database_connection(
    engine: Engine,
) -> None:
    """
    Fail before rendering operational data if Postgres is unavailable.
    """
    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 1"))


# -----------------------------------------------------------------------------
# Query construction
# -----------------------------------------------------------------------------


def build_latest_jobs_subquery() -> sa.Subquery:
    """
    Rank retrain jobs newest-first within each investigation.

    created_at is the primary ordering criterion. retrain_job_id is used as
    a deterministic tie-breaker if two rows share the same timestamp.
    """
    job_rank = (
        sa.func.row_number()
        .over(
            partition_by=(retrain_jobs.c.investigation_id),
            order_by=(
                retrain_jobs.c.created_at.desc(),
                retrain_jobs.c.retrain_job_id.desc(),
            ),
        )
        .label("job_rank")
    )

    return sa.select(
        retrain_jobs.c.retrain_job_id,
        retrain_jobs.c.investigation_id,
        retrain_jobs.c.job_status,
        retrain_jobs.c.attempt_count,
        retrain_jobs.c.resulting_model_version,
        retrain_jobs.c.created_at.label("job_created_at"),
        retrain_jobs.c.updated_at.label("job_updated_at"),
        job_rank,
    ).subquery("ranked_retrain_jobs")


def build_investigation_query() -> sa.Select:
    """
    Build the read model used by the Dashboard.

    A LEFT JOIN keeps investigations visible before a retrain job exists.
    The ranked subquery ensures that at most one retrain job is joined to
    each investigation.
    """
    latest_jobs = build_latest_jobs_subquery()

    return (
        sa.select(
            investigations.c.investigation_id.label("investigation_id"),
            investigations.c.model_name.label("model_name"),
            investigations.c.model_version.label("model_version"),
            investigations.c.current_report_id.label("report_id"),
            investigations.c.status.label("investigation_status"),
            investigations.c.current_severity.label("severity"),
            latest_jobs.c.retrain_job_id.label("latest_job_id"),
            latest_jobs.c.job_status.label("latest_job_status"),
            latest_jobs.c.attempt_count.label("attempt_count"),
            latest_jobs.c.resulting_model_version.label("resulting_model_version"),
            investigations.c.updated_at.label("updated_at"),
        )
        .select_from(
            investigations.outerjoin(
                latest_jobs,
                sa.and_(
                    latest_jobs.c.investigation_id == investigations.c.investigation_id,
                    latest_jobs.c.job_rank == 1,
                ),
            )
        )
        .order_by(
            investigations.c.updated_at.desc(),
            investigations.c.investigation_id.desc(),
        )
    )


def load_investigations(
    engine: Engine,
) -> list[dict[str, Any]]:
    """
    Load the latest Dashboard read model from Postgres.

    Results are intentionally not cached: each Streamlit rerun should read
    current operational state.
    """
    statement = build_investigation_query()

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    return [dict(row) for row in rows]


# -----------------------------------------------------------------------------
# Presentation model
# -----------------------------------------------------------------------------

DISPLAY_COLUMNS: Final[Mapping[str, str]] = {
    "investigation_id": "Investigation ID",
    "model_name": "Model",
    "model_version": "Model Version",
    "report_id": "Report ID",
    "investigation_status": "Investigation Status",
    "severity": "Severity",
    "latest_job_id": "Latest Job ID",
    "latest_job_status": "Latest Job Status",
    "attempt_count": "Attempts",
    "resulting_model_version": "Resulting Model Version",
    "updated_at": "Updated At (UTC)",
}


def format_utc_timestamp(
    value: Any,
) -> str:
    """
    Render timestamps consistently in UTC.

    PostgreSQL timezone-aware values are expected, but naive values are
    treated as UTC defensively.
    """
    if value is None:
        return EMPTY_VALUE

    if not isinstance(value, datetime):
        return str(value)

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_dataframe(
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Convert database rows into the stable, human-facing table schema.
    """
    internal_columns = list(DISPLAY_COLUMNS.keys())

    if not rows:
        return pd.DataFrame(columns=list(DISPLAY_COLUMNS.values()))

    dataframe = pd.DataFrame(
        rows,
        columns=internal_columns,
    )

    dataframe["updated_at"] = dataframe["updated_at"].map(format_utc_timestamp)

    dataframe["attempt_count"] = dataframe["attempt_count"].astype("Int64")

    dataframe = dataframe.rename(columns=DISPLAY_COLUMNS)

    return dataframe.fillna(EMPTY_VALUE)


@dataclass(frozen=True)
class DashboardMetrics:
    investigations: int
    open_investigations: int
    blocked_investigations: int
    completed_jobs: int


def calculate_metrics(
    dataframe: pd.DataFrame,
) -> DashboardMetrics:
    if dataframe.empty:
        return DashboardMetrics(
            investigations=0,
            open_investigations=0,
            blocked_investigations=0,
            completed_jobs=0,
        )

    statuses = dataframe["Investigation Status"]

    job_statuses = dataframe["Latest Job Status"]

    blocked_statuses = {
        "checkpoint_failed",
        "dispatch_failed",
    }

    return DashboardMetrics(
        investigations=len(dataframe),
        open_investigations=int((statuses == INVESTIGATION_STATUS_OPEN).sum()),
        blocked_investigations=int(statuses.isin(blocked_statuses).sum()),
        completed_jobs=int((job_statuses == JOB_STATUS_COMPLETED).sum()),
    )


# -----------------------------------------------------------------------------
# Prediction form
#
# The Dashboard never loads MLflow or the sklearn pipeline itself. It is an
# HTTP client to Model Service's /predict endpoint, same as it is a read
# client to Postgres for investigations — inference ownership stays inside
# Model Service.
# -----------------------------------------------------------------------------

JOB_OPTIONS: Final[tuple[str, ...]] = (
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
)

MARITAL_OPTIONS: Final[tuple[str, ...]] = ("divorced", "married", "single", "unknown")

EDUCATION_OPTIONS: Final[tuple[str, ...]] = (
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
)

YES_NO_UNKNOWN_OPTIONS: Final[tuple[str, ...]] = ("no", "unknown", "yes")

CONTACT_OPTIONS: Final[tuple[str, ...]] = ("cellular", "telephone")

MONTH_OPTIONS: Final[tuple[str, ...]] = (
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

DAY_OF_WEEK_OPTIONS: Final[tuple[str, ...]] = ("mon", "tue", "wed", "thu", "fri")

POUTCOME_OPTIONS: Final[tuple[str, ...]] = ("failure", "nonexistent", "success")

# duration is part of Model Service's /predict schema (the registered
# pipeline's domain transformer requires the column to be present), but it
# is dropped internally before the classifier ever sees it — a known
# leakage feature, since call duration is only known after a call ends.
# Asking a user to guess it for a not-yet-made call would be meaningless,
# so the form does not expose it and a fixed placeholder is sent instead.
IGNORED_DURATION_PLACEHOLDER: Final[int] = 0

NEVER_PREVIOUSLY_CONTACTED_SENTINEL: Final[int] = 999


class PredictionRequestError(Exception):
    """
    Raised when Model Service's /predict call fails for any reason.

    Carries a message already safe to show directly in the UI.
    """


def build_prediction_payload(
    *,
    age: int,
    job: str,
    marital: str,
    education: str,
    default: str,
    housing: str,
    loan: str,
    contact: str,
    month: str,
    day_of_week: str,
    campaign: int,
    pdays: int,
    previous: int,
    poutcome: str,
    emp_var_rate: float,
    cons_price_idx: float,
    cons_conf_idx: float,
    euribor3m: float,
    nr_employed: float,
) -> dict[str, object]:
    """
    Build the exact raw payload Model Service's PredictionRequest expects.

    A pure function, kept separate from the Streamlit form so the field
    mapping (in particular the dotted keys the registered pipeline expects)
    is directly testable without a running Streamlit session.
    """
    return {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": IGNORED_DURATION_PLACEHOLDER,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }


def call_predict_endpoint(
    *,
    base_url: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """
    Call Model Service's /predict endpoint and return the parsed JSON body.

    Raises PredictionRequestError on any failure — network error, non-2xx
    response, or an unparseable body — so the caller has one error path to
    render rather than several exception types to catch.
    """
    url = f"{base_url.rstrip('/')}/predict"

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise PredictionRequestError(
            f"Model Service rejected the request (HTTP {exc.response.status_code}): "
            f"{detail}"
        ) from exc
    except httpx.RequestError as exc:
        raise PredictionRequestError(f"Could not reach Model Service: {exc}") from exc

    return response.json()


def render_prediction_form(
    model_service_url: str,
) -> None:
    st.subheader("Try the Model")

    st.caption(
        "Score a hypothetical customer against the currently registered "
        "model. Inference happens inside Model Service, not here — the "
        "Dashboard only sends the request and renders the response."
    )

    with st.form("prediction_form"):
        column_1, column_2, column_3, column_4 = st.columns(4)

        with column_1:
            age = st.number_input("Age", min_value=17, max_value=120, value=35)
            job = st.selectbox(
                "Job", JOB_OPTIONS, index=JOB_OPTIONS.index("technician")
            )
            marital = st.selectbox(
                "Marital status",
                MARITAL_OPTIONS,
                index=MARITAL_OPTIONS.index("married"),
            )

        with column_2:
            education = st.selectbox(
                "Education",
                EDUCATION_OPTIONS,
                index=EDUCATION_OPTIONS.index("university.degree"),
            )
            default = st.selectbox(
                "Credit in default?",
                YES_NO_UNKNOWN_OPTIONS,
                index=YES_NO_UNKNOWN_OPTIONS.index("no"),
            )
            housing = st.selectbox(
                "Housing loan?",
                YES_NO_UNKNOWN_OPTIONS,
                index=YES_NO_UNKNOWN_OPTIONS.index("yes"),
            )

        with column_3:
            loan = st.selectbox(
                "Personal loan?",
                YES_NO_UNKNOWN_OPTIONS,
                index=YES_NO_UNKNOWN_OPTIONS.index("no"),
            )
            contact = st.selectbox(
                "Contact type", CONTACT_OPTIONS, index=CONTACT_OPTIONS.index("cellular")
            )
            month = st.selectbox(
                "Month of last contact", MONTH_OPTIONS, index=MONTH_OPTIONS.index("may")
            )

        with column_4:
            day_of_week = st.selectbox(
                "Day of last contact",
                DAY_OF_WEEK_OPTIONS,
                index=DAY_OF_WEEK_OPTIONS.index("mon"),
            )
            campaign = st.number_input("Contacts this campaign", min_value=1, value=2)
            previous = st.number_input(
                "Contacts before this campaign", min_value=0, value=0
            )

        st.divider()

        column_5, column_6 = st.columns(2)

        with column_5:
            never_previously_contacted = st.checkbox(
                "Never previously contacted", value=True
            )

            if never_previously_contacted:
                pdays = NEVER_PREVIOUSLY_CONTACTED_SENTINEL
            else:
                pdays = st.number_input(
                    "Days since last previous contact",
                    min_value=0,
                    max_value=998,
                    value=3,
                )

            poutcome = st.selectbox(
                "Outcome of previous campaign",
                POUTCOME_OPTIONS,
                index=POUTCOME_OPTIONS.index("nonexistent"),
            )

        with column_6:
            st.caption(
                "Economic indicators at the time of contact "
                "(quarterly / monthly / daily figures)."
            )
            emp_var_rate = st.number_input(
                "Employment variation rate", value=1.1, format="%.2f"
            )
            cons_price_idx = st.number_input(
                "Consumer price index", value=93.994, format="%.3f"
            )
            cons_conf_idx = st.number_input(
                "Consumer confidence index", value=-36.4, format="%.1f"
            )
            euribor3m = st.number_input(
                "Euribor 3-month rate", value=4.857, format="%.3f"
            )
            nr_employed = st.number_input(
                "Number of employees", value=5191.0, format="%.1f"
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    payload = build_prediction_payload(
        age=int(age),
        job=job,
        marital=marital,
        education=education,
        default=default,
        housing=housing,
        loan=loan,
        contact=contact,
        month=month,
        day_of_week=day_of_week,
        campaign=int(campaign),
        pdays=int(pdays),
        previous=int(previous),
        poutcome=poutcome,
        emp_var_rate=float(emp_var_rate),
        cons_price_idx=float(cons_price_idx),
        cons_conf_idx=float(cons_conf_idx),
        euribor3m=float(euribor3m),
        nr_employed=float(nr_employed),
    )

    try:
        result = call_predict_endpoint(
            base_url=model_service_url,
            payload=payload,
        )
    except PredictionRequestError as exc:
        logger.error("Prediction request failed: %s", exc)
        st.error(str(exc))
        return

    render_prediction_result(result)


def render_prediction_result(
    result: dict[str, object],
) -> None:
    probability = float(result["probability"])
    threshold = float(result["threshold"])
    prediction_label = str(result["prediction_label"])

    st.divider()
    st.subheader("Prediction")

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric("Probability of subscription", f"{probability:.2%}")
    column_2.metric("Operating threshold", f"{threshold:.2%}")

    with column_3:
        if prediction_label == "yes":
            st.success(f"Decision: {prediction_label.upper()}")
        else:
            st.info(f"Decision: {prediction_label.upper()}")

    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(f"Model: {result['model_name']} version {result['model_version']}")


# -----------------------------------------------------------------------------
# Streamlit rendering
# -----------------------------------------------------------------------------


def render_header() -> None:
    st.title(APP_TITLE)

    st.caption(
        "Read-only operational view of drift investigations "
        "and their latest retrain execution."
    )


def render_metrics(
    metrics: DashboardMetrics,
) -> None:
    columns = st.columns(
        4,
        border=True,
    )

    columns[0].metric(
        "Investigations",
        metrics.investigations,
    )

    columns[1].metric(
        "Open",
        metrics.open_investigations,
    )

    columns[2].metric(
        "Blocked",
        metrics.blocked_investigations,
    )

    columns[3].metric(
        "Completed Jobs",
        metrics.completed_jobs,
    )


def render_investigation_table(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        st.info(
            "No investigations exist yet. Trigger the Model "
            "Service drift endpoint to exercise the walking skeleton."
        )
        return

    st.dataframe(
        dataframe,
        width="stretch",
        height="auto",
        hide_index=True,
        column_order=[
            "Investigation ID",
            "Model",
            "Model Version",
            "Report ID",
            "Severity",
            "Investigation Status",
            "Latest Job ID",
            "Latest Job Status",
            "Attempts",
            "Resulting Model Version",
            "Updated At (UTC)",
        ],
        column_config={
            "Investigation ID": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Model": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Model Version": (
                st.column_config.TextColumn(
                    width="small",
                )
            ),
            "Report ID": (
                st.column_config.TextColumn(
                    width="large",
                )
            ),
            "Severity": (
                st.column_config.TextColumn(
                    width="small",
                )
            ),
            "Investigation Status": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Latest Job ID": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Latest Job Status": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Attempts": (
                st.column_config.NumberColumn(
                    width="small",
                    format="%d",
                )
            ),
            "Resulting Model Version": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
            "Updated At (UTC)": (
                st.column_config.TextColumn(
                    width="medium",
                )
            ),
        },
    )


def render_last_refresh_time() -> None:
    refreshed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    st.caption(f"Last refreshed: {refreshed_at}")


# -----------------------------------------------------------------------------
# Application entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    render_header()

    # Streamlit reruns the script on button interaction. The button does not
    # require custom callback logic because the database query is intentionally
    # executed on every rerun.
    st.button(
        "Refresh",
        type="primary",
        width="content",
    )

    try:
        settings = Settings.from_environment()
        engine = create_database_engine(settings.database_url)

        verify_database_connection(engine)

        rows = load_investigations(engine)

    except RuntimeError as exc:
        logger.error(
            "Dashboard configuration error: %s",
            exc,
        )

        st.error(
            "Dashboard configuration is incomplete. Check the service logs for details."
        )

        st.stop()

    except SQLAlchemyError:
        logger.exception("Database operation failed while loading Dashboard data")

        st.error("Postgres is currently unavailable or the Dashboard query failed.")

        st.stop()

    except Exception:
        logger.exception("Unexpected Dashboard failure")

        st.error(
            "The Dashboard encountered an unexpected error. "
            "Check the service logs for details."
        )

        st.stop()

    dataframe = build_dataframe(rows)

    metrics = calculate_metrics(dataframe)

    render_metrics(metrics)

    st.divider()

    render_investigation_table(dataframe)

    st.divider()

    render_prediction_form(settings.model_service_url)

    render_last_refresh_time()


if __name__ == "__main__":
    main()
