# dashboard/main.py

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

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


@dataclass(frozen=True)
class Settings:
    """
    Runtime configuration loaded from environment variables.

    The Dashboard has one mandatory dependency: Postgres.
    """

    database_url: str

    @classmethod
    def from_environment(cls) -> Settings:
        database_url = os.getenv("DATABASE_URL", "").strip()

        if not database_url:
            raise RuntimeError("DATABASE_URL is not configured")

        return cls(
            database_url=database_url,
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

    render_last_refresh_time()


if __name__ == "__main__":
    main()
