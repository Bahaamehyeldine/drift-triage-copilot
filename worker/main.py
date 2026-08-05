# worker/main.py

import logging
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass

import redis
import sqlalchemy as sa
from redis import Redis
from redis.exceptions import ResponseError
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

RETRAIN_JOB_STREAM = os.getenv(
    "RETRAIN_JOB_STREAM",
    "async-tools:retrain",
)

RETRAIN_CONSUMER_GROUP = os.getenv(
    "RETRAIN_CONSUMER_GROUP",
    "retrain-workers",
)

RETRAIN_CONSUMER_NAME = os.getenv(
    "RETRAIN_CONSUMER_NAME",
    "worker-1",
)

REDIS_BLOCK_MS = int(os.getenv("REDIS_BLOCK_MS", "5000"))

STUB_EXECUTION_SECONDS = float(os.getenv("STUB_EXECUTION_SECONDS", "1"))


# ---------------------------------------------------------------------------
# Database metadata
# ---------------------------------------------------------------------------

metadata = sa.MetaData()

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
        server_default="queued",
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
        server_default=sa.text("0"),
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
        server_default=sa.func.now(),
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
)


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainJobMessage:
    retrain_job_id: str
    investigation_id: str
    report_id: str
    model_name: str
    source_model_version: str


@dataclass(frozen=True)
class JobClaimResult:
    claimed: bool
    existing_status: str | None = None


# ---------------------------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------------------------

shutdown_requested = False


def request_shutdown(
    signal_number: int,
    _frame: object,
) -> None:
    global shutdown_requested

    shutdown_requested = True

    logger.info(
        "Shutdown requested: signal=%s",
        signal_number,
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def validate_configuration() -> None:
    missing = [
        name
        for name, value in {
            "DATABASE_URL": DATABASE_URL,
            "REDIS_URL": REDIS_URL,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required Worker configuration: " + ", ".join(missing)
        )

    if REDIS_BLOCK_MS <= 0:
        raise RuntimeError("REDIS_BLOCK_MS must be greater than zero")

    if STUB_EXECUTION_SECONDS < 0:
        raise RuntimeError("STUB_EXECUTION_SECONDS cannot be negative")


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def create_database_engine() -> sa.Engine:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")

    return sa.create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def create_redis_client() -> Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is not configured")

    return redis.from_url(
        REDIS_URL,
        decode_responses=True,
    )


def verify_database_connection(
    engine: sa.Engine,
) -> None:
    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 1"))


def ensure_consumer_group(
    redis_client: Redis,
) -> None:
    """
    Create the consumer group once.

    id="0" ensures the group can consume entries that were added before the
    Worker first started. mkstream=True creates the stream when it does not
    yet exist.
    """
    try:
        redis_client.xgroup_create(
            name=RETRAIN_JOB_STREAM,
            groupname=RETRAIN_CONSUMER_GROUP,
            id="0",
            mkstream=True,
        )

        logger.info(
            "Created Redis consumer group: stream=%s group=%s",
            RETRAIN_JOB_STREAM,
            RETRAIN_CONSUMER_GROUP,
        )

    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise

        logger.info(
            "Redis consumer group already exists: stream=%s group=%s",
            RETRAIN_JOB_STREAM,
            RETRAIN_CONSUMER_GROUP,
        )


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------


def require_message_field(
    fields: Mapping[str, str],
    field_name: str,
) -> str:
    value = fields.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid stream field: {field_name}")

    return value


def parse_retrain_job_message(
    fields: Mapping[str, str],
) -> RetrainJobMessage:
    job_type = require_message_field(
        fields,
        "job_type",
    )

    if job_type != "retrain":
        raise ValueError(f"Unsupported job_type: {job_type}")

    return RetrainJobMessage(
        retrain_job_id=require_message_field(
            fields,
            "retrain_job_id",
        ),
        investigation_id=require_message_field(
            fields,
            "investigation_id",
        ),
        report_id=require_message_field(
            fields,
            "report_id",
        ),
        model_name=require_message_field(
            fields,
            "model_name",
        ),
        source_model_version=require_message_field(
            fields,
            "source_model_version",
        ),
    )


# ---------------------------------------------------------------------------
# Durable idempotency claim
# ---------------------------------------------------------------------------


def claim_retrain_job(
    engine: sa.Engine,
    job: RetrainJobMessage,
) -> JobClaimResult:
    """
    Atomically claim a logical retrain job.

    PostgreSQL's primary-key constraint on retrain_job_id is the durable
    idempotency boundary. ON CONFLICT DO NOTHING ensures that concurrent
    workers cannot both claim the same logical job.
    """
    statement = (
        pg_insert(retrain_jobs)
        .values(
            retrain_job_id=job.retrain_job_id,
            investigation_id=job.investigation_id,
            model_name=job.model_name,
            source_model_version=job.source_model_version,
            job_status="running",
            worker_claimed_at=sa.func.now(),
            started_at=sa.func.now(),
            attempt_count=1,
        )
        .on_conflict_do_nothing(
            index_elements=[
                retrain_jobs.c.retrain_job_id,
            ],
        )
        .returning(
            retrain_jobs.c.retrain_job_id,
        )
    )

    with engine.begin() as connection:
        claimed_job_id = connection.execute(statement).scalar_one_or_none()

        if claimed_job_id is not None:
            return JobClaimResult(
                claimed=True,
            )

        existing_status = connection.execute(
            sa.select(
                retrain_jobs.c.job_status,
            ).where(retrain_jobs.c.retrain_job_id == job.retrain_job_id)
        ).scalar_one_or_none()

    return JobClaimResult(
        claimed=False,
        existing_status=existing_status,
    )


# ---------------------------------------------------------------------------
# Stub execution
# ---------------------------------------------------------------------------


def execute_stub_retraining(
    job: RetrainJobMessage,
) -> str:
    """
    Walking-skeleton retraining implementation.

    No model is trained. The Worker performs deterministic placeholder work
    and returns a synthetic resulting version so the full queue-to-database
    path can be verified.
    """
    logger.info(
        "Executing stub retraining: retrain_job_id=%s model=%s source_version=%s",
        job.retrain_job_id,
        job.model_name,
        job.source_model_version,
    )

    if STUB_EXECUTION_SECONDS:
        time.sleep(STUB_EXECUTION_SECONDS)

    return f"stub-{job.source_model_version}-retrained"


def mark_job_completed(
    engine: sa.Engine,
    *,
    retrain_job_id: str,
    resulting_model_version: str,
) -> None:
    statement = (
        sa.update(retrain_jobs)
        .where(retrain_jobs.c.retrain_job_id == retrain_job_id)
        .where(retrain_jobs.c.job_status == "running")
        .values(
            job_status="completed",
            completed_at=sa.func.now(),
            resulting_model_version=(resulting_model_version),
            failure_details=None,
            updated_at=sa.func.now(),
        )
    )

    with engine.begin() as connection:
        result = connection.execute(statement)

        if result.rowcount != 1:
            raise RuntimeError(
                "Job completion update affected "
                f"{result.rowcount} rows for "
                f"retrain_job_id={retrain_job_id}"
            )


def mark_job_failed(
    engine: sa.Engine,
    *,
    retrain_job_id: str,
    failure_details: str,
) -> None:
    statement = (
        sa.update(retrain_jobs)
        .where(retrain_jobs.c.retrain_job_id == retrain_job_id)
        .where(retrain_jobs.c.job_status == "running")
        .values(
            job_status="failed",
            completed_at=sa.func.now(),
            failure_details=failure_details[:4000],
            updated_at=sa.func.now(),
        )
    )

    with engine.begin() as connection:
        result = connection.execute(statement)

        if result.rowcount != 1:
            raise RuntimeError(
                "Job failure update affected "
                f"{result.rowcount} rows for "
                f"retrain_job_id={retrain_job_id}"
            )


# ---------------------------------------------------------------------------
# Redis acknowledgment
# ---------------------------------------------------------------------------


def acknowledge_message(
    redis_client: Redis,
    message_id: str,
) -> None:
    acknowledged_count = redis_client.xack(
        RETRAIN_JOB_STREAM,
        RETRAIN_CONSUMER_GROUP,
        message_id,
    )

    if acknowledged_count != 1:
        raise RuntimeError(
            "Redis XACK affected "
            f"{acknowledged_count} entries for "
            f"message_id={message_id}"
        )


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------


def process_message(
    *,
    engine: sa.Engine,
    redis_client: Redis,
    message_id: str,
    fields: Mapping[str, str],
) -> None:
    try:
        job = parse_retrain_job_message(fields)

    except ValueError:
        logger.exception(
            "Invalid retrain stream message left pending: message_id=%s fields=%s",
            message_id,
            dict(fields),
        )
        return

    try:
        claim = claim_retrain_job(
            engine,
            job,
        )

    except Exception:
        logger.exception(
            "Failed to claim retrain job; message left pending: "
            "message_id=%s retrain_job_id=%s",
            message_id,
            job.retrain_job_id,
        )
        return

    # ------------------------------------------------------------------
    # Duplicate delivery
    # ------------------------------------------------------------------

    if not claim.claimed:
        if claim.existing_status in {
            "running",
            "completed",
        }:
            logger.info(
                "Skipping duplicate retrain delivery: "
                "message_id=%s retrain_job_id=%s status=%s",
                message_id,
                job.retrain_job_id,
                claim.existing_status,
            )

            try:
                acknowledge_message(
                    redis_client,
                    message_id,
                )

            except Exception:
                logger.exception(
                    "Failed to acknowledge duplicate message: "
                    "message_id=%s retrain_job_id=%s",
                    message_id,
                    job.retrain_job_id,
                )

            return

        logger.warning(
            "Retrain job exists with unexpected status; "
            "message left pending for recovery: "
            "message_id=%s retrain_job_id=%s status=%s",
            message_id,
            job.retrain_job_id,
            claim.existing_status,
        )
        return

    # ------------------------------------------------------------------
    # First execution
    # ------------------------------------------------------------------

    try:
        resulting_model_version = execute_stub_retraining(job)

        mark_job_completed(
            engine,
            retrain_job_id=job.retrain_job_id,
            resulting_model_version=(resulting_model_version),
        )

    except Exception as exc:
        logger.exception(
            "Retrain job execution failed; "
            "message will remain pending: "
            "message_id=%s retrain_job_id=%s",
            message_id,
            job.retrain_job_id,
        )

        try:
            mark_job_failed(
                engine,
                retrain_job_id=job.retrain_job_id,
                failure_details=repr(exc),
            )

        except Exception:
            logger.exception(
                "Failed to persist retrain job failure: retrain_job_id=%s",
                job.retrain_job_id,
            )

        return

    # Persist completion before acknowledging Redis. If acknowledgment fails,
    # a later redelivery is safe because Postgres already records completed.
    try:
        acknowledge_message(
            redis_client,
            message_id,
        )

    except Exception:
        logger.exception(
            "Job completed but Redis acknowledgment failed; "
            "a later redelivery will be deduplicated: "
            "message_id=%s retrain_job_id=%s",
            message_id,
            job.retrain_job_id,
        )
        return

    logger.info(
        "Retrain job completed and acknowledged: "
        "message_id=%s retrain_job_id=%s "
        "resulting_model_version=%s",
        message_id,
        job.retrain_job_id,
        resulting_model_version,
    )


# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------


def consume_forever(
    *,
    engine: sa.Engine,
    redis_client: Redis,
) -> None:
    logger.info(
        "Worker started: stream=%s group=%s consumer=%s",
        RETRAIN_JOB_STREAM,
        RETRAIN_CONSUMER_GROUP,
        RETRAIN_CONSUMER_NAME,
    )

    while not shutdown_requested:
        try:
            messages = redis_client.xreadgroup(
                groupname=RETRAIN_CONSUMER_GROUP,
                consumername=RETRAIN_CONSUMER_NAME,
                streams={
                    RETRAIN_JOB_STREAM: ">",
                },
                count=1,
                block=REDIS_BLOCK_MS,
            )

        except Exception:
            logger.exception("Redis stream read failed; retrying")
            time.sleep(1)
            continue

        if not messages:
            continue

        for _stream_name, stream_messages in messages:
            for message_id, fields in stream_messages:
                process_message(
                    engine=engine,
                    redis_client=redis_client,
                    message_id=message_id,
                    fields=fields,
                )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    validate_configuration()

    signal.signal(
        signal.SIGTERM,
        request_shutdown,
    )
    signal.signal(
        signal.SIGINT,
        request_shutdown,
    )

    engine = create_database_engine()
    redis_client = create_redis_client()

    try:
        verify_database_connection(engine)
        redis_client.ping()

        logger.info("Worker infrastructure connections verified")

        ensure_consumer_group(redis_client)

        consume_forever(
            engine=engine,
            redis_client=redis_client,
        )

    finally:
        try:
            redis_client.close()
        except Exception:
            logger.exception("Failed to close Redis client cleanly")

        try:
            engine.dispose()
        except Exception:
            logger.exception("Failed to dispose database engine cleanly")

        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
