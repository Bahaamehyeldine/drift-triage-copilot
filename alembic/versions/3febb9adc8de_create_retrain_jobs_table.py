"""create retrain_jobs table

Revision ID: 3febb9adc8de
Revises: 8965ec030cda
Create Date: 2026-07-22 16:50:03.857577

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3febb9adc8de'
down_revision: Union[str, Sequence[str], None] = '8965ec030cda'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrain_jobs",
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
                name="fk_retrain_jobs_investigation_id",
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
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="ck_retrain_jobs_attempt_count_nonnegative",
        ),
    )


def downgrade() -> None:
    op.drop_table("retrain_jobs")
