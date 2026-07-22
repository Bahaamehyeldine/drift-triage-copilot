"""create investigations table

Revision ID: b6417944f888
Revises: 
Create Date: 2026-07-22 16:15:55.946008

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b6417944f888'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    op.create_table(
        "investigations",
        sa.Column("investigation_id", sa.String(length=64), primary_key=True),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("model_uri", sa.String(length=1024), nullable=False),
        sa.Column("triggering_report_id", sa.String(length=128), nullable=False),
        sa.Column("current_report_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="open",
        ),
        sa.Column("current_severity", sa.String(length=16), nullable=False),
        sa.Column("recommended_action", sa.String(length=64), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("stale_reason", sa.Text(), nullable=True),
        sa.Column("invalidation_reason", sa.Text(), nullable=True),
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
        sa.Column(
            "resolved_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "thread_id",
            name="uq_investigations_thread_id",
        ),
        sa.UniqueConstraint(
            "triggering_report_id",
            name="uq_investigations_triggering_report_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("investigations")