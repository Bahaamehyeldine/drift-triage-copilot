"""create webhook_receipts table

Revision ID: 8965ec030cda
Revises: b6417944f888
Create Date: 2026-07-22 16:38:05.899711

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8965ec030cda'
down_revision: Union[str, Sequence[str], None] = 'b6417944f888'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    op.create_table(
        "webhook_receipts",
        sa.Column(
            "report_id",
            sa.String(length=128),
            primary_key=True,
        ),
        sa.Column(
            "received_timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "processing_status",
            sa.String(length=32),
            nullable=False,
            server_default="received",
        ),
        sa.Column(
            "investigation_id",
            sa.String(length=64),
            sa.ForeignKey(
                "investigations.investigation_id",
                name="fk_webhook_receipts_investigation_id",
                ondelete="SET NULL",
            ),
            nullable=True,
        ),
        sa.Column(
            "failure_reason",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "processed_timestamp",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("webhook_receipts")