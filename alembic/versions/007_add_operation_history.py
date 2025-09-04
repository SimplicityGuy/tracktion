"""Add operation history table for admin operations tracking.

Revision ID: 007
Revises: 006
Create Date: 2025-01-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create admin_operation_history table for tracking admin operations."""
    op.create_table(
        "admin_operation_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("operation_id", sa.String(100), nullable=False),
        sa.Column("operation_type", sa.String(50), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("admin_user", sa.String(100), nullable=False),
        sa.Column("details", postgresql.JSONB, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("affected_resources", postgresql.JSONB, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_id"),
    )

    # Create indexes for efficient querying
    op.create_index("ix_admin_operation_history_operation_id", "admin_operation_history", ["operation_id"])
    op.create_index("ix_admin_operation_history_operation_type", "admin_operation_history", ["operation_type"])
    op.create_index("ix_admin_operation_history_timestamp", "admin_operation_history", ["timestamp"])
    op.create_index("ix_admin_operation_history_admin_user", "admin_operation_history", ["admin_user"])

    # Create composite indexes for common queries
    op.create_index(
        "ix_admin_operation_history_user_type",
        "admin_operation_history",
        ["admin_user", "operation_type"],
    )
    op.create_index(
        "ix_admin_operation_history_timestamp_type",
        "admin_operation_history",
        ["timestamp", "operation_type"],
    )


def downgrade() -> None:
    """Drop admin_operation_history table."""
    # Drop indexes
    op.drop_index("ix_admin_operation_history_timestamp_type", "admin_operation_history")
    op.drop_index("ix_admin_operation_history_user_type", "admin_operation_history")
    op.drop_index("ix_admin_operation_history_admin_user", "admin_operation_history")
    op.drop_index("ix_admin_operation_history_timestamp", "admin_operation_history")
    op.drop_index("ix_admin_operation_history_operation_type", "admin_operation_history")
    op.drop_index("ix_admin_operation_history_operation_id", "admin_operation_history")

    # Drop table
    op.drop_table("admin_operation_history")
