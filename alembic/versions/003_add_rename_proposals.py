"""Add rename proposals table

Revision ID: 003
Revises: 002_add_processing_fields
Create Date: 2025-08-19

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create rename_proposals table."""
    op.create_table(
        "rename_proposals",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, primary_key=True),
        sa.Column("recording_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_path", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("proposed_filename", sa.Text(), nullable=False),
        sa.Column("full_proposed_path", sa.Text(), nullable=False),
        sa.Column("confidence_score", sa.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("conflicts", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("warnings", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.ForeignKeyConstraint(
            ["recording_id"],
            ["recordings.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for better query performance
    op.create_index("ix_rename_proposals_recording_id", "rename_proposals", ["recording_id"])
    op.create_index("ix_rename_proposals_status", "rename_proposals", ["status"])
    op.create_index(
        "ix_rename_proposals_full_proposed_path",
        "rename_proposals",
        ["full_proposed_path"],
    )
    op.create_index("ix_rename_proposals_created_at", "rename_proposals", ["created_at"])


def downgrade() -> None:
    """Drop rename_proposals table."""
    op.drop_index("ix_rename_proposals_created_at", table_name="rename_proposals")
    op.drop_index("ix_rename_proposals_full_proposed_path", table_name="rename_proposals")
    op.drop_index("ix_rename_proposals_status", table_name="rename_proposals")
    op.drop_index("ix_rename_proposals_recording_id", table_name="rename_proposals")
    op.drop_table("rename_proposals")
