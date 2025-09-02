"""Add job tracking table.

Revision ID: 006
Revises: 005
Create Date: 2025-01-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create jobs table for tracking jobs across services."""
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("job_type", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="created"),
        sa.Column("service_name", sa.String(100), nullable=True),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("parent_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("context", postgresql.JSONB, nullable=True),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("progress", sa.Integer, nullable=True, server_default="0"),
        sa.Column("total_items", sa.Integer, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["parent_job_id"], ["jobs.id"], ondelete="CASCADE"),
    )

    # Create indexes for efficient querying
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_service_name", "jobs", ["service_name"])
    op.create_index("ix_jobs_correlation_id", "jobs", ["correlation_id"])
    op.create_index("ix_jobs_parent_job_id", "jobs", ["parent_job_id"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])
    op.create_index("ix_jobs_completed_at", "jobs", ["completed_at"])

    # Create composite index for common queries
    op.create_index("ix_jobs_status_service", "jobs", ["status", "service_name"])


def downgrade() -> None:
    """Drop jobs table."""
    # Drop indexes
    op.drop_index("ix_jobs_status_service", "jobs")
    op.drop_index("ix_jobs_completed_at", "jobs")
    op.drop_index("ix_jobs_created_at", "jobs")
    op.drop_index("ix_jobs_parent_job_id", "jobs")
    op.drop_index("ix_jobs_correlation_id", "jobs")
    op.drop_index("ix_jobs_service_name", "jobs")
    op.drop_index("ix_jobs_status", "jobs")
    op.drop_index("ix_jobs_job_type", "jobs")

    # Drop table
    op.drop_table("jobs")
