"""Add analysis results table

Revision ID: 005
Revises: 004
Create Date: 2025-09-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add analysis results table."""
    # Create analysis_results table
    op.create_table(
        "analysis_results",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "recording_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "analysis_type",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "result_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "confidence_score",
            sa.Numeric(precision=5, scale=4),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "processing_time_ms",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["recording_id"],
            ["recordings.id"],
            name=op.f("fk_analysis_results_recording_id_recordings"),
            ondelete="CASCADE",
        ),
    )

    # Create indexes for performance
    op.create_index(
        op.f("ix_analysis_results_recording_id"),
        "analysis_results",
        ["recording_id"],
    )
    op.create_index(
        op.f("ix_analysis_results_analysis_type"),
        "analysis_results",
        ["analysis_type"],
    )
    op.create_index(
        op.f("ix_analysis_results_status"),
        "analysis_results",
        ["status"],
    )
    op.create_index(
        op.f("ix_analysis_results_created_at"),
        "analysis_results",
        ["created_at"],
    )

    # Composite index for common queries
    op.create_index(
        op.f("ix_analysis_results_recording_type_status"),
        "analysis_results",
        ["recording_id", "analysis_type", "status"],
    )


def downgrade() -> None:
    """Remove analysis results table."""
    # Drop indexes
    op.drop_index(op.f("ix_analysis_results_recording_type_status"))
    op.drop_index(op.f("ix_analysis_results_created_at"))
    op.drop_index(op.f("ix_analysis_results_status"))
    op.drop_index(op.f("ix_analysis_results_analysis_type"))
    op.drop_index(op.f("ix_analysis_results_recording_id"))

    # Drop table
    op.drop_table("analysis_results")