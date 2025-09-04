"""Add tracklist model with confidence_score and timestamps

Revision ID: 001
Revises:
Create Date: 2025-08-27 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add tracklist model with enhanced schema."""
    # Create tracklists table with new fields
    op.create_table(
        "tracklists",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("audio_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column("tracks", sa.JSON(), nullable=False, default=sa.text("'[]'")),
        sa.Column("cue_file_path", sa.Text(), nullable=True),
        sa.Column("cue_file_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=False, default=1.0),
    )

    # Add foreign key constraint to recordings table
    op.create_foreign_key(
        "fk_tracklists_recording_id",
        "tracklists",
        "recordings",
        ["audio_file_id"],
        ["id"],
    )

    # Add indexes for performance
    op.create_index("ix_tracklists_audio_file_id", "tracklists", ["audio_file_id"])
    op.create_index("ix_tracklists_source", "tracklists", ["source"])
    op.create_index("ix_tracklists_created_at", "tracklists", ["created_at"])


def downgrade() -> None:
    """Remove tracklist model."""
    op.drop_index("ix_tracklists_created_at")
    op.drop_index("ix_tracklists_source")
    op.drop_index("ix_tracklists_audio_file_id")
    op.drop_constraint("fk_tracklists_recording_id", "tracklists", type_="foreignkey")
    op.drop_table("tracklists")
