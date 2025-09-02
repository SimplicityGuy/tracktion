"""Add soft delete support to recordings table

Revision ID: 063942b5b3ea
Revises: 004_add_cue_file_models
Create Date: 2025-08-28 15:40:36.122375

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "063942b5b3ea"
down_revision: str | Sequence[str] | None = "004_add_cue_file_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add deleted_at column to recordings table
    op.add_column("recordings", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    # Create index on deleted_at for efficient queries
    op.create_index("ix_recordings_deleted_at", "recordings", ["deleted_at"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop index
    op.drop_index("ix_recordings_deleted_at", table_name="recordings")

    # Remove deleted_at column
    op.drop_column("recordings", "deleted_at")
