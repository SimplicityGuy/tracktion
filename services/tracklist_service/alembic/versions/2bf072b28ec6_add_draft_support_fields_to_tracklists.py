"""Add draft support fields to tracklists

Revision ID: 2bf072b28ec6
Revises: 001
Create Date: 2025-08-27 17:01:03.244508

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op  # type: ignore[attr-defined]  # Alembic adds op at runtime

# revision identifiers, used by Alembic.
revision: str = "2bf072b28ec6"
down_revision: str | Sequence[str] | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add columns to existing tracklists table
    op.add_column("tracklists", sa.Column("draft_version", sa.Integer(), nullable=True))
    op.add_column(
        "tracklists",
        sa.Column("is_draft", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "tracklists",
        sa.Column(
            "parent_tracklist_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Add foreign key constraint for parent_tracklist_id
    op.create_foreign_key(
        "fk_tracklists_parent_tracklist_id",
        "tracklists",
        "tracklists",
        ["parent_tracklist_id"],
        ["id"],
    )

    # Add indexes for draft queries
    op.create_index("idx_tracklists_draft", "tracklists", ["is_draft", "audio_file_id"])
    op.create_index("idx_tracklists_versions", "tracklists", ["parent_tracklist_id"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index("idx_tracklists_versions", table_name="tracklists")
    op.drop_index("idx_tracklists_draft", table_name="tracklists")

    # Drop foreign key constraint
    op.drop_constraint("fk_tracklists_parent_tracklist_id", "tracklists", type_="foreignkey")

    # Drop columns
    op.drop_column("tracklists", "parent_tracklist_id")
    op.drop_column("tracklists", "is_draft")
    op.drop_column("tracklists", "draft_version")
