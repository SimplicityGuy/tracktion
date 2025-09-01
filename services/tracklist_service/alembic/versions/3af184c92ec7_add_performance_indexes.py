"""Add performance indexes for tracklist queries.

Revision ID: 3af184c92ec7
Revises: 2bf072b28ec6
Create Date: 2025-08-28 14:00:00.000000

"""

from collections.abc import Sequence

from alembic import op  # type: ignore[attr-defined]  # Alembic adds op at runtime

# revision identifiers, used by Alembic.
revision: str = "3af184c92ec7"
down_revision: str | None = "2bf072b28ec6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add performance indexes."""
    # Index for draft queries (most common query pattern)
    op.create_index(
        "ix_tracklists_draft_lookup",
        "tracklists",
        ["audio_file_id", "is_draft", "draft_version"],
        postgresql_using="btree",
    )

    # Index for published tracklist queries
    op.create_index(
        "ix_tracklists_published_lookup",
        "tracklists",
        ["audio_file_id", "is_draft"],
        postgresql_where="is_draft = false",
        postgresql_using="btree",
    )

    # Index for parent tracklist references (version history)
    op.create_index(
        "ix_tracklists_parent",
        "tracklists",
        ["parent_tracklist_id"],
        postgresql_using="btree",
    )

    # Index for source-based queries
    op.create_index(
        "ix_tracklists_source",
        "tracklists",
        ["source"],
        postgresql_using="btree",
    )

    # Index for updated_at (for cache invalidation and recent drafts)
    op.create_index(
        "ix_tracklists_updated_at",
        "tracklists",
        ["updated_at"],
        postgresql_using="btree",
    )

    # Composite index for draft listing queries
    op.create_index(
        "ix_tracklists_draft_list",
        "tracklists",
        ["audio_file_id", "is_draft", "parent_tracklist_id", "draft_version"],
        postgresql_using="btree",
    )


def downgrade() -> None:
    """Remove performance indexes."""
    op.drop_index("ix_tracklists_draft_list", table_name="tracklists")
    op.drop_index("ix_tracklists_updated_at", table_name="tracklists")
    op.drop_index("ix_tracklists_source", table_name="tracklists")
    op.drop_index("ix_tracklists_parent", table_name="tracklists")
    op.drop_index("ix_tracklists_published_lookup", table_name="tracklists")
    op.drop_index("ix_tracklists_draft_lookup", table_name="tracklists")
