"""Add synchronization models for tracklist versioning.

Revision ID: 004
Revises: 003_add_rename_proposals
Create Date: 2025-08-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic
revision: str = "004"
down_revision: str | None = "003_add_rename_proposals"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create tables for synchronization and versioning."""
    # Create tracklist_versions table
    op.create_table(
        "tracklist_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tracklist_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("change_type", sa.String(length=50), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=False),
        sa.Column("tracks_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("version_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, default=False),
        sa.ForeignKeyConstraint(
            ["tracklist_id"],
            ["tracklists.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_tracklist_versions_tracklist_id"),
        "tracklist_versions",
        ["tracklist_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tracklist_versions_created_at"),
        "tracklist_versions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tracklist_versions_is_current"),
        "tracklist_versions",
        ["is_current"],
        unique=False,
    )

    # Create sync_configurations table
    op.create_table(
        "sync_configurations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tracklist_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, default=True),
        sa.Column("sync_sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sync_frequency", sa.String(length=20), nullable=True),
        sa.Column("auto_accept_threshold", sa.Float(), nullable=False, default=0.9),
        sa.Column(
            "conflict_resolution",
            sa.String(length=20),
            nullable=False,
            default="manual",
        ),
        sa.Column("last_sync_at", sa.DateTime(), nullable=True),
        sa.Column("next_sync_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["tracklist_id"],
            ["tracklists.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tracklist_id"),
    )

    # Create sync_events table
    op.create_table(
        "sync_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tracklist_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("changes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("conflict_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("resolution", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ["tracklist_id"],
            ["tracklists.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_sync_events_tracklist_id"),
        "sync_events",
        ["tracklist_id"],
        unique=False,
    )
    op.create_index(op.f("ix_sync_events_created_at"), "sync_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_sync_events_status"), "sync_events", ["status"], unique=False)

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column("changes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("audit_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_logs_entity_id"), "audit_logs", ["entity_id"], unique=False)
    op.create_index(op.f("ix_audit_logs_timestamp"), "audit_logs", ["timestamp"], unique=False)
    op.create_index(op.f("ix_audit_logs_entity_type"), "audit_logs", ["entity_type"], unique=False)


def downgrade() -> None:
    """Drop synchronization and versioning tables."""
    op.drop_index(op.f("ix_audit_logs_entity_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_timestamp"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_entity_id"), table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index(op.f("ix_sync_events_status"), table_name="sync_events")
    op.drop_index(op.f("ix_sync_events_created_at"), table_name="sync_events")
    op.drop_index(op.f("ix_sync_events_tracklist_id"), table_name="sync_events")
    op.drop_table("sync_events")

    op.drop_table("sync_configurations")

    op.drop_index(op.f("ix_tracklist_versions_is_current"), table_name="tracklist_versions")
    op.drop_index(op.f("ix_tracklist_versions_created_at"), table_name="tracklist_versions")
    op.drop_index(op.f("ix_tracklist_versions_tracklist_id"), table_name="tracklist_versions")
    op.drop_table("tracklist_versions")
