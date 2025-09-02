"""add_cue_file_models

Revision ID: 004_add_cue_file_models
Revises: 3af184c92ec7
Create Date: 2025-08-28 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "004_add_cue_file_models"
down_revision = "3af184c92ec7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""

    # Add default_cue_format column to tracklists table
    op.add_column("tracklists", sa.Column("default_cue_format", sa.String(20), nullable=True))

    # Create cue_files table
    op.create_table(
        "cue_files",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=sa.func.gen_random_uuid(),
        ),
        sa.Column("tracklist_id", UUID(as_uuid=True), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("format", sa.String(20), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), default=1, nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("format_metadata", sa.JSON(), nullable=False, default=sa.text("'{}'::json")),
        sa.ForeignKeyConstraint(["tracklist_id"], ["tracklists.id"]),
    )

    # Create cue_generation_jobs table
    op.create_table(
        "cue_generation_jobs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=sa.func.gen_random_uuid(),
        ),
        sa.Column("tracklist_id", UUID(as_uuid=True), nullable=False),
        sa.Column("format", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("cue_file_id", UUID(as_uuid=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("validation_report", sa.JSON(), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False, default=sa.text("'{}'::json")),
        sa.Column("progress", sa.Integer(), default=0, nullable=False),
        sa.ForeignKeyConstraint(["tracklist_id"], ["tracklists.id"]),
        sa.ForeignKeyConstraint(["cue_file_id"], ["cue_files.id"]),
    )

    # Create indexes for performance
    op.create_index("ix_cue_files_tracklist_id", "cue_files", ["tracklist_id"])
    op.create_index("ix_cue_files_format", "cue_files", ["format"])
    op.create_index("ix_cue_files_is_active", "cue_files", ["is_active"])
    op.create_index("ix_cue_files_created_at", "cue_files", ["created_at"])

    op.create_index("ix_cue_generation_jobs_tracklist_id", "cue_generation_jobs", ["tracklist_id"])
    op.create_index("ix_cue_generation_jobs_status", "cue_generation_jobs", ["status"])
    op.create_index("ix_cue_generation_jobs_created_at", "cue_generation_jobs", ["created_at"])

    # Add constraints
    op.create_check_constraint(
        "ck_cue_files_format",
        "cue_files",
        "format IN ('standard', 'cdj', 'traktor', 'serato', 'rekordbox', 'kodi')",
    )

    op.create_check_constraint(
        "ck_cue_generation_jobs_format",
        "cue_generation_jobs",
        "format IN ('standard', 'cdj', 'traktor', 'serato', 'rekordbox', 'kodi')",
    )

    op.create_check_constraint(
        "ck_cue_generation_jobs_status",
        "cue_generation_jobs",
        "status IN ('pending', 'processing', 'completed', 'failed')",
    )

    op.create_check_constraint(
        "ck_cue_generation_jobs_progress",
        "cue_generation_jobs",
        "progress >= 0 AND progress <= 100",
    )

    op.create_check_constraint("ck_cue_files_file_size", "cue_files", "file_size >= 0")

    op.create_check_constraint("ck_cue_files_version", "cue_files", "version >= 1")


def downgrade() -> None:
    """Downgrade database schema."""

    # Drop constraints
    op.drop_constraint("ck_cue_files_version", "cue_files")
    op.drop_constraint("ck_cue_files_file_size", "cue_files")
    op.drop_constraint("ck_cue_generation_jobs_progress", "cue_generation_jobs")
    op.drop_constraint("ck_cue_generation_jobs_status", "cue_generation_jobs")
    op.drop_constraint("ck_cue_generation_jobs_format", "cue_generation_jobs")
    op.drop_constraint("ck_cue_files_format", "cue_files")

    # Drop indexes
    op.drop_index("ix_cue_generation_jobs_created_at")
    op.drop_index("ix_cue_generation_jobs_status")
    op.drop_index("ix_cue_generation_jobs_tracklist_id")

    op.drop_index("ix_cue_files_created_at")
    op.drop_index("ix_cue_files_is_active")
    op.drop_index("ix_cue_files_format")
    op.drop_index("ix_cue_files_tracklist_id")

    # Drop tables
    op.drop_table("cue_generation_jobs")
    op.drop_table("cue_files")

    # Remove column from tracklists
    op.drop_column("tracklists", "default_cue_format")
