"""add_referential_integrity_constraints

Revision ID: e05db30dd934
Revises: 063942b5b3ea
Create Date: 2025-08-28 17:00:30.829257

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "e05db30dd934"
down_revision: str | Sequence[str] | None = "063942b5b3ea"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add referential integrity constraints and database triggers."""

    # Add ON DELETE CASCADE to foreign key constraints if not already present
    # Note: These may already exist from model definitions, but we ensure they're set

    # Drop existing foreign key constraints to recreate with CASCADE
    op.drop_constraint("metadata_recording_id_fkey", "metadata", type_="foreignkey")
    op.drop_constraint("tracklists_recording_id_fkey", "tracklists", type_="foreignkey")
    op.drop_constraint("rename_proposals_recording_id_fkey", "rename_proposals", type_="foreignkey")

    # Recreate foreign keys with ON DELETE CASCADE
    op.create_foreign_key(
        "metadata_recording_id_fkey",
        "metadata",
        "recordings",
        ["recording_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "tracklists_recording_id_fkey",
        "tracklists",
        "recordings",
        ["recording_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "rename_proposals_recording_id_fkey",
        "rename_proposals",
        "recordings",
        ["recording_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Create trigger function for cascade operations logging
    op.execute(
        """
        CREATE OR REPLACE FUNCTION log_cascade_deletions()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Log deletion to a system table or external log
            -- For now, we'll just ensure proper cleanup
            RAISE NOTICE 'Cascade deletion: Table=%, RecordingId=%', TG_TABLE_NAME, OLD.recording_id;
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Create triggers on child tables to log cascade deletions
    op.execute(
        """
        CREATE TRIGGER metadata_cascade_delete_trigger
        BEFORE DELETE ON metadata
        FOR EACH ROW
        EXECUTE FUNCTION log_cascade_deletions();
    """
    )

    op.execute(
        """
        CREATE TRIGGER tracklists_cascade_delete_trigger
        BEFORE DELETE ON tracklists
        FOR EACH ROW
        EXECUTE FUNCTION log_cascade_deletions();
    """
    )

    op.execute(
        """
        CREATE TRIGGER rename_proposals_cascade_delete_trigger
        BEFORE DELETE ON rename_proposals
        FOR EACH ROW
        EXECUTE FUNCTION log_cascade_deletions();
    """
    )

    # Create function to check for orphaned records
    op.execute(
        """
        CREATE OR REPLACE FUNCTION check_orphaned_records()
        RETURNS TABLE(
            table_name TEXT,
            orphaned_count BIGINT
        ) AS $$
        BEGIN
            -- Check for metadata without parent recordings
            RETURN QUERY
            SELECT 'metadata'::TEXT, COUNT(*)
            FROM metadata m
            LEFT JOIN recordings r ON m.recording_id = r.id
            WHERE r.id IS NULL;

            -- Check for tracklists without parent recordings
            RETURN QUERY
            SELECT 'tracklists'::TEXT, COUNT(*)
            FROM tracklists t
            LEFT JOIN recordings r ON t.recording_id = r.id
            WHERE r.id IS NULL;

            -- Check for rename_proposals without parent recordings
            RETURN QUERY
            SELECT 'rename_proposals'::TEXT, COUNT(*)
            FROM rename_proposals rp
            LEFT JOIN recordings r ON rp.recording_id = r.id
            WHERE r.id IS NULL;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Create function to clean orphaned records (if any exist)
    op.execute(
        """
        CREATE OR REPLACE FUNCTION clean_orphaned_records()
        RETURNS TABLE(
            table_name TEXT,
            records_deleted BIGINT
        ) AS $$
        DECLARE
            deleted_count BIGINT;
        BEGIN
            -- Delete orphaned metadata
            DELETE FROM metadata m
            WHERE NOT EXISTS (
                SELECT 1 FROM recordings r WHERE r.id = m.recording_id
            );
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN QUERY SELECT 'metadata'::TEXT, deleted_count;

            -- Delete orphaned tracklists
            DELETE FROM tracklists t
            WHERE NOT EXISTS (
                SELECT 1 FROM recordings r WHERE r.id = t.recording_id
            );
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN QUERY SELECT 'tracklists'::TEXT, deleted_count;

            -- Delete orphaned rename_proposals
            DELETE FROM rename_proposals rp
            WHERE NOT EXISTS (
                SELECT 1 FROM recordings r WHERE r.id = rp.recording_id
            );
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN QUERY SELECT 'rename_proposals'::TEXT, deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Add check constraints to ensure data integrity
    op.create_check_constraint("ck_recordings_file_path_not_empty", "recordings", sa.text("file_path != ''"))

    op.create_check_constraint("ck_recordings_file_name_not_empty", "recordings", sa.text("file_name != ''"))

    op.create_check_constraint("ck_metadata_key_not_empty", "metadata", sa.text("key != ''"))

    # Create indexes for better query performance on foreign keys
    op.create_index("idx_metadata_recording_id", "metadata", ["recording_id"])
    op.create_index("idx_tracklists_recording_id", "tracklists", ["recording_id"])
    op.create_index("idx_rename_proposals_recording_id", "rename_proposals", ["recording_id"])


def downgrade() -> None:
    """Remove referential integrity constraints and triggers."""

    # Drop indexes
    op.drop_index("idx_rename_proposals_recording_id", "rename_proposals")
    op.drop_index("idx_tracklists_recording_id", "tracklists")
    op.drop_index("idx_metadata_recording_id", "metadata")

    # Drop check constraints
    op.drop_constraint("ck_metadata_key_not_empty", "metadata", type_="check")
    op.drop_constraint("ck_recordings_file_name_not_empty", "recordings", type_="check")
    op.drop_constraint("ck_recordings_file_path_not_empty", "recordings", type_="check")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS clean_orphaned_records();")
    op.execute("DROP FUNCTION IF EXISTS check_orphaned_records();")

    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS rename_proposals_cascade_delete_trigger ON rename_proposals;")
    op.execute("DROP TRIGGER IF EXISTS tracklists_cascade_delete_trigger ON tracklists;")
    op.execute("DROP TRIGGER IF EXISTS metadata_cascade_delete_trigger ON metadata;")

    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS log_cascade_deletions();")

    # Drop and recreate foreign keys without CASCADE
    op.drop_constraint("rename_proposals_recording_id_fkey", "rename_proposals", type_="foreignkey")
    op.drop_constraint("tracklists_recording_id_fkey", "tracklists", type_="foreignkey")
    op.drop_constraint("metadata_recording_id_fkey", "metadata", type_="foreignkey")

    op.create_foreign_key("metadata_recording_id_fkey", "metadata", "recordings", ["recording_id"], ["id"])

    op.create_foreign_key(
        "tracklists_recording_id_fkey",
        "tracklists",
        "recordings",
        ["recording_id"],
        ["id"],
    )

    op.create_foreign_key(
        "rename_proposals_recording_id_fkey",
        "rename_proposals",
        "recordings",
        ["recording_id"],
        ["id"],
    )
