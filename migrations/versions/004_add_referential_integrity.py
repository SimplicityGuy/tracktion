"""Add referential integrity constraints and triggers.

Revision ID: 004_add_referential_integrity
Revises: 003_add_deleted_at_column
Create Date: 2024-01-28 10:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "004_add_referential_integrity"
down_revision = "003_add_deleted_at_column"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add referential integrity constraints and cascade triggers."""

    # Add ON DELETE CASCADE to foreign key constraints if not already present
    # Note: PostgreSQL automatically handles CASCADE when defined in the model,
    # but we'll add explicit constraints for safety

    # 1. Ensure metadata cascades on recording deletion
    op.execute(
        """
        ALTER TABLE metadata
        DROP CONSTRAINT IF EXISTS metadata_recording_id_fkey;
    """
    )

    op.execute(
        """
        ALTER TABLE metadata
        ADD CONSTRAINT metadata_recording_id_fkey
        FOREIGN KEY (recording_id)
        REFERENCES recordings(id)
        ON DELETE CASCADE;
    """
    )

    # 2. Ensure tracklist cascades on recording deletion
    op.execute(
        """
        ALTER TABLE tracklist
        DROP CONSTRAINT IF EXISTS tracklist_recording_id_fkey;
    """
    )

    op.execute(
        """
        ALTER TABLE tracklist
        ADD CONSTRAINT tracklist_recording_id_fkey
        FOREIGN KEY (recording_id)
        REFERENCES recordings(id)
        ON DELETE CASCADE;
    """
    )

    # 3. Ensure rename_proposals cascades on recording deletion
    op.execute(
        """
        ALTER TABLE rename_proposals
        DROP CONSTRAINT IF EXISTS rename_proposals_recording_id_fkey;
    """
    )

    op.execute(
        """
        ALTER TABLE rename_proposals
        ADD CONSTRAINT rename_proposals_recording_id_fkey
        FOREIGN KEY (recording_id)
        REFERENCES recordings(id)
        ON DELETE CASCADE;
    """
    )

    # 4. Create a trigger to clean up orphaned records automatically
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cleanup_orphaned_metadata()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Clean up metadata entries that reference non-existent recordings
            DELETE FROM metadata
            WHERE recording_id NOT IN (SELECT id FROM recordings WHERE deleted_at IS NULL);

            -- Clean up tracklist entries that reference non-existent recordings
            DELETE FROM tracklist
            WHERE recording_id NOT IN (SELECT id FROM recordings WHERE deleted_at IS NULL);

            -- Clean up rename proposals that reference non-existent recordings
            DELETE FROM rename_proposals
            WHERE recording_id NOT IN (SELECT id FROM recordings WHERE deleted_at IS NULL);

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # 5. Create trigger for soft delete cascade
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cascade_soft_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            -- When a recording is soft-deleted, mark related rename proposals as cancelled
            IF NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL THEN
                UPDATE rename_proposals
                SET status = 'cancelled',
                    updated_at = CURRENT_TIMESTAMP
                WHERE recording_id = NEW.id
                AND status = 'pending';
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # 6. Attach the soft delete trigger to recordings table
    op.execute(
        """
        CREATE TRIGGER trigger_cascade_soft_delete
        AFTER UPDATE OF deleted_at ON recordings
        FOR EACH ROW
        EXECUTE FUNCTION cascade_soft_delete();
    """
    )

    # 7. Create a function to validate referential integrity
    op.execute(
        """
        CREATE OR REPLACE FUNCTION validate_referential_integrity()
        RETURNS TABLE(
            table_name TEXT,
            orphaned_count BIGINT,
            details JSONB
        ) AS $$
        BEGIN
            -- Check for orphaned metadata
            RETURN QUERY
            SELECT
                'metadata'::TEXT as table_name,
                COUNT(*) as orphaned_count,
                jsonb_agg(jsonb_build_object(
                    'id', m.id,
                    'recording_id', m.recording_id,
                    'key', m.key
                )) as details
            FROM metadata m
            LEFT JOIN recordings r ON m.recording_id = r.id
            WHERE r.id IS NULL OR r.deleted_at IS NOT NULL
            GROUP BY table_name
            HAVING COUNT(*) > 0;

            -- Check for orphaned tracklist
            RETURN QUERY
            SELECT
                'tracklist'::TEXT as table_name,
                COUNT(*) as orphaned_count,
                jsonb_agg(jsonb_build_object(
                    'id', t.id,
                    'recording_id', t.recording_id,
                    'source', t.source
                )) as details
            FROM tracklist t
            LEFT JOIN recordings r ON t.recording_id = r.id
            WHERE r.id IS NULL OR r.deleted_at IS NOT NULL
            GROUP BY table_name
            HAVING COUNT(*) > 0;

            -- Check for orphaned rename_proposals
            RETURN QUERY
            SELECT
                'rename_proposals'::TEXT as table_name,
                COUNT(*) as orphaned_count,
                jsonb_agg(jsonb_build_object(
                    'id', rp.id,
                    'recording_id', rp.recording_id,
                    'status', rp.status
                )) as details
            FROM rename_proposals rp
            LEFT JOIN recordings r ON rp.recording_id = r.id
            WHERE r.id IS NULL OR r.deleted_at IS NOT NULL
            GROUP BY table_name
            HAVING COUNT(*) > 0;

            RETURN;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # 8. Create indexes for better performance on foreign key lookups
    op.create_index(
        "idx_metadata_recording_id_key",
        "metadata",
        ["recording_id", "key"],
        if_not_exists=True,
    )

    op.create_index("idx_tracklist_recording_id", "tracklist", ["recording_id"], if_not_exists=True)

    op.create_index(
        "idx_rename_proposals_recording_id_status",
        "rename_proposals",
        ["recording_id", "status"],
        if_not_exists=True,
    )

    # 9. Create a scheduled cleanup function for old soft-deleted records
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cleanup_old_soft_deleted_records(days_to_keep INTEGER DEFAULT 30)
        RETURNS TABLE(
            deleted_recordings BIGINT,
            deleted_metadata BIGINT,
            deleted_tracklist BIGINT,
            deleted_proposals BIGINT
        ) AS $$
        DECLARE
            cutoff_date TIMESTAMP WITH TIME ZONE;
            rec_count BIGINT;
            meta_count BIGINT;
            track_count BIGINT;
            prop_count BIGINT;
        BEGIN
            cutoff_date := CURRENT_TIMESTAMP - (days_to_keep || ' days')::INTERVAL;

            -- Delete old metadata for soft-deleted recordings
            WITH deleted_recordings AS (
                SELECT id FROM recordings
                WHERE deleted_at IS NOT NULL
                AND deleted_at < cutoff_date
            )
            DELETE FROM metadata
            WHERE recording_id IN (SELECT id FROM deleted_recordings);
            GET DIAGNOSTICS meta_count = ROW_COUNT;

            -- Delete old tracklist for soft-deleted recordings
            WITH deleted_recordings AS (
                SELECT id FROM recordings
                WHERE deleted_at IS NOT NULL
                AND deleted_at < cutoff_date
            )
            DELETE FROM tracklist
            WHERE recording_id IN (SELECT id FROM deleted_recordings);
            GET DIAGNOSTICS track_count = ROW_COUNT;

            -- Delete old proposals for soft-deleted recordings
            WITH deleted_recordings AS (
                SELECT id FROM recordings
                WHERE deleted_at IS NOT NULL
                AND deleted_at < cutoff_date
            )
            DELETE FROM rename_proposals
            WHERE recording_id IN (SELECT id FROM deleted_recordings);
            GET DIAGNOSTICS prop_count = ROW_COUNT;

            -- Finally, delete the old soft-deleted recordings
            DELETE FROM recordings
            WHERE deleted_at IS NOT NULL
            AND deleted_at < cutoff_date;
            GET DIAGNOSTICS rec_count = ROW_COUNT;

            RETURN QUERY SELECT rec_count, meta_count, track_count, prop_count;
        END;
        $$ LANGUAGE plpgsql;
    """
    )


def downgrade() -> None:
    """Remove referential integrity constraints and triggers."""

    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS trigger_cascade_soft_delete ON recordings;")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS cascade_soft_delete();")
    op.execute("DROP FUNCTION IF EXISTS cleanup_orphaned_metadata();")
    op.execute("DROP FUNCTION IF EXISTS validate_referential_integrity();")
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_soft_deleted_records(INTEGER);")

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_metadata_recording_id_key;")
    op.execute("DROP INDEX IF EXISTS idx_tracklist_recording_id;")
    op.execute("DROP INDEX IF EXISTS idx_rename_proposals_recording_id_status;")

    # Note: We don't remove the foreign key constraints as they're part of the base schema
