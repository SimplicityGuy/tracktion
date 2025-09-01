"""Unit tests for integrity validation functionality."""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from services.tracklist_service.src.utils.integrity_validator import IntegrityValidator


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return Mock()


@pytest.fixture
def validator(mock_session):
    """Create an IntegrityValidator instance with mock session."""
    return IntegrityValidator(mock_session)


class TestIntegrityValidator:
    """Test suite for IntegrityValidator."""

    def test_check_orphaned_records_no_orphans(self, validator, mock_session):
        """Test checking for orphaned records when none exist."""
        # Mock the database function result
        mock_result = [("metadata", 0), ("tracklists", 0), ("rename_proposals", 0)]
        mock_session.execute.return_value = mock_result

        result = validator.check_orphaned_records()

        assert result == {"metadata": 0, "tracklists": 0, "rename_proposals": 0}
        mock_session.execute.assert_called_once()

    def test_check_orphaned_records_with_orphans(self, validator, mock_session):
        """Test checking for orphaned records when some exist."""
        # Mock the database function result with orphaned records
        mock_result = [("metadata", 5), ("tracklists", 2), ("rename_proposals", 0)]
        mock_session.execute.return_value = mock_result

        with patch("services.tracklist_service.src.utils.integrity_validator.logger") as mock_logger:
            result = validator.check_orphaned_records()

        assert result == {"metadata": 5, "tracklists": 2, "rename_proposals": 0}

        # Verify warnings were logged for orphaned records
        assert mock_logger.warning.call_count == 2

    def test_check_orphaned_records_database_error(self, validator, mock_session):
        """Test handling of database errors when checking orphaned records."""
        mock_session.execute.side_effect = SQLAlchemyError("Database error")

        with pytest.raises(SQLAlchemyError):
            validator.check_orphaned_records()

    def test_clean_orphaned_records_dry_run(self, validator, mock_session):
        """Test cleaning orphaned records in dry run mode."""
        # Mock check_orphaned_records to be called in dry run
        mock_result = [("metadata", 3), ("tracklists", 1), ("rename_proposals", 0)]
        mock_session.execute.return_value = mock_result

        result = validator.clean_orphaned_records(dry_run=True)

        assert result == {"metadata": 3, "tracklists": 1, "rename_proposals": 0}
        # Verify only check was called, not clean
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_not_called()

    def test_clean_orphaned_records_actual_clean(self, validator, mock_session):
        """Test actually cleaning orphaned records."""
        # Mock the clean function result
        mock_result = [("metadata", 3), ("tracklists", 1), ("rename_proposals", 0)]
        mock_session.execute.return_value = mock_result

        with patch("services.tracklist_service.src.utils.integrity_validator.logger") as mock_logger:
            result = validator.clean_orphaned_records(dry_run=False)

        assert result == {"metadata": 3, "tracklists": 1, "rename_proposals": 0}

        # Verify commit was called
        mock_session.commit.assert_called_once()

        # Verify info logs for cleaned records
        assert mock_logger.info.call_count == 2

    def test_clean_orphaned_records_rollback_on_error(self, validator, mock_session):
        """Test that clean operation rolls back on error."""
        mock_session.execute.side_effect = SQLAlchemyError("Database error")

        with pytest.raises(SQLAlchemyError):
            validator.clean_orphaned_records(dry_run=False)

        mock_session.rollback.assert_called_once()

    def test_validate_foreign_keys(self, validator, mock_session):
        """Test foreign key validation."""
        # Mock the query result
        mock_result = [
            ("metadata_recording_id_fkey", "metadata", "c"),  # CASCADE
            ("tracklists_recording_id_fkey", "tracklists", "c"),  # CASCADE
            (
                "rename_proposals_recording_id_fkey",
                "rename_proposals",
                "a",
            ),  # NO ACTION
        ]
        mock_session.execute.return_value = mock_result

        with patch("services.tracklist_service.src.utils.integrity_validator.logger") as mock_logger:
            result = validator.validate_foreign_keys()

        assert len(result) == 3
        assert result[0] == ("metadata_recording_id_fkey", "metadata", True)
        assert result[1] == ("tracklists_recording_id_fkey", "tracklists", True)
        assert result[2] == (
            "rename_proposals_recording_id_fkey",
            "rename_proposals",
            False,
        )

        # Verify warning for non-CASCADE constraint
        mock_logger.warning.assert_called_once()

    def test_validate_check_constraints(self, validator, mock_session):
        """Test check constraint validation."""
        # Mock the query result
        mock_result = [
            ("ck_recordings_file_path_not_empty", "recordings"),
            ("ck_recordings_file_name_not_empty", "recordings"),
            ("ck_metadata_key_not_empty", "metadata"),
        ]
        mock_session.execute.return_value = mock_result

        result = validator.validate_check_constraints()

        assert len(result) == 3
        for _, _, is_valid in result:
            assert is_valid is True

    def test_validate_indexes(self, validator, mock_session):
        """Test index validation."""
        # Mock the query result with some missing indexes
        mock_result = [
            ("idx_metadata_recording_id", "metadata"),
            ("idx_tracklists_recording_id", "tracklists"),
            # Missing: idx_rename_proposals_recording_id
        ]
        mock_session.execute.return_value = mock_result

        with patch("services.tracklist_service.src.utils.integrity_validator.logger") as mock_logger:
            result = validator.validate_indexes()

        assert len(result) == 3
        assert result[0] == ("idx_metadata_recording_id", "metadata", True)
        assert result[1] == ("idx_tracklists_recording_id", "tracklists", True)
        assert result[2] == (
            "idx_rename_proposals_recording_id",
            "rename_proposals",
            False,
        )

        # Verify warning for missing index
        mock_logger.warning.assert_called_once()

    def test_run_full_validation_all_valid(self, validator, mock_session):
        """Test full validation when everything is valid."""
        # Mock all validation methods to return valid results
        with patch.object(validator, "check_orphaned_records") as mock_check:
            mock_check.return_value = {
                "metadata": 0,
                "tracklists": 0,
                "rename_proposals": 0,
            }

            with patch.object(validator, "validate_foreign_keys") as mock_fk:
                mock_fk.return_value = [
                    ("metadata_recording_id_fkey", "metadata", True),
                    ("tracklists_recording_id_fkey", "tracklists", True),
                ]

                with patch.object(validator, "validate_check_constraints") as mock_check_const:
                    mock_check_const.return_value = [
                        ("ck_recordings_file_path_not_empty", "recordings", True),
                    ]

                    with patch.object(validator, "validate_indexes") as mock_idx:
                        mock_idx.return_value = [
                            ("idx_metadata_recording_id", "metadata", True),
                        ]

                        result = validator.run_full_validation()

        assert result["is_valid"] is True
        assert result["orphaned_records"] == {
            "metadata": 0,
            "tracklists": 0,
            "rename_proposals": 0,
        }

    def test_run_full_validation_with_issues(self, validator, mock_session):
        """Test full validation when issues are found."""
        # Mock validation methods to return issues
        with patch.object(validator, "check_orphaned_records") as mock_check:
            mock_check.return_value = {
                "metadata": 5,
                "tracklists": 0,
                "rename_proposals": 0,
            }

            with patch.object(validator, "validate_foreign_keys") as mock_fk:
                mock_fk.return_value = [
                    ("metadata_recording_id_fkey", "metadata", False),  # Invalid
                ]

                with patch.object(validator, "validate_check_constraints") as mock_check_const:
                    mock_check_const.return_value = []

                    with patch.object(validator, "validate_indexes") as mock_idx:
                        mock_idx.return_value = [
                            ("idx_metadata_recording_id", "metadata", False),  # Missing
                        ]

                        result = validator.run_full_validation()

        assert result["is_valid"] is False
        assert result["orphaned_records"]["metadata"] == 5
