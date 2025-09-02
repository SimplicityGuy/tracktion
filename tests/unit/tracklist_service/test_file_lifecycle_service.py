"""Unit tests for file lifecycle service."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.services.file_lifecycle_service import FileLifecycleService
from shared.core_types.src.models import Recording


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def lifecycle_service(mock_session):
    """Create file lifecycle service instance."""
    return FileLifecycleService(session=mock_session)


@pytest.fixture
def sample_recording():
    """Create a sample recording."""
    recording = MagicMock(spec=Recording)
    recording.id = uuid4()
    recording.file_path = "/music/test.mp3"
    recording.file_name = "test.mp3"
    recording.sha256_hash = "abc123"
    recording.xxh128_hash = "def456"
    recording.file_size = 1024
    recording.deleted_at = None
    recording.processing_status = "completed"
    return recording


class TestFileLifecycleService:
    """Test FileLifecycleService methods."""

    @pytest.mark.asyncio
    async def test_handle_file_created_new(self, lifecycle_service, mock_session):
        """Test handling creation of a new file."""
        # Setup: No existing recording
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_created(
            file_path="/music/new.mp3",
            sha256_hash="hash123",
            xxh128_hash="hash456",
            file_size=2048,
        )

        # Verify
        assert success is True
        assert error is None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_created_existing(self, lifecycle_service, mock_session, sample_recording):
        """Test handling creation of a file that already exists."""
        # Setup: Existing recording found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_created(
            file_path="/music/different.mp3",
            sha256_hash="abc123",  # Same hash as existing
            xxh128_hash="def456",
            file_size=1024,
        )

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.file_path == "/music/different.mp3"
        assert sample_recording.file_name == "different.mp3"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_modified_existing(self, lifecycle_service, mock_session, sample_recording):
        """Test handling modification of an existing file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_modified(
            file_path="/music/test.mp3",
            sha256_hash="newhash123",
            xxh128_hash="newhash456",
            file_size=4096,
        )

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.sha256_hash == "newhash123"
        assert sample_recording.xxh128_hash == "newhash456"
        assert sample_recording.file_size == 4096
        assert sample_recording.processing_status == "pending"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_modified_not_found(self, lifecycle_service, mock_session):
        """Test handling modification of a file not in database."""
        # Setup: No recording found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test - should call handle_file_created
        with patch.object(lifecycle_service, "handle_file_created") as mock_create:
            mock_create.return_value = (True, None)

            success, error = await lifecycle_service.handle_file_modified(
                file_path="/music/new.mp3",
                sha256_hash="hash123",
                xxh128_hash="hash456",
                file_size=2048,
            )

            # Verify
            mock_create.assert_called_once_with("/music/new.mp3", "hash123", "hash456", 2048)

    @pytest.mark.asyncio
    async def test_handle_file_deleted_soft(self, lifecycle_service, mock_session, sample_recording):
        """Test soft deletion of a file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_deleted(file_path="/music/test.mp3", soft_delete=True)

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.deleted_at is not None
        assert sample_recording.processing_status == "deleted"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_deleted_hard(self, lifecycle_service, mock_session, sample_recording):
        """Test hard deletion of a file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_deleted(file_path="/music/test.mp3", soft_delete=False)

        # Verify
        assert success is True
        assert error is None
        mock_session.delete.assert_called_once_with(sample_recording)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_deleted_not_found(self, lifecycle_service, mock_session):
        """Test deletion of a file not in database."""
        # Setup: No recording found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_deleted(file_path="/music/missing.mp3", soft_delete=True)

        # Verify - should still succeed but with warning
        assert success is True
        assert error is None
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_moved_existing(self, lifecycle_service, mock_session, sample_recording):
        """Test handling file move for existing recording."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_moved(
            old_path="/music/test.mp3",
            new_path="/music/moved/test.mp3",
            sha256_hash="abc123",
            xxh128_hash="def456",
        )

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.file_path == "/music/moved/test.mp3"
        assert sample_recording.file_name == "test.mp3"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_moved_not_found(self, lifecycle_service, mock_session):
        """Test handling file move for non-existent recording."""
        # Setup: No recording found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test - should call handle_file_created
        with patch.object(lifecycle_service, "handle_file_created") as mock_create:
            mock_create.return_value = (True, None)

            success, error = await lifecycle_service.handle_file_moved(
                old_path="/music/old.mp3",
                new_path="/music/new.mp3",
                sha256_hash="hash123",
                xxh128_hash="hash456",
            )

            # Verify
            mock_create.assert_called_once_with("/music/new.mp3", "hash123", "hash456")

    @pytest.mark.asyncio
    async def test_handle_file_renamed(self, lifecycle_service, mock_session, sample_recording):
        """Test handling file rename (delegates to move)."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.handle_file_renamed(
            old_path="/music/test.mp3",
            new_path="/music/renamed.mp3",
            sha256_hash="abc123",
            xxh128_hash="def456",
        )

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.file_path == "/music/renamed.mp3"
        assert sample_recording.file_name == "renamed.mp3"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_soft_deleted(self, lifecycle_service, mock_session, sample_recording):
        """Test recovering a soft-deleted file."""
        # Setup: Recording is soft-deleted
        sample_recording.deleted_at = datetime.now(UTC)
        sample_recording.processing_status = "deleted"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_recording
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.recover_soft_deleted(file_path="/music/test.mp3")

        # Verify
        assert success is True
        assert error is None
        assert sample_recording.deleted_at is None
        assert sample_recording.processing_status == "pending"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_soft_deleted_not_found(self, lifecycle_service, mock_session):
        """Test recovering a file that isn't soft-deleted."""
        # Setup: No soft-deleted recording found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        success, error = await lifecycle_service.recover_soft_deleted(file_path="/music/missing.mp3")

        # Verify
        assert success is False
        assert error == "Recording not found"

    @pytest.mark.asyncio
    async def test_cleanup_old_soft_deletes(self, lifecycle_service, mock_session):
        """Test cleanup of old soft-deleted records."""
        # Setup: Create old soft-deleted recordings
        old_recording1 = MagicMock(spec=Recording)
        old_recording1.file_path = "/music/old1.mp3"
        old_recording1.deleted_at = datetime.now(UTC) - timedelta(days=31)

        old_recording2 = MagicMock(spec=Recording)
        old_recording2.file_path = "/music/old2.mp3"
        old_recording2.deleted_at = datetime.now(UTC) - timedelta(days=45)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            old_recording1,
            old_recording2,
        ]
        mock_session.execute.return_value = mock_result

        # Test
        count = await lifecycle_service.cleanup_old_soft_deletes(days_old=30)

        # Verify
        assert count == 2
        assert mock_session.delete.call_count == 2
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_created_error(self, lifecycle_service, mock_session):
        """Test error handling in file creation."""
        # Setup: Database error
        mock_session.execute.side_effect = Exception("Database error")

        # Test
        success, error = await lifecycle_service.handle_file_created(
            file_path="/music/error.mp3", sha256_hash="hash123"
        )

        # Verify
        assert success is False
        assert "Database error" in error
        mock_session.rollback.assert_called_once()
