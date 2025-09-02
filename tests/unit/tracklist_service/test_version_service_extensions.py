"""Tests for version service extensions (Task 6)."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.messaging.sync_event_consumer import SyncEventConsumer
from services.tracklist_service.src.messaging.sync_message_schemas import VersionRollbackRequest
from services.tracklist_service.src.models.synchronization import TracklistVersion
from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.version_service import VersionService


class TestVersionServiceExtensions:
    """Test version service extension methods."""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def version_service(self, mock_session):
        """Create version service with mock session."""
        return VersionService(mock_session)

    @pytest.mark.asyncio
    async def test_get_version_by_id_success(self, version_service, mock_session):
        """Test getting version by UUID successfully."""
        # Setup mock version
        version_id = uuid4()
        mock_version = Mock(spec=TracklistVersion)
        mock_version.id = version_id
        mock_version.version_number = 3
        mock_version.tracklist_id = uuid4()

        # Mock query execution
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_version
        mock_session.execute.return_value = mock_result

        # Test
        result = await version_service.get_version_by_id(version_id)

        # Verify
        assert result == mock_version
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_version_by_id_not_found(self, version_service, mock_session):
        """Test getting version by UUID when not found."""
        version_id = uuid4()

        # Mock query execution - no result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        result = await version_service.get_version_by_id(version_id)

        # Verify
        assert result is None
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_to_version_by_id_success(self, version_service, mock_session):
        """Test rollback to version by UUID successfully."""
        # Setup
        tracklist_id = uuid4()
        version_id = uuid4()
        version_number = 5

        # Mock version
        mock_version = Mock(spec=TracklistVersion)
        mock_version.id = version_id
        mock_version.version_number = version_number
        mock_version.tracklist_id = tracklist_id
        mock_version.tracks_snapshot = [{"position": 1, "title": "Test Track"}]

        # Mock tracklist
        mock_tracklist = Mock(spec=TracklistDB)
        mock_tracklist.id = tracklist_id
        mock_tracklist.tracks = []

        # Setup get_version_by_id mock
        with (
            patch.object(version_service, "get_version_by_id", return_value=mock_version),
            patch.object(version_service, "rollback_to_version", return_value=mock_tracklist) as mock_rollback,
        ):
            # Test
            result = await version_service.rollback_to_version_by_id(tracklist_id, version_id)

            # Verify
            assert result == mock_tracklist
            mock_rollback.assert_called_once_with(tracklist_id, version_number)

    @pytest.mark.asyncio
    async def test_rollback_to_version_by_id_version_not_found(self, version_service):
        """Test rollback when version not found."""
        tracklist_id = uuid4()
        version_id = uuid4()

        # Setup get_version_by_id to return None
        with patch.object(version_service, "get_version_by_id", return_value=None):
            # Test and expect ValueError
            with pytest.raises(ValueError) as exc_info:
                await version_service.rollback_to_version_by_id(tracklist_id, version_id)

            assert f"Version {version_id} not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rollback_to_version_by_id_wrong_tracklist(self, version_service):
        """Test rollback when version belongs to different tracklist."""
        tracklist_id = uuid4()
        other_tracklist_id = uuid4()
        version_id = uuid4()

        # Mock version with different tracklist_id
        mock_version = Mock(spec=TracklistVersion)
        mock_version.id = version_id
        mock_version.tracklist_id = other_tracklist_id

        # Setup get_version_by_id mock
        with patch.object(version_service, "get_version_by_id", return_value=mock_version):
            # Test and expect ValueError
            with pytest.raises(ValueError) as exc_info:
                await version_service.rollback_to_version_by_id(tracklist_id, version_id)

            assert f"Version {version_id} does not belong to tracklist {tracklist_id}" in str(exc_info.value)


class TestSyncEventConsumerVersionFix:
    """Test sync event consumer version mismatch fix."""

    @pytest.mark.asyncio
    async def test_handle_version_rollback_with_uuid(self):
        """Test version rollback using UUID instead of version number."""

        # Create consumer
        consumer = SyncEventConsumer(SessionLocal=AsyncMock)

        # Create request with version_id (UUID)
        tracklist_id = uuid4()
        version_id = uuid4()
        request = VersionRollbackRequest(
            tracklist_id=tracklist_id,
            version_id=version_id,
            create_backup=True,
            actor="test_user",
            reason="Test rollback",
        )

        # Mock session and version service
        mock_session = AsyncMock()
        mock_tracklist = Mock(spec=TracklistDB)

        with patch.object(consumer, "SessionLocal") as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session

            with patch(
                "services.tracklist_service.src.messaging.sync_event_consumer.VersionService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service.rollback_to_version_by_id.return_value = mock_tracklist
                mock_service_class.return_value = mock_service

                # Test
                await consumer._handle_version_rollback(request.model_dump())

                # Verify the new method is called with version_id
                mock_service.rollback_to_version_by_id.assert_called_once_with(
                    tracklist_id=tracklist_id,
                    version_id=version_id,
                )
