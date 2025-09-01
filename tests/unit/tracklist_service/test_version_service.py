"""Unit tests for version management service."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import TracklistVersion
from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.version_service import VersionService


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def version_service(mock_session):
    """Create version service instance."""
    return VersionService(mock_session)


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracklist = MagicMock(spec=TracklistDB)
    tracklist.id = uuid4()
    tracklist.tracks = [
        {"position": 1, "artist": "Artist 1", "title": "Track 1"},
        {"position": 2, "artist": "Artist 2", "title": "Track 2"},
    ]
    return tracklist


@pytest.fixture
def sample_version():
    """Create a sample version."""
    version = MagicMock(spec=TracklistVersion)
    version.id = uuid4()
    version.tracklist_id = uuid4()
    version.version_number = 1
    version.is_current = True
    version.tracks_snapshot = [{"position": 1, "artist": "Artist 1", "title": "Track 1"}]
    return version


class TestVersionService:
    """Test VersionService methods."""

    @pytest.mark.asyncio
    async def test_create_first_version(self, version_service, mock_session, sample_tracklist):
        """Test creating the first version of a tracklist."""
        tracklist_id = uuid4()

        # Mock no existing version
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session.get.return_value = sample_tracklist

        # Create version
        await version_service.create_version(
            tracklist_id=tracklist_id,
            change_type="manual_edit",
            change_summary="Initial version",
            created_by="user123",
        )

        # Verify session calls
        mock_session.add.assert_called()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

        # Verify version was added with correct attributes
        added_version = mock_session.add.call_args[0][0]
        assert isinstance(added_version, TracklistVersion)
        assert added_version.version_number == 1
        assert added_version.is_current is True
        assert added_version.change_type == "manual_edit"

    @pytest.mark.asyncio
    async def test_create_subsequent_version(self, version_service, mock_session, sample_version):
        """Test creating a subsequent version."""
        tracklist_id = sample_version.tracklist_id

        # Mock existing version
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none = Mock(return_value=sample_version)
        mock_session.execute = AsyncMock(return_value=mock_execute_result)

        # Create new version
        await version_service.create_version(
            tracklist_id=tracklist_id,
            change_type="import_update",
            change_summary="Updated from import",
            tracks_snapshot=[{"position": 1, "artist": "New", "title": "Track"}],
        )

        # Verify previous version marked as not current
        assert sample_version.is_current is False

        # Verify new version created
        assert mock_session.add.call_count == 2  # Previous + new
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_version(self, version_service, mock_session, sample_version):
        """Test getting the latest version."""
        tracklist_id = uuid4()

        # Mock query result
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none = Mock(return_value=sample_version)
        mock_session.execute = AsyncMock(return_value=mock_execute_result)

        result = await version_service.get_latest_version(tracklist_id)

        assert result == sample_version
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_specific_version(self, version_service, mock_session, sample_version):
        """Test getting a specific version by number."""
        tracklist_id = uuid4()
        version_number = 2

        # Mock query result
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none = Mock(return_value=sample_version)
        mock_session.execute = AsyncMock(return_value=mock_execute_result)

        result = await version_service.get_version(tracklist_id, version_number)

        assert result == sample_version
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_versions(self, version_service, mock_session):
        """Test listing versions for a tracklist."""
        tracklist_id = uuid4()

        # Create mock versions
        versions = [
            MagicMock(version_number=3),
            MagicMock(version_number=2),
            MagicMock(version_number=1),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = versions
        mock_session.execute.return_value = mock_result

        result = await version_service.list_versions(tracklist_id, limit=10)

        assert len(result) == 3
        assert result == versions
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, version_service, mock_session, sample_version, sample_tracklist):
        """Test rolling back to a previous version."""
        tracklist_id = sample_tracklist.id
        version_number = 1

        # Setup mocks
        sample_version.tracks_snapshot = [{"position": 1, "artist": "Old", "title": "Version"}]

        # Mock get_version
        with patch.object(version_service, "get_version", return_value=sample_version):
            # Mock get tracklist
            mock_session.get.return_value = sample_tracklist

            # Mock create_version
            with patch.object(version_service, "create_version") as mock_create:
                await version_service.rollback_to_version(tracklist_id, version_number)

                # Verify rollback version created
                mock_create.assert_called_once_with(
                    tracklist_id=tracklist_id,
                    change_type="rollback",
                    change_summary=f"Rolled back to version {version_number}",
                    tracks_snapshot=sample_version.tracks_snapshot,
                )

                # Verify tracklist updated
                assert sample_tracklist.tracks == sample_version.tracks_snapshot
                mock_session.add.assert_called_with(sample_tracklist)
                mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_version_not_found(self, version_service, mock_session):
        """Test rollback when version doesn't exist."""
        tracklist_id = uuid4()

        with (
            patch.object(version_service, "get_version", return_value=None),
            pytest.raises(ValueError, match="Version .* not found"),
        ):
            await version_service.rollback_to_version(tracklist_id, 99)

    @pytest.mark.asyncio
    async def test_get_version_diff(self, version_service):
        """Test getting differences between versions."""
        tracklist_id = uuid4()

        # Create two versions with different tracks
        v1 = MagicMock()
        v1.tracks_snapshot = [
            {"position": 1, "artist": "A1", "title": "T1"},
            {"position": 2, "artist": "A2", "title": "T2"},
        ]

        v2 = MagicMock()
        v2.tracks_snapshot = [
            {"position": 1, "artist": "A1", "title": "T1-modified"},  # Modified
            {"position": 3, "artist": "A3", "title": "T3"},  # Added
        ]

        with patch.object(version_service, "get_version") as mock_get:
            mock_get.side_effect = [v1, v2]

            diff = await version_service.get_version_diff(tracklist_id, 1, 2)

            assert len(diff["added"]) == 1  # Position 3
            assert len(diff["removed"]) == 1  # Position 2
            assert len(diff["modified"]) == 1  # Position 1
            assert diff["total_changes"] == 3

    @pytest.mark.asyncio
    async def test_get_version_diff_missing_version(self, version_service):
        """Test diff when one version doesn't exist."""
        tracklist_id = uuid4()

        with patch.object(version_service, "get_version") as mock_get:
            mock_get.side_effect = [None, MagicMock()]

            with pytest.raises(ValueError, match="One or both versions not found"):
                await version_service.get_version_diff(tracklist_id, 1, 2)

    @pytest.mark.asyncio
    async def test_prune_old_versions(self, version_service, mock_session):
        """Test pruning old versions based on retention policy."""
        tracklist_id = uuid4()
        cutoff = datetime.now(UTC) - timedelta(days=100)

        # Create mock old versions to delete
        old_versions = [
            MagicMock(id=uuid4(), created_at=cutoff - timedelta(days=10)),
            MagicMock(id=uuid4(), created_at=cutoff - timedelta(days=20)),
        ]

        # Mock keep versions query
        keep_result = MagicMock()
        keep_result.__iter__ = lambda x: iter([(uuid4(),), (uuid4(),)])

        # Mock delete query
        delete_result = MagicMock()
        delete_result.scalars.return_value.all.return_value = old_versions

        mock_session.execute.side_effect = [keep_result, delete_result]

        count = await version_service.prune_old_versions(tracklist_id, keep_count=5, keep_days=90)

        assert count == 2
        assert mock_session.delete.call_count == 2
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_version_without_tracklist(self, version_service, mock_session):
        """Test creating version when tracklist doesn't exist."""
        tracklist_id = uuid4()

        # Mock no existing version and no tracklist
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session.get.return_value = None

        with pytest.raises(ValueError, match="Tracklist .* not found"):
            await version_service.create_version(
                tracklist_id=tracklist_id,
                change_type="manual_edit",
                change_summary="Test",
            )
