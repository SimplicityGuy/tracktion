"""Unit tests for 1001tracklists sync service."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import SyncConfiguration
from services.tracklist_service.src.models.tracklist import TrackEntry, TracklistDB
from services.tracklist_service.src.services.tracklists_sync_service import TracklistsSyncService


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_import_service():
    """Create a mock import service."""
    return MagicMock()


@pytest.fixture
def mock_version_service():
    """Create a mock version service."""
    service = MagicMock()
    service.create_version = AsyncMock()
    return service


@pytest.fixture
def mock_audit_service():
    """Create a mock audit service."""
    service = MagicMock()
    service.log_tracklist_change = AsyncMock()
    return service


@pytest.fixture
def sync_service(mock_session, mock_import_service, mock_version_service, mock_audit_service):
    """Create sync service instance."""
    return TracklistsSyncService(
        session=mock_session,
        import_service=mock_import_service,
        version_service=mock_version_service,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracklist = MagicMock(spec=TracklistDB)
    tracklist.id = uuid4()
    tracklist.source = "1001tracklists"
    tracklist.import_url = "https://1001tracklists.com/tracklist/test"
    tracklist.tracks = [
        {"position": 1, "artist": "Artist 1", "title": "Track 1", "start_time": 0},
        {"position": 2, "artist": "Artist 2", "title": "Track 2", "start_time": 300},
    ]
    return tracklist


@pytest.fixture
def sample_tracks():
    """Create sample track entries."""
    return [
        TrackEntry(
            position=1,
            start_time=timedelta(seconds=0),
            artist="Artist 1",
            title="Track 1",
        ),
        TrackEntry(
            position=2,
            start_time=timedelta(seconds=300),
            artist="Artist 2",
            title="Track 2",
        ),
    ]


class TestTracklistsSyncService:
    """Test TracklistsSyncService methods."""

    @pytest.mark.asyncio
    async def test_check_for_updates_no_changes(
        self,
        sync_service,
        mock_session,
        mock_import_service,
        sample_tracklist,
        sample_tracks,
    ):
        """Test checking for updates when there are no changes."""
        tracklist_id = sample_tracklist.id

        # Mock database get
        mock_session.get.return_value = sample_tracklist

        # Mock scraped tracklist
        mock_scraped = MagicMock()
        mock_scraped.tracks = []
        mock_import_service.fetch_tracklist_from_1001.return_value = mock_scraped
        mock_import_service.transform_to_track_entries.return_value = sample_tracks

        result = await sync_service.check_for_updates(tracklist_id)

        assert result is None  # No changes detected

    @pytest.mark.asyncio
    async def test_check_for_updates_with_changes(
        self, sync_service, mock_session, mock_import_service, sample_tracklist
    ):
        """Test checking for updates when there are changes."""
        tracklist_id = sample_tracklist.id

        # Mock database get
        mock_session.get.return_value = sample_tracklist

        # Mock scraped tracklist with changes
        mock_scraped = MagicMock()
        mock_import_service.fetch_tracklist_from_1001.return_value = mock_scraped

        # Different tracks to simulate changes
        updated_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                artist="Artist 1",
                title="Track 1 (Extended)",  # Modified title
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(seconds=300),
                artist="Artist 2",
                title="Track 2",
            ),
            TrackEntry(
                position=3,
                start_time=timedelta(seconds=600),
                artist="Artist 3",
                title="Track 3",  # New track
            ),
        ]
        mock_import_service.transform_to_track_entries.return_value = updated_tracks

        result = await sync_service.check_for_updates(tracklist_id)

        assert result is not None
        assert result["has_updates"] is True
        assert len(result["changes"]["tracks_added"]) == 1  # Track 3
        assert len(result["changes"]["tracks_modified"]) == 1  # Track 1
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_check_for_updates_non_1001_source(self, sync_service, mock_session):
        """Test checking updates for non-1001tracklists source."""
        tracklist = MagicMock()
        tracklist.source = "manual"  # Not from 1001tracklists

        mock_session.get.return_value = tracklist

        result = await sync_service.check_for_updates(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_check_for_updates_no_url(self, sync_service, mock_session, sample_tracklist):
        """Test checking updates when no import URL is available."""
        sample_tracklist.import_url = None
        mock_session.get.return_value = sample_tracklist

        result = await sync_service.check_for_updates(sample_tracklist.id)

        assert result is None

    def test_compare_tracklists_no_changes(self, sync_service, sample_tracks):
        """Test comparing tracklists with no changes."""
        changes = sync_service._compare_tracklists(sample_tracks, sample_tracks)

        assert changes["has_changes"] is False
        assert changes["total_changes"] == 0

    def test_compare_tracklists_with_additions(self, sync_service, sample_tracks):
        """Test comparing tracklists with added tracks."""
        current = sample_tracks[:1]  # Only first track
        latest = sample_tracks  # Both tracks

        changes = sync_service._compare_tracklists(current, latest)

        assert changes["has_changes"] is True
        assert len(changes["tracks_added"]) == 1
        assert changes["tracks_added"][0]["position"] == 2

    def test_compare_tracklists_with_removals(self, sync_service, sample_tracks):
        """Test comparing tracklists with removed tracks."""
        current = sample_tracks  # Both tracks
        latest = sample_tracks[:1]  # Only first track

        changes = sync_service._compare_tracklists(current, latest)

        assert changes["has_changes"] is True
        assert len(changes["tracks_removed"]) == 1
        assert changes["tracks_removed"][0]["position"] == 2

    def test_compare_tracklists_with_modifications(self, sync_service):
        """Test comparing tracklists with modified tracks."""
        current = [
            TrackEntry(position=1, start_time=timedelta(0), artist="A1", title="T1"),
        ]
        latest = [
            TrackEntry(position=1, start_time=timedelta(0), artist="A1", title="T1 (Remix)"),
        ]

        changes = sync_service._compare_tracklists(current, latest)

        assert changes["has_changes"] is True
        assert len(changes["tracks_modified"]) == 1

    def test_track_differs(self, sync_service):
        """Test track difference detection."""
        track1 = TrackEntry(
            position=1,
            start_time=timedelta(0),
            artist="Artist",
            title="Title",
        )
        track2 = TrackEntry(
            position=1,
            start_time=timedelta(0),
            artist="Artist",
            title="Title (Remix)",  # Different
        )

        assert sync_service._track_differs(track1, track2) is True
        assert sync_service._track_differs(track1, track1) is False

    def test_calculate_change_confidence(self, sync_service):
        """Test confidence calculation for changes."""
        # No changes = high confidence
        changes = {"has_changes": False}
        assert sync_service._calculate_change_confidence(changes) == 1.0

        # Minor modifications = decent confidence
        changes = {
            "has_changes": True,
            "tracks_added": [],
            "tracks_removed": [],
            "tracks_modified": [1, 2],
        }
        confidence = sync_service._calculate_change_confidence(changes)
        assert 0.8 <= confidence <= 1.0

        # Major changes = lower confidence
        changes = {
            "has_changes": True,
            "tracks_added": [1, 2, 3, 4, 5, 6],
            "tracks_removed": [1, 2, 3, 4],
            "tracks_modified": [],
        }
        confidence = sync_service._calculate_change_confidence(changes)
        assert confidence < 0.8

    @pytest.mark.asyncio
    async def test_apply_updates_success(self, sync_service, mock_session, sample_tracklist):
        """Test successfully applying updates."""
        tracklist_id = sample_tracklist.id
        updates = {
            "confidence": 0.95,
            "changes": {
                "total_changes": 1,
                "tracks_added": [],
                "tracks_removed": [],
                "tracks_modified": [
                    {
                        "position": 1,
                        "old": {"position": 1, "title": "Old"},
                        "new": {"position": 1, "title": "New"},
                    }
                ],
            },
        }

        # Mock database operations
        mock_session.get.return_value = sample_tracklist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No existing config
        mock_session.execute.return_value = mock_result

        success, error = await sync_service.apply_updates(tracklist_id, updates, auto=False)

        assert success is True
        assert error is None
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_apply_updates_low_confidence_auto(self, sync_service, mock_session, sample_tracklist):
        """Test auto-update with low confidence gets queued."""
        tracklist_id = sample_tracklist.id
        updates = {
            "confidence": 0.5,  # Low confidence
            "changes": {"total_changes": 5},
        }

        # Mock sync configuration
        mock_config = MagicMock(spec=SyncConfiguration)
        mock_config.auto_accept_threshold = 0.9

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute.return_value = mock_result

        success, error = await sync_service.apply_updates(tracklist_id, updates, auto=True)

        assert success is False
        assert "queued for manual review" in error

    def test_apply_changes_to_tracks(self, sync_service):
        """Test applying changes to tracks list."""
        current = [
            {"position": 1, "title": "Track 1"},
            {"position": 2, "title": "Track 2"},
            {"position": 3, "title": "Track 3"},
        ]

        changes = {
            "tracks_added": [{"position": 4, "title": "Track 4"}],
            "tracks_removed": [{"position": 2}],
            "tracks_modified": [
                {
                    "position": 1,
                    "new": {"position": 1, "title": "Track 1 (Modified)"},
                }
            ],
        }

        result = sync_service._apply_changes_to_tracks(current, changes)

        # Check results
        positions = [t["position"] for t in result]
        assert 2 not in positions  # Track 2 removed
        assert 4 in positions  # Track 4 added
        assert result[0]["title"] == "Track 1 (Modified)"  # Track 1 modified
