"""Unit tests for conflict resolution service."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.conflict_resolution_service import (
    ConflictResolutionService,
    ConflictType,
    ResolutionStrategy,
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


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
def conflict_service(mock_session, mock_version_service, mock_audit_service):
    """Create conflict resolution service instance."""
    return ConflictResolutionService(
        session=mock_session,
        version_service=mock_version_service,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracklist = MagicMock(spec=TracklistDB)
    tracklist.id = uuid4()
    tracklist.tracks = [
        {"position": 1, "artist": "Artist 1", "title": "Track 1"},
        {"position": 2, "artist": "Artist 2", "title": "Track 2"},
        {"position": 3, "artist": "Artist 3", "title": "Track 3"},
    ]
    return tracklist


class TestConflictResolutionService:
    """Test ConflictResolutionService methods."""

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(self, conflict_service):
        """Test detecting conflicts when there are none."""
        tracklist_id = uuid4()
        current_state = {"tracks": 3}
        proposed_changes = {
            "tracks_added": [],
            "tracks_removed": [],
            "tracks_modified": [],
        }

        conflicts = await conflict_service.detect_conflicts(tracklist_id, current_state, proposed_changes)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_major_restructure(self, conflict_service):
        """Test detecting major restructure conflict."""
        tracklist_id = uuid4()
        current_state = {"tracks": 10}
        proposed_changes = {
            "tracks_added": [{"position": i} for i in range(11, 18)],  # 7 added
            "tracks_removed": [{"position": i} for i in range(1, 7)],  # 6 removed
            "tracks_modified": [],
        }

        conflicts = await conflict_service.detect_conflicts(tracklist_id, current_state, proposed_changes)

        assert len(conflicts) > 0
        assert conflicts[0]["type"] == ConflictType.MAJOR_RESTRUCTURE.value
        assert conflicts[0]["severity"] == "high"
        assert not conflicts[0]["auto_resolvable"]

    @pytest.mark.asyncio
    async def test_detect_conflicts_track_modifications(self, conflict_service):
        """Test detecting track modification conflicts."""
        tracklist_id = uuid4()
        current_state = {"tracks": 3}
        proposed_changes = {
            "tracks_added": [],
            "tracks_removed": [],
            "tracks_modified": [
                {
                    "position": 1,
                    "old": {"artist": "Artist 1", "title": "Track 1"},
                    "new": {"artist": "Artist 1", "title": "Track 1 (Extended)"},
                }
            ],
        }

        conflicts = await conflict_service.detect_conflicts(tracklist_id, current_state, proposed_changes)

        assert len(conflicts) == 1
        assert conflicts[0]["type"] == ConflictType.TRACK_MODIFIED.value

    @pytest.mark.asyncio
    async def test_detect_conflicts_critical_track_removal(self, conflict_service):
        """Test detecting critical track removal."""
        tracklist_id = uuid4()
        current_state = {"tracks": 3}
        proposed_changes = {
            "tracks_added": [],
            "tracks_removed": [
                {"position": 1, "title": "Opening Track"}  # First track is critical
            ],
            "tracks_modified": [],
        }

        conflicts = await conflict_service.detect_conflicts(tracklist_id, current_state, proposed_changes)

        assert len(conflicts) == 1
        assert conflicts[0]["type"] == ConflictType.TRACK_REMOVED.value
        assert conflicts[0]["recommended_strategy"] == ResolutionStrategy.KEEP_CURRENT.value

    def test_is_major_restructure(self, conflict_service):
        """Test major restructure detection."""
        # Not major - few changes
        changes = {
            "tracks_added": [1, 2],
            "tracks_removed": [1],
            "tracks_modified": [1],
        }
        assert not conflict_service._is_major_restructure(changes)

        # Major - many changes
        changes = {
            "tracks_added": [1, 2, 3, 4, 5, 6, 7],
            "tracks_removed": [1, 2, 3, 4, 5, 6],
            "tracks_modified": [],
        }
        assert conflict_service._is_major_restructure(changes)

    def test_calculate_field_confidence(self, conflict_service):
        """Test field confidence calculation."""
        # Adding missing data
        confidence = conflict_service._calculate_field_confidence("label", None, "New Label")
        assert confidence == 0.9

        # Minor change in major field
        confidence = conflict_service._calculate_field_confidence("title", "Track 1", "Track 1 (Extended)")
        assert confidence == 0.85

        # Complete change in major field
        confidence = conflict_service._calculate_field_confidence("artist", "Artist A", "Artist B")
        assert confidence == 0.3

    def test_is_critical_track(self, conflict_service):
        """Test critical track detection."""
        # First track is critical
        assert conflict_service._is_critical_track({"position": 1, "title": "Any"})

        # Intro track is critical
        assert conflict_service._is_critical_track({"position": 5, "title": "Intro Mix"})

        # Regular track is not critical
        assert not conflict_service._is_critical_track({"position": 5, "title": "Regular Track"})

    def test_recommend_strategy(self, conflict_service):
        """Test strategy recommendation."""
        # High confidence - use proposed
        detail = {"confidence": 0.9, "severity": "low"}
        assert conflict_service._recommend_strategy(detail) == ResolutionStrategy.USE_PROPOSED.value

        # Low confidence - keep current
        detail = {"confidence": 0.3, "severity": "high"}
        assert conflict_service._recommend_strategy(detail) == ResolutionStrategy.KEEP_CURRENT.value

        # Medium confidence - manual edit
        detail = {"confidence": 0.6, "severity": "medium"}
        assert conflict_service._recommend_strategy(detail) == ResolutionStrategy.MANUAL_EDIT.value

    @pytest.mark.asyncio
    async def test_prepare_conflict_ui_data(self, conflict_service):
        """Test preparing conflict data for UI."""
        tracklist_id = uuid4()
        conflicts = [
            {
                "id": "conflict1",
                "type": ConflictType.TRACK_MODIFIED.value,
                "severity": "medium",
                "description": "Track 1 modified",
                "auto_resolvable": True,
                "recommended_strategy": ResolutionStrategy.USE_PROPOSED.value,
                "details": {
                    "position": 1,
                    "changed_fields": [
                        {
                            "field": "title",
                            "old_value": "Track 1",
                            "new_value": "Track 1 (Extended)",
                            "confidence": 0.85,
                        }
                    ],
                },
            }
        ]
        current_state = {"tracks": 3}
        proposed_changes = {
            "tracks_added": [],
            "tracks_removed": [],
            "tracks_modified": [1],
        }

        ui_data = await conflict_service.prepare_conflict_ui_data(
            tracklist_id, conflicts, current_state, proposed_changes
        )

        assert ui_data["tracklist_id"] == str(tracklist_id)
        assert ui_data["total_conflicts"] == 1
        assert ui_data["auto_resolvable_count"] == 1
        assert len(ui_data["conflicts"]) == 1

    @pytest.mark.asyncio
    async def test_resolve_conflicts_success(
        self, conflict_service, mock_session, mock_version_service, mock_audit_service, sample_tracklist
    ):
        """Test successful conflict resolution."""
        tracklist_id = sample_tracklist.id
        resolutions = [
            {
                "conflict_id": "conflict1",
                "strategy": ResolutionStrategy.USE_PROPOSED.value,
                "proposed_data": {"tracks": sample_tracklist.tracks},
            }
        ]

        mock_session.get.return_value = sample_tracklist

        success, error = await conflict_service.resolve_conflicts(tracklist_id, resolutions, "user")

        assert success is True
        assert error is None
        mock_version_service.create_version.assert_called_once()
        mock_audit_service.log_tracklist_change.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_conflicts_tracklist_not_found(self, conflict_service, mock_session):
        """Test conflict resolution when tracklist not found."""
        tracklist_id = uuid4()
        resolutions = []

        mock_session.get.return_value = None

        success, error = await conflict_service.resolve_conflicts(tracklist_id, resolutions, "user")

        assert success is False
        assert error == "Tracklist not found"

    @pytest.mark.asyncio
    async def test_auto_resolve_conflicts(self, conflict_service):
        """Test automatic conflict resolution."""
        tracklist_id = uuid4()
        conflicts = [
            {
                "id": "conflict1",
                "auto_resolvable": True,
                "recommended_strategy": ResolutionStrategy.USE_PROPOSED.value,
                "type": ConflictType.TRACK_MODIFIED.value,
                "details": {"position": 1, "confidence": 0.9},
            },
            {
                "id": "conflict2",
                "auto_resolvable": False,
                "recommended_strategy": ResolutionStrategy.MANUAL_EDIT.value,
            },
        ]
        proposed_changes = {"tracks_modified": [{"position": 1, "new": {"position": 1, "title": "New Title"}}]}

        resolutions = await conflict_service.auto_resolve_conflicts(tracklist_id, conflicts, proposed_changes)

        assert len(resolutions) == 1  # Only auto-resolvable conflict
        assert resolutions[0]["conflict_id"] == "conflict1"
        assert resolutions[0]["automated"] is True

    def test_extract_proposed_data(self, conflict_service):
        """Test extracting proposed data for a conflict."""
        conflict = {
            "type": ConflictType.TRACK_MODIFIED.value,
            "details": {"position": 2},
        }
        proposed_changes = {
            "tracks_modified": [
                {"position": 1, "new": {"title": "Track 1"}},
                {"position": 2, "new": {"title": "Track 2 Modified"}},
            ]
        }

        data = conflict_service._extract_proposed_data(conflict, proposed_changes)

        assert "tracks" in data
        assert data["tracks"][0]["title"] == "Track 2 Modified"
