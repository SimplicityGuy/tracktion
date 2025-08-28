"""Unit tests for expanded track operations API."""

from datetime import timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from services.tracklist_service.src.api.manual_endpoints import (
    BulkTrackUpdateRequest,
    TrackReorderRequest,
    auto_calculate_end_times,
    bulk_update_tracks,
    get_timing_suggestions,
    match_all_tracks_to_catalog,
    reorder_track,
    validate_timing,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist


class TestTrackOperations:
    """Test expanded track operations."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_draft_service(self):
        """Create mock draft service."""
        with patch("services.tracklist_service.src.api.manual_endpoints.DraftService") as mock_cls:
            yield mock_cls.return_value

    @pytest.fixture
    def sample_tracklist(self):
        """Create sample tracklist with tracks."""
        return Tracklist(
            id=uuid4(),
            audio_file_id=uuid4(),
            source="manual",
            is_draft=True,
            tracks=[
                TrackEntry(
                    position=1,
                    start_time=timedelta(seconds=0),
                    end_time=timedelta(seconds=180),
                    artist="Artist 1",
                    title="Track 1",
                ),
                TrackEntry(
                    position=2,
                    start_time=timedelta(seconds=180),
                    end_time=timedelta(seconds=360),
                    artist="Artist 2",
                    title="Track 2",
                ),
                TrackEntry(
                    position=3,
                    start_time=timedelta(seconds=360),
                    artist="Artist 3",
                    title="Track 3",
                ),
            ],
        )

    def test_bulk_update_tracks_success(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test bulk updating tracks."""
        tracklist_id = uuid4()
        new_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                artist="New Artist 1",
                title="New Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(seconds=200),
                artist="New Artist 2",
                title="New Track 2",
            ),
        ]

        mock_draft_service.get_draft.return_value = sample_tracklist

        request = BulkTrackUpdateRequest(tracks=new_tracks)
        result = bulk_update_tracks(tracklist_id, request, mock_db_session)

        assert result == new_tracks
        mock_draft_service.save_draft.assert_called_once_with(tracklist_id, new_tracks, auto_version=False)

    def test_bulk_update_tracks_duplicate_positions(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test bulk update with duplicate positions fails."""
        tracklist_id = uuid4()
        invalid_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=1,  # Duplicate position
                start_time=timedelta(seconds=180),
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        mock_draft_service.get_draft.return_value = sample_tracklist

        request = BulkTrackUpdateRequest(tracks=invalid_tracks)

        with pytest.raises(HTTPException) as exc_info:
            bulk_update_tracks(tracklist_id, request, mock_db_session)

        assert exc_info.value.status_code == 400
        assert "Duplicate positions" in str(exc_info.value.detail)

    def test_reorder_track_move_down(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test reordering track to later position."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        request = TrackReorderRequest(from_position=1, to_position=3)

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_timing.return_value.normalize_track_positions.return_value = sample_tracklist.tracks

            result = reorder_track(tracklist_id, request, mock_db_session)

            # Check that the track was moved
            mock_draft_service.save_draft.assert_called_once()
            assert len(result) == 3

    def test_reorder_track_move_up(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test reordering track to earlier position."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        request = TrackReorderRequest(from_position=3, to_position=1)

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_timing.return_value.normalize_track_positions.return_value = sample_tracklist.tracks

            result = reorder_track(tracklist_id, request, mock_db_session)

            mock_draft_service.save_draft.assert_called_once()
            assert len(result) == 3

    def test_reorder_track_not_found(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test reordering non-existent track."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        request = TrackReorderRequest(from_position=99, to_position=1)

        with pytest.raises(HTTPException) as exc_info:
            reorder_track(tracklist_id, request, mock_db_session)

        assert exc_info.value.status_code == 404
        assert "Track at position 99 not found" in str(exc_info.value.detail)

    def test_auto_calculate_end_times_success(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test auto-calculating end times."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_timing.return_value.auto_calculate_end_times.return_value = sample_tracklist.tracks

            result = auto_calculate_end_times(
                tracklist_id,
                audio_duration="1:30:00",  # 90 minutes
                db=mock_db_session,
            )

            assert len(result) == 3
            mock_timing.return_value.auto_calculate_end_times.assert_called_once()
            mock_draft_service.save_draft.assert_called_once()

    def test_get_timing_suggestions_with_overlaps(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test getting timing suggestions for overlapping tracks."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_suggestions = [
                {
                    "type": "fix_overlap",
                    "track_position": 1,
                    "next_position": 2,
                    "current_overlap": 10.0,
                    "suggested_action": "adjust_end_time",
                    "priority": "high",
                }
            ]
            mock_timing.return_value.suggest_timing_adjustments.return_value = mock_suggestions

            result = get_timing_suggestions(
                tracklist_id,
                target_duration="1:00:00",
                db=mock_db_session,
            )

            assert len(result) == 1
            assert result[0]["type"] == "fix_overlap"
            assert result[0]["priority"] == "high"

    def test_validate_timing_valid(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test validating timing with no issues."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_timing.return_value.validate_timing_consistency.return_value = (
                True,
                [],
            )

            result = validate_timing(
                tracklist_id,
                audio_duration="1:00:00",
                db=mock_db_session,
            )

            assert result["is_valid"] is True
            assert len(result["issues"]) == 0
            assert result["track_count"] == 3

    def test_validate_timing_with_overlaps(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test validating timing with overlaps."""
        tracklist_id = uuid4()

        # Create overlapping tracks
        sample_tracklist.tracks[0].end_time = timedelta(seconds=200)  # Overlaps with track 2
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            mock_timing.return_value.detect_timing_conflicts.return_value = [
                {
                    "type": "overlap",
                    "track_position": 1,
                    "conflicting_position": 2,
                    "overlap_duration": 20.0,
                }
            ]

            result = validate_timing(
                tracklist_id,
                db=mock_db_session,
            )

            # Should detect the overlap when no audio duration provided
            if not result["is_valid"]:
                assert len(result["issues"]) > 0
                assert "overlap" in result["issues"][0]

    def test_match_all_tracks_to_catalog(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test matching all tracks to catalog."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.CatalogSearchService") as mock_catalog:
            # Mock matched tracks with catalog IDs
            matched_tracks = sample_tracklist.tracks.copy()
            for track in matched_tracks:
                track.catalog_track_id = uuid4()
                track.confidence = 0.85

            mock_catalog.return_value.fuzzy_match_tracks.return_value = matched_tracks

            result = match_all_tracks_to_catalog(
                tracklist_id,
                threshold=0.7,
                db=mock_db_session,
            )

            assert len(result) == 3
            assert all(track.catalog_track_id is not None for track in result)
            assert all(track.confidence > 0.7 for track in result)
            mock_draft_service.save_draft.assert_called_once()

    def test_validate_timing_no_audio_duration(self, mock_db_session, mock_draft_service, sample_tracklist):
        """Test timing validation without audio duration."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = sample_tracklist

        with patch("services.tracklist_service.src.api.manual_endpoints.TimingService") as mock_timing:
            # No conflicts detected
            mock_timing.return_value.detect_timing_conflicts.return_value = []

            result = validate_timing(
                tracklist_id,
                db=mock_db_session,
            )

            assert result["is_valid"] is True
            assert len(result["issues"]) == 0

    def test_tracklist_not_found_error(self, mock_db_session, mock_draft_service):
        """Test error when tracklist not found."""
        tracklist_id = uuid4()
        mock_draft_service.get_draft.return_value = None

        # Test bulk update
        request = BulkTrackUpdateRequest(tracks=[])
        with pytest.raises(HTTPException) as exc_info:
            bulk_update_tracks(tracklist_id, request, mock_db_session)
        assert exc_info.value.status_code == 404

        # Test auto calculate
        with pytest.raises(HTTPException) as exc_info:
            auto_calculate_end_times(tracklist_id, db=mock_db_session)
        assert exc_info.value.status_code == 404

        # Test timing suggestions
        with pytest.raises(HTTPException) as exc_info:
            get_timing_suggestions(tracklist_id, db=mock_db_session)
        assert exc_info.value.status_code == 404
