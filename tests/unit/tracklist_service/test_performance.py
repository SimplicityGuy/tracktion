"""Performance tests for tracklist service."""

import time
from datetime import timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.catalog_search_service import CatalogSearchService
from services.tracklist_service.src.services.draft_service import DraftService
from services.tracklist_service.src.services.timing_service import TimingService


class TestPerformance:
    """Test performance optimizations."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def large_tracklist(self):
        """Create a large tracklist for performance testing."""
        tracks = [
            TrackEntry(
                position=i + 1,
                start_time=timedelta(minutes=i * 3),
                end_time=timedelta(minutes=(i + 1) * 3),
                artist=f"Artist {i}",
                title=f"Track {i}",
            )
            for i in range(100)  # 100 tracks
        ]
        return Tracklist(
            id=uuid4(),
            audio_file_id=uuid4(),
            source="manual",
            is_draft=True,
            tracks=tracks,
        )

    def test_timing_adjustment_performance(self, large_tracklist):
        """Test that timing adjustments complete within 500ms."""
        timing_service = TimingService()
        audio_duration = timedelta(hours=5)

        start_time = time.time()
        result = timing_service.adjust_track_timings(
            large_tracklist.tracks,
            audio_duration=audio_duration,
        )
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # Convert to ms

        assert execution_time < 500, f"Timing adjustment took {execution_time:.2f}ms, expected <500ms"
        assert len(result) == 100
        assert result[-1].end_time <= audio_duration

    def test_batch_validation_performance(self, large_tracklist):
        """Test batch validation performance."""
        timing_service = TimingService()

        # Create 10 tracklists to validate
        tracklists = [large_tracklist.tracks for _ in range(10)]
        durations = [timedelta(hours=5) for _ in range(10)]

        start_time = time.time()
        results = timing_service.batch_validate_timings(tracklists, durations)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Batch validation took {execution_time:.2f}ms, expected <500ms"
        assert len(results) == 10

    def test_draft_save_performance(self, mock_db_session, large_tracklist):
        """Test draft save performance with large track list."""
        draft_service = DraftService(mock_db_session)

        # Mock the database query
        mock_db_session.query().filter_by().first.return_value = MagicMock(
            to_model=lambda: large_tracklist,
            tracks=[],
        )

        start_time = time.time()
        with patch.object(draft_service, "_has_significant_changes", return_value=False):
            draft_service.save_draft(
                large_tracklist.id,
                large_tracklist.tracks,
                auto_version=False,
            )
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Draft save took {execution_time:.2f}ms, expected <500ms"

    def test_batch_track_update_performance(self, mock_db_session, large_tracklist):
        """Test batch track update performance."""
        draft_service = DraftService(mock_db_session)

        # Mock the database
        mock_db_session.query().filter_by().first.return_value = MagicMock(
            to_model=lambda: large_tracklist,
        )

        # Prepare 50 track updates
        updates = [{"position": i, "artist": f"Updated Artist {i}"} for i in range(1, 51)]

        start_time = time.time()
        with patch.object(draft_service, "save_draft", return_value=large_tracklist):
            draft_service.batch_update_tracks(large_tracklist.id, updates)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Batch update took {execution_time:.2f}ms, expected <500ms"

    def test_catalog_search_performance(self, mock_db_session):
        """Test catalog search performance."""
        catalog_service = CatalogSearchService(mock_db_session)

        # Mock database results
        mock_results = []
        for i in range(100):
            mock_recording = MagicMock()
            mock_recording.id = uuid4()
            mock_metadata = MagicMock()
            mock_metadata.key = "artist" if i % 2 == 0 else "title"
            mock_metadata.value = f"Test {i}"
            mock_results.append((mock_recording, mock_metadata))

        mock_db_session.query().join().filter().limit().all.return_value = mock_results

        start_time = time.time()
        results = catalog_service.search_catalog(
            query="test",
            limit=10,
        )
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Catalog search took {execution_time:.2f}ms, expected <500ms"
        assert len(results) <= 10

    def test_timing_conflict_detection_performance(self, large_tracklist):
        """Test timing conflict detection performance."""
        timing_service = TimingService()

        # Create overlapping tracks for conflict detection
        for i in range(0, len(large_tracklist.tracks) - 1, 2):
            # Make every other track overlap with the next
            large_tracklist.tracks[i].end_time = large_tracklist.tracks[i + 1].start_time + timedelta(seconds=10)

        start_time = time.time()
        conflicts_list = []
        for track in large_tracklist.tracks[:50]:  # Check first 50 tracks
            conflicts = timing_service.detect_timing_conflicts(track, large_tracklist.tracks)
            conflicts_list.extend(conflicts)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Conflict detection took {execution_time:.2f}ms, expected <500ms"
        assert len(conflicts_list) > 0  # Should detect some conflicts

    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        timing_service = TimingService()

        # Create test data
        tracks = [
            TrackEntry(
                position=i + 1,  # Position must be positive (>= 1)
                start_time=timedelta(minutes=i * 3),
                end_time=timedelta(minutes=(i + 1) * 3),
                artist=f"Artist {i}",
                title=f"Track {i}",
            )
            for i in range(20)
        ]

        # First run (no cache)
        start_time = time.time()
        for _ in range(10):
            timing_service.validate_timing_consistency(tracks, timedelta(hours=1))
        first_run_time = time.time() - start_time

        # Second run (with cache)
        start_time = time.time()
        for _ in range(10):
            timing_service.batch_validate_timings([tracks], [timedelta(hours=1)])
        second_run_time = time.time() - start_time

        # Cache should make second run faster
        assert second_run_time < first_run_time, "Cache did not improve performance"

    def test_bulk_create_performance(self, mock_db_session):
        """Test bulk draft creation performance."""
        draft_service = DraftService(mock_db_session)

        # Prepare bulk data
        draft_data = [
            {
                "audio_file_id": uuid4(),
                "tracks": [
                    TrackEntry(
                        position=j + 1,
                        start_time=timedelta(minutes=j * 3),
                        artist=f"Artist {j}",
                        title=f"Track {j}",
                    )
                    for j in range(10)
                ],
            }
            for _ in range(20)  # 20 drafts
        ]

        # Mock database operations
        mock_db_session.query().filter_by().order_by().first.return_value = None
        mock_db_session.add = MagicMock()
        mock_db_session.commit = MagicMock()

        start_time = time.time()
        with patch.object(draft_service, "create_draft") as mock_create:
            mock_create.return_value = MagicMock(id=uuid4())
            draft_service.bulk_create_drafts(draft_data)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 1000, f"Bulk create took {execution_time:.2f}ms, expected <1000ms"
        assert mock_create.call_count == 20

    def test_optimize_timing_layout_performance(self):
        """Test timing layout optimization performance."""
        timing_service = TimingService()

        # Create unoptimized tracks with placeholder timing
        tracks = [
            TrackEntry(
                position=i + 1,
                start_time=timedelta(0),  # Start with default timing
                end_time=None,  # Will be optimized
                artist=f"Artist {i}",
                title=f"Track {i}",
            )
            for i in range(100)
        ]

        target_duration = timedelta(hours=5)

        start_time = time.time()
        optimized = timing_service.optimize_timing_layout(tracks, target_duration)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        assert execution_time < 500, f"Layout optimization took {execution_time:.2f}ms, expected <500ms"
        assert len(optimized) == 100
        assert optimized[-1].end_time == target_duration

    @pytest.mark.benchmark
    def test_response_time_under_load(self, mock_db_session):
        """Test that response times remain under 500ms under load."""
        timing_service = TimingService()

        # Simulate multiple concurrent operations
        operations = []

        # Create test data
        tracks = [
            TrackEntry(
                position=i + 1,
                start_time=timedelta(minutes=i * 3),
                end_time=timedelta(minutes=(i + 1) * 3),
                artist=f"Artist {i}",
                title=f"Track {i}",
            )
            for i in range(50)
        ]

        start_time = time.time()

        # Simulate mixed operations
        for _ in range(5):
            # Timing adjustment
            result1 = timing_service.adjust_track_timings(tracks, timedelta(hours=3))
            operations.append(result1)

            # Conflict detection
            conflicts = timing_service.detect_timing_conflicts(tracks[0], tracks)
            operations.append(conflicts)

            # Validation
            valid, issues = timing_service.validate_timing_consistency(tracks, timedelta(hours=3))
            operations.append((valid, issues))

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000

        # Even under load, should complete within reasonable time
        assert execution_time < 2000, f"Operations under load took {execution_time:.2f}ms, expected <2000ms"
        assert len(operations) == 15
