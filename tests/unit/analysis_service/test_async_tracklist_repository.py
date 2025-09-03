"""Comprehensive unit tests for AsyncTracklistRepository class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from services.analysis_service.src.repositories import AsyncTracklistRepository
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import Recording, Tracklist


class TestAsyncTracklistRepository:
    """Test async tracklist repository operations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        manager = MagicMock(spec=AsyncDatabaseManager)

        # Create async context manager mock
        async_session_mock = AsyncMock()
        async_session_mock.__aenter__ = AsyncMock(return_value=async_session_mock)
        async_session_mock.__aexit__ = AsyncMock(return_value=None)

        manager.get_db_session.return_value = async_session_mock
        return manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        """Create repository with mock DB."""
        return AsyncTracklistRepository(mock_db_manager)

    @pytest.fixture
    def sample_recording_id(self):
        """Create a sample recording ID for testing."""
        return uuid4()

    @pytest.fixture
    def sample_tracks(self):
        """Create sample track data for testing."""
        return [
            {
                "track_number": 1,
                "title": "Opening Track",
                "artist": "DJ Example",
                "start_time": "00:00:00",
                "duration": "03:45",
                "bpm": 128.0,
            },
            {
                "track_number": 2,
                "title": "Second Track",
                "artist": "Producer Test",
                "start_time": "03:45:00",
                "duration": "04:15",
                "bpm": 130.0,
            },
            {
                "track_number": 3,
                "title": "Final Track",
                "artist": "Artist Sample",
                "start_time": "08:00:00",
                "duration": "05:30",
                "bpm": 126.0,
            },
        ]

    @pytest.fixture
    def sample_tracklist(self, sample_recording_id, sample_tracks):
        """Create a sample tracklist for testing."""
        return Tracklist(
            id=uuid4(),
            recording_id=sample_recording_id,
            source="manual",
            tracks=sample_tracks,
            cue_file_path="/path/to/test.cue",
        )

    @pytest.fixture
    def sample_recording(self, sample_recording_id):
        """Create a sample recording for testing."""
        return Recording(
            id=sample_recording_id,
            file_path="/test/sample.mp3",
            file_name="sample.mp3",
            processing_status="completed",
        )

    # Basic CRUD Operations Tests

    @pytest.mark.asyncio
    async def test_create_tracklist(self, repository, mock_db_manager, sample_recording_id, sample_tracks):
        """Test creating a new tracklist."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create tracklist
        result = await repository.create(
            recording_id=sample_recording_id, source="manual", tracks=sample_tracks, cue_file_path="/path/to/test.cue"
        )

        # Verify
        assert result is not None
        assert result.recording_id == sample_recording_id
        assert result.source == "manual"
        assert result.tracks == sample_tracks
        assert result.cue_file_path == "/path/to/test.cue"

        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tracklist_minimal(self, repository, mock_db_manager, sample_recording_id, sample_tracks):
        """Test creating tracklist with minimal required fields."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with minimal data (no CUE file)
        result = await repository.create(
            recording_id=sample_recording_id, source="1001tracklists", tracks=sample_tracks
        )

        # Verify defaults
        assert result.recording_id == sample_recording_id
        assert result.source == "1001tracklists"
        assert result.tracks == sample_tracks
        assert result.cue_file_path is None

    @pytest.mark.asyncio
    async def test_create_tracklist_empty_tracks(self, repository, mock_db_manager, sample_recording_id):
        """Test creating tracklist with empty tracks list."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with empty tracks
        result = await repository.create(recording_id=sample_recording_id, source="auto", tracks=[])

        # Verify
        assert result.recording_id == sample_recording_id
        assert result.source == "auto"
        assert result.tracks == []
        assert result.cue_file_path is None

    @pytest.mark.asyncio
    async def test_get_by_recording_id(self, repository, mock_db_manager, sample_tracklist):
        """Test getting tracklist by recording ID."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_tracklist
        session.execute = AsyncMock(return_value=mock_result)

        # Get tracklist
        result = await repository.get_by_recording_id(sample_tracklist.recording_id)

        # Verify
        assert result is not None
        assert result.recording_id == sample_tracklist.recording_id
        assert result.source == "manual"
        assert len(result.tracks) == 3
        assert result.cue_file_path == "/path/to/test.cue"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_recording_id_not_found(self, repository, mock_db_manager):
        """Test getting non-existent tracklist by recording ID."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Get non-existent tracklist
        result = await repository.get_by_recording_id(uuid4())

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_tracks(self, repository, mock_db_manager, sample_recording_id):
        """Test updating tracks for a tracklist."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()

        # Create updated tracks
        updated_tracks = [
            {
                "track_number": 1,
                "title": "Updated Track 1",
                "artist": "New Artist",
                "start_time": "00:00:00",
                "duration": "04:00",
                "bpm": 135.0,
            },
            {
                "track_number": 2,
                "title": "Updated Track 2",
                "artist": "Another Artist",
                "start_time": "04:00:00",
                "duration": "03:30",
                "bpm": 140.0,
            },
        ]

        # Create expected updated tracklist
        updated_tracklist = Tracklist(
            id=uuid4(), recording_id=sample_recording_id, source="manual", tracks=updated_tracks, cue_file_path=None
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = updated_tracklist
        session.execute = AsyncMock(return_value=mock_result)

        # Update tracks
        result = await repository.update_tracks(sample_recording_id, updated_tracks)

        # Verify
        assert result is not None
        assert result.recording_id == sample_recording_id
        assert result.tracks == updated_tracks
        assert len(result.tracks) == 2
        assert result.tracks[0]["title"] == "Updated Track 1"
        assert result.tracks[1]["bpm"] == 140.0
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_tracks_not_found(self, repository, mock_db_manager):
        """Test updating tracks for non-existent tracklist."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Update non-existent tracklist
        result = await repository.update_tracks(uuid4(), [{"track_number": 1, "title": "Test Track"}])

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_tracks_empty_list(self, repository, mock_db_manager, sample_recording_id):
        """Test updating tracks with empty list."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()

        # Create tracklist with empty tracks
        empty_tracklist = Tracklist(
            id=uuid4(), recording_id=sample_recording_id, source="manual", tracks=[], cue_file_path=None
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = empty_tracklist
        session.execute = AsyncMock(return_value=mock_result)

        # Update with empty tracks
        result = await repository.update_tracks(sample_recording_id, [])

        # Verify
        assert result is not None
        assert result.tracks == []
        session.execute.assert_called_once()

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_create_database_error(self, repository, mock_db_manager, sample_recording_id, sample_tracks):
        """Test handling database errors during creation."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock(side_effect=IntegrityError("", "", ""))
        session.rollback = AsyncMock()

        # Attempt to create (should raise exception)
        with pytest.raises(IntegrityError):
            await repository.create(recording_id=sample_recording_id, source="manual", tracks=sample_tracks)

    @pytest.mark.asyncio
    async def test_get_database_error(self, repository, mock_db_manager):
        """Test handling database errors during retrieval."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Database connection failed"))

        # Attempt to get (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_recording_id(uuid4())

    @pytest.mark.asyncio
    async def test_update_database_error(self, repository, mock_db_manager):
        """Test handling database errors during update."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Update failed"))

        # Attempt to update (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.update_tracks(uuid4(), [])

    # Data Validation and Edge Cases Tests

    @pytest.mark.asyncio
    async def test_create_with_complex_tracks_data(self, repository, mock_db_manager, sample_recording_id):
        """Test creating tracklist with complex track data."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create complex track data
        complex_tracks = [
            {
                "track_number": 1,
                "title": "Complex Track with Unicode: 🎵",
                "artist": "Artïst Nâmé with Accénts",
                "album": "Test Album",
                "start_time": "00:00:00.123",
                "end_time": "03:45:12.456",
                "duration": "03:45:12.333",
                "bpm": 128.5,
                "key": "A minor",
                "genre": ["House", "Electronic"],
                "energy": 0.8,
                "danceability": 0.9,
                "valence": 0.7,
                "metadata": {
                    "isrc": "USUM71234567",
                    "label": "Test Records",
                    "year": 2023,
                    "custom_field": "custom_value",
                },
            },
            {
                "track_number": 2,
                "title": "Track with Long Description",
                "artist": "Very Long Artist Name That Goes On And On",
                "description": (
                    "This is a very long description that contains multiple sentences. It describes the track in "
                    "great detail and includes information about the production, the inspiration, and the "
                    "technical aspects of the recording."
                ),
                "tags": ["uplifting", "progressive", "melodic", "atmospheric"],
                "waveform_data": [0.1, 0.2, 0.3, 0.4, 0.5] * 1000,  # Large array
                "cue_points": [
                    {"name": "intro", "time": 0.0},
                    {"name": "verse", "time": 32.5},
                    {"name": "chorus", "time": 64.0},
                    {"name": "breakdown", "time": 128.0},
                    {"name": "outro", "time": 192.0},
                ],
            },
        ]

        result = await repository.create(
            recording_id=sample_recording_id,
            source="advanced_analysis",
            tracks=complex_tracks,
            cue_file_path="/complex/path/with spaces/test.cue",
        )

        # Verify creation with complex data
        assert result is not None
        assert result.tracks == complex_tracks
        assert result.tracks[0]["title"] == "Complex Track with Unicode: 🎵"
        assert result.tracks[0]["metadata"]["isrc"] == "USUM71234567"
        assert len(result.tracks[1]["waveform_data"]) == 5000

    @pytest.mark.asyncio
    async def test_create_with_various_sources(self, repository, mock_db_manager, sample_recording_id, sample_tracks):
        """Test creating tracklists with different source types."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        sources = ["manual", "1001tracklists", "beatport", "discogs", "spotify", "soundcloud", "auto", "ai_generated"]

        for source in sources:
            result = await repository.create(
                recording_id=uuid4(),  # Different recording for each
                source=source,
                tracks=sample_tracks,
            )

            assert result.source == source

    @pytest.mark.asyncio
    async def test_tracks_json_serialization(self, repository, mock_db_manager, sample_recording_id):
        """Test that complex track data is properly handled as JSON."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create tracks with various data types
        json_tracks = [
            {
                "track_number": 1,
                "title": "JSON Test Track",
                "bpm": 128.0,  # float
                "duration_seconds": 225,  # int
                "has_vocals": True,  # bool
                "analysis_confidence": None,  # None
                "nested_object": {"level1": {"level2": {"deep_value": "test"}}},
                "array_of_objects": [
                    {"time": 0.0, "value": 0.5},
                    {"time": 1.0, "value": 0.7},
                    {"time": 2.0, "value": 0.3},
                ],
            }
        ]

        result = await repository.create(recording_id=sample_recording_id, source="json_test", tracks=json_tracks)

        # Verify complex JSON data is preserved
        assert result is not None
        assert result.tracks == json_tracks
        assert result.tracks[0]["nested_object"]["level1"]["level2"]["deep_value"] == "test"
        assert len(result.tracks[0]["array_of_objects"]) == 3

    @pytest.mark.asyncio
    async def test_cue_file_path_variations(self, repository, mock_db_manager, sample_recording_id, sample_tracks):
        """Test various CUE file path formats."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        cue_paths = [
            "/absolute/path/to/file.cue",
            "./relative/path/file.cue",
            "../parent/dir/file.cue",
            "/path with spaces/file name.cue",
            "/path/with-special_chars@file.cue",
            "/very/long/path/that/goes/on/for/many/directories/and/has/a/very/long/filename.cue",
            "C:\\Windows\\Path\\file.cue",  # Windows path
            None,  # No CUE file
        ]

        for cue_path in cue_paths:
            result = await repository.create(
                recording_id=uuid4(),  # Different recording for each
                source="cue_test",
                tracks=sample_tracks,
                cue_file_path=cue_path,
            )

            assert result.cue_file_path == cue_path

    # Concurrency and Transaction Tests

    @pytest.mark.asyncio
    async def test_concurrent_creates(self, mock_db_manager):
        """Test handling concurrent create operations."""
        # Create multiple repositories
        repos = [AsyncTracklistRepository(mock_db_manager) for _ in range(5)]

        # Setup separate sessions
        sessions = []
        for _ in range(5):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            session.add = MagicMock()
            session.flush = AsyncMock()
            session.refresh = AsyncMock()
            session.commit = AsyncMock()
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent creates
        recording_ids = [uuid4() for _ in range(5)]
        tasks = []
        for i, (repo, recording_id) in enumerate(zip(repos, recording_ids, strict=False)):
            task = repo.create(
                recording_id=recording_id,
                source=f"concurrent_source_{i}",
                tracks=[{"track_number": i + 1, "title": f"Track {i + 1}"}],
            )
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.recording_id == recording_ids[i]
            assert result.source == f"concurrent_source_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, mock_db_manager):
        """Test handling concurrent read operations."""
        # Create multiple repositories
        repos = [AsyncTracklistRepository(mock_db_manager) for _ in range(10)]

        # Setup sessions with different mock results
        sessions = []
        expected_results = []
        for i in range(10):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            # Create unique result for each session
            tracklist = Tracklist(
                id=uuid4(),
                recording_id=uuid4(),
                source=f"source_{i}",
                tracks=[{"track_number": i, "title": f"Track {i}"}],
                cue_file_path=f"/path/{i}.cue" if i % 2 == 0 else None,
            )
            expected_results.append(tracklist)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = tracklist
            session.execute = AsyncMock(return_value=mock_result)
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent reads
        tasks = []
        for i, repo in enumerate(repos):
            task = repo.get_by_recording_id(expected_results[i].recording_id)
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.recording_id == expected_results[i].recording_id
            assert result.source == f"source_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, mock_db_manager):
        """Test handling concurrent update operations."""
        # Create multiple repositories
        repos = [AsyncTracklistRepository(mock_db_manager) for _ in range(5)]

        # Setup separate sessions
        sessions = []
        for idx in range(5):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            # Create updated tracklist result
            updated_tracklist = Tracklist(
                id=uuid4(),
                recording_id=uuid4(),
                source="updated",
                tracks=[{"track_number": idx, "title": f"Updated Track {idx}"}],
                cue_file_path=None,
            )

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = updated_tracklist
            session.execute = AsyncMock(return_value=mock_result)
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent updates
        recording_ids = [uuid4() for _ in range(5)]
        tasks = []
        for i, (repo, recording_id) in enumerate(zip(repos, recording_ids, strict=False)):
            task = repo.update_tracks(
                recording_id=recording_id, tracks=[{"track_number": i, "title": f"Updated Track {i}"}]
            )
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result is not None
            assert result.tracks[0]["title"] == f"Updated Track {i}"

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, mock_db_manager):
        """Test transaction isolation between operations."""
        # Create two repositories
        repo1 = AsyncTracklistRepository(mock_db_manager)
        repo2 = AsyncTracklistRepository(mock_db_manager)

        # Setup separate sessions
        session1 = AsyncMock()
        session1.__aenter__ = AsyncMock(return_value=session1)
        session1.__aexit__ = AsyncMock(return_value=None)

        session2 = AsyncMock()
        session2.__aenter__ = AsyncMock(return_value=session2)
        session2.__aexit__ = AsyncMock(return_value=None)

        mock_db_manager.get_db_session.side_effect = [session1, session2]

        # Setup different results
        tracklist1 = Tracklist(
            id=uuid4(),
            recording_id=uuid4(),
            source="source1",
            tracks=[{"track_number": 1, "title": "Track 1"}],
            cue_file_path="/path1.cue",
        )

        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = tracklist1
        session1.execute = AsyncMock(return_value=result1)

        result2 = MagicMock()
        result2.scalar_one_or_none.return_value = None  # Not found
        session2.execute = AsyncMock(return_value=result2)

        # Execute operations in different transactions
        recording_id1 = uuid4()
        recording_id2 = uuid4()

        get_result1 = await repo1.get_by_recording_id(recording_id1)
        get_result2 = await repo2.get_by_recording_id(recording_id2)

        # Verify isolation - each got different results
        assert get_result1 is not None
        assert get_result1.source == "source1"
        assert get_result2 is None
        assert session1.execute.call_count == 1
        assert session2.execute.call_count == 1

    # Edge Cases and Special Scenarios

    @pytest.mark.asyncio
    async def test_create_with_null_tracks(self, repository, mock_db_manager, sample_recording_id):
        """Test creating tracklist with null tracks."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with null tracks (should work as tracks is nullable)
        result = await repository.create(recording_id=sample_recording_id, source="empty", tracks=None)

        # Verify
        assert result.recording_id == sample_recording_id
        assert result.source == "empty"
        assert result.tracks is None
        assert result.cue_file_path is None

    @pytest.mark.asyncio
    async def test_update_tracks_with_malformed_data(self, repository, mock_db_manager, sample_recording_id):
        """Test updating tracks with various malformed data."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()

        # Different types of malformed data should still be stored (validation happens at app level)
        malformed_tracks_examples = [
            # Missing required fields
            [{"title": "No track number"}],
            # Wrong data types
            [{"track_number": "one", "title": 123, "bpm": "fast"}],
            # Very nested structure
            [{"track": {"info": {"details": {"number": 1}}}}],
            # Empty objects
            [{}],
            # Mixed valid/invalid
            [{"track_number": 1, "title": "Valid Track"}, {"invalid": "data"}],
        ]

        for _, malformed_tracks in enumerate(malformed_tracks_examples):
            updated_tracklist = Tracklist(
                id=uuid4(),
                recording_id=sample_recording_id,
                source="malformed",
                tracks=malformed_tracks,
                cue_file_path=None,
            )

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = updated_tracklist
            session.execute = AsyncMock(return_value=mock_result)

            # Update with malformed data (should not fail - JSON can store anything)
            result = await repository.update_tracks(sample_recording_id, malformed_tracks)

            # Verify update succeeded with malformed data
            assert result is not None
            assert result.tracks == malformed_tracks

    @pytest.mark.asyncio
    async def test_session_context_manager_exception_handling(self, mock_db_manager):
        """Test that session context manager properly handles exceptions."""
        repository = AsyncTracklistRepository(mock_db_manager)

        # Setup mock session that raises exception
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(side_effect=Exception("Connection lost"))

        mock_db_manager.get_db_session.return_value = session

        # Verify exception is propagated and context manager cleanup occurs
        with pytest.raises(Exception, match="Connection lost"):
            await repository.get_by_recording_id(uuid4())

        # Verify context manager cleanup was called
        session.__aenter__.assert_called_once()
        session.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization with database manager."""
        mock_manager = MagicMock(spec=AsyncDatabaseManager)

        repository = AsyncTracklistRepository(mock_manager)

        assert repository.db is mock_manager
        assert hasattr(repository, "db")

    # Tracklist-Specific Operations Tests

    @pytest.mark.asyncio
    async def test_track_ordering_preservation(self, repository, mock_db_manager, sample_recording_id):
        """Test that track ordering is preserved in JSON storage."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create tracks with specific ordering
        ordered_tracks = [
            {"track_number": i, "title": f"Track {i:02d}", "start_time": f"{i:02d}:00:00"}
            for i in range(10, 0, -1)  # Reverse order
        ]

        result = await repository.create(
            recording_id=sample_recording_id, source="ordering_test", tracks=ordered_tracks
        )

        # Verify ordering is preserved
        assert len(result.tracks) == 10
        for i, track in enumerate(result.tracks):
            expected_number = 10 - i  # Reverse order
            assert track["track_number"] == expected_number
            assert track["title"] == f"Track {expected_number:02d}"

    @pytest.mark.asyncio
    async def test_cue_data_handling(self, repository, mock_db_manager, sample_recording_id):
        """Test handling of CUE file data and metadata."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create tracks with CUE-style data
        cue_tracks = [
            {
                "track_number": 1,
                "title": "First Track",
                "performer": "Artist One",
                "index_00": "00:00:00",
                "index_01": "00:02:15",
                "isrc": "USUM71500001",
            },
            {
                "track_number": 2,
                "title": "Second Track",
                "performer": "Artist Two",
                "index_00": "03:45:20",
                "index_01": "03:47:35",
                "isrc": "USUM71500002",
            },
        ]

        result = await repository.create(
            recording_id=sample_recording_id, source="cue", tracks=cue_tracks, cue_file_path="/path/to/mixtape.cue"
        )

        # Verify CUE-specific data is preserved
        assert result.cue_file_path == "/path/to/mixtape.cue"
        assert result.tracks[0]["index_00"] == "00:00:00"
        assert result.tracks[0]["index_01"] == "00:02:15"
        assert result.tracks[0]["isrc"] == "USUM71500001"
        assert result.tracks[1]["performer"] == "Artist Two"

    @pytest.mark.asyncio
    async def test_large_tracklist_handling(self, repository, mock_db_manager, sample_recording_id):
        """Test handling of very large tracklists."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create a large tracklist (100 tracks)
        large_tracklist = []
        for i in range(1, 101):
            minutes = (i - 1) * 3
            seconds = (i - 1) * 30 % 60
            large_tracklist.append(
                {
                    "track_number": i,
                    "title": f"Track {i:03d} - Long Title With Many Words And Details",
                    "artist": f"Artist Name {i}",
                    "album": f"Album Name {i}",
                    "start_time": f"{minutes:02d}:{seconds:02d}:00",
                    "duration": "03:30",
                    "bpm": 120.0 + (i % 20),
                    "key": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][i % 12],
                    "genre": ["House", "Techno", "Progressive", "Trance"][i % 4],
                    "energy": (i % 100) / 100.0,
                    "tags": [f"tag{j}" for j in range(i % 5)],
                    "metadata": {
                        "catalog_number": f"CAT{i:05d}",
                        "release_year": 2020 + (i % 4),
                        "label": f"Label {i % 10}",
                    },
                }
            )

        result = await repository.create(recording_id=sample_recording_id, source="large_set", tracks=large_tracklist)

        # Verify large tracklist is handled correctly
        assert len(result.tracks) == 100
        assert result.tracks[0]["track_number"] == 1
        assert result.tracks[99]["track_number"] == 100
        assert result.tracks[50]["title"] == "Track 051 - Long Title With Many Words And Details"

    @pytest.mark.asyncio
    async def test_multiple_source_types(self, repository, mock_db_manager, sample_tracks):
        """Test tracklists from different sources have appropriate characteristics."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Test different source types with appropriate data
        source_tests = [
            {"source": "manual", "tracks": sample_tracks, "cue_file_path": None},
            {
                "source": "1001tracklists",
                "tracks": [
                    {"track_number": i, "title": f"Web Track {i}", "url": f"https://1001tracklists.com/track/{i}"}
                    for i in range(1, 4)
                ],
                "cue_file_path": None,
            },
            {
                "source": "cue",
                "tracks": [
                    {"track_number": i, "title": f"CUE Track {i}", "index_01": f"0{i}:00:00"} for i in range(1, 4)
                ],
                "cue_file_path": "/path/to/file.cue",
            },
            {
                "source": "beatport",
                "tracks": [{"track_number": i, "title": f"BP Track {i}", "beatport_id": f"bp{i}"} for i in range(1, 4)],
                "cue_file_path": None,
            },
        ]

        for test_case in source_tests:
            result = await repository.create(
                recording_id=uuid4(),
                source=test_case["source"],
                tracks=test_case["tracks"],
                cue_file_path=test_case["cue_file_path"],
            )

            assert result.source == test_case["source"]
            assert result.tracks == test_case["tracks"]
            assert result.cue_file_path == test_case["cue_file_path"]


class TestAsyncTracklistRepositoryIntegration:
    """Integration-style tests that test multiple repository methods together."""

    @pytest.fixture
    def mock_db_manager_integration(self):
        """Create a mock database manager for integration tests."""
        manager = MagicMock(spec=AsyncDatabaseManager)

        # Create async context manager mock that can be reused
        async_session_mock = AsyncMock()
        async_session_mock.__aenter__ = AsyncMock(return_value=async_session_mock)
        async_session_mock.__aexit__ = AsyncMock(return_value=None)
        async_session_mock.add = MagicMock()
        async_session_mock.flush = AsyncMock()
        async_session_mock.refresh = AsyncMock()
        async_session_mock.commit = AsyncMock()

        manager.get_db_session.return_value = async_session_mock
        return manager, async_session_mock

    @pytest.mark.asyncio
    async def test_create_then_get_workflow(self, mock_db_manager_integration):
        """Test complete workflow: create tracklist then retrieve it."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncTracklistRepository(mock_db_manager)

        # Setup for create operation
        recording_id = uuid4()
        tracklist_id = uuid4()

        tracks = [{"track_number": 1, "title": "Test Track 1"}, {"track_number": 2, "title": "Test Track 2"}]

        # Mock the created tracklist
        created_tracklist = Tracklist(
            id=tracklist_id, recording_id=recording_id, source="test", tracks=tracks, cue_file_path="/test.cue"
        )

        # Create tracklist
        with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.side_effect = lambda obj: setattr(obj, "id", tracklist_id)

            result = await repository.create(
                recording_id=recording_id, source="test", tracks=tracks, cue_file_path="/test.cue"
            )

        # Verify creation
        assert result.recording_id == recording_id
        assert result.source == "test"
        assert len(result.tracks) == 2

        # Setup for get operation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = created_tracklist
        session.execute = AsyncMock(return_value=mock_result)

        # Get the created tracklist
        retrieved = await repository.get_by_recording_id(recording_id)

        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == tracklist_id
        assert retrieved.source == "test"
        assert len(retrieved.tracks) == 2

    @pytest.mark.asyncio
    async def test_create_update_get_workflow(self, mock_db_manager_integration):
        """Test workflow: create, update, then get tracklist."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncTracklistRepository(mock_db_manager)

        recording_id = uuid4()
        tracklist_id = uuid4()

        initial_tracks = [{"track_number": 1, "title": "Initial Track"}]
        updated_tracks = [{"track_number": 1, "title": "Updated Track 1"}, {"track_number": 2, "title": "New Track 2"}]

        # Step 1: Create
        with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.side_effect = lambda obj: setattr(obj, "id", tracklist_id)

            await repository.create(recording_id=recording_id, source="test", tracks=initial_tracks)

        # Step 2: Update
        updated_tracklist = Tracklist(
            id=tracklist_id, recording_id=recording_id, source="test", tracks=updated_tracks, cue_file_path=None
        )

        mock_update_result = MagicMock()
        mock_update_result.scalar_one_or_none.return_value = updated_tracklist
        session.execute = AsyncMock(return_value=mock_update_result)

        updated = await repository.update_tracks(recording_id, updated_tracks)

        assert updated is not None
        assert len(updated.tracks) == 2

        # Step 3: Get updated result
        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = updated_tracklist
        session.execute = AsyncMock(return_value=mock_get_result)

        retrieved = await repository.get_by_recording_id(recording_id)

        # Verify final state
        assert retrieved is not None
        assert len(retrieved.tracks) == 2
        assert retrieved.tracks[0]["title"] == "Updated Track 1"
        assert retrieved.tracks[1]["title"] == "New Track 2"

    @pytest.mark.asyncio
    async def test_multiple_recordings_workflow(self, mock_db_manager_integration):
        """Test workflow with tracklists for multiple recordings."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncTracklistRepository(mock_db_manager)

        # Create tracklists for multiple recordings
        recording_ids = [uuid4() for _ in range(3)]
        tracklist_data = [
            {"source": "manual", "tracks": [{"track_number": 1, "title": "Manual Track"}]},
            {"source": "cue", "tracks": [{"track_number": 1, "title": "CUE Track"}], "cue_file_path": "/test.cue"},
            {"source": "web", "tracks": [{"track_number": 1, "title": "Web Track"}]},
        ]

        # Create all tracklists
        for i, recording_id in enumerate(recording_ids):
            data = tracklist_data[i]

            with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
                mock_refresh.side_effect = lambda obj: setattr(obj, "id", uuid4())

                await repository.create(
                    recording_id=recording_id,
                    source=data["source"],
                    tracks=data["tracks"],
                    cue_file_path=data.get("cue_file_path"),
                )

        # Verify each can be retrieved independently
        for i, recording_id in enumerate(recording_ids):
            expected_tracklist = Tracklist(
                id=uuid4(),
                recording_id=recording_id,
                source=tracklist_data[i]["source"],
                tracks=tracklist_data[i]["tracks"],
                cue_file_path=tracklist_data[i].get("cue_file_path"),
            )

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = expected_tracklist
            session.execute = AsyncMock(return_value=mock_result)

            retrieved = await repository.get_by_recording_id(recording_id)

            assert retrieved is not None
            assert retrieved.source == tracklist_data[i]["source"]
            assert retrieved.tracks == tracklist_data[i]["tracks"]

    @pytest.mark.asyncio
    async def test_tracklist_evolution_workflow(self, mock_db_manager_integration):
        """Test tracklist evolving from empty to complete."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncTracklistRepository(mock_db_manager)

        recording_id = uuid4()

        # Step 1: Create empty tracklist
        with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
            tracklist_id = uuid4()
            mock_refresh.side_effect = lambda obj: setattr(obj, "id", tracklist_id)

            await repository.create(recording_id=recording_id, source="auto", tracks=[])

        # Step 2: Add basic track info
        basic_tracks = [{"track_number": 1, "title": "Track 1"}, {"track_number": 2, "title": "Track 2"}]

        updated_tracklist = Tracklist(
            id=tracklist_id, recording_id=recording_id, source="auto", tracks=basic_tracks, cue_file_path=None
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = updated_tracklist
        session.execute = AsyncMock(return_value=mock_result)

        updated1 = await repository.update_tracks(recording_id, basic_tracks)
        assert len(updated1.tracks) == 2

        # Step 3: Add detailed metadata
        detailed_tracks = [
            {
                "track_number": 1,
                "title": "Track 1",
                "artist": "Artist 1",
                "bpm": 128.0,
                "key": "C major",
                "start_time": "00:00:00",
                "duration": "03:45",
            },
            {
                "track_number": 2,
                "title": "Track 2",
                "artist": "Artist 2",
                "bpm": 132.0,
                "key": "G minor",
                "start_time": "03:45:00",
                "duration": "04:20",
            },
        ]

        final_tracklist = Tracklist(
            id=tracklist_id,
            recording_id=recording_id,
            source="manual",  # Source changed to manual
            tracks=detailed_tracks,
            cue_file_path="/final.cue",
        )

        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = final_tracklist
        session.execute = AsyncMock(return_value=mock_result2)

        updated2 = await repository.update_tracks(recording_id, detailed_tracks)

        # Verify evolution completed successfully
        assert len(updated2.tracks) == 2
        assert updated2.tracks[0]["bpm"] == 128.0
        assert updated2.tracks[1]["key"] == "G minor"
        assert "artist" in updated2.tracks[0]
        assert "duration" in updated2.tracks[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
