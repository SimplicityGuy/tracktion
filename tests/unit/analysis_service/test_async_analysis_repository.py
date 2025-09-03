"""Comprehensive unit tests for AsyncAnalysisResultRepository class."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from services.analysis_service.src.repositories import AsyncAnalysisResultRepository
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import AnalysisResult, Recording


class TestAsyncAnalysisResultRepository:
    """Test async analysis result repository operations."""

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
        return AsyncAnalysisResultRepository(mock_db_manager)

    @pytest.fixture
    def sample_analysis_result(self):
        """Create a sample analysis result for testing."""
        return AnalysisResult(
            id=uuid4(),
            recording_id=uuid4(),
            analysis_type="bpm",
            result_data={"bpm": 128.0, "confidence": 0.95},
            confidence_score=Decimal("0.9500"),
            status="completed",
            processing_time_ms=1500,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_recording(self):
        """Create a sample recording for testing."""
        return Recording(
            id=uuid4(),
            file_path="/test/sample.mp3",
            file_name="sample.mp3",
            processing_status="completed",
        )

    # Basic CRUD Operations Tests

    @pytest.mark.asyncio
    async def test_create_analysis_result(self, repository, mock_db_manager):
        """Test creating a new analysis result."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create analysis result
        recording_id = uuid4()
        result = await repository.create(
            recording_id=recording_id,
            analysis_type="bpm",
            result_data={"bpm": 128.0, "confidence": 0.95},
            confidence_score=0.95,
            status="completed",
            processing_time_ms=1500,
        )

        # Verify
        assert result is not None
        assert result.recording_id == recording_id
        assert result.analysis_type == "bpm"
        assert result.result_data == {"bpm": 128.0, "confidence": 0.95}
        assert result.confidence_score == 0.95
        assert result.status == "completed"
        assert result.processing_time_ms == 1500

        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_analysis_result_minimal(self, repository, mock_db_manager):
        """Test creating analysis result with minimal required fields."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with minimal data
        recording_id = uuid4()
        result = await repository.create(
            recording_id=recording_id,
            analysis_type="key",
        )

        # Verify defaults
        assert result.recording_id == recording_id
        assert result.analysis_type == "key"
        assert result.result_data is None
        assert result.confidence_score is None
        assert result.status == "pending"
        assert result.processing_time_ms is None

    @pytest.mark.asyncio
    async def test_get_by_id(self, repository, mock_db_manager, sample_analysis_result):
        """Test getting analysis result by ID."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_analysis_result
        session.execute = AsyncMock(return_value=mock_result)

        # Get analysis result
        result = await repository.get_by_id(sample_analysis_result.id)

        # Verify
        assert result is not None
        assert result.id == sample_analysis_result.id
        assert result.analysis_type == "bpm"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_db_manager):
        """Test getting non-existent analysis result by ID."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Get non-existent analysis result
        result = await repository.get_by_id(uuid4())

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_recording_id(self, repository, mock_db_manager, sample_analysis_result):
        """Test getting all analysis results for a recording."""
        # Setup mock to return multiple results
        session = await mock_db_manager.get_db_session().__aenter__()
        analysis_results = [
            sample_analysis_result,
            AnalysisResult(
                id=uuid4(),
                recording_id=sample_analysis_result.recording_id,
                analysis_type="key",
                result_data={"key": "C major"},
                status="completed",
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = analysis_results
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get results
        results = await repository.get_by_recording_id(sample_analysis_result.recording_id)

        # Verify
        assert len(results) == 2
        assert results[0] == sample_analysis_result
        assert results[1].analysis_type == "key"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_recording_and_type(self, repository, mock_db_manager, sample_analysis_result):
        """Test getting analysis result by recording ID and type."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_analysis_result
        session.execute = AsyncMock(return_value=mock_result)

        # Get result
        result = await repository.get_by_recording_and_type(sample_analysis_result.recording_id, "bpm")

        # Verify
        assert result is not None
        assert result.analysis_type == "bpm"
        assert result.status == "completed"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_recording_and_type_not_found(self, repository, mock_db_manager):
        """Test getting non-existent analysis result by recording and type."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Get non-existent result
        result = await repository.get_by_recording_and_type(uuid4(), "nonexistent")

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_completed_results_for_recording(self, repository, mock_db_manager):
        """Test getting formatted completed analysis results."""
        # Setup mock data
        session = await mock_db_manager.get_db_session().__aenter__()

        # Create multiple analysis results
        analysis_results = [
            AnalysisResult(
                id=uuid4(), recording_id=uuid4(), analysis_type="bpm", result_data={"bpm": 128.0}, status="completed"
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=uuid4(),
                analysis_type="key",
                result_data={"key": "C major"},
                status="completed",
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=uuid4(),
                analysis_type="mood",
                result_data={"mood": "happy"},
                status="completed",
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = analysis_results
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get results
        recording_id = uuid4()
        results = await repository.get_completed_results_for_recording(recording_id)

        # Verify
        assert isinstance(results, dict)
        assert results["bpm"] == 128.0
        assert results["key"] == "C major"
        assert results["mood"] == "happy"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_completed_results_empty(self, repository, mock_db_manager):
        """Test getting completed results when none exist."""
        # Setup mock to return empty list
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get results
        results = await repository.get_completed_results_for_recording(uuid4())

        # Verify
        assert results == {}
        session.execute.assert_called_once()

    # Update Operations Tests

    @pytest.mark.asyncio
    async def test_update_status(self, repository, mock_db_manager):
        """Test updating analysis result status."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        # Update status
        analysis_id = uuid4()
        result = await repository.update_status(
            analysis_id=analysis_id, status="completed", result_data={"bpm": 130.0}, processing_time_ms=2000
        )

        # Verify
        assert result is True
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, repository, mock_db_manager):
        """Test updating status for non-existent analysis result."""
        # Setup mock to return 0 affected rows
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)

        # Update non-existent analysis
        result = await repository.update_status(analysis_id=uuid4(), status="completed")

        # Verify
        assert result is False
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_with_error(self, repository, mock_db_manager):
        """Test updating status with error message."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        # Update with error
        result = await repository.update_status(
            analysis_id=uuid4(), status="failed", error_message="Processing failed due to invalid audio format"
        )

        # Verify
        assert result is True
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_result(self, repository, mock_db_manager):
        """Test updating analysis result with final data."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        # Update result
        result = await repository.update_result(
            analysis_result_id=uuid4(),
            result_data={"bpm": 125.5, "confidence": 0.92},
            confidence_score=0.92,
            status="completed",
            processing_time_ms=1800,
        )

        # Verify
        assert result is True
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_result_not_found(self, repository, mock_db_manager):
        """Test updating result for non-existent analysis."""
        # Setup mock to return 0 affected rows
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)

        # Update non-existent analysis
        result = await repository.update_result(analysis_result_id=uuid4(), result_data={"bpm": 128.0})

        # Verify
        assert result is False
        session.execute.assert_called_once()

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_create_database_error(self, repository, mock_db_manager):
        """Test handling database errors during creation."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock(side_effect=IntegrityError("", "", ""))
        session.rollback = AsyncMock()

        # Attempt to create (should raise exception)
        with pytest.raises(IntegrityError):
            await repository.create(recording_id=uuid4(), analysis_type="bpm")

    @pytest.mark.asyncio
    async def test_get_by_id_database_error(self, repository, mock_db_manager):
        """Test handling database errors during retrieval."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Database connection failed"))

        # Attempt to get (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_id(uuid4())

    @pytest.mark.asyncio
    async def test_update_database_error(self, repository, mock_db_manager):
        """Test handling database errors during update."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Update failed"))

        # Attempt to update (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.update_status(uuid4(), "completed")

    # Concurrency and Transaction Tests

    @pytest.mark.asyncio
    async def test_concurrent_creates(self, mock_db_manager):
        """Test handling concurrent create operations."""
        # Create multiple repositories
        repos = [AsyncAnalysisResultRepository(mock_db_manager) for _ in range(5)]

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
            task = repo.create(recording_id=recording_id, analysis_type=f"type_{i}", status="pending")
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.recording_id == recording_ids[i]
            assert result.analysis_type == f"type_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, mock_db_manager):
        """Test handling concurrent read operations."""
        # Create multiple repositories
        repos = [AsyncAnalysisResultRepository(mock_db_manager) for _ in range(10)]

        # Setup sessions with different mock results
        sessions = []
        expected_results = []
        for i in range(10):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            # Create unique result for each session
            analysis_result = AnalysisResult(
                id=uuid4(), recording_id=uuid4(), analysis_type=f"type_{i}", status="completed"
            )
            expected_results.append(analysis_result)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = analysis_result
            session.execute = AsyncMock(return_value=mock_result)
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent reads
        tasks = []
        for i, repo in enumerate(repos):
            task = repo.get_by_id(expected_results[i].id)
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.id == expected_results[i].id
            assert result.analysis_type == f"type_{i}"

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, mock_db_manager):
        """Test transaction isolation between operations."""
        # Create two repositories
        repo1 = AsyncAnalysisResultRepository(mock_db_manager)
        repo2 = AsyncAnalysisResultRepository(mock_db_manager)

        # Setup separate sessions
        session1 = AsyncMock()
        session1.__aenter__ = AsyncMock(return_value=session1)
        session1.__aexit__ = AsyncMock(return_value=None)

        session2 = AsyncMock()
        session2.__aenter__ = AsyncMock(return_value=session2)
        session2.__aexit__ = AsyncMock(return_value=None)

        mock_db_manager.get_db_session.side_effect = [session1, session2]

        # Setup different update results
        result1 = MagicMock()
        result1.rowcount = 1
        session1.execute = AsyncMock(return_value=result1)

        result2 = MagicMock()
        result2.rowcount = 0  # Not found
        session2.execute = AsyncMock(return_value=result2)

        # Execute operations in different transactions
        analysis_id1 = uuid4()
        analysis_id2 = uuid4()

        update_result1 = await repo1.update_status(analysis_id1, "completed")
        update_result2 = await repo2.update_status(analysis_id2, "completed")

        # Verify isolation - each got different results
        assert update_result1 is True
        assert update_result2 is False
        assert session1.execute.call_count == 1
        assert session2.execute.call_count == 1

    # Data Validation Tests

    @pytest.mark.asyncio
    async def test_create_with_invalid_confidence_score(self, repository, mock_db_manager):
        """Test creating analysis with invalid confidence score."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with confidence score out of valid range
        result = await repository.create(
            recording_id=uuid4(),
            analysis_type="bpm",
            confidence_score=1.5,  # Invalid - should be <= 1.0
        )

        # Verify creation still works (database constraint would handle validation)
        assert result is not None
        assert result.confidence_score == 1.5

    @pytest.mark.asyncio
    async def test_create_with_large_result_data(self, repository, mock_db_manager):
        """Test creating analysis with large result data."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create large result data
        large_data = {
            "bpm": 128.0,
            "beats": [i * 0.5 for i in range(1000)],  # Large array
            "analysis_details": {"algorithm": "complex", "parameters": {f"param_{i}": i for i in range(100)}},
        }

        result = await repository.create(recording_id=uuid4(), analysis_type="bpm", result_data=large_data)

        # Verify creation with large data
        assert result is not None
        assert result.result_data == large_data

    @pytest.mark.asyncio
    async def test_result_data_json_serialization(self, repository, mock_db_manager):
        """Test that complex result data is properly handled."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create complex nested data structure
        complex_data = {
            "analysis": {
                "bpm": 128.0,
                "confidence": 0.95,
                "timestamps": [1.0, 2.0, 3.0],
                "metadata": {
                    "algorithm_version": "2.1",
                    "processing_options": {"window_size": 1024, "hop_length": 512, "use_onset_detection": True},
                },
            },
            "quality_metrics": {"snr": 45.2, "dynamic_range": 12.8},
        }

        result = await repository.create(recording_id=uuid4(), analysis_type="bpm", result_data=complex_data)

        # Verify complex data is preserved
        assert result is not None
        assert result.result_data == complex_data

    # Edge Cases and Special Scenarios

    @pytest.mark.asyncio
    async def test_get_completed_results_with_null_data(self, repository, mock_db_manager):
        """Test getting completed results when some have null result_data."""
        # Setup mock data with mixed null/valid data
        session = await mock_db_manager.get_db_session().__aenter__()

        analysis_results = [
            AnalysisResult(
                id=uuid4(),
                recording_id=uuid4(),
                analysis_type="bpm",
                result_data={"bpm": 128.0},  # Valid data
                status="completed",
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=uuid4(),
                analysis_type="key",
                result_data=None,  # Null data
                status="completed",
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=uuid4(),
                analysis_type="mood",
                result_data={"tempo": "fast"},  # Missing expected key
                status="completed",
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = analysis_results
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get results
        results = await repository.get_completed_results_for_recording(uuid4())

        # Verify only valid data is included
        assert "bpm" in results
        assert results["bpm"] == 128.0
        assert "key" not in results  # Null data excluded
        assert "mood" not in results  # Missing expected key excluded

    @pytest.mark.asyncio
    async def test_update_status_partial_update(self, repository, mock_db_manager):
        """Test partial status update with only some fields."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        # Update with only status (no other fields)
        result = await repository.update_status(
            analysis_id=uuid4(),
            status="in_progress",
            # No result_data, error_message, or processing_time_ms
        )

        # Verify update succeeded
        assert result is True
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_analysis_types_same_recording(self, repository, mock_db_manager):
        """Test handling multiple analysis types for the same recording."""
        # Setup mock session for multiple creates
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        recording_id = uuid4()
        analysis_types = ["bpm", "key", "mood", "tempo", "energy"]

        # Create multiple analysis results for same recording
        results = []
        for analysis_type in analysis_types:
            result = await repository.create(recording_id=recording_id, analysis_type=analysis_type, status="pending")
            results.append(result)

        # Verify all were created successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.recording_id == recording_id
            assert result.analysis_type == analysis_types[i]

    @pytest.mark.asyncio
    async def test_session_context_manager_exception_handling(self, mock_db_manager):
        """Test that session context manager properly handles exceptions."""
        repository = AsyncAnalysisResultRepository(mock_db_manager)

        # Setup mock session that raises exception
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(side_effect=Exception("Connection lost"))

        mock_db_manager.get_db_session.return_value = session

        # Verify exception is propagated and context manager cleanup occurs
        with pytest.raises(Exception, match="Connection lost"):
            await repository.get_by_id(uuid4())

        # Verify context manager cleanup was called
        session.__aenter__.assert_called_once()
        session.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization with database manager."""
        mock_manager = MagicMock(spec=AsyncDatabaseManager)

        repository = AsyncAnalysisResultRepository(mock_manager)

        assert repository.db is mock_manager
        assert hasattr(repository, "db")


class TestAsyncAnalysisRepositoryIntegration:
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
        """Test complete workflow: create analysis then retrieve it."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncAnalysisResultRepository(mock_db_manager)

        # Setup for create operation
        recording_id = uuid4()
        analysis_id = uuid4()

        # Mock the created analysis result
        created_analysis = AnalysisResult(
            id=analysis_id, recording_id=recording_id, analysis_type="bpm", result_data={"bpm": 128.0}, status="pending"
        )

        # Create analysis
        with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.side_effect = lambda obj: setattr(obj, "id", analysis_id)

            result = await repository.create(recording_id=recording_id, analysis_type="bpm", result_data={"bpm": 128.0})

        # Verify creation
        assert result.recording_id == recording_id
        assert result.analysis_type == "bpm"

        # Setup for get operation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = created_analysis
        session.execute = AsyncMock(return_value=mock_result)

        # Get the created analysis
        retrieved = await repository.get_by_id(analysis_id)

        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == analysis_id
        assert retrieved.analysis_type == "bpm"

    @pytest.mark.asyncio
    async def test_create_update_get_workflow(self, mock_db_manager_integration):
        """Test workflow: create, update, then get analysis result."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncAnalysisResultRepository(mock_db_manager)

        recording_id = uuid4()
        analysis_id = uuid4()

        # Step 1: Create
        with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.side_effect = lambda obj: setattr(obj, "id", analysis_id)

            await repository.create(recording_id=recording_id, analysis_type="bpm", status="pending")

        # Step 2: Update
        mock_update_result = MagicMock()
        mock_update_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update_result)

        updated = await repository.update_status(
            analysis_id=analysis_id,
            status="completed",
            result_data={"bpm": 128.0, "confidence": 0.95},
            processing_time_ms=1500,
        )

        assert updated is True

        # Step 3: Get updated result
        updated_analysis = AnalysisResult(
            id=analysis_id,
            recording_id=recording_id,
            analysis_type="bpm",
            result_data={"bpm": 128.0, "confidence": 0.95},
            status="completed",
            processing_time_ms=1500,
        )

        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = updated_analysis
        session.execute = AsyncMock(return_value=mock_get_result)

        retrieved = await repository.get_by_id(analysis_id)

        # Verify final state
        assert retrieved is not None
        assert retrieved.status == "completed"
        assert retrieved.result_data == {"bpm": 128.0, "confidence": 0.95}
        assert retrieved.processing_time_ms == 1500

    @pytest.mark.asyncio
    async def test_batch_analysis_workflow(self, mock_db_manager_integration):
        """Test workflow with multiple analysis types for one recording."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncAnalysisResultRepository(mock_db_manager)

        recording_id = uuid4()
        analysis_types = ["bpm", "key", "mood"]

        # Create multiple analysis results
        created_analyses = []
        for analysis_type in analysis_types:
            analysis_id = uuid4()

            with patch.object(session, "refresh", new_callable=AsyncMock) as mock_refresh:
                mock_refresh.side_effect = lambda obj, aid=analysis_id: setattr(obj, "id", aid)

                created = await repository.create(
                    recording_id=recording_id, analysis_type=analysis_type, status="pending"
                )
                created_analyses.append(created)

        # Get all results for recording
        mock_analysis_results = [
            AnalysisResult(
                id=uuid4(),
                recording_id=recording_id,
                analysis_type="bpm",
                result_data={"bpm": 128.0},
                status="completed",
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=recording_id,
                analysis_type="key",
                result_data={"key": "C major"},
                status="completed",
            ),
            AnalysisResult(
                id=uuid4(),
                recording_id=recording_id,
                analysis_type="mood",
                result_data={"mood": "energetic"},
                status="completed",
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_analysis_results
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        results = await repository.get_by_recording_id(recording_id)

        # Verify all analyses were retrieved
        assert len(results) == 3
        types_found = {r.analysis_type for r in results}
        assert types_found == {"bpm", "key", "mood"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
