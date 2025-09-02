"""
Integration tests for AsyncAnalysisResultRepository with real database operations.

Tests comprehensive CRUD operations, transaction rollback, error handling,
and data persistence using real PostgreSQL database.
"""

import asyncio
import logging
import os
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from services.analysis_service.src.repositories import AsyncAnalysisResultRepository
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import AnalysisResult, Recording

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://tracktion:tracktion_password@localhost:5432/tracktion_test"
)


@pytest.fixture(scope="module")
async def db_manager():
    """Create async database manager for tests."""
    manager = AsyncDatabaseManager()
    # Initialize with test database
    await manager.initialize(TEST_DATABASE_URL)

    # Ensure tables exist
    async with manager.get_db_session() as session:
        # Check if analysis_results table exists
        result = await session.execute(
            text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'analysis_results')")
        )
        table_exists = result.scalar()
        if not table_exists:
            logger.warning("analysis_results table does not exist, tests may fail")

    yield manager
    await manager.close()


@pytest.fixture
async def repository(db_manager):
    """Create repository instance with database manager."""
    return AsyncAnalysisResultRepository(db_manager)


@pytest.fixture
async def test_recording(db_manager):
    """Create a test recording for analysis results."""
    recording = Recording(
        file_path="/test/path/recording.wav",
        file_name="recording.wav",
        file_size=1024000,
        processing_status="pending",
        sha256_hash="test_hash_123",
        xxh128_hash="test_xxh_456",
    )

    async with db_manager.get_db_session() as session:
        session.add(recording)
        await session.flush()
        await session.refresh(recording)
        await session.commit()

    yield recording

    # Cleanup
    async with db_manager.get_db_session() as session:
        await session.execute(
            text("DELETE FROM analysis_results WHERE recording_id = :recording_id"), {"recording_id": recording.id}
        )
        await session.execute(text("DELETE FROM recordings WHERE id = :recording_id"), {"recording_id": recording.id})
        await session.commit()


class TestAsyncAnalysisResultRepositoryCRUD:
    """Test basic CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_analysis_result(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test creating a new analysis result."""
        result_data = {"bpm": 128.5, "confidence": 0.95, "method": "beattrack"}

        analysis_result = await repository.create(
            recording_id=test_recording.id,
            analysis_type="bpm",
            result_data=result_data,
            confidence_score=0.95,
            status="completed",
            processing_time_ms=1500,
        )

        assert analysis_result is not None
        assert analysis_result.id is not None
        assert analysis_result.recording_id == test_recording.id
        assert analysis_result.analysis_type == "bpm"
        assert analysis_result.result_data == result_data
        assert analysis_result.confidence_score == 0.95
        assert analysis_result.status == "completed"
        assert analysis_result.processing_time_ms == 1500
        assert analysis_result.created_at is not None
        assert analysis_result.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_analysis_result_by_id(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test retrieving analysis result by ID."""
        # Create test data
        result_data = {"key": "C major", "confidence": 0.89}
        created_result = await repository.create(
            recording_id=test_recording.id,
            analysis_type="key",
            result_data=result_data,
            confidence_score=0.89,
            status="completed",
        )

        # Retrieve by ID
        retrieved_result = await repository.get_by_id(created_result.id)

        assert retrieved_result is not None
        assert retrieved_result.id == created_result.id
        assert retrieved_result.recording_id == test_recording.id
        assert retrieved_result.analysis_type == "key"
        assert retrieved_result.result_data == result_data
        assert retrieved_result.confidence_score == 0.89
        assert retrieved_result.status == "completed"

        # Test with relationships loaded
        assert hasattr(retrieved_result, "recording")

    @pytest.mark.asyncio
    async def test_get_analysis_results_by_recording_id(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test retrieving all analysis results for a recording."""
        # Create multiple analysis results
        bpm_result = await repository.create(
            recording_id=test_recording.id, analysis_type="bpm", result_data={"bpm": 130.0}, status="completed"
        )

        key_result = await repository.create(
            recording_id=test_recording.id, analysis_type="key", result_data={"key": "D minor"}, status="completed"
        )

        mood_result = await repository.create(
            recording_id=test_recording.id, analysis_type="mood", result_data={"mood": "energetic"}, status="pending"
        )

        # Retrieve all results for recording
        results = await repository.get_by_recording_id(test_recording.id)

        assert len(results) == 3
        result_ids = {result.id for result in results}
        assert bpm_result.id in result_ids
        assert key_result.id in result_ids
        assert mood_result.id in result_ids

        # Verify ordering (should be by created_at desc)
        assert results[0].created_at >= results[1].created_at
        assert results[1].created_at >= results[2].created_at

    @pytest.mark.asyncio
    async def test_get_analysis_result_by_recording_and_type(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test retrieving analysis result by recording ID and type."""
        # Create multiple BPM results (older and newer)
        await repository.create(
            recording_id=test_recording.id, analysis_type="bpm", result_data={"bpm": 120.0}, status="completed"
        )

        # Wait a moment to ensure different timestamps
        await asyncio.sleep(0.1)

        new_bpm = await repository.create(
            recording_id=test_recording.id, analysis_type="bpm", result_data={"bpm": 125.0}, status="completed"
        )

        # Should return the most recent completed result
        result = await repository.get_by_recording_and_type(test_recording.id, "bpm")

        assert result is not None
        assert result.id == new_bpm.id
        assert result.result_data["bpm"] == 125.0

        # Test with non-existent type
        no_result = await repository.get_by_recording_and_type(test_recording.id, "nonexistent")
        assert no_result is None

    @pytest.mark.asyncio
    async def test_get_completed_results_for_recording(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test retrieving formatted completed results."""
        # Create multiple completed results
        await repository.create(
            recording_id=test_recording.id, analysis_type="bpm", result_data={"bpm": 128.5}, status="completed"
        )

        await repository.create(
            recording_id=test_recording.id, analysis_type="key", result_data={"key": "C major"}, status="completed"
        )

        await repository.create(
            recording_id=test_recording.id, analysis_type="mood", result_data={"mood": "uplifting"}, status="completed"
        )

        # Create a pending result (should not be included)
        await repository.create(
            recording_id=test_recording.id, analysis_type="tempo", result_data={"tempo": "fast"}, status="pending"
        )

        results_dict = await repository.get_completed_results_for_recording(test_recording.id)

        assert "bpm" in results_dict
        assert "key" in results_dict
        assert "mood" in results_dict
        assert "tempo" not in results_dict  # Pending result should not be included

        assert results_dict["bpm"] == 128.5
        assert results_dict["key"] == "C major"
        assert results_dict["mood"] == "uplifting"


class TestAsyncAnalysisResultRepositoryUpdates:
    """Test update operations."""

    @pytest.mark.asyncio
    async def test_update_status(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test updating analysis result status."""
        # Create test result
        analysis_result = await repository.create(
            recording_id=test_recording.id, analysis_type="bpm", result_data={"bpm": 130.0}, status="pending"
        )

        # Update status to completed
        result_data = {"bpm": 130.0, "confidence": 0.92}
        success = await repository.update_status(
            analysis_result.id, status="completed", result_data=result_data, processing_time_ms=2000
        )

        assert success is True

        # Verify update
        updated_result = await repository.get_by_id(analysis_result.id)
        assert updated_result is not None
        assert updated_result.status == "completed"
        assert updated_result.result_data == result_data
        assert updated_result.processing_time_ms == 2000
        assert updated_result.updated_at > analysis_result.updated_at

    @pytest.mark.asyncio
    async def test_update_status_with_error(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test updating analysis result status with error."""
        # Create test result
        analysis_result = await repository.create(recording_id=test_recording.id, analysis_type="key", status="pending")

        # Update status to failed with error
        success = await repository.update_status(
            analysis_result.id, status="failed", error_message="Analysis timeout after 30 seconds"
        )

        assert success is True

        # Verify update
        updated_result = await repository.get_by_id(analysis_result.id)
        assert updated_result is not None
        assert updated_result.status == "failed"
        assert updated_result.error_message == "Analysis timeout after 30 seconds"

    @pytest.mark.asyncio
    async def test_update_result(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test updating analysis result with final data."""
        # Create test result
        analysis_result = await repository.create(
            recording_id=test_recording.id, analysis_type="mood", status="pending"
        )

        # Update with final result data
        final_data = {"mood": "energetic", "energy_level": 0.85, "valence": 0.72}

        success = await repository.update_result(
            analysis_result.id,
            result_data=final_data,
            confidence_score=0.88,
            status="completed",
            processing_time_ms=3500,
        )

        assert success is True

        # Verify update
        updated_result = await repository.get_by_id(analysis_result.id)
        assert updated_result is not None
        assert updated_result.status == "completed"
        assert updated_result.result_data == final_data
        assert updated_result.confidence_score == 0.88
        assert updated_result.processing_time_ms == 3500

    @pytest.mark.asyncio
    async def test_update_nonexistent_result(self, repository: AsyncAnalysisResultRepository):
        """Test updating non-existent analysis result."""
        fake_id = uuid.uuid4()

        success = await repository.update_status(fake_id, status="completed", result_data={"test": "data"})

        assert success is False


class TestAsyncAnalysisResultRepositoryTransactions:
    """Test transaction behavior and rollback scenarios."""

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, db_manager: AsyncDatabaseManager, test_recording: Recording):
        """Test that database transactions rollback properly on errors."""
        repository = AsyncAnalysisResultRepository(db_manager)

        # Count initial results
        initial_results = await repository.get_by_recording_id(test_recording.id)
        initial_count = len(initial_results)

        # Attempt to create invalid analysis result (this should fail)
        try:
            async with db_manager.get_db_session() as session:
                # Create valid result first
                analysis_result = AnalysisResult(
                    recording_id=test_recording.id,
                    analysis_type="test",
                    result_data={"test": "data"},
                    status="completed",
                )
                session.add(analysis_result)
                await session.flush()

                # Force an error by violating a constraint
                # This simulates a database constraint violation
                await session.execute(
                    text(
                        "INSERT INTO analysis_results (id, recording_id, analysis_type, status) "
                        "VALUES (:id, :recording_id, :analysis_type, :status)"
                    ),
                    {
                        "id": analysis_result.id,  # Duplicate ID should cause error
                        "recording_id": test_recording.id,
                        "analysis_type": "duplicate",
                        "status": "pending",
                    },
                )
                await session.commit()
        except Exception:
            # Expected to fail
            pass

        # Verify no results were created due to rollback
        final_results = await repository.get_by_recording_id(test_recording.id)
        final_count = len(final_results)
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_concurrent_access(self, db_manager: AsyncDatabaseManager, test_recording: Recording):
        """Test concurrent access to the same recording."""
        repository1 = AsyncAnalysisResultRepository(db_manager)
        repository2 = AsyncAnalysisResultRepository(db_manager)

        async def create_analysis_result(repo, analysis_type: str, delay: float):
            """Create an analysis result with delay."""
            await asyncio.sleep(delay)
            return await repo.create(
                recording_id=test_recording.id,
                analysis_type=analysis_type,
                result_data={analysis_type: "test_value"},
                status="completed",
            )

        # Create analysis results concurrently
        tasks = [
            create_analysis_result(repository1, "concurrent_test_1", 0.1),
            create_analysis_result(repository2, "concurrent_test_2", 0.05),
            create_analysis_result(repository1, "concurrent_test_3", 0.15),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result.id is not None

        # Verify all were created
        all_results = await repository1.get_by_recording_id(test_recording.id)
        concurrent_results = [r for r in all_results if r.analysis_type.startswith("concurrent_test_")]
        assert len(concurrent_results) == 3


class TestAsyncAnalysisResultRepositoryErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_recording_id(self, repository: AsyncAnalysisResultRepository):
        """Test creating analysis result with invalid recording ID."""
        fake_recording_id = uuid.uuid4()

        with pytest.raises(IntegrityError):  # Should raise foreign key constraint error
            await repository.create(
                recording_id=fake_recording_id, analysis_type="bpm", result_data={"bpm": 120.0}, status="completed"
            )

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test creating analysis result with missing required fields."""
        with pytest.raises(IntegrityError):
            await repository.create(
                recording_id=test_recording.id,
                analysis_type="",  # Empty analysis type should fail
                status="completed",
            )

    @pytest.mark.asyncio
    async def test_database_connection_error(self, test_recording: Recording):
        """Test behavior when database connection fails."""
        # Create repository with invalid database URL
        invalid_manager = AsyncDatabaseManager()
        repository = AsyncAnalysisResultRepository(invalid_manager)

        with pytest.raises(OperationalError):  # Should raise connection error
            await repository.get_by_recording_id(test_recording.id)


class TestAsyncAnalysisResultRepositoryPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test performance of bulk operations."""

        # Create multiple analysis results
        start_time = time.time()

        analysis_types = ["bpm", "key", "mood", "tempo", "genre"]
        created_results = []

        for i, analysis_type in enumerate(analysis_types):
            result = await repository.create(
                recording_id=test_recording.id,
                analysis_type=f"{analysis_type}_{i}",
                result_data={analysis_type: f"value_{i}"},
                status="completed",
                confidence_score=0.8 + (i * 0.02),
            )
            created_results.append(result)

        creation_time = time.time() - start_time

        # Retrieve all results
        start_time = time.time()
        retrieved_results = await repository.get_by_recording_id(test_recording.id)
        retrieval_time = time.time() - start_time

        # Performance assertions (adjust thresholds as needed)
        assert creation_time < 2.0  # Should create 5 results in under 2 seconds
        assert retrieval_time < 0.5  # Should retrieve results in under 500ms
        assert len(retrieved_results) >= len(created_results)

        logger.info(f"Created {len(created_results)} results in {creation_time:.3f}s")
        logger.info(f"Retrieved {len(retrieved_results)} results in {retrieval_time:.3f}s")

    @pytest.mark.asyncio
    async def test_large_result_data_handling(
        self, repository: AsyncAnalysisResultRepository, test_recording: Recording
    ):
        """Test handling of large result data."""
        # Create large result data (simulating detailed analysis)
        large_result_data = {
            "detailed_analysis": {
                "samples": list(range(1000)),  # 1000 data points
                "features": {f"feature_{i}": i * 0.1 for i in range(100)},
                "metadata": {
                    "analysis_version": "2.1.0",
                    "parameters": {f"param_{i}": f"value_{i}" for i in range(50)},
                    "timestamps": [f"2024-01-01T00:{i:02d}:00Z" for i in range(60)],
                },
            }
        }

        # Create analysis result with large data
        start_time = time.time()
        analysis_result = await repository.create(
            recording_id=test_recording.id,
            analysis_type="detailed_spectrum",
            result_data=large_result_data,
            status="completed",
        )
        creation_time = time.time() - start_time

        # Retrieve and verify
        start_time = time.time()
        retrieved_result = await repository.get_by_id(analysis_result.id)
        retrieval_time = time.time() - start_time

        assert retrieved_result is not None
        assert retrieved_result.result_data == large_result_data
        assert len(retrieved_result.result_data["detailed_analysis"]["samples"]) == 1000

        # Performance should still be reasonable
        assert creation_time < 5.0  # Large data creation under 5 seconds
        assert retrieval_time < 2.0  # Large data retrieval under 2 seconds

        logger.info(f"Created large result data in {creation_time:.3f}s")
        logger.info(f"Retrieved large result data in {retrieval_time:.3f}s")


@pytest.mark.integration
@pytest.mark.requires_docker
class TestAsyncAnalysisResultRepositoryIntegration:
    """Integration tests requiring full database setup."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, repository: AsyncAnalysisResultRepository, test_recording: Recording):
        """Test complete analysis workflow from creation to completion."""
        # Step 1: Create pending analysis
        pending_result = await repository.create(
            recording_id=test_recording.id, analysis_type="comprehensive", status="pending"
        )

        assert pending_result.status == "pending"
        assert pending_result.result_data is None
        assert pending_result.processing_time_ms is None

        # Step 2: Update to running
        await repository.update_status(pending_result.id, status="running")

        running_result = await repository.get_by_id(pending_result.id)
        assert running_result.status == "running"

        # Step 3: Complete with results
        final_data = {
            "bpm": 126.8,
            "key": "A minor",
            "mood": "contemplative",
            "energy": 0.65,
            "danceability": 0.72,
            "analysis_details": {
                "peak_frequencies": [60, 250, 1000, 4000],
                "spectral_centroid": 2500.5,
                "zero_crossing_rate": 0.12,
            },
        }

        await repository.update_result(
            pending_result.id,
            result_data=final_data,
            confidence_score=0.91,
            status="completed",
            processing_time_ms=15000,
        )

        # Step 4: Verify final state
        completed_result = await repository.get_by_id(pending_result.id)
        assert completed_result.status == "completed"
        assert completed_result.result_data == final_data
        assert completed_result.confidence_score == 0.91
        assert completed_result.processing_time_ms == 15000

        # Step 5: Verify in completed results
        completed_results = await repository.get_completed_results_for_recording(test_recording.id)
        assert "bpm" in completed_results
        assert "key" in completed_results
        assert "mood" in completed_results
        assert completed_results["bpm"] == 126.8
        assert completed_results["key"] == "A minor"
        assert completed_results["mood"] == "contemplative"
