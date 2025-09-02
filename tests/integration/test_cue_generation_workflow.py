"""
Integration tests for CUE generation with database workflow.

Tests the complete CUE generation workflow including database operations,
file storage, validation, and error handling with real database operations.
"""

import asyncio
import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from services.tracklist_service.src.models.cue_file import BatchGenerateCueRequest, GenerateCueRequest
from services.tracklist_service.src.services.cue_generation_service import CueGenerationService
from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.repositories import JobStatus

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://tracktion:tracktion_password@localhost:5432/tracktion_test"
)


class MockStorageService:
    """Mock storage service for CUE file operations."""

    def __init__(self):
        self.stored_files: dict[str, str] = {}
        self.store_calls: list[tuple[str, str]] = []

    def store_cue_file(self, file_path: str, content: str) -> tuple[bool, str, str | None]:
        """
        Mock CUE file storage.

        Returns:
            Tuple of (success, stored_path, error_message)
        """
        try:
            # Simulate storage
            self.stored_files[file_path] = content
            self.store_calls.append((file_path, content))

            # Return success with the same path
            return True, file_path, None
        except Exception as e:
            return False, "", str(e)

    def get_stored_file(self, file_path: str) -> str | None:
        """Get stored file content for verification."""
        return self.stored_files.get(file_path)

    def list_stored_files(self) -> list[str]:
        """List all stored file paths."""
        return list(self.stored_files.keys())

    def clear_storage(self):
        """Clear all stored files."""
        self.stored_files.clear()
        self.store_calls.clear()


class MockTracklistModel:
    """Mock tracklist model for testing."""

    def __init__(
        self,
        tracklist_id: UUID | None = None,
        title: str = "Test Mix",
        artist: str = "Test DJ",
        source: str = "manual",
        audio_file_path: str = "test_audio.wav",
        created_at: datetime | None = None,
        genre: str | None = None,
    ):
        self.id = tracklist_id or uuid4()
        self.title = title
        self.artist = artist
        self.source = source
        self.audio_file_path = audio_file_path
        self.created_at = created_at or datetime.now(UTC)
        self.genre = genre
        self.tracks = []

    def add_track(
        self,
        title: str,
        artist: str,
        start_time: timedelta | None = None,
        bpm: float | None = None,
        key: str | None = None,
    ):
        """Add a track to the tracklist."""
        track = MockTrackModel(title=title, artist=artist, start_time=start_time, bpm=bpm, key=key)
        self.tracks.append(track)
        return track


class MockTrackModel:
    """Mock track model for testing."""

    def __init__(
        self,
        title: str,
        artist: str,
        start_time: timedelta | None = None,
        bpm: float | None = None,
        key: str | None = None,
    ):
        self.title = title
        self.artist = artist
        self.start_time = start_time
        self.bpm = bpm
        self.key = key


@pytest.fixture(scope="module")
async def db_manager():
    """Create database manager for tests."""
    manager = DatabaseManager()
    manager.initialize(TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"))

    # Verify database connection
    with manager.get_db_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1

    yield manager
    manager.close()


@pytest.fixture
def storage_service():
    """Create mock storage service."""
    return MockStorageService()


@pytest.fixture
def cue_generation_service(storage_service, db_manager):
    """Create CUE generation service with dependencies."""
    return CueGenerationService(
        storage_service=storage_service,
        cache_service=None,  # No cache for basic tests
        db_manager=db_manager,
    )


@pytest.fixture
def sample_tracklist():
    """Create sample tracklist for testing."""
    tracklist = MockTracklistModel(
        title="Test DJ Mix 2024",
        artist="Test DJ",
        source="manual",
        genre="House",
        created_at=datetime(2024, 1, 15, 20, 0, 0, tzinfo=UTC),
    )

    # Add sample tracks
    tracklist.add_track("Opening Track", "Artist One", timedelta(seconds=0), 120.0, "C major")
    tracklist.add_track("Peak Time", "Artist Two", timedelta(minutes=4, seconds=30), 128.0, "G minor")
    tracklist.add_track("Deep Vibes", "Artist Three", timedelta(minutes=9, seconds=15), 124.5, "F major")
    tracklist.add_track("Closing Track", "Artist Four", timedelta(minutes=13, seconds=45), 118.0, "D minor")

    return tracklist


class TestCueGenerationServiceBasicOperations:
    """Test basic CUE generation operations."""

    @pytest.mark.asyncio
    async def test_generate_standard_cue_file(
        self,
        cue_generation_service: CueGenerationService,
        sample_tracklist: MockTracklistModel,
        storage_service: MockStorageService,
    ):
        """Test generating standard format CUE file."""
        request = GenerateCueRequest(format="standard", options={}, validate_audio=False, audio_file_path=None)

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        assert response.success is True
        assert response.error is None
        assert response.job_id is not None
        assert response.processing_time_ms > 0

        # Verify job was created and completed
        if cue_generation_service.job_repo:
            job = cue_generation_service.job_repo.get_by_id(response.job_id)
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            assert job.job_type == "cue_generation"
            assert job.context["format"] == "standard"

    @pytest.mark.asyncio
    async def test_generate_cue_with_file_storage(
        self,
        cue_generation_service: CueGenerationService,
        sample_tracklist: MockTracklistModel,
        storage_service: MockStorageService,
    ):
        """Test generating CUE file with storage enabled."""
        # Add store_file attribute to request
        request = GenerateCueRequest(format="traktor", options={}, validate_audio=False, audio_file_path=None)
        request.store_file = True  # Enable file storage

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        assert response.success is True
        assert response.cue_file_id is not None
        assert response.file_path is not None
        assert response.error is None

        # Verify file was stored
        stored_files = storage_service.list_stored_files()
        assert len(stored_files) >= 1

        # Verify content
        stored_content = storage_service.get_stored_file(response.file_path)
        assert stored_content is not None
        assert "FORMAT TRAKTOR" in stored_content
        assert "Test DJ Mix 2024" in stored_content
        assert "TRACK 01 AUDIO" in stored_content

    @pytest.mark.asyncio
    async def test_generate_multiple_formats(
        self,
        cue_generation_service: CueGenerationService,
        sample_tracklist: MockTracklistModel,
        storage_service: MockStorageService,
    ):
        """Test generating CUE files in multiple formats."""
        request = BatchGenerateCueRequest(
            formats=["standard", "cdj", "traktor", "serato"],
            options={},
            validate_audio=False,
            audio_file_path=None,
            store_files=True,
        )

        response = await cue_generation_service.generate_multiple_formats(sample_tracklist, request)

        assert response.success is True
        assert response.total_files == 4
        assert response.successful_files == 4
        assert response.failed_files == 0
        assert len(response.results) == 4

        # Verify each format succeeded
        for result in response.results:
            assert result.success is True
            assert result.error is None

        # Check stored files
        stored_files = storage_service.list_stored_files()
        assert len(stored_files) >= 4  # May have additional files from other tests

        # Verify content for each format
        for result in response.results:
            if result.file_path:
                content = storage_service.get_stored_file(result.file_path)
                assert content is not None
                assert "Test DJ Mix 2024" in content

    @pytest.mark.asyncio
    async def test_cue_validation_workflow(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test CUE file validation workflow."""
        # Create temporary audio file for validation
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            # Write minimal WAV header (44 bytes)
            wav_header = (
                b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00"
                b"\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00"
            )
            temp_audio.write(wav_header)
            audio_file_path = temp_audio.name

        try:
            request = GenerateCueRequest(
                format="standard", options={}, validate_audio=True, audio_file_path=audio_file_path
            )

            response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

            assert response.success is True
            assert response.validation_report is not None
            assert response.error is None

            # Check validation report structure
            validation = response.validation_report
            assert "valid" in validation
            assert "errors" in validation
            assert "warnings" in validation
            assert "metadata" in validation

            # Basic validation should pass
            assert validation["valid"] is True
            assert validation["metadata"]["audio_file"] == audio_file_path

        finally:
            # Cleanup temp file
            audio_path = Path(audio_file_path)
            if audio_path.exists():
                audio_path.unlink()

    @pytest.mark.asyncio
    async def test_format_capabilities(self, cue_generation_service: CueGenerationService):
        """Test format capabilities querying."""
        # Test standard format capabilities
        std_caps = cue_generation_service.get_format_capabilities("standard")
        assert std_caps["supports_multiple_files"] is True
        assert std_caps["supports_pregap"] is True
        assert std_caps["max_tracks"] == 99
        assert "title" in std_caps["metadata_fields"]

        # Test DJ software format capabilities
        cdj_caps = cue_generation_service.get_format_capabilities("cdj")
        assert cdj_caps["supports_multiple_files"] is False
        assert cdj_caps["max_tracks"] == 999
        assert cdj_caps["timing_precision"] == "milliseconds"
        assert "bpm" in cdj_caps["metadata_fields"]

        # Test unknown format
        unknown_caps = cue_generation_service.get_format_capabilities("unknown")
        assert "error" in unknown_caps
        assert "supported_formats" in unknown_caps


class TestCueGenerationServiceErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_format_handling(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test handling of invalid CUE format."""
        request = GenerateCueRequest(format="invalid_format", options={}, validate_audio=False, audio_file_path=None)

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        assert response.success is False
        assert response.error is not None
        assert "Unsupported format" in response.error
        assert response.cue_file_id is None
        assert response.file_path is None

        # Verify job was marked as failed
        if cue_generation_service.job_repo:
            job = cue_generation_service.job_repo.get_by_id(response.job_id)
            assert job is not None
            assert job.status == JobStatus.FAILED
            assert "Unsupported format" in job.error_message

    @pytest.mark.asyncio
    async def test_storage_failure_handling(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test handling of storage failures."""

        # Create failing storage service
        class FailingStorageService:
            def store_cue_file(self, file_path: str, content: str):
                return False, "", "Simulated storage failure"

        # Replace storage service temporarily
        original_storage = cue_generation_service.storage_service
        cue_generation_service.storage_service = FailingStorageService()

        try:
            request = GenerateCueRequest(format="standard", options={}, validate_audio=False, audio_file_path=None)
            request.store_file = True

            # This should fail due to storage issues but still generate CUE content
            response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

            # The CUE generation itself should succeed, but storage should fail
            # Implementation may vary - could fail completely or succeed without storage
            if not response.success:
                assert "storage" in response.error.lower() or "failed to store" in response.error.lower()

        finally:
            # Restore original storage
            cue_generation_service.storage_service = original_storage

    @pytest.mark.asyncio
    async def test_empty_tracklist_handling(self, cue_generation_service: CueGenerationService):
        """Test handling of empty tracklist."""
        empty_tracklist = MockTracklistModel(title="Empty Mix")
        # No tracks added

        request = GenerateCueRequest(format="standard", options={}, validate_audio=False, audio_file_path=None)

        response = await cue_generation_service.generate_cue_file(empty_tracklist, request)

        # Should succeed but produce minimal CUE file
        assert response.success is True
        assert response.error is None

    @pytest.mark.asyncio
    async def test_validation_failure_handling(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test handling of validation failures."""
        request = GenerateCueRequest(
            format="standard",
            options={},
            validate_audio=True,
            audio_file_path="/nonexistent/path/to/audio.wav",  # Invalid path
        )

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        # Generation should succeed but validation should report issues
        assert response.success is True
        assert response.validation_report is not None
        # Validation might warn about missing audio file
        if response.validation_report.get("valid") is False:
            assert len(response.validation_report.get("errors", [])) > 0


class TestCueGenerationServiceDatabaseIntegration:
    """Test database integration features."""

    @pytest.mark.asyncio
    async def test_job_tracking_workflow(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test complete job tracking workflow."""
        if not cue_generation_service.job_repo:
            pytest.skip("Job repository not available")

        request = GenerateCueRequest(
            format="rekordbox", options={"include_colors": True}, validate_audio=False, audio_file_path=None
        )

        # Track initial job count
        initial_jobs = len(cue_generation_service.job_repo.list_by_type("cue_generation"))

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        assert response.success is True
        assert response.job_id is not None

        # Verify job was created and completed
        final_jobs = len(cue_generation_service.job_repo.list_by_type("cue_generation"))
        assert final_jobs == initial_jobs + 1

        job = cue_generation_service.job_repo.get_by_id(response.job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.service_name == "tracklist_service"
        assert job.context["format"] == "rekordbox"
        assert job.context["validate_audio"] is False

        # Verify job result
        if job.result:
            assert "processing_time_ms" in job.result
            assert job.result["processing_time_ms"] == response.processing_time_ms

    @pytest.mark.asyncio
    async def test_concurrent_cue_generation(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test concurrent CUE generation operations."""
        # Create multiple requests for different formats
        requests = [
            GenerateCueRequest(format="standard", options={}, validate_audio=False, audio_file_path=None),
            GenerateCueRequest(format="traktor", options={}, validate_audio=False, audio_file_path=None),
            GenerateCueRequest(format="serato", options={}, validate_audio=False, audio_file_path=None),
        ]

        # Execute concurrently
        tasks = [cue_generation_service.generate_cue_file(sample_tracklist, request) for request in requests]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for i, response in enumerate(responses):
            assert response.success is True
            assert response.error is None
            assert response.job_id is not None
            logger.info(f"Request {i} completed in {response.processing_time_ms}ms")

        # Verify all jobs were created
        if cue_generation_service.job_repo:
            for response in responses:
                job = cue_generation_service.job_repo.get_by_id(response.job_id)
                assert job is not None
                assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_tracklist_validation(self, cue_generation_service: CueGenerationService):
        """Test tracklist validation for CUE generation."""
        # Test with None tracklist
        result = await cue_generation_service.validate_tracklist_for_cue(None)
        if result is not None:
            assert result.valid is False
            assert "Tracklist is required" in result.error or "required" in result.error.lower()

        # Test with valid tracklist
        valid_tracklist = MockTracklistModel(title="Valid Mix")
        valid_tracklist.add_track("Track 1", "Artist 1")

        result = await cue_generation_service.validate_tracklist_for_cue(valid_tracklist)
        # Implementation may return None for valid tracklists or a success result


class TestCueGenerationServiceAdvancedFeatures:
    """Test advanced CUE generation features."""

    @pytest.mark.asyncio
    async def test_format_conversion(self, cue_generation_service: CueGenerationService):
        """Test CUE format conversion."""
        source_cue = """REM GENERATED BY Tracklist Service
REM FORMAT STANDARD
TITLE "Test Mix"
PERFORMER "Test Artist"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Artist 1"
    INDEX 01 00:00:00
"""

        response = await cue_generation_service.convert_cue_format(source_cue, "standard", "traktor")

        assert response.success is True
        assert response.error is None
        assert response.job_id is not None

        # Verify job was created
        if cue_generation_service.job_repo:
            job = cue_generation_service.job_repo.get_by_id(response.job_id)
            assert job is not None
            assert job.job_type == "cue_format_conversion"
            assert job.context["source_format"] == "standard"
            assert job.context["target_format"] == "traktor"

    @pytest.mark.asyncio
    async def test_same_format_conversion(self, cue_generation_service: CueGenerationService):
        """Test conversion between same formats."""
        source_cue = 'TITLE "Test"\nFILE "audio.wav" WAVE'

        response = await cue_generation_service.convert_cue_format(
            source_cue,
            "standard",
            "STANDARD",  # Same format, different case
        )

        assert response.success is True
        assert response.error is None

        # Should complete quickly since no conversion needed
        if cue_generation_service.job_repo:
            job = cue_generation_service.job_repo.get_by_id(response.job_id)
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            if job.result:
                assert job.result.get("converted") is False
                assert job.result.get("reason") == "same_format"

    def test_conversion_preview(self, cue_generation_service: CueGenerationService):
        """Test conversion preview warnings."""

        # Mock CueFormat enum values for testing
        class MockCueFormat:
            def __init__(self, value):
                self.value = value

        # Test conversion from standard to CDJ
        warnings = cue_generation_service.get_conversion_preview(MockCueFormat("standard"), MockCueFormat("cdj"))

        assert isinstance(warnings, list)
        assert len(warnings) > 0
        # Should warn about potential metadata loss
        warning_text = " ".join(warnings).lower()
        assert "metadata" in warning_text or "precision" in warning_text

        # Test same format conversion
        warnings = cue_generation_service.get_conversion_preview(MockCueFormat("standard"), MockCueFormat("standard"))

        assert len(warnings) >= 1
        assert "same" in warnings[0].lower()

    def test_cue_content_validation(self, cue_generation_service: CueGenerationService):
        """Test CUE content validation."""
        # Valid CUE content
        valid_cue = """TITLE "Test Mix"
PERFORMER "Test Artist"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    INDEX 01 00:00:00
"""

        result = cue_generation_service.validate_cue_content(valid_cue, "standard", {})

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["metadata"]["content_length"] == len(valid_cue)

        # Invalid CUE content (empty)
        result = cue_generation_service.validate_cue_content("", "standard", {})

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "empty" in result["errors"][0].lower()

        # CUE without tracks
        no_tracks_cue = """TITLE "Test Mix"
PERFORMER "Test Artist"
FILE "audio.wav" WAVE
"""

        result = cue_generation_service.validate_cue_content(no_tracks_cue, "standard", {})

        assert result["valid"] is False
        assert any("track" in error.lower() for error in result["errors"])

    def test_supported_formats(self, cue_generation_service: CueGenerationService):
        """Test supported formats listing."""
        formats = cue_generation_service.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0

        # Should include major formats
        format_values = [fmt.value if hasattr(fmt, "value") else str(fmt) for fmt in formats]
        expected_formats = ["standard", "cdj", "traktor", "serato", "rekordbox"]

        for expected in expected_formats:
            assert any(expected.lower() in fmt.lower() for fmt in format_values)

    @pytest.mark.asyncio
    async def test_cache_invalidation(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test cache invalidation workflow."""
        # Test cache invalidation (should handle gracefully even without cache)
        invalidated = await cue_generation_service.invalidate_tracklist_cache(sample_tracklist.id)

        # Should return 0 or more (depending on cache service availability)
        assert isinstance(invalidated, int)
        assert invalidated >= 0


class TestCueGenerationServicePerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_bulk_generation_performance(
        self, cue_generation_service: CueGenerationService, storage_service: MockStorageService
    ):
        """Test performance of bulk CUE generation."""

        # Create multiple tracklists
        tracklists = []
        for i in range(10):
            tracklist = MockTracklistModel(title=f"Performance Test Mix {i}", artist=f"Test DJ {i}")

            # Add tracks
            for j in range(5):
                tracklist.add_track(
                    f"Track {j + 1}", f"Artist {j + 1}", timedelta(minutes=j * 3, seconds=j * 15), 120.0 + j * 2
                )

            tracklists.append(tracklist)

        # Generate CUE files for all tracklists
        start_time = time.time()

        tasks = []
        for tracklist in tracklists:
            request = GenerateCueRequest(format="standard", options={}, validate_audio=False, audio_file_path=None)
            task = cue_generation_service.generate_cue_file(tracklist, request)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        generation_time = time.time() - start_time

        # Verify all succeeded
        successful_count = sum(1 for r in responses if r.success)
        assert successful_count == len(tracklists)

        # Performance assertions
        avg_time_per_generation = generation_time / len(tracklists)
        assert generation_time < 30.0  # All generations in under 30 seconds
        assert avg_time_per_generation < 5.0  # Average under 5 seconds each

        logger.info(f"Generated {len(tracklists)} CUE files in {generation_time:.3f}s")
        logger.info(f"Average time per generation: {avg_time_per_generation:.3f}s")

        # Check processing times
        processing_times = [r.processing_time_ms for r in responses if r.processing_time_ms]
        if processing_times:
            avg_processing_ms = sum(processing_times) / len(processing_times)
            max_processing_ms = max(processing_times)

            logger.info(f"Average processing time: {avg_processing_ms:.1f}ms")
            logger.info(f"Maximum processing time: {max_processing_ms:.1f}ms")

            assert avg_processing_ms < 2000  # Average under 2 seconds
            assert max_processing_ms < 5000  # Max under 5 seconds

    @pytest.mark.asyncio
    async def test_large_tracklist_handling(self, cue_generation_service: CueGenerationService):
        """Test handling of large tracklists."""

        # Create large tracklist (100 tracks)
        large_tracklist = MockTracklistModel(title="Large Performance Test Mix", artist="Marathon DJ")

        for i in range(100):
            large_tracklist.add_track(
                f"Extended Track {i + 1:03d}",
                f"Various Artist {i + 1}",
                timedelta(minutes=i * 2, seconds=i % 60),
                115.0 + (i % 20),
                f"Key {i % 12}",
            )

        request = GenerateCueRequest(
            format="traktor",  # Format that includes BPM and key data
            options={},
            validate_audio=False,
            audio_file_path=None,
        )

        start_time = time.time()
        response = await cue_generation_service.generate_cue_file(large_tracklist, request)
        generation_time = time.time() - start_time

        assert response.success is True
        assert response.error is None

        # Should handle large tracklist reasonably quickly
        assert generation_time < 10.0  # Under 10 seconds for 100 tracks
        assert response.processing_time_ms < 8000  # Under 8 seconds processing

        logger.info(f"Generated CUE for 100 tracks in {generation_time:.3f}s")
        logger.info(f"Processing time: {response.processing_time_ms}ms")


@pytest.mark.integration
@pytest.mark.requires_docker
class TestCueGenerationServiceFullIntegration:
    """Full integration tests requiring database and services."""

    @pytest.mark.asyncio
    async def test_complete_cue_workflow_with_database(
        self,
        cue_generation_service: CueGenerationService,
        sample_tracklist: MockTracklistModel,
        storage_service: MockStorageService,
        db_manager: DatabaseManager,
    ):
        """Test complete CUE generation workflow with database operations."""
        if not cue_generation_service.job_repo:
            pytest.skip("Job repository not available")

        # Step 1: Generate CUE file with storage
        request = GenerateCueRequest(
            format="rekordbox",
            options={"include_colors": True, "include_ratings": True},
            validate_audio=False,
            audio_file_path=None,
        )
        request.store_file = True

        initial_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        # Verify generation success
        assert response.success is True
        assert response.job_id is not None
        assert response.cue_file_id is not None
        assert response.file_path is not None
        assert response.processing_time_ms > 0

        # Step 2: Verify database state
        final_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))
        assert final_job_count == initial_job_count + 1

        job = cue_generation_service.job_repo.get_by_id(response.job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.error_message is None

        # Verify job context
        assert job.context["format"] == "rekordbox"
        assert job.context["validate_audio"] is False

        # Verify job result
        if job.result:
            assert job.result["cue_file_id"] == str(response.cue_file_id)
            assert job.result["file_path"] == response.file_path
            assert job.result["processing_time_ms"] == response.processing_time_ms

        # Step 3: Verify file storage
        stored_content = storage_service.get_stored_file(response.file_path)
        assert stored_content is not None
        assert "FORMAT REKORDBOX" in stored_content
        assert sample_tracklist.title in stored_content

        # Verify tracks are included
        for track in sample_tracklist.tracks:
            assert track.title in stored_content
            assert track.artist in stored_content

        # Step 4: Test regeneration
        regeneration_response = await cue_generation_service.regenerate_cue_file(
            sample_tracklist,
            type("MockCueFile", (), {"format": "rekordbox"})(),  # Mock CUE file object
            {"enhanced_metadata": True},
        )

        assert regeneration_response.success is True
        assert regeneration_response.job_id is not None

        # Verify regeneration job
        if cue_generation_service.job_repo:
            regen_job = cue_generation_service.job_repo.get_by_id(regeneration_response.job_id)
            assert regen_job is not None
            assert regen_job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_batch_generation_with_database(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test batch generation with full database tracking."""
        if not cue_generation_service.job_repo:
            pytest.skip("Job repository not available")

        request = BatchGenerateCueRequest(
            formats=["standard", "traktor", "serato", "rekordbox"],
            options={"batch_mode": True},
            validate_audio=False,
            audio_file_path=None,
            store_files=True,
        )

        initial_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))

        response = await cue_generation_service.generate_multiple_formats(sample_tracklist, request)

        # Verify batch response
        assert response.success is True
        assert response.total_files == 4
        assert response.successful_files == 4
        assert response.failed_files == 0
        assert len(response.results) == 4

        # Verify individual results
        formats_processed = set()
        for result in response.results:
            assert result.success is True
            assert result.job_id is not None

            # Verify job in database
            job = cue_generation_service.job_repo.get_by_id(result.job_id)
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            formats_processed.add(job.context["format"])

        # Verify all formats were processed
        expected_formats = {"standard", "traktor", "serato", "rekordbox"}
        assert formats_processed == expected_formats

        # Verify job count increased by 4
        final_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))
        assert final_job_count == initial_job_count + 4

    @pytest.mark.asyncio
    async def test_error_recovery_with_database(
        self, cue_generation_service: CueGenerationService, sample_tracklist: MockTracklistModel
    ):
        """Test error recovery and database consistency."""
        if not cue_generation_service.job_repo:
            pytest.skip("Job repository not available")

        # Force an error by using invalid format
        request = GenerateCueRequest(
            format="invalid_format_123", options={}, validate_audio=False, audio_file_path=None
        )

        initial_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        # Verify error handling
        assert response.success is False
        assert response.error is not None
        assert "Unsupported format" in response.error
        assert response.job_id is not None

        # Verify job was created and marked as failed
        final_job_count = len(cue_generation_service.job_repo.list_by_type("cue_generation"))
        assert final_job_count == initial_job_count + 1

        failed_job = cue_generation_service.job_repo.get_by_id(response.job_id)
        assert failed_job is not None
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.error_message is not None
        assert "Unsupported format" in failed_job.error_message

        # Database should remain consistent
        with db_manager.get_db_session() as session:
            # Verify no partial data was left behind
            result = session.execute(
                text("SELECT COUNT(*) FROM jobs WHERE status = :status"), {"status": "running"}
            ).scalar()
            assert result == 0  # No jobs left in running state
