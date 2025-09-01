"""
Unit tests for CUE file models.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.cue_file import (
    BatchCueGenerationResponse,
    BatchGenerateCueRequest,
    ConvertCueRequest,
    ConvertCueResponse,
    CueFile,
    CueFileDB,
    CueFormat,
    CueGenerationJob,
    CueGenerationJobDB,
    CueGenerationResponse,
    CueGenerationStatus,
    GenerateCueRequest,
    ValidationResult,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist


class TestCueFile:
    """Test CueFile model."""

    def test_cue_file_creation(self):
        """Test creating a CUE file model."""
        tracklist_id = uuid4()
        cue_file = CueFile(
            tracklist_id=tracklist_id,
            file_path="/data/cue_files/test.cue",
            format=CueFormat.STANDARD,
            file_size=1024,
            checksum="abc123def456",
        )

        assert cue_file.tracklist_id == tracklist_id
        assert cue_file.format == CueFormat.STANDARD
        assert cue_file.file_size == 1024
        assert cue_file.version == 1
        assert cue_file.is_active is True
        assert isinstance(cue_file.metadata, dict)

    def test_cue_file_validation_negative_file_size(self):
        """Test validation fails for negative file size."""
        tracklist_id = uuid4()

        with pytest.raises(ValueError, match="File size cannot be negative"):
            CueFile(
                tracklist_id=tracklist_id,
                file_path="/data/cue_files/test.cue",
                format=CueFormat.STANDARD,
                file_size=-1,
                checksum="abc123def456",
            )

    def test_cue_file_validation_file_size_too_large(self):
        """Test validation fails for file size over 1MB."""
        tracklist_id = uuid4()

        with pytest.raises(ValueError, match="CUE file size cannot exceed 1MB"):
            CueFile(
                tracklist_id=tracklist_id,
                file_path="/data/cue_files/test.cue",
                format=CueFormat.STANDARD,
                file_size=2 * 1024 * 1024,  # 2MB
                checksum="abc123def456",
            )

    def test_cue_file_validation_invalid_version(self):
        """Test validation fails for version less than 1."""
        tracklist_id = uuid4()

        with pytest.raises(ValueError, match="Version must be positive"):
            CueFile(
                tracklist_id=tracklist_id,
                file_path="/data/cue_files/test.cue",
                format=CueFormat.STANDARD,
                file_size=1024,
                checksum="abc123def456",
                version=0,
            )


class TestCueGenerationJob:
    """Test CueGenerationJob model."""

    def test_cue_generation_job_creation(self):
        """Test creating a CUE generation job."""
        tracklist_id = uuid4()
        job = CueGenerationJob(tracklist_id=tracklist_id, format=CueFormat.CDJ)

        assert job.tracklist_id == tracklist_id
        assert job.format == CueFormat.CDJ
        assert job.status == CueGenerationStatus.PENDING
        assert job.progress == 0
        assert isinstance(job.options, dict)

    def test_cue_generation_job_with_validation_report(self):
        """Test job with validation report."""
        tracklist_id = uuid4()
        validation_report = ValidationResult(valid=True, audio_duration=3600.0, tracklist_duration=3580.0)

        job = CueGenerationJob(
            tracklist_id=tracklist_id,
            format=CueFormat.TRAKTOR,
            validation_report=validation_report,
        )

        assert job.validation_report.valid is True
        assert job.validation_report.audio_duration == 3600.0


class TestValidationResult:
    """Test ValidationResult model."""

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert result.error is None
        assert len(result.warnings) == 0
        assert isinstance(result.metadata, dict)

    def test_validation_result_invalid(self):
        """Test invalid validation result with error."""
        result = ValidationResult(
            valid=False,
            error="Track timing exceeds audio duration",
            warnings=["Gap detected between tracks 3 and 4"],
        )

        assert result.valid is False
        assert result.error == "Track timing exceeds audio duration"
        assert len(result.warnings) == 1


class TestRequestModels:
    """Test API request models."""

    def test_generate_cue_request(self):
        """Test CUE generation request."""
        request = GenerateCueRequest(
            format=CueFormat.SERATO,
            options={"include_bpm": True},
            audio_file_path="/audio/mix.mp3",
        )

        assert request.format == CueFormat.SERATO
        assert request.options["include_bpm"] is True
        assert request.validate_audio is True

    def test_batch_generate_cue_request(self):
        """Test batch CUE generation request."""
        formats = [CueFormat.STANDARD, CueFormat.CDJ, CueFormat.TRAKTOR]
        request = BatchGenerateCueRequest(formats=formats)

        assert len(request.formats) == 3
        assert CueFormat.STANDARD in request.formats

    def test_batch_generate_cue_request_empty_formats(self):
        """Test batch request fails with empty formats."""
        with pytest.raises(ValueError, match="At least one format must be specified"):
            BatchGenerateCueRequest(formats=[])

    def test_batch_generate_cue_request_too_many_formats(self):
        """Test batch request fails with too many formats."""
        formats = [CueFormat.STANDARD] * 7  # More than 6

        with pytest.raises(ValueError, match="Too many formats specified"):
            BatchGenerateCueRequest(formats=formats)

    def test_convert_cue_request(self):
        """Test CUE conversion request."""
        request = ConvertCueRequest(target_format=CueFormat.REKORDBOX, preserve_metadata=False)

        assert request.target_format == CueFormat.REKORDBOX
        assert request.preserve_metadata is False


class TestResponseModels:
    """Test API response models."""

    def test_cue_generation_response(self):
        """Test CUE generation response."""
        job_id = uuid4()
        cue_file_id = uuid4()

        response = CueGenerationResponse(
            success=True,
            job_id=job_id,
            cue_file_id=cue_file_id,
            file_path="/data/cue_files/test.cue",
        )

        assert response.success is True
        assert response.job_id == job_id
        assert response.cue_file_id == cue_file_id

    def test_batch_cue_generation_response(self):
        """Test batch CUE generation response."""
        results = [CueGenerationResponse(success=True, job_id=uuid4(), cue_file_id=uuid4())]

        response = BatchCueGenerationResponse(
            success=True,
            results=results,
            total_files=3,
            successful_files=1,
            failed_files=2,
        )

        assert response.success is True
        assert len(response.results) == 1
        assert response.total_files == 3

    def test_convert_cue_response(self):
        """Test CUE conversion response."""
        cue_file_id = uuid4()

        response = ConvertCueResponse(
            success=True,
            cue_file_id=cue_file_id,
            file_path="/data/cue_files/converted.cue",
            warnings=["Some metadata was lost during conversion"],
        )

        assert response.success is True
        assert response.cue_file_id == cue_file_id
        assert len(response.warnings) == 1


class TestDatabaseModels:
    """Test SQLAlchemy database models."""

    def test_cue_file_db_to_model(self):
        """Test converting CueFileDB to Pydantic model."""
        tracklist_id = uuid4()
        cue_file_id = uuid4()
        now = datetime.now(UTC)

        db_model = CueFileDB(
            id=cue_file_id,
            tracklist_id=tracklist_id,
            file_path="/data/test.cue",
            format="standard",
            file_size=1024,
            checksum="abc123",
            created_at=now,
            updated_at=now,
            version=1,
            is_active=True,
            format_metadata={"encoding": "UTF-8"},
        )

        pydantic_model = db_model.to_model()

        assert isinstance(pydantic_model, CueFile)
        assert pydantic_model.id == cue_file_id
        assert pydantic_model.format == CueFormat.STANDARD
        assert pydantic_model.metadata["encoding"] == "UTF-8"

    def test_cue_file_from_model(self):
        """Test converting Pydantic model to CueFileDB."""
        tracklist_id = uuid4()

        pydantic_model = CueFile(
            tracklist_id=tracklist_id,
            file_path="/data/test.cue",
            format=CueFormat.CDJ,
            file_size=2048,
            checksum="def456",
        )

        db_model = CueFileDB.from_model(pydantic_model)

        assert db_model.tracklist_id == tracklist_id
        assert db_model.format == "cdj"
        assert db_model.file_size == 2048

    def test_cue_generation_job_db_to_model(self):
        """Test converting CueGenerationJobDB to Pydantic model."""
        tracklist_id = uuid4()
        job_id = uuid4()
        now = datetime.now(UTC)

        db_model = CueGenerationJobDB(
            id=job_id,
            tracklist_id=tracklist_id,
            format="traktor",
            status="processing",
            created_at=now,
            progress=50,
            options={"include_bpm": True},
        )

        pydantic_model = db_model.to_model()

        assert isinstance(pydantic_model, CueGenerationJob)
        assert pydantic_model.id == job_id
        assert pydantic_model.status == CueGenerationStatus.PROCESSING
        assert pydantic_model.progress == 50

    def test_cue_generation_job_from_model(self):
        """Test converting Pydantic model to CueGenerationJobDB."""
        tracklist_id = uuid4()

        pydantic_model = CueGenerationJob(
            tracklist_id=tracklist_id,
            format=CueFormat.SERATO,
            status=CueGenerationStatus.COMPLETED,
            progress=100,
        )

        db_model = CueGenerationJobDB.from_model(pydantic_model)

        assert db_model.tracklist_id == tracklist_id
        assert db_model.format == "serato"
        assert db_model.status == "completed"
        assert db_model.progress == 100

    def test_cue_generation_job_db_with_invalid_validation_report(self):
        """Test handling invalid validation report in database."""
        tracklist_id = uuid4()
        job_id = uuid4()
        now = datetime.now(UTC)

        db_model = CueGenerationJobDB(
            id=job_id,
            tracklist_id=tracklist_id,
            format="standard",
            status="pending",
            created_at=now,
            progress=0,
            validation_report={"invalid": "data"},  # Invalid structure
        )

        pydantic_model = db_model.to_model()

        # Should handle invalid data gracefully
        assert pydantic_model.validation_report is None


class TestEnums:
    """Test enum definitions."""

    def test_cue_format_values(self):
        """Test CueFormat enum values."""
        assert CueFormat.STANDARD == "standard"
        assert CueFormat.CDJ == "cdj"
        assert CueFormat.TRAKTOR == "traktor"
        assert CueFormat.SERATO == "serato"
        assert CueFormat.REKORDBOX == "rekordbox"
        assert CueFormat.KODI == "kodi"

    def test_cue_generation_status_values(self):
        """Test CueGenerationStatus enum values."""
        assert CueGenerationStatus.PENDING == "pending"
        assert CueGenerationStatus.PROCESSING == "processing"
        assert CueGenerationStatus.COMPLETED == "completed"
        assert CueGenerationStatus.FAILED == "failed"


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist for testing."""

    tracks = [
        TrackEntry(
            position=1,
            start_time=timedelta(minutes=0),
            end_time=timedelta(minutes=5),
            artist="Test Artist 1",
            title="Test Track 1",
        ),
        TrackEntry(
            position=2,
            start_time=timedelta(minutes=5),
            end_time=timedelta(minutes=10),
            artist="Test Artist 2",
            title="Test Track 2",
        ),
    ]

    return Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)


class TestIntegration:
    """Integration tests with related models."""

    def test_cue_file_with_tracklist(self, sample_tracklist):
        """Test CUE file integration with tracklist."""

        cue_file = CueFile(
            tracklist_id=sample_tracklist.id,
            file_path="/data/test.cue",
            format=CueFormat.STANDARD,
            file_size=1024,
            checksum="abc123",
        )

        assert cue_file.tracklist_id == sample_tracklist.id
        assert cue_file.format == CueFormat.STANDARD
