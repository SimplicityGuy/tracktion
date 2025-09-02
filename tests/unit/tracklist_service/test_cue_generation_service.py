"""
Unit tests for CUE generation service.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.cue_file import (
    BatchGenerateCueRequest,
    CueFormat,
    GenerateCueRequest,
    ValidationResult,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.cue_generation_service import CueGenerationService
from services.tracklist_service.src.services.storage_service import StorageResult


class TestCueGenerationService:
    """Test CueGenerationService class."""

    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service."""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage_service):
        """Create CueGenerationService instance."""
        return CueGenerationService(mock_storage_service)

    @pytest.fixture
    def sample_tracklist(self):
        """Create a sample tracklist."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=3, seconds=30),
                artist="Test Artist 1",
                title="Test Track 1",
                remix="Original Mix",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3, seconds=30),
                end_time=timedelta(minutes=7, seconds=15),
                artist="Test Artist 2",
                title="Test Track 2",
            ),
        ]

        return Tracklist(
            audio_file_id=uuid4(),
            source="manual",
            tracks=tracks,
            confidence_score=0.87,
            is_draft=False,
        )

    @pytest.fixture
    def generation_request(self):
        """Create a sample generation request."""
        return GenerateCueRequest(
            format=CueFormat.STANDARD,
            options={},
            validate_audio=True,
            audio_file_path="test_mix.wav",
        )

    @patch("services.tracklist_service.src.services.cue_generation_service.CueIntegrationService")
    @pytest.mark.asyncio
    async def test_generate_cue_file_success(
        self,
        mock_integration_class,
        service,
        sample_tracklist,
        generation_request,
        mock_storage_service,
    ):
        """Test successful CUE file generation."""
        # Setup mocks
        mock_integration = MagicMock()
        mock_integration.generate_cue_content.return_value = (
            True,
            "MOCK CUE CONTENT",
            None,
        )
        mock_integration.validate_cue_content.return_value = ValidationResult(valid=True)
        mock_integration_class.return_value = mock_integration

        mock_storage_service.store_cue_file = AsyncMock(
            return_value=StorageResult(success=True, file_path="cue_files/test_file.cue")
        )

        # Recreate service with mocked integration
        service = CueGenerationService(mock_storage_service)
        service.cue_integration = mock_integration

        # Test
        response = await service.generate_cue_file(sample_tracklist, generation_request)

        assert response.success is True
        assert response.job_id is not None
        assert response.cue_file_id is not None
        assert response.file_path == "cue_files/test_file.cue"
        assert response.validation_report is not None
        assert response.validation_report.valid is True

        # Verify storage was called
        mock_storage_service.store_cue_file.assert_called_once()
        storage_call = mock_storage_service.store_cue_file.call_args
        assert storage_call[1]["content"] == "MOCK CUE CONTENT"
        assert storage_call[1]["cue_format"] == "standard"

    @patch("services.tracklist_service.src.services.cue_generation_service.CueIntegrationService")
    @pytest.mark.asyncio
    async def test_generate_cue_file_generation_failure(
        self, mock_integration_class, service, sample_tracklist, generation_request
    ):
        """Test CUE file generation failure."""
        # Setup mock to fail generation
        mock_integration = MagicMock()
        mock_integration.generate_cue_content.return_value = (
            False,
            "",
            "Generation failed",
        )
        mock_integration_class.return_value = mock_integration

        # Recreate service with mocked integration
        service.cue_integration = mock_integration

        # Test
        response = await service.generate_cue_file(sample_tracklist, generation_request)

        assert response.success is False
        assert response.error == "Generation failed"
        assert response.cue_file_id is None

    @patch("services.tracklist_service.src.services.cue_generation_service.CueIntegrationService")
    @pytest.mark.asyncio
    async def test_generate_cue_file_validation_failure(
        self,
        mock_integration_class,
        service,
        sample_tracklist,
        generation_request,
        mock_storage_service,
    ):
        """Test CUE file validation failure."""
        # Setup mocks
        mock_integration = MagicMock()
        mock_integration.generate_cue_content.return_value = (
            True,
            "INVALID CUE CONTENT",
            None,
        )
        mock_integration.validate_cue_content.return_value = ValidationResult(valid=False, error="Invalid CUE format")
        mock_integration_class.return_value = mock_integration

        # Recreate service with mocked integration
        service = CueGenerationService(mock_storage_service)
        service.cue_integration = mock_integration

        # Test
        response = await service.generate_cue_file(sample_tracklist, generation_request)

        assert response.success is False
        assert "Validation failed" in response.error

    @patch("services.tracklist_service.src.services.cue_generation_service.CueIntegrationService")
    @pytest.mark.asyncio
    async def test_generate_cue_file_storage_failure(
        self,
        mock_integration_class,
        service,
        sample_tracklist,
        generation_request,
        mock_storage_service,
    ):
        """Test CUE file storage failure."""
        # Setup mocks
        mock_integration = MagicMock()
        mock_integration.generate_cue_content.return_value = (
            True,
            "MOCK CUE CONTENT",
            None,
        )
        mock_integration.validate_cue_content.return_value = ValidationResult(valid=True)
        mock_integration_class.return_value = mock_integration

        mock_storage_service.store_cue_file = AsyncMock(
            return_value=StorageResult(success=False, error="Storage failed")
        )

        # Recreate service with mocked integration
        service = CueGenerationService(mock_storage_service)
        service.cue_integration = mock_integration

        # Test
        response = await service.generate_cue_file(sample_tracklist, generation_request)

        assert response.success is False
        assert "Storage failed" in response.error

    @pytest.mark.asyncio
    async def test_generate_cue_file_no_validation(self, service, sample_tracklist, mock_storage_service):
        """Test CUE file generation without validation."""
        # Create request without validation
        request = GenerateCueRequest(format=CueFormat.STANDARD, validate_audio=False)

        mock_storage_service.store_cue_file = AsyncMock(
            return_value=StorageResult(success=True, file_path="test_file.cue")
        )

        with patch.object(service.cue_integration, "generate_cue_content") as mock_generate:
            mock_generate.return_value = (True, "SIMPLE CUE CONTENT", None)

            response = await service.generate_cue_file(sample_tracklist, request)

            assert response.success is True
            assert response.cue_file_id is not None
            assert response.validation_report is None

    @pytest.mark.asyncio
    async def test_generate_multiple_formats_success(self, service, sample_tracklist):
        """Test successful bulk generation."""
        bulk_request = BatchGenerateCueRequest(formats=[CueFormat.STANDARD, CueFormat.CDJ], options={})

        with patch.object(service, "generate_cue_file") as mock_generate:
            # Mock successful responses for both formats
            mock_generate.side_effect = [
                MagicMock(success=True, job_id=uuid4()),
                MagicMock(success=True, job_id=uuid4()),
            ]

            response = await service.generate_multiple_formats(sample_tracklist, bulk_request)

            assert response.success is True
            assert response.total_files == 2
            assert response.successful_files == 2
            assert response.failed_files == 0
            assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_generate_multiple_formats_with_failure(self, service, sample_tracklist):
        """Test bulk generation with some failures."""
        bulk_request = BatchGenerateCueRequest(formats=[CueFormat.STANDARD, CueFormat.CDJ], options={})

        with patch.object(service, "generate_cue_file") as mock_generate:
            # Mock one success and one failure
            mock_generate.side_effect = [
                MagicMock(success=True, job_id=uuid4()),
                MagicMock(success=False, error="Second format failed"),
            ]

            response = await service.generate_multiple_formats(sample_tracklist, bulk_request)

            assert response.success is False
            assert response.total_files == 2
            assert response.successful_files == 1
            assert response.failed_files == 1
            assert len(response.results) == 2

    def test_get_format_capabilities(self, service):
        """Test getting format capabilities."""
        with patch.object(service.cue_integration, "get_format_capabilities") as mock_caps:
            mock_caps.return_value = {"max_tracks": 99, "supports_isrc": True}

            caps = service.get_format_capabilities(CueFormat.STANDARD)

            assert caps["max_tracks"] == 99
            assert caps["supports_isrc"] is True

    def test_get_conversion_preview(self, service):
        """Test getting conversion warnings."""
        with patch.object(service.cue_integration, "get_conversion_warnings") as mock_warnings:
            mock_warnings.return_value = [
                "ISRC codes will be lost",
                "REM fields may be truncated",
            ]

            warnings = service.get_conversion_preview(CueFormat.STANDARD, CueFormat.CDJ)

            assert len(warnings) == 2
            assert "ISRC codes will be lost" in warnings

    def test_get_supported_formats(self, service):
        """Test getting supported formats."""
        with patch.object(service.cue_integration, "get_supported_formats") as mock_formats:
            mock_formats.return_value = [
                CueFormat.STANDARD,
                CueFormat.CDJ,
                CueFormat.TRAKTOR,
            ]

            formats = service.get_supported_formats()

            assert len(formats) == 3
            assert CueFormat.STANDARD in formats

    @pytest.mark.asyncio
    async def test_validate_tracklist_for_cue_success(self, service, sample_tracklist):
        """Test successful tracklist validation."""
        with patch.object(service, "get_format_capabilities") as mock_caps:
            mock_caps.return_value = {"max_tracks": 99}

            result = await service.validate_tracklist_for_cue(sample_tracklist, CueFormat.STANDARD)

            assert result.valid is True
            assert len(result.warnings) == 0
            assert result.metadata["track_count"] == 2

    @pytest.mark.asyncio
    async def test_validate_tracklist_for_cue_too_many_tracks(self, service, sample_tracklist):
        """Test tracklist validation with too many tracks."""
        with patch.object(service, "get_format_capabilities") as mock_caps:
            mock_caps.return_value = {"max_tracks": 1}

            result = await service.validate_tracklist_for_cue(sample_tracklist, CueFormat.CDJ)

            assert result.valid is True  # Still valid, just warnings
            assert len(result.warnings) == 1
            assert "exceeds" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_validate_tracklist_missing_fields(self, service):
        """Test tracklist validation with missing required fields."""
        # Create tracklist with missing fields
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                artist="",  # Missing artist
                title="Test Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3),
                artist="Test Artist 2",
                title="",  # Missing title
            ),
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        with patch.object(service, "get_format_capabilities") as mock_caps:
            mock_caps.return_value = {"max_tracks": 99}

            result = await service.validate_tracklist_for_cue(tracklist, CueFormat.STANDARD)

            assert result.valid is False  # Invalid due to missing required fields
            assert len(result.warnings) >= 2  # At least missing artist and title
            assert any("missing artist" in w for w in result.warnings)
            assert any("missing title" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_validate_tracklist_timing_overlap(self, service):
        """Test tracklist validation with timing overlaps."""
        # Create tracks with overlapping times
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=4),  # Overlaps with next track
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3),  # Starts before previous ends
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        with patch.object(service, "get_format_capabilities") as mock_caps:
            mock_caps.return_value = {"max_tracks": 99}

            result = await service.validate_tracklist_for_cue(tracklist, CueFormat.STANDARD)

            assert result.valid is True  # Valid but with warnings
            assert len(result.warnings) >= 1
            assert any("overlaps" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_regenerate_cue_file_not_implemented(self, service, sample_tracklist):
        """Test regeneration method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Regeneration requires CUE file repository"):
            await service.regenerate_cue_file(uuid4(), sample_tracklist)
