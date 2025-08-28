"""Integration tests for CUE file generation flow.

These tests verify the complete flow from tracklist creation through
CUE file generation, validation, and storage.
"""

import tempfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.tracklist_service.src.models.tracklist import Tracklist, TrackEntry
from services.tracklist_service.src.services.cue_generation_service import (
    CueGenerationService,
    GenerateCueRequest,
    BatchGenerateCueRequest,
)
from services.tracklist_service.src.services.draft_service import DraftService
from services.tracklist_service.src.services.cue_integration import CueIntegrationService


@pytest.fixture
def test_db_session():
    """Create a test database session."""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables (you would need your actual models here)
    # Base.metadata.create_all(bind=engine)

    session = TestSessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist for testing."""
    tracklist = Tracklist(
        id=uuid4(),
        audio_file_id=uuid4(),
        source="manual",
        confidence_score=0.95,
        is_draft=False,
        tracks=[
            TrackEntry(
                position=1,
                artist="Artist 1",
                title="Track 1",
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=5, seconds=30),
                confidence=1.0,
                is_manual_entry=True,
            ),
            TrackEntry(
                position=2,
                artist="Artist 2",
                title="Track 2",
                remix="Remix Version",
                start_time=timedelta(minutes=5, seconds=30),
                end_time=timedelta(minutes=10, seconds=15),
                confidence=0.9,
                is_manual_entry=True,
            ),
            TrackEntry(
                position=3,
                artist="Artist 3",
                title="Track 3",
                label="Test Label",
                start_time=timedelta(minutes=10, seconds=15),
                end_time=timedelta(minutes=15),
                confidence=0.85,
                is_manual_entry=True,
            ),
        ],
    )
    return tracklist


@pytest.fixture
def cue_generation_service():
    """Create CUE generation service instance."""
    return CueGenerationService()


@pytest.fixture
def cue_integration_service():
    """Create CUE integration service instance."""
    return CueIntegrationService()


class TestCueGenerationFlow:
    """Test complete CUE generation flow."""

    @pytest.mark.asyncio
    async def test_generate_standard_cue_from_tracklist(self, sample_tracklist, cue_generation_service):
        """Test generating a standard CUE file from a tracklist."""
        request = GenerateCueRequest(
            format="standard",
            validate_audio=False,
            store_file=False,
        )

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

        assert response.success is True
        assert response.error is None
        assert response.content is not None
        assert "TRACK 01 AUDIO" in response.content
        assert "Artist 1" in response.content
        assert "Track 1" in response.content
        assert response.format == "standard"

    @pytest.mark.asyncio
    async def test_generate_multiple_formats(self, sample_tracklist, cue_generation_service):
        """Test generating CUE files in multiple formats."""
        request = BatchGenerateCueRequest(
            formats=["standard", "cdj", "traktor"],
            validate_audio=False,
            store_files=False,
        )

        response = await cue_generation_service.generate_multiple_formats(sample_tracklist, request)

        assert response.success is True
        assert len(response.results) == 3

        # Verify each format
        for result in response.results:
            assert result.success is True
            assert result.content is not None
            assert result.format in ["standard", "cdj", "traktor"]

    @pytest.mark.asyncio
    async def test_validate_cue_timing(self, sample_tracklist, cue_generation_service):
        """Test CUE file timing validation."""
        # Generate CUE file
        request = GenerateCueRequest(
            format="standard",
            validate_audio=False,
            store_file=False,
        )

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)
        assert response.success is True

        # Validate timing
        is_valid, errors = await cue_generation_service.validate_tracklist_for_cue(sample_tracklist)

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_tracklist_validation(self, cue_generation_service):
        """Test validation with invalid tracklist."""
        # Create tracklist with overlapping tracks
        invalid_tracklist = Tracklist(
            id=uuid4(),
            audio_file_id=uuid4(),
            source="manual",
            confidence_score=0.95,
            tracks=[
                TrackEntry(
                    position=1,
                    artist="Artist 1",
                    title="Track 1",
                    start_time=timedelta(minutes=0),
                    end_time=timedelta(minutes=6),  # Overlaps with track 2
                    confidence=1.0,
                ),
                TrackEntry(
                    position=2,
                    artist="Artist 2",
                    title="Track 2",
                    start_time=timedelta(minutes=5),  # Starts before track 1 ends
                    end_time=timedelta(minutes=10),
                    confidence=1.0,
                ),
            ],
        )

        is_valid, errors = await cue_generation_service.validate_tracklist_for_cue(invalid_tracklist)

        assert is_valid is False
        assert len(errors) > 0
        assert any("overlap" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_cue_file_storage(self, sample_tracklist, cue_generation_service):
        """Test storing CUE file to filesystem."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock storage path
            cue_generation_service.storage_path = Path(temp_dir)

            request = GenerateCueRequest(
                format="standard",
                validate_audio=False,
                store_file=True,
            )

            response = await cue_generation_service.generate_cue_file(sample_tracklist, request)

            assert response.success is True
            assert response.cue_file_id is not None
            assert response.file_path is not None

            # Verify file was created
            if response.file_path:
                file_path = Path(temp_dir) / response.file_path
                assert file_path.exists()

                # Verify content
                content = file_path.read_text()
                assert "TRACK 01 AUDIO" in content

    @pytest.mark.asyncio
    async def test_format_conversion(self, sample_tracklist, cue_generation_service):
        """Test converting CUE file between formats."""
        # First generate a standard CUE
        request = GenerateCueRequest(
            format="standard",
            validate_audio=False,
            store_file=False,
        )

        response = await cue_generation_service.generate_cue_file(sample_tracklist, request)
        assert response.success is True
        original_content = response.content

        # Convert to CDJ format
        conversion_response = await cue_generation_service.convert_cue_format(original_content, "standard", "cdj")

        assert conversion_response.success is True
        assert conversion_response.content is not None
        assert conversion_response.format == "cdj"

        # Content should be different for different formats
        assert conversion_response.content != original_content


class TestManualTracklistToCueFlow:
    """Test flow from manual tracklist creation to CUE generation."""

    @pytest.mark.asyncio
    async def test_draft_to_published_to_cue(self, test_db_session):
        """Test complete flow from draft creation to CUE generation."""
        draft_service = DraftService()
        cue_generation_service = CueGenerationService()

        # Create draft tracklist
        draft = draft_service.create_draft(
            test_db_session,
            audio_file_id=uuid4(),
            tracks=[
                TrackEntry(
                    position=1,
                    artist="Test Artist",
                    title="Test Track",
                    start_time=timedelta(0),
                    end_time=timedelta(minutes=3),
                    confidence=1.0,
                    is_manual_entry=True,
                )
            ],
        )

        assert draft.is_draft is True

        # Publish draft
        published = draft_service.publish_draft(test_db_session, draft.id)
        assert published.is_draft is False

        # Generate CUE file
        request = GenerateCueRequest(
            format="standard",
            validate_audio=False,
            store_file=False,
        )

        response = await cue_generation_service.generate_cue_file(published, request)

        assert response.success is True
        assert response.content is not None
        assert "Test Artist" in response.content
        assert "Test Track" in response.content


class TestCueIntegrationService:
    """Test CUE integration service functionality."""

    def test_generate_cue_content(self, sample_tracklist, cue_integration_service):
        """Test generating CUE content through integration service."""
        from services.tracklist_service.src.models.cue_file import CueFormat

        success, content, error = cue_integration_service.generate_cue_content(
            sample_tracklist,
            CueFormat.STANDARD,
            audio_filename="test_mix.wav",
        )

        assert success is True
        assert content is not None
        assert error is None
        assert "TRACK 01 AUDIO" in content
        assert "test_mix.wav" in content

    def test_validate_cue_content(self, sample_tracklist, cue_integration_service):
        """Test validating CUE content."""
        from services.tracklist_service.src.models.cue_file import CueFormat

        # Generate content first
        success, content, _ = cue_integration_service.generate_cue_content(
            sample_tracklist,
            CueFormat.STANDARD,
        )
        assert success is True

        # Validate it
        validation_result = cue_integration_service.validate_cue_content(
            content,
            CueFormat.STANDARD,
        )

        assert validation_result.valid is True
        assert validation_result.error is None

    def test_format_conversion(self, sample_tracklist, cue_integration_service):
        """Test format conversion through integration service."""
        from services.tracklist_service.src.models.cue_file import CueFormat

        # Generate standard format
        success, standard_content, _ = cue_integration_service.generate_cue_content(
            sample_tracklist,
            CueFormat.STANDARD,
        )
        assert success is True

        # Convert to CDJ format
        success, cdj_content, warnings, error = cue_integration_service.convert_cue_format(
            standard_content,
            CueFormat.STANDARD,
            CueFormat.CDJ,
            preserve_metadata=True,
        )

        assert success is True
        assert cdj_content is not None
        assert error is None

    def test_get_supported_formats(self, cue_integration_service):
        """Test getting list of supported formats."""
        formats = cue_integration_service.get_supported_formats()

        assert len(formats) > 0
        assert any(f.value == "standard" for f in formats)
        assert any(f.value == "cdj" for f in formats)

    def test_extract_metadata(self, sample_tracklist, cue_integration_service):
        """Test extracting metadata from CUE content."""
        from services.tracklist_service.src.models.cue_file import CueFormat

        # Generate content
        success, content, _ = cue_integration_service.generate_cue_content(
            sample_tracklist,
            CueFormat.STANDARD,
        )
        assert success is True

        # Extract metadata
        metadata = cue_integration_service.extract_metadata_from_content(content)

        assert metadata.get("track_count") == 3
        assert metadata.get("has_file_reference") is True
        assert metadata.get("has_rem_fields") is True


@pytest.mark.asyncio
async def test_full_integration_flow():
    """Test the complete integration flow end-to-end."""
    # This test would require actual service instances and database
    # It's marked as integration test for when services are available

    # 1. Create manual tracklist
    # 2. Add tracks with timing
    # 3. Validate timing
    # 4. Publish draft
    # 5. Generate CUE in multiple formats
    # 6. Validate CUE files
    # 7. Store CUE files
    # 8. Verify storage and retrieval

    pass  # Implementation depends on actual service availability
