"""Tests for metadata integration in file rename proposal service."""

from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from services.analysis_service.src.file_rename_proposal.batch_processor import (
    BatchProcessor,
)
from services.analysis_service.src.file_rename_proposal.message_interface import (
    MessageTypes,
    RenameProposalMessageInterface,
)
from services.analysis_service.src.file_rename_proposal.proposal_generator import (
    ProposalGenerator,
    RenameProposal,
)
from shared.core_types.src.models import Metadata, Recording


class TestMetadataIntegration:
    """Test metadata integration in file rename proposal components."""

    @pytest.fixture
    def mock_metadata_repo(self):
        """Create mock metadata repository."""
        return Mock()

    @pytest.fixture
    def mock_recording_repo(self):
        """Create mock recording repository."""
        return Mock()

    @pytest.fixture
    def mock_proposal_repo(self):
        """Create mock proposal repository."""
        return Mock()

    @pytest.fixture
    def mock_proposal_generator(self):
        """Create mock proposal generator."""
        return Mock(spec=ProposalGenerator)

    @pytest.fixture
    def mock_conflict_detector(self):
        """Create mock conflict detector."""
        detector = Mock()
        detector.detect_conflicts.return_value = {"conflicts": [], "warnings": []}
        return detector

    @pytest.fixture
    def mock_confidence_scorer(self):
        """Create mock confidence scorer."""
        scorer = Mock()
        scorer.calculate_confidence.return_value = (0.85, {"metadata": 0.9, "pattern": 0.8})
        return scorer

    def test_message_interface_uses_metadata_repository(
        self,
        mock_proposal_generator,
        mock_conflict_detector,
        mock_confidence_scorer,
        mock_proposal_repo,
        mock_recording_repo,
        mock_metadata_repo,
    ):
        """Test that message interface properly uses metadata repository."""
        # Setup
        recording_id = uuid4()
        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/test/path/file.mp3"
        recording.file_name = "file.mp3"

        # Mock metadata
        metadata_list = [
            Mock(spec=Metadata, key="artist", value="Test Artist"),
            Mock(spec=Metadata, key="title", value="Test Title"),
            Mock(spec=Metadata, key="album", value="Test Album"),
            Mock(spec=Metadata, key="bpm", value="128"),
        ]

        mock_recording_repo.get_by_id.return_value = recording
        mock_metadata_repo.get_by_recording.return_value = metadata_list

        # Create a mock proposal
        mock_proposal = RenameProposal(
            recording_id=recording_id,
            original_path="/test/path/file.mp3",
            original_filename="file.mp3",
            proposed_filename="Test Artist - Test Title.mp3",
            full_proposed_path="/test/path/Test Artist - Test Title.mp3",
            confidence_score=0.85,
            metadata_source="complete",
            pattern_used="[artist] - [title]",
        )
        mock_proposal_generator.generate_proposal.return_value = mock_proposal

        # Create a mock created proposal for the repository
        mock_created_proposal = Mock()
        mock_created_proposal.id = uuid4()
        mock_created_proposal.recording_id = recording_id
        mock_created_proposal.original_filename = "file.mp3"
        mock_created_proposal.proposed_filename = "Test Artist - Test Title.mp3"
        mock_created_proposal.full_proposed_path = "/test/path/Test Artist - Test Title.mp3"
        mock_created_proposal.confidence_score = 0.85
        mock_created_proposal.status = "pending"
        mock_created_proposal.conflicts = []
        mock_created_proposal.warnings = []
        mock_created_proposal.created_at = datetime.now(UTC)
        mock_proposal_repo.create.return_value = mock_created_proposal

        # Create interface
        interface = RenameProposalMessageInterface(
            proposal_generator=mock_proposal_generator,
            conflict_detector=mock_conflict_detector,
            confidence_scorer=mock_confidence_scorer,
            proposal_repo=mock_proposal_repo,
            recording_repo=mock_recording_repo,
            metadata_repo=mock_metadata_repo,
        )

        # Execute
        message = {
            "type": MessageTypes.GENERATE_PROPOSAL,
            "recording_id": str(recording_id),
            "request_id": "test_request_123",
        }

        response = interface.process_message(message)

        # Verify
        assert response["type"] == MessageTypes.PROPOSAL_GENERATED
        assert "proposal" in response

        # Verify metadata was fetched
        mock_metadata_repo.get_by_recording.assert_called_once_with(recording_id)

        # Verify proposal generator received metadata dict
        expected_metadata = {
            "artist": "Test Artist",
            "title": "Test Title",
            "album": "Test Album",
            "bpm": "128",
        }
        mock_proposal_generator.generate_proposal.assert_called_once_with(
            recording_id=recording_id,
            original_path="/test/path/file.mp3",
            metadata=expected_metadata,
            file_extension="mp3",
        )

        # Verify confidence scorer received metadata
        mock_confidence_scorer.calculate_confidence.assert_called_once()
        call_args = mock_confidence_scorer.calculate_confidence.call_args[1]
        assert call_args["metadata"] == expected_metadata

    def test_batch_processor_uses_metadata_repository(
        self,
        mock_proposal_generator,
        mock_conflict_detector,
        mock_confidence_scorer,
        mock_proposal_repo,
        mock_recording_repo,
        mock_metadata_repo,
    ):
        """Test that batch processor properly uses metadata repository."""
        # Setup
        recording_id = uuid4()
        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/test/path/file.flac"
        recording.file_name = "file.flac"

        # Mock metadata
        metadata_list = [
            Mock(spec=Metadata, key="artist", value="Batch Artist"),
            Mock(spec=Metadata, key="title", value="Batch Title"),
            Mock(spec=Metadata, key="genre", value="Electronic"),
        ]

        mock_recording_repo.get_by_id.return_value = recording
        mock_metadata_repo.get_by_recording.return_value = metadata_list
        mock_proposal_repo.get_pending_proposals.return_value = []

        # Create a mock proposal
        mock_proposal = RenameProposal(
            recording_id=recording_id,
            original_path="/test/path/file.flac",
            original_filename="file.flac",
            proposed_filename="Batch Artist - Batch Title.flac",
            full_proposed_path="/test/path/Batch Artist - Batch Title.flac",
            confidence_score=0.75,
            metadata_source="id3",
            pattern_used="[artist] - [title]",
        )
        mock_proposal_generator.generate_proposal.return_value = mock_proposal

        # Mock the created proposal
        mock_created_proposal = Mock()
        mock_created_proposal.id = uuid4()
        mock_proposal_repo.create.return_value = mock_created_proposal

        # Create batch processor
        processor = BatchProcessor(
            proposal_generator=mock_proposal_generator,
            conflict_detector=mock_conflict_detector,
            confidence_scorer=mock_confidence_scorer,
            proposal_repo=mock_proposal_repo,
            recording_repo=mock_recording_repo,
            metadata_repo=mock_metadata_repo,
        )

        # Execute single recording processing
        job = Mock()
        job.job_id = "test_job"
        job.options = {"enable_conflict_resolution": False}

        result = processor._process_single_recording(recording_id, job, {})

        # Verify
        assert result["success"] is True

        # Verify metadata was fetched
        mock_metadata_repo.get_by_recording.assert_called_once_with(recording_id)

        # Verify proposal generator received metadata dict
        expected_metadata = {
            "artist": "Batch Artist",
            "title": "Batch Title",
            "genre": "Electronic",
        }
        mock_proposal_generator.generate_proposal.assert_called_once_with(
            recording_id=recording_id,
            original_path="/test/path/file.flac",
            metadata=expected_metadata,
            file_extension="flac",
        )

        # Verify confidence scorer received metadata
        mock_confidence_scorer.calculate_confidence.assert_called_once()
        call_args = mock_confidence_scorer.calculate_confidence.call_args[1]
        assert call_args["metadata"] == expected_metadata

    def test_proposal_generator_determines_metadata_source(self):
        """Test that proposal generator correctly determines metadata source."""
        # Setup
        mock_config = Mock()
        mock_config.confidence_weights = {
            "metadata_completeness": 0.3,
            "metadata_quality": 0.3,
            "pattern_match": 0.2,
            "conflicts": 0.2,
        }
        mock_config.max_filename_length = 255
        mock_config.max_path_length = 4096
        mock_config.replacement_char = "_"

        mock_pattern_manager = Mock()
        mock_pattern_manager.apply_pattern.return_value = "Artist - Title"
        mock_pattern_manager.get_pattern_for_type.return_value = "[artist] - [title]"

        mock_validator = Mock()
        mock_validator.sanitize_filename.side_effect = lambda x: x

        generator = ProposalGenerator(
            config=mock_config,
            pattern_manager=mock_pattern_manager,
            validator=mock_validator,
        )

        # Test complete metadata
        metadata_complete = {
            "artist": "Complete Artist",
            "title": "Complete Title",
            "album": "Complete Album",
            "bpm": "128",
            "key": "Am",
        }
        assert generator._determine_metadata_source(metadata_complete) == "complete"

        # Test ID3 metadata
        metadata_id3 = {
            "artist": "ID3 Artist",
            "title": "ID3 Title",
            "album": "ID3 Album",
        }
        assert generator._determine_metadata_source(metadata_id3) == "id3"

        # Test analysis metadata
        metadata_analysis = {
            "bpm": "130",
            "key": "C",
        }
        assert generator._determine_metadata_source(metadata_analysis) == "analysis"

        # Test partial metadata
        metadata_partial = {
            "artist": "Partial Artist",
        }
        assert generator._determine_metadata_source(metadata_partial) == "partial"

        # Test empty metadata
        assert generator._determine_metadata_source({}) == "empty"

        # Test inferred metadata (unknown values)
        metadata_unknown = {
            "artist": "unknown",
            "title": "",
        }
        assert generator._determine_metadata_source(metadata_unknown) == "inferred"

    def test_metadata_integration_with_empty_metadata(
        self,
        mock_proposal_generator,
        mock_conflict_detector,
        mock_confidence_scorer,
        mock_proposal_repo,
        mock_recording_repo,
        mock_metadata_repo,
    ):
        """Test handling of empty metadata list."""
        # Setup
        recording_id = uuid4()
        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/test/path/nodata.mp3"
        recording.file_name = "nodata.mp3"

        # Mock empty metadata
        mock_recording_repo.get_by_id.return_value = recording
        mock_metadata_repo.get_by_recording.return_value = []

        # Create a mock proposal
        mock_proposal = RenameProposal(
            recording_id=recording_id,
            original_path="/test/path/nodata.mp3",
            original_filename="nodata.mp3",
            proposed_filename="nodata.mp3",
            full_proposed_path="/test/path/nodata.mp3",
            confidence_score=0.3,
            metadata_source="empty",
            pattern_used="default",
        )
        mock_proposal_generator.generate_proposal.return_value = mock_proposal

        # Mock the created proposal
        mock_created_proposal = Mock()
        mock_created_proposal.id = uuid4()
        mock_created_proposal.recording_id = recording_id
        mock_created_proposal.original_filename = "nodata.mp3"
        mock_created_proposal.proposed_filename = "nodata.mp3"
        mock_created_proposal.full_proposed_path = "/test/path/nodata.mp3"
        mock_created_proposal.confidence_score = 0.3
        mock_created_proposal.status = "pending"
        mock_created_proposal.conflicts = []
        mock_created_proposal.warnings = []
        mock_created_proposal.created_at = datetime.now(UTC)
        mock_proposal_repo.create.return_value = mock_created_proposal

        # Create interface
        interface = RenameProposalMessageInterface(
            proposal_generator=mock_proposal_generator,
            conflict_detector=mock_conflict_detector,
            confidence_scorer=mock_confidence_scorer,
            proposal_repo=mock_proposal_repo,
            recording_repo=mock_recording_repo,
            metadata_repo=mock_metadata_repo,
        )

        # Execute
        message = {
            "type": MessageTypes.GENERATE_PROPOSAL,
            "recording_id": str(recording_id),
            "request_id": "test_empty_metadata",
        }

        response = interface.process_message(message)

        # Verify
        assert response["type"] == MessageTypes.PROPOSAL_GENERATED

        # Verify empty dict was passed
        mock_proposal_generator.generate_proposal.assert_called_once_with(
            recording_id=recording_id,
            original_path="/test/path/nodata.mp3",
            metadata={},
            file_extension="mp3",
        )
