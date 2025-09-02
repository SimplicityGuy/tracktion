"""Tests for CUE validation and conversion handlers (Task 7)."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from aio_pika import AbstractIncomingMessage

from services.tracklist_service.src.messaging.cue_generation_handler import CueGenerationHandler
from services.tracklist_service.src.messaging.message_schemas import (
    CueConversionMessage,
    CueValidationMessage,
)


class TestCueValidationHandler:
    """Test CUE validation handler implementation."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory."""
        factory = AsyncMock()
        session = AsyncMock()
        factory.return_value.__aenter__.return_value = session
        return factory, session

    @pytest.fixture
    def cue_handler(self, mock_session_factory):
        """Create CUE generation handler."""
        factory, _ = mock_session_factory
        return CueGenerationHandler(session_factory=factory)

    @pytest.mark.asyncio
    async def test_handle_cue_validation_success(self, cue_handler, mock_session_factory):
        """Test successful CUE validation."""
        factory, mock_session = mock_session_factory

        # Create validation message
        cue_file_id = uuid4()
        validation_job_id = uuid4()
        message = CueValidationMessage(
            cue_file_id=cue_file_id,
            validation_job_id=validation_job_id,
            audio_file_path="/path/to/audio.wav",
            validation_options={},
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock CUE file from database
        mock_cue_file = Mock()
        mock_cue_file.file_path = "/path/to/test.cue"
        mock_cue_file.tracklist_id = uuid4()
        mock_session.get.return_value = mock_cue_file

        # Mock validation result
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.errors = []
        mock_validation_result.warnings = [Mock(line=10, message="Minor formatting issue", severity="WARNING")]

        with patch(
            "services.tracklist_service.src.messaging.cue_generation_handler.CueValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            # Test
            await cue_handler.handle_cue_validation(message, rabbit_msg)

            # Verify
            mock_validator.validate.assert_called_once_with(mock_cue_file.file_path)
            assert mock_cue_file.validation_status == "valid"
            assert len(mock_cue_file.validation_warnings) == 1
            assert mock_cue_file.validation_warnings[0]["message"] == "Minor formatting issue"
            mock_session.commit.assert_called_once()
            rabbit_msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_cue_validation_invalid_file(self, cue_handler, mock_session_factory):
        """Test CUE validation with invalid file."""
        factory, mock_session = mock_session_factory

        # Create validation message
        cue_file_id = uuid4()
        message = CueValidationMessage(
            cue_file_id=cue_file_id,
            validation_job_id=uuid4(),
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock CUE file from database
        mock_cue_file = Mock()
        mock_cue_file.file_path = "/path/to/test.cue"
        mock_session.get.return_value = mock_cue_file

        # Mock validation result with errors
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.errors = [
            Mock(line=5, message="Invalid timestamp format", severity="ERROR"),
            Mock(line=12, message="Track overlap detected", severity="ERROR"),
        ]
        mock_validation_result.warnings = []

        with patch(
            "services.tracklist_service.src.messaging.cue_generation_handler.CueValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            # Test
            await cue_handler.handle_cue_validation(message, rabbit_msg)

            # Verify
            assert mock_cue_file.validation_status == "invalid"
            assert len(mock_cue_file.validation_errors) == 2
            assert mock_cue_file.validation_errors[0]["message"] == "Invalid timestamp format"
            mock_session.commit.assert_called_once()
            rabbit_msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_cue_validation_file_not_found(self, cue_handler, mock_session_factory):
        """Test CUE validation when file not found in database."""
        factory, mock_session = mock_session_factory

        # Create validation message
        message = CueValidationMessage(
            cue_file_id=uuid4(),
            validation_job_id=uuid4(),
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock CUE file not found
        mock_session.get.return_value = None

        # Test
        await cue_handler.handle_cue_validation(message, rabbit_msg)

        # Verify
        rabbit_msg.reject.assert_called_once_with(requeue=False)
        mock_session.commit.assert_not_called()


class TestCueConversionHandler:
    """Test CUE conversion handler implementation."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory."""
        factory = AsyncMock()
        session = AsyncMock()
        factory.return_value.__aenter__.return_value = session
        return factory, session

    @pytest.fixture
    def cue_handler(self, mock_session_factory):
        """Create CUE generation handler."""
        factory, _ = mock_session_factory
        return CueGenerationHandler(session_factory=factory)

    @pytest.mark.asyncio
    async def test_handle_cue_conversion_success(self, cue_handler, mock_session_factory):
        """Test successful CUE conversion."""
        factory, mock_session = mock_session_factory

        # Create conversion message
        source_cue_id = uuid4()
        conversion_job_id = uuid4()
        message = CueConversionMessage(
            source_cue_file_id=source_cue_id,
            conversion_job_id=conversion_job_id,
            target_format="STANDARD",
            preserve_metadata=True,
            conversion_options={},
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock source CUE file from database
        mock_source_cue = Mock()
        mock_source_cue.file_path = "/path/to/source.cue"
        mock_source_cue.tracklist_id = uuid4()
        mock_session.get.return_value = mock_source_cue

        # Mock conversion report
        mock_conversion_report = Mock()
        mock_conversion_report.success = True
        mock_conversion_report.changes = [Mock(change_type="modified", command="INDEX", reason="Format compatibility")]
        mock_conversion_report.warnings = ["Some metadata may be altered"]
        mock_conversion_report.errors = []

        # Mock Path operations
        with patch("services.tracklist_service.src.messaging.cue_generation_handler.Path") as mock_path_class:
            mock_path = Mock()
            mock_path.with_suffix.return_value = Path("/path/to/source.standard.cue")
            mock_path_class.return_value = mock_path

            # Mock file existence check
            mock_output_path = Mock()
            mock_output_path.exists.return_value = True
            mock_output_path.stat.return_value.st_size = 1024
            mock_path.with_suffix.return_value = mock_output_path

            with patch(
                "services.tracklist_service.src.messaging.cue_generation_handler.CueConverter"
            ) as mock_converter_class:
                mock_converter = Mock()
                mock_converter.convert.return_value = mock_conversion_report
                mock_converter_class.return_value = mock_converter

                with patch(
                    "services.tracklist_service.src.messaging.cue_generation_handler.CueFormat"
                ) as mock_format_enum:
                    mock_format_enum.__getitem__.return_value = "STANDARD"

                    # Test
                    await cue_handler.handle_cue_conversion(message, rabbit_msg)

                    # Verify
                    mock_converter.convert.assert_called_once()
                    # Check that a new CUE file record was added
                    assert mock_session.add.called
                    mock_session.commit.assert_called_once()
                    rabbit_msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_cue_conversion_invalid_format(self, cue_handler, mock_session_factory):
        """Test CUE conversion with invalid target format."""
        factory, mock_session = mock_session_factory

        # Create conversion message with invalid format
        message = CueConversionMessage(
            source_cue_file_id=uuid4(),
            conversion_job_id=uuid4(),
            target_format="INVALID_FORMAT",
            preserve_metadata=True,
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock source CUE file
        mock_source_cue = Mock()
        mock_source_cue.file_path = "/path/to/source.cue"
        mock_session.get.return_value = mock_source_cue

        with patch("services.tracklist_service.src.messaging.cue_generation_handler.CueFormat") as mock_format_enum:
            # Make format lookup fail
            mock_format_enum.__getitem__.side_effect = KeyError("Invalid format")

            # Test
            await cue_handler.handle_cue_conversion(message, rabbit_msg)

            # Verify
            rabbit_msg.reject.assert_called_once_with(requeue=False)
            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_cue_conversion_failure(self, cue_handler, mock_session_factory):
        """Test CUE conversion when conversion fails."""
        factory, mock_session = mock_session_factory

        # Create conversion message
        message = CueConversionMessage(
            source_cue_file_id=uuid4(),
            conversion_job_id=uuid4(),
            target_format="STANDARD",
        )

        # Mock RabbitMQ message
        rabbit_msg = AsyncMock(spec=AbstractIncomingMessage)

        # Mock source CUE file
        mock_source_cue = Mock()
        mock_source_cue.file_path = "/path/to/source.cue"
        mock_session.get.return_value = mock_source_cue

        # Mock failed conversion report
        mock_conversion_report = Mock()
        mock_conversion_report.success = False
        mock_conversion_report.errors = ["Incompatible format", "Data loss would occur"]

        with (
            patch("services.tracklist_service.src.messaging.cue_generation_handler.Path"),
            patch(
                "services.tracklist_service.src.messaging.cue_generation_handler.CueConverter"
            ) as mock_converter_class,
        ):
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_conversion_report
            mock_converter_class.return_value = mock_converter

            with patch("services.tracklist_service.src.messaging.cue_generation_handler.CueFormat") as mock_format_enum:
                mock_format_enum.__getitem__.return_value = "STANDARD"

                # Test
                await cue_handler.handle_cue_conversion(message, rabbit_msg)

                # Verify
                # No new CUE file should be added on failure
                mock_session.add.assert_not_called()
                mock_session.commit.assert_not_called()
                rabbit_msg.ack.assert_called_once()
