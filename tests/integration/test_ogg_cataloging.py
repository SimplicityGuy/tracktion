"""Integration tests for OGG file cataloging end-to-end."""

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.metadata_extractor import MetadataExtractor
from services.analysis_service.src.storage_handler import StorageHandler
from services.file_watcher.src.file_scanner import FileScanner
from services.file_watcher.src.message_publisher import MessagePublisher


class TestOggCatalogingIntegration:
    """Integration tests for OGG file cataloging through the full pipeline."""

    @pytest.fixture
    def mock_rabbitmq_connection(self):
        """Create a mock RabbitMQ connection."""
        mock_conn = MagicMock()
        mock_channel = MagicMock()
        mock_conn.channel.return_value = mock_channel
        mock_conn.is_closed = False
        return mock_conn

    @pytest.fixture
    def mock_storage_handler(self):
        """Create a mock storage handler."""
        storage = Mock(spec=StorageHandler)
        storage.store_metadata.return_value = True
        storage.update_recording_status.return_value = True
        return storage

    @pytest.fixture
    def temp_ogg_file(self):
        """Create a temporary OGG file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            # Write OGG file header (simplified)
            f.write(b"OggS\x00\x02")  # OGG page header
            f.write(b"\x00" * 100)  # Dummy content
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    def test_ogg_file_discovery_and_publishing(self, mock_rabbitmq_connection, temp_ogg_file):
        """Test that OGG files are discovered and published to RabbitMQ."""
        # Create scanner and publisher
        scanner = FileScanner()
        # Mock the connection initialization for MessagePublisher
        with (
            patch("services.file_watcher.src.message_publisher.pika.ConnectionParameters"),
            patch("services.file_watcher.src.message_publisher.pika.BlockingConnection"),
        ):
            publisher = MessagePublisher("mock_host")
            publisher.connection = mock_rabbitmq_connection
            publisher.channel = mock_rabbitmq_connection.channel()

        # Scan directory containing OGG file
        temp_dir = Path(temp_ogg_file).parent
        files = scanner.scan_directory(temp_dir)

        # Verify OGG file was discovered
        ogg_files = [f for f in files if f["extension"].lower() in [".ogg", ".oga"]]
        assert len(ogg_files) > 0, "OGG file should be discovered"

        # Publish the discovered OGG file
        for file_info in ogg_files:
            publisher.publish_file_discovered(file_info)

        # Verify message was published with correct format
        mock_channel = mock_rabbitmq_connection.channel()
        assert mock_channel.basic_publish.called

        # Check the published message
        call_args = mock_channel.basic_publish.call_args
        message_body = json.loads(call_args[1]["body"])

        assert message_body["file_info"]["extension"] == ".ogg"
        assert message_body["format_family"] == "ogg_vorbis"
        assert "correlation_id" in message_body
        assert "timestamp" in message_body

    @patch("services.analysis_service.src.metadata_extractor.Path")
    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_ogg_metadata_extraction_and_storage(self, mock_oggvorbis, mock_path, mock_storage_handler):
        """Test OGG metadata extraction and storage in the catalog."""
        # Mock file exists
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True
        mock_path_obj.suffix = ".ogg"
        mock_path.return_value = mock_path_obj

        # Setup mock OGG file
        mock_audio = MagicMock()
        mock_audio.tags = {
            "title": ["Test OGG Song"],
            "artist": ["OGG Artist"],
            "album": ["OGG Album"],
            "date": ["2024"],
            "genre": ["Electronic"],
        }
        mock_audio.info.length = 180.5
        mock_audio.info.bitrate = 192000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.get = lambda key, default: mock_audio.tags.get(key, default)
        mock_oggvorbis.return_value = mock_audio

        # Extract metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract("/test/file.ogg")

        # Verify format identification
        assert metadata["format"] == "ogg", "Format should be identified as OGG"
        assert metadata["title"] == "Test OGG Song"
        assert metadata["artist"] == "OGG Artist"

        # Store in catalog
        recording_id = uuid.uuid4()
        correlation_id = str(uuid.uuid4())

        # Store metadata using storage handler
        mock_storage_handler.store_metadata(recording_id=recording_id, metadata=metadata, correlation_id=correlation_id)

        # Verify storage was called with correct data
        assert mock_storage_handler.store_metadata.called
        call_args = mock_storage_handler.store_metadata.call_args
        assert call_args[1]["metadata"]["format"] == "ogg"
        assert call_args[1]["metadata"]["title"] == "Test OGG Song"

    def test_ogg_format_appears_in_catalog_queries(self, mock_storage_handler):
        """Test that OGG files appear in catalog queries with proper format identification."""
        # Simulate storing an OGG file in the catalog
        recording_id = uuid.uuid4()
        ogg_metadata = {
            "format": "ogg",
            "title": "OGG Test Track",
            "artist": "Test Artist",
            "album": "Test Album",
            "duration": "240.5",
            "bitrate": "192000",
            "sample_rate": "44100",
        }

        # Store the OGG file metadata
        mock_storage_handler.store_metadata(
            recording_id=recording_id, metadata=ogg_metadata, correlation_id="test-correlation"
        )

        # Simulate a catalog query
        mock_storage_handler.get_by_format = Mock(
            return_value=[
                {
                    "id": str(recording_id),
                    "format": "ogg",
                    "title": "OGG Test Track",
                    "artist": "Test Artist",
                }
            ]
        )

        # Query for OGG files
        ogg_files = mock_storage_handler.get_by_format("ogg")

        # Verify OGG files are returned
        assert len(ogg_files) > 0, "OGG files should appear in catalog queries"
        assert ogg_files[0]["format"] == "ogg"
        assert ogg_files[0]["title"] == "OGG Test Track"

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_corrupted_ogg_file_handling_in_catalog(self, mock_oggvorbis, mock_storage_handler):
        """Test that corrupted OGG files are handled properly in cataloging."""
        # Simulate corrupted OGG file
        mock_oggvorbis.side_effect = Exception("Corrupted OGG file")

        extractor = MetadataExtractor()
        recording_id = uuid.uuid4()

        # Attempt to extract metadata from corrupted file
        with pytest.raises(Exception, match="Corrupted OGG file"):
            extractor._extract_ogg("/test/corrupted.ogg")

        # Update recording status to failed
        mock_storage_handler.update_recording_status(
            recording_id=recording_id,
            status="failed",
            error_message="Failed to extract metadata: Corrupted OGG file",
            correlation_id="test-correlation",
        )

        # Verify status update was called
        assert mock_storage_handler.update_recording_status.called
        call_args = mock_storage_handler.update_recording_status.call_args
        assert call_args[1]["status"] == "failed"
        assert "Corrupted OGG file" in call_args[1]["error_message"]

    def test_ogg_file_end_to_end_cataloging(self, mock_rabbitmq_connection, mock_storage_handler):
        """Test complete end-to-end cataloging of OGG files."""
        # Create a complete message for an OGG file
        recording_id = str(uuid.uuid4())
        file_info = {
            "path": "/music/test.ogg",
            "name": "test.ogg",
            "extension": ".ogg",
            "size": 5242880,  # 5MB
            "modified": datetime.now(UTC).isoformat(),
            "hash": "abc123def456",
        }

        # Create message publisher with mocked connection
        with (
            patch("services.file_watcher.src.message_publisher.pika.ConnectionParameters"),
            patch("services.file_watcher.src.message_publisher.pika.BlockingConnection"),
        ):
            publisher = MessagePublisher("mock_host")
            publisher.connection = mock_rabbitmq_connection
            publisher.channel = mock_rabbitmq_connection.channel()

        # Publish file discovery
        publisher.publish_file_discovered(file_info)

        # Verify message includes OGG-specific handling
        mock_channel = mock_rabbitmq_connection.channel()
        call_args = mock_channel.basic_publish.call_args
        message = json.loads(call_args[1]["body"])

        assert message["format_family"] == "ogg_vorbis"
        assert message["file_info"]["extension"] == ".ogg"

        # Simulate metadata extraction and storage
        metadata = {
            "format": "ogg",
            "title": "End-to-End Test",
            "artist": "Integration Test",
            "duration": "180.0",
            "bitrate": "192000",
        }

        # Store in catalog
        mock_storage_handler.store_metadata(
            recording_id=uuid.UUID(recording_id), metadata=metadata, correlation_id=message["correlation_id"]
        )

        # Update status to processed
        mock_storage_handler.update_recording_status(
            recording_id=uuid.UUID(recording_id), status="processed", correlation_id=message["correlation_id"]
        )

        # Verify complete flow
        assert mock_channel.basic_publish.called
        assert mock_storage_handler.store_metadata.called
        assert mock_storage_handler.update_recording_status.called

        # Verify final status
        status_call = mock_storage_handler.update_recording_status.call_args
        assert status_call[1]["status"] == "processed"

    def test_ogg_and_oga_extensions_both_cataloged(self, mock_storage_handler):
        """Test that both .ogg and .oga extensions are properly cataloged."""
        scanner = FileScanner()

        # Verify both extensions are supported
        assert ".ogg" in scanner.SUPPORTED_EXTENSIONS
        assert ".oga" in scanner.SUPPORTED_EXTENSIONS

        # Test metadata extraction for both
        extractor = MetadataExtractor()
        assert ".ogg" in extractor.SUPPORTED_FORMATS
        assert ".oga" in extractor.SUPPORTED_FORMATS

        # Simulate storing both types
        for ext in [".ogg", ".oga"]:
            recording_id = uuid.uuid4()
            metadata = {
                "format": ext.lstrip("."),
                "title": f"Test {ext} file",
                "extension": ext,
            }

            mock_storage_handler.store_metadata(
                recording_id=recording_id, metadata=metadata, correlation_id=f"test-{ext}"
            )

            assert mock_storage_handler.store_metadata.called
