"""Unit tests for watchdog event handler."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from watchdog.events import (
    DirCreatedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from services.file_watcher.src.watchdog_handler import TracktionEventHandler


class TestTracktionEventHandler:
    """Test suite for TracktionEventHandler."""

    @pytest.fixture
    def mock_publisher(self):
        """Create a mock message publisher."""
        mock = MagicMock()
        mock.publish_file_event.return_value = True
        return mock

    @pytest.fixture
    def handler(self, mock_publisher):
        """Create an event handler instance with mock publisher."""
        return TracktionEventHandler(message_publisher=mock_publisher)

    @pytest.fixture
    def handler_no_publisher(self):
        """Create an event handler instance without publisher."""
        return TracktionEventHandler(message_publisher=None)

    def test_is_audio_file_valid_extensions(self, handler):
        """Test that valid audio file extensions are recognized."""
        valid_files = [
            "/path/to/file.mp3",
            "/path/to/file.MP3",
            "/path/to/file.flac",
            "/path/to/file.wav",
            "/path/to/file.wave",
            "/path/to/file.m4a",
            "/path/to/file.ogg",
            "/path/to/file.oga",
        ]
        for file_path in valid_files:
            assert handler.is_audio_file(file_path), f"{file_path} should be recognized as audio"

    def test_is_audio_file_invalid_extensions(self, handler):
        """Test that non-audio files are not recognized."""
        invalid_files = [
            "/path/to/file.txt",
            "/path/to/file.doc",
            "/path/to/file.jpg",
            "/path/to/file",
            "/path/to/file.mp3.txt",
        ]
        for file_path in invalid_files:
            assert not handler.is_audio_file(file_path), f"{file_path} should not be recognized as audio"

    def test_on_created_audio_file(self, handler, mock_publisher):
        """Test handling of audio file creation event."""
        event = FileCreatedEvent("/path/to/new_song.mp3")
        handler.on_created(event)

        # Verify publisher was called
        mock_publisher.publish_file_event.assert_called_once()
        call_args = mock_publisher.publish_file_event.call_args
        assert call_args[0][0]["event_type"] == "created"
        assert call_args[0][0]["path"].endswith("new_song.mp3")
        assert call_args[0][1] == "created"

    def test_on_created_non_audio_file(self, handler, mock_publisher):
        """Test that non-audio file creation is ignored."""
        event = FileCreatedEvent("/path/to/document.txt")
        handler.on_created(event)

        # Verify publisher was not called
        mock_publisher.publish_file_event.assert_not_called()

    def test_on_created_directory(self, handler, mock_publisher):
        """Test that directory creation is ignored."""
        event = DirCreatedEvent("/path/to/new_dir")
        handler.on_created(event)

        # Verify publisher was not called
        mock_publisher.publish_file_event.assert_not_called()

    def test_on_modified_audio_file(self, handler, mock_publisher):
        """Test handling of audio file modification event."""
        event = FileModifiedEvent("/path/to/song.flac")
        handler.on_modified(event)

        # Verify publisher was called
        mock_publisher.publish_file_event.assert_called_once()
        call_args = mock_publisher.publish_file_event.call_args
        assert call_args[0][0]["event_type"] == "modified"
        assert call_args[0][1] == "modified"

    def test_on_deleted_audio_file(self, handler, mock_publisher):
        """Test handling of audio file deletion event."""
        event = FileDeletedEvent("/path/to/old_song.wav")
        handler.on_deleted(event)

        # Verify publisher was called
        mock_publisher.publish_file_event.assert_called_once()
        call_args = mock_publisher.publish_file_event.call_args
        assert call_args[0][0]["event_type"] == "deleted"
        assert call_args[0][1] == "deleted"

    def test_on_moved_rename_same_directory(self, handler, mock_publisher):
        """Test handling of file rename (move within same directory)."""
        event = FileMovedEvent("/path/to/old_name.mp3", "/path/to/new_name.mp3")
        handler.on_moved(event)

        # Verify publisher was called with rename event
        mock_publisher.publish_file_event.assert_called_once()
        call_args = mock_publisher.publish_file_event.call_args
        assert call_args[0][0]["event_type"] == "renamed"
        assert call_args[0][0]["old_path"] == "/path/to/old_name.mp3"
        assert call_args[0][1] == "renamed"

    def test_on_moved_different_directory(self, handler, mock_publisher):
        """Test handling of file move to different directory."""
        event = FileMovedEvent("/path/from/song.mp3", "/path/to/song.mp3")
        handler.on_moved(event)

        # Verify publisher was called with move event
        mock_publisher.publish_file_event.assert_called_once()
        call_args = mock_publisher.publish_file_event.call_args
        assert call_args[0][0]["event_type"] == "moved"
        assert call_args[0][0]["old_path"] == "/path/from/song.mp3"
        assert call_args[0][1] == "moved"

    def test_on_moved_directory(self, handler, mock_publisher):
        """Test that directory move is ignored."""
        event = DirMovedEvent("/path/from/dir", "/path/to/dir")
        handler.on_moved(event)

        # Verify publisher was not called
        mock_publisher.publish_file_event.assert_not_called()

    def test_on_moved_non_audio_file(self, handler, mock_publisher):
        """Test that non-audio file move is ignored."""
        event = FileMovedEvent("/path/from/doc.txt", "/path/to/doc.txt")
        handler.on_moved(event)

        # Verify publisher was not called
        mock_publisher.publish_file_event.assert_not_called()

    def test_send_event_without_publisher(self, handler_no_publisher):
        """Test that events without publisher are handled gracefully."""
        event = FileCreatedEvent("/path/to/song.mp3")
        # Should not raise exception
        handler_no_publisher.on_created(event)

    def test_send_event_with_existing_file(self, handler, mock_publisher):
        """Test sending event with a file that exists."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"test audio content")

        try:
            event = FileCreatedEvent(tmp_path)
            handler.on_created(event)

            # Verify file stats were included
            call_args = mock_publisher.publish_file_event.call_args
            file_info = call_args[0][0]
            assert "size_bytes" in file_info
            assert "modified_time" in file_info
        finally:
            Path(tmp_path).unlink()

    def test_send_event_publisher_failure(self, handler, mock_publisher):
        """Test handling of publisher failure."""
        mock_publisher.publish_file_event.return_value = False
        event = FileCreatedEvent("/path/to/song.mp3")

        # Should not raise exception
        handler.on_created(event)

        # Verify publisher was called despite failure
        mock_publisher.publish_file_event.assert_called_once()

    def test_send_event_publisher_exception(self, handler, mock_publisher):
        """Test handling of publisher exception."""
        mock_publisher.publish_file_event.side_effect = Exception("Publishing error")
        event = FileCreatedEvent("/path/to/song.mp3")

        # Should not raise exception
        handler.on_created(event)

        # Verify publisher was called despite exception
        mock_publisher.publish_file_event.assert_called_once()

    def test_ogg_file_handling(self, handler, mock_publisher):
        """Test special handling of OGG files."""
        event = FileCreatedEvent("/path/to/audio.ogg")
        handler.on_created(event)

        # Verify OGG file was properly handled
        call_args = mock_publisher.publish_file_event.call_args
        file_info = call_args[0][0]
        assert file_info["extension"] == ".ogg"

    def test_file_path_absolute(self, handler, mock_publisher):
        """Test that file paths are converted to absolute paths."""
        event = FileCreatedEvent("relative/path/song.mp3")
        handler.on_created(event)

        # Verify path was made absolute
        call_args = mock_publisher.publish_file_event.call_args
        file_info = call_args[0][0]
        assert Path(file_info["path"]).is_absolute()
