"""Integration tests for file watcher service with watchdog."""

import os

# Import components to test
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "file_watcher" / "src"))
from main import FileWatcherService
from watchdog_handler import TracktionEventHandler


@pytest.mark.integration
class TestFileWatcherIntegration:
    """Integration tests for file watcher with real file operations."""

    @pytest.fixture
    def temp_watch_dir(self):
        """Create a temporary directory for watching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_publisher(self):
        """Create a mock publisher for testing."""
        mock = MagicMock()
        mock.publish_file_event.return_value = True
        return mock

    def test_real_file_creation_detection(self, temp_watch_dir, mock_publisher):
        """Test that real file creation is detected by watchdog."""

        # Create event handler and observer
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=False)
        observer.start()

        try:
            # Create a test audio file
            test_file = temp_watch_dir / "test_song.mp3"
            test_file.write_text("test audio content")

            # Give watchdog time to detect the change
            time.sleep(0.5)

            # Verify the event was detected
            mock_publisher.publish_file_event.assert_called()
            call_args = mock_publisher.publish_file_event.call_args
            assert call_args[0][0]["event_type"] == "created"
            assert "test_song.mp3" in call_args[0][0]["path"]
            assert call_args[0][1] == "created"
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_real_file_modification_detection(self, temp_watch_dir, mock_publisher):
        """Test that file modification is detected."""

        # Create a test file first
        test_file = temp_watch_dir / "existing_song.flac"
        test_file.write_text("initial content")

        # Create event handler and observer
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=False)
        observer.start()

        try:
            # Give observer time to settle
            time.sleep(0.2)

            # Modify the file
            test_file.write_text("modified content")

            # Give watchdog time to detect the change
            time.sleep(0.5)

            # Verify the modification was detected
            assert mock_publisher.publish_file_event.called
            # Find the modify event
            for call in mock_publisher.publish_file_event.call_args_list:
                if call[0][1] == "modified":
                    assert "existing_song.flac" in call[0][0]["path"]
                    break
            else:
                pytest.fail("No modification event detected")
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_real_file_deletion_detection(self, temp_watch_dir, mock_publisher):
        """Test that file deletion is detected."""

        # Create a test file first
        test_file = temp_watch_dir / "delete_me.wav"
        test_file.write_text("to be deleted")

        # Create event handler and observer
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=False)
        observer.start()

        try:
            # Give observer time to settle
            time.sleep(0.2)

            # Delete the file
            test_file.unlink()

            # Give watchdog time to detect the change
            time.sleep(0.5)

            # Verify the deletion was detected
            assert mock_publisher.publish_file_event.called
            # Find the delete event
            for call in mock_publisher.publish_file_event.call_args_list:
                if call[0][1] == "deleted":
                    assert "delete_me.wav" in call[0][0]["path"]
                    break
            else:
                pytest.fail("No deletion event detected")
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_real_file_rename_detection(self, temp_watch_dir, mock_publisher):
        """Test that file rename is detected."""

        # Create a test file first
        old_file = temp_watch_dir / "old_name.ogg"
        old_file.write_text("ogg content")

        # Create event handler and observer
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=False)
        observer.start()

        try:
            # Give observer time to settle
            time.sleep(0.2)

            # Rename the file
            new_file = temp_watch_dir / "new_name.ogg"
            old_file.rename(new_file)

            # Give watchdog time to detect the change
            time.sleep(0.5)

            # Verify the rename was detected
            assert mock_publisher.publish_file_event.called
            # Find the rename event
            for call in mock_publisher.publish_file_event.call_args_list:
                if call[0][1] == "renamed":
                    assert "new_name.ogg" in call[0][0]["path"]
                    assert "old_name.ogg" in call[0][0].get("old_path", "")
                    break
            else:
                # Some systems might report as moved
                for call in mock_publisher.publish_file_event.call_args_list:
                    if call[0][1] == "moved":
                        assert "new_name.ogg" in call[0][0]["path"]
                        break
                else:
                    pytest.fail("No rename/move event detected")
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_recursive_directory_monitoring(self, temp_watch_dir, mock_publisher):
        """Test that subdirectories are monitored recursively."""

        # Create a subdirectory
        subdir = temp_watch_dir / "subdir"
        subdir.mkdir()

        # Create event handler and observer with recursive=True
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=True)
        observer.start()

        try:
            # Give observer time to settle
            time.sleep(0.2)

            # Create a file in the subdirectory
            test_file = subdir / "nested_song.m4a"
            test_file.write_text("nested audio")

            # Give watchdog time to detect the change
            time.sleep(0.5)

            # Verify the event was detected
            mock_publisher.publish_file_event.assert_called()
            call_args = mock_publisher.publish_file_event.call_args
            assert call_args[0][0]["event_type"] == "created"
            assert "nested_song.m4a" in call_args[0][0]["path"]
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_non_audio_files_ignored(self, temp_watch_dir, mock_publisher):
        """Test that non-audio files are ignored."""

        # Create event handler and observer
        handler = TracktionEventHandler(mock_publisher)
        observer = Observer()
        observer.schedule(handler, str(temp_watch_dir), recursive=False)
        observer.start()

        try:
            # Create non-audio files
            (temp_watch_dir / "document.txt").write_text("text")
            (temp_watch_dir / "image.jpg").write_bytes(b"\xff\xd8\xff")  # JPEG header
            (temp_watch_dir / "data.json").write_text('{"key": "value"}')

            # Give watchdog time to process
            time.sleep(0.5)

            # Verify no events were published
            mock_publisher.publish_file_event.assert_not_called()
        finally:
            observer.stop()
            observer.join(timeout=2)

    @patch("main.MessagePublisher")
    @patch.dict(os.environ, {"FILE_WATCHER_SCAN_PATH": ""})
    def test_service_with_real_files(self, mock_publisher_class, temp_watch_dir):
        """Test the full service with real file operations."""
        # Set the scan path to our temp directory
        os.environ["FILE_WATCHER_SCAN_PATH"] = str(temp_watch_dir)

        mock_publisher = MagicMock()
        mock_publisher.publish_file_event.return_value = True
        mock_publisher_class.return_value = mock_publisher

        # Create service
        service = FileWatcherService()
        assert str(service.scan_path) == str(temp_watch_dir)

        # Note: We can't fully test the service.start() method in integration
        # because it runs an infinite loop. This would require threading
        # and more complex test setup. For now, we verify initialization.
