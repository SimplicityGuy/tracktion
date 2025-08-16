"""Unit tests for the File Watcher service."""

import os
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../services/file_watcher/src'))

from main import FileWatcherService


class TestFileWatcherService(unittest.TestCase):
    """Test cases for FileWatcherService."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.env_patcher = patch.dict(
            os.environ,
            {
                "FILE_WATCHER_SCAN_PATH": "/test/path",
                "FILE_WATCHER_SCAN_INTERVAL": "30",
            },
        )
        self.env_patcher.start()

    def tearDown(self) -> None:
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_initialization(self) -> None:
        """Test service initialization with environment variables."""
        service = FileWatcherService()
        self.assertEqual(str(service.scan_path), "/test/path")
        self.assertEqual(service.scan_interval, 30)
        self.assertFalse(service.running)

    @patch("main.time.sleep")
    def test_start_stop(self, mock_sleep: MagicMock) -> None:
        """Test service start and stop behavior."""
        service = FileWatcherService()
        
        # Simulate immediate shutdown
        mock_sleep.side_effect = lambda x: setattr(service, "running", False)
        
        service.start()
        self.assertFalse(service.running)
        mock_sleep.assert_called()

    def test_handle_shutdown(self) -> None:
        """Test shutdown signal handling."""
        service = FileWatcherService()
        service.running = True
        service._handle_shutdown(15, None)  # SIGTERM
        self.assertFalse(service.running)


if __name__ == "__main__":
    unittest.main()