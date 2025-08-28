"""Unit tests for main file watcher service."""

import signal

# Import the service to test
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services" / "file_watcher" / "src"))
from main import FileWatcherService


class TestFileWatcherService:
    """Test suite for FileWatcherService."""

    @pytest.fixture
    def mock_env(self, monkeypatch, tmp_path):
        """Set up test environment variables."""
        test_path = tmp_path / "test_path"
        test_path.mkdir()
        monkeypatch.setenv("FILE_WATCHER_SCAN_PATH", str(test_path))
        monkeypatch.setenv("RABBITMQ_HOST", "test_host")
        monkeypatch.setenv("RABBITMQ_PORT", "5672")
        monkeypatch.setenv("RABBITMQ_USER", "test_user")
        monkeypatch.setenv("RABBITMQ_PASS", "test_pass")
        return test_path

    def test_initialization(self, mock_env):
        """Test service initialization."""
        service = FileWatcherService()
        assert str(service.scan_path) == str(mock_env)
        assert service.rabbitmq_host == "test_host"
        assert service.rabbitmq_port == 5672
        assert service.running is False
        assert service.observer is None
        assert service.publisher is None

    @patch("main.signal.signal")  # Mock signal handling
    @patch("main.MessagePublisher")
    @patch("main.Observer")
    @patch("main.TracktionEventHandler")
    def test_start_with_successful_connection(
        self, mock_handler_class, mock_observer_class, mock_publisher_class, mock_signal, mock_env
    ):
        """Test successful service start."""
        # Setup mocks
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        mock_observer_class.return_value = mock_observer

        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler

        service = FileWatcherService()

        # Start service in a thread and stop it after a short time
        def run_service():
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Give service time to initialize
        time.sleep(0.1)

        # Verify observer was started
        assert mock_observer.schedule.called
        assert mock_observer.start.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)

        # Verify cleanup was called
        assert mock_observer.stop.called
        assert mock_observer.join.called
        assert mock_publisher.disconnect.called

    @patch("main.signal.signal")  # Mock signal handling
    @patch("main.MessagePublisher")
    @patch("main.Observer")
    def test_start_with_rabbitmq_connection_failure(
        self, mock_observer_class, mock_publisher_class, mock_signal, mock_env
    ):
        """Test service continues without RabbitMQ if connection fails."""
        # Setup mock to fail connection
        mock_publisher_class.side_effect = Exception("Connection failed")

        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        mock_observer_class.return_value = mock_observer

        service = FileWatcherService()

        # Start service in a thread
        def run_service():
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Give service time to initialize
        time.sleep(0.1)

        # Service should still be running with observer
        assert service.running is True
        assert mock_observer.start.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)

    @patch("main.signal.signal")  # Mock signal handling
    @patch("main.MessagePublisher")
    @patch("main.Observer")
    def test_observer_restart_on_failure(self, mock_observer_class, mock_publisher_class, mock_signal, mock_env):
        """Test that observer is restarted if it dies."""
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        # Create two observers - first dies, second survives
        dead_observer = MagicMock()
        dead_observer.is_alive.side_effect = [True, False]  # Dies on second check

        new_observer = MagicMock()
        new_observer.is_alive.return_value = True

        mock_observer_class.side_effect = [dead_observer, new_observer]

        service = FileWatcherService()

        # Start service in a thread
        def run_service():
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Give service time to detect failure and restart
        time.sleep(1.5)

        # Verify new observer was created and started
        assert mock_observer_class.call_count == 2
        assert new_observer.start.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)

    @patch("main.signal.signal")  # Mock signal handling
    @patch("main.MessagePublisher")
    @patch("main.Observer")
    @patch("main.Path.exists")
    @patch("main.Path.mkdir")
    def test_creates_missing_scan_path(
        self, mock_mkdir, mock_exists, mock_observer_class, mock_publisher_class, mock_signal, mock_env
    ):
        """Test that missing scan path is created."""
        mock_exists.return_value = False
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        mock_observer_class.return_value = mock_observer

        service = FileWatcherService()

        # Start service in a thread
        def run_service():
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Give service time to initialize
        time.sleep(0.1)

        # Verify directory was created
        assert mock_mkdir.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)

    def test_handle_shutdown_signal(self, mock_env):
        """Test graceful shutdown on signal."""
        service = FileWatcherService()
        service.running = True

        # Simulate SIGTERM
        service._handle_shutdown(signal.SIGTERM, None)

        assert service.running is False

    @patch("main.MessagePublisher")
    @patch("main.Observer")
    def test_cleanup_with_running_observer(self, mock_observer_class, mock_publisher_class, mock_env):
        """Test cleanup with running observer."""
        mock_publisher = MagicMock()
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True

        service = FileWatcherService()
        service.publisher = mock_publisher
        service.observer = mock_observer

        service._cleanup()

        # Verify observer was stopped properly
        assert mock_observer.stop.called
        assert mock_observer.join.called
        assert mock_publisher.disconnect.called

    @patch("main.MessagePublisher")
    @patch("main.Observer")
    def test_cleanup_with_stuck_observer(self, mock_observer_class, mock_publisher_class, mock_env):
        """Test cleanup when observer doesn't stop within timeout."""
        mock_publisher = MagicMock()
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        # Observer stays alive even after join timeout
        mock_observer.join.return_value = None

        service = FileWatcherService()
        service.publisher = mock_publisher
        service.observer = mock_observer
        service.shutdown_timeout = 0.1  # Short timeout for testing

        service._cleanup()

        # Verify stop and join were still called
        assert mock_observer.stop.called
        assert mock_observer.join.called
        mock_observer.join.assert_called_with(timeout=0.1)

    def test_cleanup_without_resources(self, mock_env):
        """Test cleanup when no resources are initialized."""
        service = FileWatcherService()
        # Should not raise any exceptions
        service._cleanup()

    @patch("main.signal.signal")  # Mock signal handling
    @patch("main.MessagePublisher")
    @patch("main.Observer")
    def test_observer_schedule_failure(self, mock_observer_class, mock_publisher_class, mock_signal, mock_env):
        """Test handling of observer schedule failure."""
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        mock_observer = MagicMock()
        mock_observer.schedule.side_effect = Exception("Schedule failed")
        mock_observer_class.return_value = mock_observer

        service = FileWatcherService()

        # Service should exit gracefully
        service.start()

        assert service.running is False
        assert mock_publisher.disconnect.called
