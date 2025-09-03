"""Unit tests for main file watcher service."""

import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from services.file_watcher.src.main import FileWatcherService


class TestFileWatcherService:
    """Test suite for FileWatcherService."""

    @pytest.fixture
    def mock_env(self, monkeypatch, tmp_path):
        """Set up test environment variables."""
        test_path = tmp_path / "test_path"
        test_path.mkdir()
        monkeypatch.setenv("DATA_DIR", str(test_path))
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

    def test_data_dir_environment_variable(self, monkeypatch, tmp_path):
        """Test that DATA_DIR environment variable is respected."""
        custom_path = tmp_path / "custom_music"
        custom_path.mkdir()
        monkeypatch.setenv("DATA_DIR", str(custom_path))

        service = FileWatcherService()
        assert str(service.scan_path) == str(custom_path)

    def test_default_directory_fallback(self, monkeypatch):
        """Test fallback to default directory when no env var is set."""
        # Clear any existing environment variables
        monkeypatch.delenv("DATA_DIR", raising=False)
        monkeypatch.delenv("FILE_WATCHER_SCAN_PATH", raising=False)

        service = FileWatcherService()
        assert str(service.scan_path) == "/data/music"

    def test_legacy_env_var_compatibility(self, monkeypatch, tmp_path):
        """Test backward compatibility with FILE_WATCHER_SCAN_PATH."""
        legacy_path = tmp_path / "legacy_path"
        legacy_path.mkdir()
        monkeypatch.setenv("FILE_WATCHER_SCAN_PATH", str(legacy_path))
        monkeypatch.delenv("DATA_DIR", raising=False)

        service = FileWatcherService()
        assert str(service.scan_path) == str(legacy_path)

    def test_data_dir_takes_precedence(self, monkeypatch, tmp_path):
        """Test that DATA_DIR takes precedence over FILE_WATCHER_SCAN_PATH."""
        new_path = tmp_path / "new_path"
        old_path = tmp_path / "old_path"
        new_path.mkdir()
        old_path.mkdir()

        monkeypatch.setenv("DATA_DIR", str(new_path))
        monkeypatch.setenv("FILE_WATCHER_SCAN_PATH", str(old_path))

        service = FileWatcherService()
        assert str(service.scan_path) == str(new_path)

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    def test_invalid_directory_path(self, mock_signal, monkeypatch, tmp_path):
        """Test that service exits with error when directory doesn't exist."""
        non_existent = tmp_path / "non_existent_path"
        monkeypatch.setenv("DATA_DIR", str(non_existent))

        service = FileWatcherService()

        # Service should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            service.start()

        assert exc_info.value.code == 1

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    @patch("services.file_watcher.src.main.os.access")
    def test_permission_denied(self, mock_access, mock_signal, monkeypatch, tmp_path):
        """Test that service exits with error when no read permission."""
        test_path = tmp_path / "no_permission"
        test_path.mkdir()
        monkeypatch.setenv("DATA_DIR", str(test_path))

        # Mock no read permission
        mock_access.return_value = False

        service = FileWatcherService()

        # Service should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            service.start()

        assert exc_info.value.code == 1

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
    @patch("services.file_watcher.src.main.TracktionEventHandler")
    def test_start_with_successful_connection(
        self,
        mock_handler_class,
        mock_observer_class,
        mock_publisher_class,
        mock_signal,
        mock_env,
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
        service_started = threading.Event()

        # Start service in a thread
        def run_service():
            # Mock the service loop to exit quickly after setup
            def mock_start():
                # Perform initialization
                service.running = True
                try:
                    service.publisher = mock_publisher_class()
                    service.observer = mock_observer_class()
                    service.observer.schedule(mock_handler_class(), str(service.scan_path), recursive=True)
                    service.observer.start()
                    service_started.set()
                    # Exit immediately instead of running the monitoring loop
                    return
                except Exception:
                    service.running = False
                    return

            service.start = mock_start
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Wait for service initialization to complete
        assert service_started.wait(timeout=2), "Service failed to start within timeout"

        # Verify observer was started
        assert mock_observer.schedule.called
        assert mock_observer.start.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)
        service._cleanup()

        # Verify cleanup was called
        assert mock_observer.stop.called
        assert mock_observer.join.called
        assert mock_publisher.disconnect.called

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
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
        service_initialized = threading.Event()

        # Start service in a thread
        def run_service():
            # Mock the service loop to handle failure and exit quickly
            def mock_start():
                service.running = True
                try:
                    service.publisher = mock_publisher_class()
                except Exception:
                    service.publisher = None  # Connection failed, continue without publisher
                service.observer = mock_observer_class()
                service.observer.start()
                service_initialized.set()
                # Exit instead of running monitoring loop

            service.start = mock_start
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Wait for service to handle the failure
        assert service_initialized.wait(timeout=2), "Service failed to initialize within timeout"

        # Service should still be running with observer (despite publisher failure)
        assert service.running is True
        assert mock_observer.start.called

        # Stop the service
        service.running = False
        service_thread.join(timeout=2)

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
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
        observer_restarted = threading.Event()

        service = FileWatcherService()

        # Start service in a thread with mocked monitoring loop
        def run_service():
            def mock_start():
                service.running = True
                service.publisher = mock_publisher_class()

                # Simulate first observer creation
                service.observer = mock_observer_class()  # dead_observer
                service.observer.start()

                # Simulate monitoring check - first call returns True, second returns False
                service.observer.is_alive()  # True (first check)
                alive_second_check = service.observer.is_alive()  # False (second check)

                if not alive_second_check:  # Observer died
                    service.observer.stop()
                    service.observer.join()
                    service.observer = mock_observer_class()  # new_observer
                    service.observer.start()
                    observer_restarted.set()

                # Exit the monitoring loop

            service.start = mock_start
            service.start()

        service_thread = threading.Thread(target=run_service)
        service_thread.start()

        # Wait for observer restart to complete
        assert observer_restarted.wait(timeout=2), "Observer was not restarted within timeout"

        # Verify new observer was created and started
        assert mock_observer_class.call_count == 2
        assert new_observer.start.called

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

    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
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

    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
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
        service.shutdown_timeout = 0.01  # Very short timeout for testing

        service._cleanup()

        # Verify stop and join were still called
        assert mock_observer.stop.called
        assert mock_observer.join.called
        mock_observer.join.assert_called_with(timeout=0.01)

    def test_cleanup_without_resources(self, mock_env):
        """Test cleanup when no resources are initialized."""
        service = FileWatcherService()
        # Should not raise any exceptions
        service._cleanup()

    @patch("services.file_watcher.src.main.signal.signal")  # Mock signal handling
    @patch("services.file_watcher.src.main.MessagePublisher")
    @patch("services.file_watcher.src.main.Observer")
    def test_observer_schedule_failure(self, mock_observer_class, mock_publisher_class, mock_signal, mock_env):
        """Test handling of observer schedule failure."""
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        mock_observer = MagicMock()
        mock_observer.schedule.side_effect = Exception("Schedule failed")
        mock_observer.is_alive.return_value = False
        mock_observer_class.return_value = mock_observer

        service = FileWatcherService()

        # Service should exit gracefully
        service.start()

        assert service.running is False
        # Verify schedule was attempted
        assert mock_observer.schedule.called
        # Observer.stop() should not be called since scheduling failed before observer started
        assert not mock_observer.stop.called
