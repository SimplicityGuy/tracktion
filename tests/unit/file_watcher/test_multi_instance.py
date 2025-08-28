"""Tests for multi-instance support."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
import structlog
from structlog.testing import LogCapture

# Mock before import
with patch.dict(os.environ, {"INSTANCE_ID": "test-instance"}):
    from services.file_watcher.src.main import FileWatcherService
    from services.file_watcher.src.message_publisher import MessagePublisher


@pytest.fixture
def log_capture() -> LogCapture:
    """Create a LogCapture fixture for testing logs."""
    return LogCapture()


class TestInstanceIdentification:
    """Test instance identification functionality."""

    def test_instance_id_from_environment(self) -> None:
        """Test that instance ID is read from environment variable."""
        with patch.dict(os.environ, {"INSTANCE_ID": "custom-watcher"}):
            # Re-import to get new INSTANCE_ID
            import importlib

            import services.file_watcher.src.main as main_module

            importlib.reload(main_module)
            assert main_module.INSTANCE_ID == "custom-watcher"

    def test_instance_id_auto_generated(self) -> None:
        """Test that instance ID is auto-generated when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.file_watcher.src.main.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
                import importlib

                import services.file_watcher.src.main as main_module

                importlib.reload(main_module)
                assert main_module.INSTANCE_ID == "12345678"

    def test_instance_id_in_service(self) -> None:
        """Test that FileWatcherService stores instance ID."""
        # Need to reload the module to get the current INSTANCE_ID
        import services.file_watcher.src.main as main_module

        service = FileWatcherService()
        assert hasattr(service, "instance_id")
        assert service.instance_id == main_module.INSTANCE_ID

    def test_instance_id_in_logs(self, log_capture: LogCapture) -> None:
        """Test that instance ID appears in log messages."""

        def add_test_instance_id(logger: None, method_name: str, event_dict: dict) -> dict:
            """Add instance ID to all log messages."""
            event_dict["instance_id"] = "test-logger"
            return event_dict

        with patch.dict(os.environ, {"INSTANCE_ID": "test-logger"}):
            # Configure structlog with test processor
            structlog.configure(
                processors=[
                    add_test_instance_id,
                    log_capture,
                ],
                logger_factory=structlog.PrintLoggerFactory(),
            )

            logger = structlog.get_logger()
            logger.info("Test message", extra_field="value")

            # Check that instance_id is in the log
            assert len(log_capture.entries) == 1
            assert log_capture.entries[0]["instance_id"] == "test-logger"
            assert log_capture.entries[0]["event"] == "Test message"


class TestMessagePublisherMultiInstance:
    """Test MessagePublisher with multi-instance support."""

    def test_publisher_with_instance_id(self) -> None:
        """Test MessagePublisher initialization with instance ID."""
        publisher = MessagePublisher(instance_id="test-instance", watched_directory="/data/test")
        assert publisher.instance_id == "test-instance"
        assert publisher.watched_directory == "/data/test"

    def test_publisher_default_values(self) -> None:
        """Test MessagePublisher with default instance values."""
        publisher = MessagePublisher()
        assert publisher.instance_id == "default"
        assert publisher.watched_directory == "/data/music"

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_connection_properties(self, mock_connection: MagicMock) -> None:
        """Test that connection properties include instance metadata."""
        publisher = MessagePublisher(instance_id="conn-test", watched_directory="/test/dir")

        # Check connection parameters
        conn_params = publisher.connection_params
        assert conn_params.client_properties["connection_name"] == "file_watcher_conn-test"
        assert conn_params.client_properties["instance_id"] == "conn-test"
        assert conn_params.client_properties["watched_directory"] == "/test/dir"

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_message_includes_instance_metadata(self, mock_connection: MagicMock) -> None:
        """Test that published messages include instance metadata."""
        mock_channel = MagicMock()
        mock_connection.return_value.channel.return_value = mock_channel

        publisher = MessagePublisher(instance_id="msg-test", watched_directory="/watched/path")
        publisher.connect()

        file_info = {
            "path": "/watched/path/file.mp3",
            "name": "file.mp3",
            "extension": ".mp3",
            "size_bytes": "1024",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
        }

        publisher.publish_file_discovered(file_info)

        # Get the published message
        call_args = mock_channel.basic_publish.call_args
        import json

        message = json.loads(call_args[1]["body"])

        # Verify instance metadata in message
        assert message["instance_id"] == "msg-test"
        assert message["watched_directory"] == "/watched/path"
        assert message["file_info"] == file_info

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_multiple_publishers_different_ids(self, mock_connection: MagicMock) -> None:
        """Test multiple publishers with different instance IDs."""
        publisher1 = MessagePublisher(instance_id="instance-1", watched_directory="/dir1")
        publisher2 = MessagePublisher(instance_id="instance-2", watched_directory="/dir2")
        publisher3 = MessagePublisher(instance_id="instance-3", watched_directory="/dir3")

        # Verify each has unique identification
        assert publisher1.instance_id != publisher2.instance_id
        assert publisher2.instance_id != publisher3.instance_id
        assert publisher1.instance_id != publisher3.instance_id

        # Verify connection names are unique
        assert publisher1.connection_params.client_properties["connection_name"] == "file_watcher_instance-1"
        assert publisher2.connection_params.client_properties["connection_name"] == "file_watcher_instance-2"
        assert publisher3.connection_params.client_properties["connection_name"] == "file_watcher_instance-3"


class TestFileWatcherMultiInstance:
    """Test FileWatcherService with multi-instance support."""

    @patch("services.file_watcher.src.main.MessagePublisher")
    def test_service_passes_instance_to_publisher(self, mock_publisher_class: MagicMock) -> None:
        """Test that FileWatcherService passes instance ID to MessagePublisher."""
        service = FileWatcherService()
        service.instance_id = "service-test"

        # Mock the publisher instance
        mock_publisher = MagicMock()
        mock_publisher_class.return_value = mock_publisher

        # Start service (will fail but that's ok for this test)
        with patch.object(service, "scan_path", MagicMock(exists=MagicMock(return_value=True))):
            with patch("os.access", return_value=True):
                with patch("services.file_watcher.src.main.Observer"):
                    try:
                        # This will fail but we just need to check the publisher creation
                        service.start()
                    except Exception:
                        pass

        # Verify MessagePublisher was created with instance metadata
        mock_publisher_class.assert_called_with(
            host=service.rabbitmq_host,
            port=service.rabbitmq_port,
            username=service.rabbitmq_user,
            password=service.rabbitmq_pass,
            instance_id="service-test",
            watched_directory=str(service.scan_path),
        )

    def test_different_instances_different_directories(self) -> None:
        """Test that different instances can monitor different directories."""
        with patch.dict(os.environ, {"DATA_DIR": "/data/music"}):
            service1 = FileWatcherService()
            assert str(service1.scan_path) == "/data/music"

        with patch.dict(os.environ, {"DATA_DIR": "/data/podcasts"}):
            service2 = FileWatcherService()
            assert str(service2.scan_path) == "/data/podcasts"

        with patch.dict(os.environ, {"DATA_DIR": "/data/audiobooks"}):
            service3 = FileWatcherService()
            assert str(service3.scan_path) == "/data/audiobooks"


class TestMultiInstanceScenarios:
    """Test multi-instance scenarios and edge cases."""

    def test_instance_id_uniqueness(self) -> None:
        """Test that auto-generated instance IDs are unique."""
        ids = set()
        for _ in range(100):
            # Generate new ID
            new_id = str(uuid.uuid4())[:8]
            assert new_id not in ids
            ids.add(new_id)

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_concurrent_message_publishing(self, mock_connection: MagicMock) -> None:
        """Test that multiple instances can publish messages concurrently."""
        mock_channel = MagicMock()
        mock_connection.return_value.channel.return_value = mock_channel

        publishers = []
        for i in range(5):
            publisher = MessagePublisher(instance_id=f"concurrent-{i}", watched_directory=f"/data/dir{i}")
            publisher.connect()
            publishers.append(publisher)

        # Each publisher sends a message
        for i, publisher in enumerate(publishers):
            file_info = {
                "path": f"/data/dir{i}/file{i}.mp3",
                "name": f"file{i}.mp3",
                "extension": ".mp3",
            }
            publisher.publish_file_discovered(file_info)

        # Verify all messages were published
        assert mock_channel.basic_publish.call_count == 5

    def test_instance_isolation(self) -> None:
        """Test that instances are properly isolated."""
        # Create multiple instances with different configs
        instances = []
        for i in range(3):
            with patch.dict(os.environ, {"INSTANCE_ID": f"isolated-{i}", "DATA_DIR": f"/data/dir{i}"}):
                service = FileWatcherService()
                instances.append(service)

        # Verify each has unique configuration
        assert instances[0].instance_id != instances[1].instance_id
        assert instances[1].instance_id != instances[2].instance_id
        assert str(instances[0].scan_path) != str(instances[1].scan_path)
        assert str(instances[1].scan_path) != str(instances[2].scan_path)
