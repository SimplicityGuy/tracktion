"""Unit tests for structured logging configuration."""

import json
import logging
import os
import sys
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.analysis_service.src.structured_logging import (
    CorrelationIdFilter,
    PerformanceLogger,
    StructuredFormatter,
    configure_structured_logging,
    get_performance_metrics,
)


class TestStructuredFormatter(unittest.TestCase):
    """Test StructuredFormatter class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.formatter = StructuredFormatter(
            service_name="test_service",
            include_hostname=True,
            include_function=True,
            include_thread=True,
        )
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)

    def test_basic_formatting(self) -> None:
        """Test basic log formatting."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        self.assertEqual(log_data["service"], "test_service")
        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["logger"], "test_logger")
        self.assertEqual(log_data["message"], "Test message")
        self.assertIn("timestamp", log_data)
        self.assertEqual(log_data["line"], 42)

    def test_formatting_with_extra_fields(self) -> None:
        """Test formatting with extra fields."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Add extra fields
        record.correlation_id = "corr123"  # type: ignore[attr-defined]
        record.user_id = "user456"  # type: ignore[attr-defined]

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        self.assertEqual(log_data["correlation_id"], "corr123")
        self.assertIn("extra", log_data)
        self.assertEqual(log_data["extra"]["user_id"], "user456")

    def test_formatting_with_exception(self) -> None:
        """Test formatting with exception info."""
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=None,
            )
            record.exc_info = sys.exc_info()

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        self.assertEqual(log_data["level"], "ERROR")
        self.assertIn("exception", log_data)
        self.assertIn("ValueError: Test error", log_data["exception"])

    def test_optional_fields(self) -> None:
        """Test optional field inclusion."""
        # Formatter without optional fields
        formatter = StructuredFormatter(
            service_name="test_service",
            include_hostname=False,
            include_function=False,
            include_thread=False,
        )

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        self.assertNotIn("hostname", log_data)
        self.assertNotIn("function", log_data)
        self.assertNotIn("thread", log_data)

    def test_performance_metrics_update(self) -> None:
        """Test that performance metrics are updated."""
        initial_metrics = get_performance_metrics()
        initial_count = initial_metrics["total_logs"]

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        self.formatter.format(record)

        updated_metrics = get_performance_metrics()
        self.assertEqual(updated_metrics["total_logs"], initial_count + 1)
        self.assertGreaterEqual(updated_metrics["avg_format_time_ms"], 0)


class TestCorrelationIdFilter(unittest.TestCase):
    """Test CorrelationIdFilter class."""

    def test_add_correlation_id(self) -> None:
        """Test adding correlation ID to log record."""
        filter_obj = CorrelationIdFilter(correlation_id="test_corr_123")

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Apply filter
        result = filter_obj.filter(record)

        self.assertTrue(result)
        self.assertEqual(record.correlation_id, "test_corr_123")  # type: ignore[attr-defined]

    def test_no_override_existing_correlation_id(self) -> None:
        """Test that existing correlation ID is not overridden."""
        filter_obj = CorrelationIdFilter(correlation_id="default_corr")

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "existing_corr"  # type: ignore[attr-defined]

        # Apply filter
        result = filter_obj.filter(record)

        self.assertTrue(result)
        self.assertEqual(record.correlation_id, "existing_corr")  # type: ignore[attr-defined]

    def test_no_correlation_id(self) -> None:
        """Test filter with no correlation ID."""
        filter_obj = CorrelationIdFilter(correlation_id=None)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Apply filter
        result = filter_obj.filter(record)

        self.assertTrue(result)
        self.assertFalse(hasattr(record, "correlation_id"))


class TestPerformanceLogger(unittest.TestCase):
    """Test PerformanceLogger context manager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.logger = MagicMock(spec=logging.Logger)

    def test_successful_operation(self) -> None:
        """Test logging successful operation."""
        with PerformanceLogger(
            self.logger,
            "test_operation",
            correlation_id="corr123",
            threshold_ms=100,
        ):
            time.sleep(0.01)  # 10ms operation

        # Should have start and completion logs
        self.assertEqual(self.logger.log.call_count, 2)

        # Check start log
        start_call = self.logger.log.call_args_list[0]
        self.assertEqual(start_call[0][0], logging.INFO)
        self.assertIn("Starting test_operation", start_call[0][1])

        # Check completion log
        end_call = self.logger.log.call_args_list[1]
        self.assertEqual(end_call[0][0], logging.INFO)
        self.assertIn("Completed test_operation", end_call[0][1])
        self.assertIn("duration_ms", end_call[1]["extra"])

    def test_slow_operation(self) -> None:
        """Test logging slow operation."""
        with PerformanceLogger(
            self.logger,
            "test_operation",
            threshold_ms=1,  # Very low threshold
        ):
            time.sleep(0.01)  # Will exceed threshold

        # Should have warning for slow operation
        self.logger.warning.assert_called_once()
        warning_call = self.logger.warning.call_args
        self.assertIn("threshold", warning_call[0][0])

    def test_failed_operation(self) -> None:
        """Test logging failed operation."""
        try:
            with PerformanceLogger(
                self.logger,
                "test_operation",
                correlation_id="corr123",
            ):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should have error log
        self.logger.error.assert_called_once()
        error_call = self.logger.error.call_args
        self.assertIn("failed", error_call[0][0])
        self.assertEqual(error_call[1]["extra"]["error"], "Test error")

    def test_no_correlation_id(self) -> None:
        """Test performance logger without correlation ID."""
        with PerformanceLogger(self.logger, "test_operation"):
            pass

        # Check that correlation_id is not in extra
        end_call = self.logger.log.call_args_list[1]
        self.assertNotIn("correlation_id", end_call[1]["extra"])


class TestConfigureStructuredLogging(unittest.TestCase):
    """Test configure_structured_logging function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Clear root logger handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Clear root logger handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []

    @patch("sys.stdout", new_callable=StringIO)
    def test_console_logging(self, mock_stdout: StringIO) -> None:
        """Test console logging configuration."""
        configure_structured_logging(
            service_name="test_service",
            log_level="INFO",
            include_console=True,
            include_hostname=False,
        )

        # Log a test message
        logger = logging.getLogger("test")
        logger.info("Test message")

        # Check output
        output = mock_stdout.getvalue()
        self.assertIn("Test message", output)

        # Parse JSON output
        log_data = json.loads(output.strip().split("\n")[-1])
        self.assertEqual(log_data["service"], "test_service")
        self.assertEqual(log_data["level"], "INFO")

    def test_file_logging(self) -> None:
        """Test file logging configuration."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp_file:
            log_file = tmp_file.name

        try:
            configure_structured_logging(
                service_name="test_service",
                log_level="DEBUG",
                log_file=log_file,
                include_console=False,
            )

            # Log test messages
            logger = logging.getLogger("test")
            logger.debug("Debug message")
            logger.info("Info message")

            # Read log file
            log_path = Path(log_file)
            with log_path.open() as f:
                lines = f.readlines()

            # Should have at least the configuration log and our test logs
            self.assertGreaterEqual(len(lines), 2)  # Configuration log + at least one test log

            # Parse last log entry
            log_data = json.loads(lines[-1])
            self.assertEqual(log_data["message"], "Info message")

        finally:
            # Clean up
            Path(log_file).unlink()

    def test_log_level_configuration(self) -> None:
        """Test log level configuration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            configure_structured_logging(
                service_name="test_service",
                log_level="WARNING",
                include_console=True,
            )

            # Create a new logger with explicit level
            logger = logging.getLogger("test.submodule")
            logger.setLevel(logging.WARNING)

            logger.debug("Debug message")  # Should not appear
            logger.info("Info message")  # Should not appear
            logger.warning("Warning message")  # Should appear

            output = mock_stdout.getvalue()
            self.assertNotIn("Debug message", output)
            self.assertNotIn("Info message", output)
            self.assertIn("Warning message", output)

    def test_env_variable_override(self) -> None:
        """Test that LOG_LEVEL environment variable overrides parameter."""
        with (
            patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            configure_structured_logging(
                service_name="test_service",
                log_level="DEBUG",  # This should be overridden
                include_console=True,
            )

            logger = logging.getLogger("test")
            logger.warning("Warning message")  # Should not appear
            logger.error("Error message")  # Should appear

            output = mock_stdout.getvalue()
            self.assertNotIn("Warning message", output)
            self.assertIn("Error message", output)

    def test_library_log_levels(self) -> None:
        """Test that noisy library loggers are quieted."""

        configure_structured_logging(
            service_name="test_service",
            log_level="DEBUG",
        )

        # Check that specific loggers have WARNING level
        pika_logger = logging.getLogger("pika")
        redis_logger = logging.getLogger("redis")
        urllib3_logger = logging.getLogger("urllib3")

        self.assertEqual(pika_logger.level, logging.WARNING)
        self.assertEqual(redis_logger.level, logging.WARNING)
        self.assertEqual(urllib3_logger.level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
