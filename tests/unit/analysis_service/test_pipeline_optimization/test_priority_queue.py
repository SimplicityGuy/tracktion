"""Unit tests for priority queue functionality."""

import unittest
from unittest.mock import MagicMock

from services.analysis_service.src.priority_queue import (
    MessagePriority,
    PriorityCalculator,
    PriorityConfig,
    add_priority_to_message,
    setup_priority_queue,
)


class TestMessagePriority(unittest.TestCase):
    """Test MessagePriority enum."""

    def test_priority_values(self) -> None:
        """Test that priority values are correctly defined."""
        self.assertEqual(MessagePriority.CRITICAL, 10)
        self.assertEqual(MessagePriority.HIGH, 7)
        self.assertEqual(MessagePriority.NORMAL, 5)
        self.assertEqual(MessagePriority.LOW, 3)
        self.assertEqual(MessagePriority.BACKGROUND, 1)

    def test_priority_comparison(self) -> None:
        """Test that priorities can be compared."""
        self.assertGreater(MessagePriority.CRITICAL, MessagePriority.HIGH)
        self.assertGreater(MessagePriority.HIGH, MessagePriority.NORMAL)
        self.assertGreater(MessagePriority.NORMAL, MessagePriority.LOW)
        self.assertGreater(MessagePriority.LOW, MessagePriority.BACKGROUND)


class TestPriorityConfig(unittest.TestCase):
    """Test PriorityConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PriorityConfig()
        self.assertTrue(config.enable_priority)
        self.assertEqual(config.max_priority, 10)
        self.assertEqual(config.default_priority, MessagePriority.NORMAL)
        self.assertEqual(config.small_file_threshold, 10.0)
        self.assertEqual(config.large_file_threshold, 100.0)
        self.assertEqual(config.small_file_boost, 2)
        self.assertEqual(config.large_file_penalty, 2)
        self.assertEqual(config.retry_boost, 3)
        self.assertEqual(config.user_request_boost, 3)

    def test_format_priorities_initialization(self) -> None:
        """Test that format priorities are initialized."""
        config = PriorityConfig()
        self.assertIsNotNone(config.format_priorities)
        if config.format_priorities:  # Type guard for mypy
            self.assertEqual(config.format_priorities["mp3"], MessagePriority.NORMAL)
            self.assertEqual(config.format_priorities["wav"], MessagePriority.LOW)

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        custom_formats = {"mp3": 8, "flac": 6}
        config = PriorityConfig(
            enable_priority=False,
            max_priority=15,
            default_priority=7,
            format_priorities=custom_formats,
        )
        self.assertFalse(config.enable_priority)
        self.assertEqual(config.max_priority, 15)
        self.assertEqual(config.default_priority, 7)
        self.assertEqual(config.format_priorities, custom_formats)


class TestPriorityCalculator(unittest.TestCase):
    """Test PriorityCalculator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = PriorityConfig()
        self.calculator = PriorityCalculator(self.config)

    def test_disabled_priority(self) -> None:
        """Test that disabled priority returns default."""
        config = PriorityConfig(enable_priority=False, default_priority=7)
        calculator = PriorityCalculator(config)
        priority = calculator.calculate_priority("test.mp3")
        self.assertEqual(priority, 7)

    def test_custom_priority_override(self) -> None:
        """Test that custom priority overrides calculation."""
        priority = self.calculator.calculate_priority("test.mp3", custom_priority=8)
        self.assertEqual(priority, 8)

    def test_custom_priority_clamping(self) -> None:
        """Test that custom priority is clamped to valid range."""
        # Test upper bound
        priority = self.calculator.calculate_priority("test.mp3", custom_priority=20)
        self.assertEqual(priority, self.config.max_priority)

        # Test lower bound
        priority = self.calculator.calculate_priority("test.mp3", custom_priority=0)
        self.assertEqual(priority, 1)

    def test_format_based_priority(self) -> None:
        """Test priority based on file format."""
        # MP3 should get normal priority
        priority = self.calculator.calculate_priority("test.mp3")
        self.assertEqual(priority, MessagePriority.NORMAL)

        # WAV should get low priority
        priority = self.calculator.calculate_priority("test.wav")
        self.assertEqual(priority, MessagePriority.LOW)

        # Unknown format should get default
        priority = self.calculator.calculate_priority("test.xyz")
        self.assertEqual(priority, self.config.default_priority)

    def test_small_file_boost(self) -> None:
        """Test that small files get priority boost."""
        priority = self.calculator.calculate_priority(
            "test.mp3",
            file_size_mb=5.0,  # Small file
        )
        expected = MessagePriority.NORMAL + self.config.small_file_boost
        self.assertEqual(priority, expected)

    def test_large_file_penalty(self) -> None:
        """Test that large files get priority penalty."""
        priority = self.calculator.calculate_priority(
            "test.mp3",
            file_size_mb=150.0,  # Large file
        )
        expected = MessagePriority.NORMAL - self.config.large_file_penalty
        self.assertEqual(priority, expected)

    def test_retry_boost(self) -> None:
        """Test that retries get priority boost."""
        priority = self.calculator.calculate_priority("test.mp3", is_retry=True)
        expected = MessagePriority.NORMAL + self.config.retry_boost
        self.assertEqual(priority, expected)

    def test_user_request_boost(self) -> None:
        """Test that user requests get priority boost."""
        priority = self.calculator.calculate_priority("test.mp3", is_user_request=True)
        expected = MessagePriority.NORMAL + self.config.user_request_boost
        self.assertEqual(priority, expected)

    def test_combined_adjustments(self) -> None:
        """Test that multiple adjustments combine correctly."""
        priority = self.calculator.calculate_priority(
            "test.mp3",
            file_size_mb=5.0,  # Small file: +2
            is_retry=True,  # Retry: +3
            is_user_request=True,  # User request: +3
        )
        expected = MessagePriority.NORMAL + 2 + 3 + 3
        expected = min(expected, self.config.max_priority)
        self.assertEqual(priority, expected)

    def test_priority_clamping(self) -> None:
        """Test that final priority is clamped to valid range."""
        # Test that boosted priority doesn't exceed max
        priority = self.calculator.calculate_priority(
            "test.mp3",
            file_size_mb=5.0,
            is_retry=True,
            is_user_request=True,
        )
        self.assertLessEqual(priority, self.config.max_priority)

        # Test that penalized priority doesn't go below 1
        priority = self.calculator.calculate_priority(
            "test.wav",  # Low priority format
            file_size_mb=200.0,  # Large file penalty
        )
        self.assertGreaterEqual(priority, 1)


class TestSetupPriorityQueue(unittest.TestCase):
    """Test setup_priority_queue function."""

    def test_setup_priority_queue(self) -> None:
        """Test that priority queue is set up correctly."""
        mock_channel = MagicMock()

        setup_priority_queue(mock_channel, "test_queue", max_priority=15)

        mock_channel.queue_declare.assert_called_once_with(
            queue="test_queue",
            durable=True,
            arguments={
                "x-max-priority": 15,
                "x-message-ttl": 3600000,
            },
        )


class TestAddPriorityToMessage(unittest.TestCase):
    """Test add_priority_to_message function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = PriorityConfig()
        self.calculator = PriorityCalculator(self.config)

    def test_add_priority_to_message(self) -> None:
        """Test that priority is added to message."""
        message = {
            "file_path": "test.mp3",
            "recording_id": "123",
            "file_size_mb": 5.0,
        }

        result = add_priority_to_message(message, self.calculator)

        self.assertIn("priority", result)
        expected_priority = MessagePriority.NORMAL + self.config.small_file_boost
        self.assertEqual(result["priority"], expected_priority)

    def test_add_priority_with_retry(self) -> None:
        """Test that retry count affects priority."""
        message = {
            "file_path": "test.mp3",
            "recording_id": "123",
            "retry_count": 2,
        }

        result = add_priority_to_message(message, self.calculator)

        expected_priority = MessagePriority.NORMAL + self.config.retry_boost
        self.assertEqual(result["priority"], expected_priority)

    def test_add_priority_with_user_request(self) -> None:
        """Test that user request flag affects priority."""
        message = {
            "file_path": "test.mp3",
            "recording_id": "123",
            "user_request": True,
        }

        result = add_priority_to_message(message, self.calculator)

        expected_priority = MessagePriority.NORMAL + self.config.user_request_boost
        self.assertEqual(result["priority"], expected_priority)

    def test_add_priority_preserves_existing(self) -> None:
        """Test that existing priority is preserved if custom."""
        message = {
            "file_path": "test.mp3",
            "recording_id": "123",
            "priority": 9,
        }

        result = add_priority_to_message(message, self.calculator)

        # Custom priority should be used
        self.assertEqual(result["priority"], 9)


if __name__ == "__main__":
    unittest.main()
