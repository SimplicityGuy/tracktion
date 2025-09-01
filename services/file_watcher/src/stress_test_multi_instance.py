#!/usr/bin/env python3
"""Stress test for multi-instance file watcher support."""

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pika
import structlog

logger = structlog.get_logger()


@dataclass
class StressTestConfig:
    """Configuration for stress test."""

    num_instances: int = 3
    files_per_instance: int = 10
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_pass: str = "guest"


class MultiInstanceStressTest:
    """Stress test for multi-instance file watcher deployment."""

    def __init__(self, config: StressTestConfig | None = None) -> None:
        """Initialize stress test.

        Args:
            config: Stress test configuration object

        """
        self.config = config or StressTestConfig()
        self.test_dirs: list[Path] = []
        self.received_messages: list[dict[str, Any]] = []

    def setup_test_directories(self) -> None:
        """Create temporary directories for testing."""
        logger.info("Setting up test directories", num_instances=self.config.num_instances)

        for i in range(self.config.num_instances):
            temp_dir = Path(tempfile.mkdtemp(prefix=f"tracktion_test_{i}_"))
            self.test_dirs.append(temp_dir)
            logger.info("Created test directory", instance=i, path=str(temp_dir))

    def cleanup_test_directories(self) -> None:
        """Clean up temporary directories."""
        logger.info("Cleaning up test directories")

        for test_dir in self.test_dirs:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                logger.info("Removed test directory", path=str(test_dir))

    def create_test_files(self) -> None:
        """Create test audio files in each directory."""
        logger.info(
            "Creating test files",
            num_instances=self.config.num_instances,
            files_per_instance=self.config.files_per_instance,
        )

        def create_files_for_instance(instance_idx: int) -> None:
            test_dir = self.test_dirs[instance_idx]
            for file_idx in range(self.config.files_per_instance):
                # Create a simple MP3 file (just header bytes for testing)
                file_name = f"test_file_{instance_idx}_{file_idx}.mp3"
                file_path = test_dir / file_name

                # Write minimal MP3 header (ID3v2)
                with Path(file_path).open("wb") as f:
                    f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
                    f.write(b"\xff\xfb")  # MP3 sync word
                    f.write(b"\x00" * 100)  # Some dummy data

                logger.debug(
                    "Created test file",
                    instance=instance_idx,
                    file=file_name,
                    path=str(file_path),
                )

        # Create files in parallel for all instances
        with ThreadPoolExecutor(max_workers=self.config.num_instances) as executor:
            executor.map(create_files_for_instance, range(self.config.num_instances))

    def setup_rabbitmq_consumer(self) -> Any:
        """Set up RabbitMQ consumer to receive messages."""
        credentials = pika.PlainCredentials(self.config.rabbitmq_user, self.config.rabbitmq_pass)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.config.rabbitmq_host,
                port=self.config.rabbitmq_port,
                credentials=credentials,
            ),
        )
        channel = connection.channel()

        # Declare exchange and queue
        channel.exchange_declare(exchange="file_events", exchange_type="topic", durable=True)
        result = channel.queue_declare(queue="stress_test_queue", exclusive=True)
        queue_name = result.method.queue

        # Bind queue to exchange
        channel.queue_bind(exchange="file_events", queue=queue_name, routing_key="file.*")

        # Set up consumer callback
        def callback(ch: Any, method: Any, properties: Any, body: bytes) -> None:
            message = json.loads(body)
            self.received_messages.append(message)
            logger.debug(
                "Received message",
                instance_id=message.get("instance_id"),
                event_type=message.get("event_type"),
                file_path=message.get("file_info", {}).get("path"),
            )

        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

        return connection, channel

    def analyze_results(self) -> dict[str, Any]:
        """Analyze test results for conflicts and performance."""
        results: dict[str, Any] = {
            "total_messages": len(self.received_messages),
            "expected_messages": self.config.num_instances * self.config.files_per_instance,
            "instances": {},
            "conflicts": [],
            "performance": {},
        }

        # Group messages by instance
        for msg in self.received_messages:
            instance_id = msg.get("instance_id", "unknown")
            if instance_id not in results["instances"]:
                results["instances"][instance_id] = {
                    "count": 0,
                    "watched_directory": msg.get("watched_directory"),
                    "files": [],
                }

            results["instances"][instance_id]["count"] += 1
            results["instances"][instance_id]["files"].append(msg.get("file_info", {}).get("path"))

        # Check for conflicts (duplicate file paths from different instances)
        file_instance_map: dict[str, str] = {}
        for msg in self.received_messages:
            file_path = msg.get("file_info", {}).get("path")
            instance_id = msg.get("instance_id")

            if file_path and instance_id and file_path in file_instance_map:
                if file_instance_map[file_path] != instance_id:
                    results["conflicts"].append(
                        {
                            "file": file_path,
                            "instances": [file_instance_map[file_path], instance_id],
                        },
                    )
            elif file_path and instance_id:
                file_instance_map[file_path] = instance_id

        # Calculate performance metrics
        if self.received_messages:
            timestamps = [msg.get("timestamp") for msg in self.received_messages if msg.get("timestamp")]
            if timestamps:
                # Simple throughput calculation
                results["performance"]["throughput"] = len(self.received_messages)

        return results

    def run_stress_test(self, duration: int = 30) -> dict[str, Any]:
        """Run the stress test.

        Args:
            duration: Duration to run the test in seconds

        Returns:
            Test results dictionary

        """
        logger.info(
            "Starting multi-instance stress test",
            num_instances=self.config.num_instances,
            files_per_instance=self.config.files_per_instance,
            duration=duration,
        )

        try:
            # Set up test environment
            self.setup_test_directories()

            # Set up RabbitMQ consumer
            try:
                connection, channel = self.setup_rabbitmq_consumer()
                logger.info("Connected to RabbitMQ")
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                logger.info("Test will create files but won't receive messages")
                connection = None
                channel = None

            # Create test files
            self.create_test_files()

            # If connected to RabbitMQ, consume messages
            if channel:
                logger.info(f"Consuming messages for {duration} seconds...")
                # Start consuming in a non-blocking way
                connection.process_data_events(time_limit=duration)
            else:
                # Just wait if not connected
                time.sleep(duration)

            # Analyze results
            results = self.analyze_results()

            # Print summary
            logger.info("Stress test completed")
            logger.info(f"Total messages received: {results['total_messages']}")
            logger.info(f"Expected messages: {results['expected_messages']}")
            logger.info(f"Number of instances detected: {len(results['instances'])}")
            logger.info(f"Conflicts detected: {len(results['conflicts'])}")

            for instance_id, data in results["instances"].items():
                logger.info(f"Instance {instance_id}: {data['count']} messages, directory: {data['watched_directory']}")

            if results["conflicts"]:
                logger.warning(f"Conflicts found: {results['conflicts']}")

            return results

        finally:
            # Clean up
            self.cleanup_test_directories()
            if connection and not connection.is_closed:
                connection.close()


def main() -> None:
    """Main entry point for stress test."""
    parser = argparse.ArgumentParser(description="Multi-instance file watcher stress test")
    parser.add_argument(
        "--instances",
        type=int,
        default=3,
        help="Number of file watcher instances to simulate (default: 3)",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=10,
        help="Number of files per instance (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--rabbitmq-host",
        default="localhost",
        help="RabbitMQ host (default: localhost)",
    )
    parser.add_argument("--rabbitmq-port", type=int, default=5672, help="RabbitMQ port (default: 5672)")
    parser.add_argument("--rabbitmq-user", default="guest", help="RabbitMQ username (default: guest)")
    parser.add_argument("--rabbitmq-pass", default="guest", help="RabbitMQ password (default: guest)")

    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Run stress test
    config = StressTestConfig(
        num_instances=args.instances,
        files_per_instance=args.files,
        rabbitmq_host=args.rabbitmq_host,
        rabbitmq_port=args.rabbitmq_port,
        rabbitmq_user=args.rabbitmq_user,
        rabbitmq_pass=args.rabbitmq_pass,
    )
    stress_test = MultiInstanceStressTest(config)

    results = stress_test.run_stress_test(duration=args.duration)

    # Exit with appropriate code
    if results["conflicts"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
