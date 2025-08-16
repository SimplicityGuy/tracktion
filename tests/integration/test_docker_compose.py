"""Integration tests for Docker Compose setup."""

import subprocess
import time
import unittest
from typing import Any, Dict


class TestDockerCompose(unittest.TestCase):
    """Test Docker Compose configuration and service health."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        cls.compose_file = "infrastructure/docker-compose.yaml"

    def test_docker_compose_config(self) -> None:
        """Test that docker-compose configuration is valid."""
        result = subprocess.run(
            ["docker-compose", "-f", self.compose_file, "config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Config validation failed: {result.stderr}")

    def test_service_definitions(self) -> None:
        """Test that all required services are defined."""
        result = subprocess.run(
            ["docker-compose", "-f", self.compose_file, "config", "--services"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        
        services = result.stdout.strip().split("\n")
        required_services = [
            "postgres",
            "neo4j",
            "redis",
            "rabbitmq",
            "file_watcher",
            "cataloging_service",
            "analysis_service",
            "tracklist_service",
        ]
        
        for service in required_services:
            self.assertIn(service, services, f"Service {service} not found in docker-compose")


if __name__ == "__main__":
    unittest.main()