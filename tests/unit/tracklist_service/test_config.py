"""Tests for tracklist service configuration."""

import os
from unittest.mock import patch

from services.tracklist_service.src.config import (
    APIConfig,
    CacheConfig,
    MessageQueueConfig,
    ScrapingConfig,
    ServiceConfig,
    get_config,
    reset_config,
    set_config,
)


class TestScrapingConfig:
    """Test scraping configuration."""

    def test_default_values(self):
        """Test default scraping configuration values."""
        config = ScrapingConfig()

        assert config.base_url == "https://1001tracklists.com"
        assert len(config.user_agents) == 3
        assert config.request_timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay_base == 1.0
        assert config.rate_limit_delay == 2.0
        assert config.session_timeout == 3600
        assert config.respect_robots_txt is True


class TestAPIConfig:
    """Test API configuration."""

    def test_default_values(self):
        """Test default API configuration values."""
        config = APIConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is False
        assert config.log_level == "info"
        assert config.api_prefix == "/api/v1"
        assert config.docs_enabled is True
        assert config.pagination_default_limit == 20
        assert config.pagination_max_limit == 100


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_values(self):
        """Test default cache configuration values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 1
        assert config.redis_password is None
        assert config.search_ttl_hours == 24
        assert config.failed_search_ttl_minutes == 30
        assert config.key_prefix == "tracklist:"


class TestMessageQueueConfig:
    """Test message queue configuration."""

    def test_default_values(self):
        """Test default message queue configuration values."""
        config = MessageQueueConfig()

        assert config.rabbitmq_url == "amqp://guest:guest@localhost:5672/"
        assert config.exchange_name == "tracktion_exchange"
        assert config.search_queue == "tracklist_search_queue"
        assert config.search_routing_key == "tracklist.search"
        assert config.result_routing_key == "tracklist.result"
        assert config.max_retries == 3
        assert config.base_delay_seconds == 2.0
        assert config.prefetch_count == 1


class TestServiceConfig:
    """Test main service configuration."""

    def test_default_values(self):
        """Test default service configuration values."""
        config = ServiceConfig()

        assert isinstance(config.scraping, ScrapingConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.message_queue, MessageQueueConfig)
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.health_check_enabled is True
        assert config.debug_mode is False

    @patch.dict(
        os.environ,
        {
            "TRACKLIST_SCRAPING_RATE_LIMIT_DELAY": "3.0",
            "TRACKLIST_API_PORT": "8001",
            "TRACKLIST_CACHE_ENABLED": "false",
            "TRACKLIST_LOG_LEVEL": "DEBUG",
        },
    )
    def test_from_env(self):
        """Test configuration creation from environment variables."""
        config = ServiceConfig.from_env()

        assert config.scraping.rate_limit_delay == 3.0
        assert config.api.port == 8001
        assert config.cache.enabled is False
        assert config.log_level == "DEBUG"

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ServiceConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        config = ServiceConfig()

        # Set invalid values
        config.scraping.request_timeout = -1
        config.api.port = 70000
        config.cache.redis_port = 0
        config.message_queue.max_retries = -1

        errors = config.validate()
        assert len(errors) > 0
        assert any("request_timeout must be positive" in error for error in errors)
        assert any("port must be between 1 and 65535" in error for error in errors)
        assert any("max_retries must be non-negative" in error for error in errors)


class TestConfigGlobals:
    """Test global configuration functions."""

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test setting custom configuration."""
        custom_config = ServiceConfig()
        custom_config.log_level = "DEBUG"

        set_config(custom_config)
        retrieved_config = get_config()

        assert retrieved_config is custom_config
        assert retrieved_config.log_level == "DEBUG"

    def test_reset_config(self):
        """Test resetting configuration."""
        # Get initial config
        initial_config = get_config()

        # Set custom config
        custom_config = ServiceConfig()
        custom_config.log_level = "DEBUG"
        set_config(custom_config)

        # Reset and get new config
        reset_config()
        new_config = get_config()

        assert new_config is not initial_config
        assert new_config is not custom_config
        assert new_config.log_level == "INFO"  # Default value
