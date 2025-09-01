"""
Configuration management for the tracklist service.

Provides centralized configuration for web scraping, API endpoints,
caching, and messaging settings.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations."""

    base_url: str = "https://1001tracklists.com"
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
    )
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay_base: float = 1.0
    rate_limit_delay: float = 2.0
    session_timeout: int = 3600
    respect_robots_txt: bool = True


@dataclass
class APIConfig:
    """Configuration for API endpoints."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    api_prefix: str = "/api/v1"
    docs_enabled: bool = True
    pagination_default_limit: int = 20
    pagination_max_limit: int = 100


@dataclass
class CacheConfig:
    """Configuration for Redis caching."""

    enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # Use different DB than analysis service
    redis_password: str | None = None
    search_ttl_hours: int = 24
    failed_search_ttl_minutes: int = 30
    key_prefix: str = "tracklist:"


@dataclass
class MessageQueueConfig:
    """Configuration for RabbitMQ messaging."""

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    exchange_name: str = "tracktion_exchange"
    search_queue: str = "tracklist_search_queue"
    search_routing_key: str = "tracklist.search"
    result_routing_key: str = "tracklist.result"
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    prefetch_count: int = 1


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database."""

    host: str = "localhost"
    port: int = 5432
    name: str = "tracktion_tracklist"
    user: str = "tracklist_user"
    password: str = "tracklist_password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo_queries: bool = False


@dataclass
class ServiceConfig:
    """Main configuration for the tracklist service."""

    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    message_queue: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Service-level settings
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    debug_mode: bool = False

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Create configuration from environment variables.

        Environment variables follow the pattern:
        TRACKLIST_<SECTION>_<SETTING>

        Examples:
        - TRACKLIST_SCRAPING_RATE_LIMIT_DELAY=3.0
        - TRACKLIST_API_PORT=8001
        - TRACKLIST_CACHE_ENABLED=false
        """
        config = cls()

        # Scraping configuration
        config.scraping.base_url = os.getenv("TRACKLIST_SCRAPING_BASE_URL", config.scraping.base_url)
        if val := os.getenv("TRACKLIST_SCRAPING_REQUEST_TIMEOUT"):
            config.scraping.request_timeout = int(val)
        if val := os.getenv("TRACKLIST_SCRAPING_MAX_RETRIES"):
            config.scraping.max_retries = int(val)
        if val := os.getenv("TRACKLIST_SCRAPING_RETRY_DELAY_BASE"):
            config.scraping.retry_delay_base = float(val)
        if val := os.getenv("TRACKLIST_SCRAPING_RATE_LIMIT_DELAY"):
            config.scraping.rate_limit_delay = float(val)
        if val := os.getenv("TRACKLIST_SCRAPING_SESSION_TIMEOUT"):
            config.scraping.session_timeout = int(val)
        if val := os.getenv("TRACKLIST_SCRAPING_RESPECT_ROBOTS_TXT"):
            config.scraping.respect_robots_txt = val.lower() in ("true", "1", "yes")

        # API configuration
        config.api.host = os.getenv("TRACKLIST_API_HOST", config.api.host)
        if val := os.getenv("TRACKLIST_API_PORT"):
            config.api.port = int(val)
        if val := os.getenv("TRACKLIST_API_WORKERS"):
            config.api.workers = int(val)
        if val := os.getenv("TRACKLIST_API_RELOAD"):
            config.api.reload = val.lower() in ("true", "1", "yes")
        config.api.log_level = os.getenv("TRACKLIST_API_LOG_LEVEL", config.api.log_level)
        config.api.api_prefix = os.getenv("TRACKLIST_API_PREFIX", config.api.api_prefix)
        if val := os.getenv("TRACKLIST_API_DOCS_ENABLED"):
            config.api.docs_enabled = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKLIST_API_PAGINATION_DEFAULT_LIMIT"):
            config.api.pagination_default_limit = int(val)
        if val := os.getenv("TRACKLIST_API_PAGINATION_MAX_LIMIT"):
            config.api.pagination_max_limit = int(val)

        # Cache configuration
        if val := os.getenv("TRACKLIST_CACHE_ENABLED"):
            config.cache.enabled = val.lower() in ("true", "1", "yes")
        config.cache.redis_host = os.getenv("TRACKLIST_CACHE_REDIS_HOST", config.cache.redis_host)
        if val := os.getenv("TRACKLIST_CACHE_REDIS_PORT"):
            config.cache.redis_port = int(val)
        if val := os.getenv("TRACKLIST_CACHE_REDIS_DB"):
            config.cache.redis_db = int(val)
        config.cache.redis_password = os.getenv("TRACKLIST_CACHE_REDIS_PASSWORD")
        if val := os.getenv("TRACKLIST_CACHE_SEARCH_TTL_HOURS"):
            config.cache.search_ttl_hours = int(val)
        if val := os.getenv("TRACKLIST_CACHE_FAILED_SEARCH_TTL_MINUTES"):
            config.cache.failed_search_ttl_minutes = int(val)
        config.cache.key_prefix = os.getenv("TRACKLIST_CACHE_KEY_PREFIX", config.cache.key_prefix)

        # Message queue configuration
        config.message_queue.rabbitmq_url = os.getenv("TRACKLIST_MQ_RABBITMQ_URL", config.message_queue.rabbitmq_url)
        config.message_queue.exchange_name = os.getenv("TRACKLIST_MQ_EXCHANGE_NAME", config.message_queue.exchange_name)
        config.message_queue.search_queue = os.getenv("TRACKLIST_MQ_SEARCH_QUEUE", config.message_queue.search_queue)
        config.message_queue.search_routing_key = os.getenv(
            "TRACKLIST_MQ_SEARCH_ROUTING_KEY", config.message_queue.search_routing_key
        )
        config.message_queue.result_routing_key = os.getenv(
            "TRACKLIST_MQ_RESULT_ROUTING_KEY", config.message_queue.result_routing_key
        )
        if val := os.getenv("TRACKLIST_MQ_MAX_RETRIES"):
            config.message_queue.max_retries = int(val)
        if val := os.getenv("TRACKLIST_MQ_BASE_DELAY_SECONDS"):
            config.message_queue.base_delay_seconds = float(val)
        if val := os.getenv("TRACKLIST_MQ_PREFETCH_COUNT"):
            config.message_queue.prefetch_count = int(val)

        # Service-level settings
        config.log_level = os.getenv("TRACKLIST_LOG_LEVEL", config.log_level)
        if val := os.getenv("TRACKLIST_METRICS_ENABLED"):
            config.metrics_enabled = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKLIST_HEALTH_CHECK_ENABLED"):
            config.health_check_enabled = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKLIST_DEBUG_MODE"):
            config.debug_mode = val.lower() in ("true", "1", "yes")

        return config

    def validate(self) -> list[str]:
        """Validate configuration and return any errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate scraping settings
        if self.scraping.request_timeout <= 0:
            errors.append("Scraping request_timeout must be positive")
        if self.scraping.max_retries < 0:
            errors.append("Scraping max_retries must be non-negative")
        if self.scraping.rate_limit_delay < 0:
            errors.append("Scraping rate_limit_delay must be non-negative")

        # Validate API settings
        if not 1 <= self.api.port <= 65535:
            errors.append("API port must be between 1 and 65535")
        if self.api.workers <= 0:
            errors.append("API workers must be positive")
        if self.api.pagination_default_limit <= 0:
            errors.append("API pagination_default_limit must be positive")
        if self.api.pagination_max_limit <= 0:
            errors.append("API pagination_max_limit must be positive")
        if self.api.pagination_default_limit > self.api.pagination_max_limit:
            errors.append("API pagination_default_limit cannot exceed pagination_max_limit")

        # Validate cache settings
        if not 1 <= self.cache.redis_port <= 65535:
            errors.append("Cache redis_port must be between 1 and 65535")
        if self.cache.search_ttl_hours <= 0:
            errors.append("Cache search_ttl_hours must be positive")
        if self.cache.failed_search_ttl_minutes <= 0:
            errors.append("Cache failed_search_ttl_minutes must be positive")

        # Validate message queue settings
        if self.message_queue.max_retries < 0:
            errors.append("Message queue max_retries must be non-negative")
        if self.message_queue.base_delay_seconds <= 0:
            errors.append("Message queue base_delay_seconds must be positive")
        if self.message_queue.prefetch_count <= 0:
            errors.append("Message queue prefetch_count must be positive")

        return errors


# Global configuration instance
_config: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    """Get the global configuration instance.

    Creates the configuration from environment variables on first call.
    """
    global _config  # noqa: PLW0603  # Global config pattern for application configuration
    if _config is None:
        _config = ServiceConfig.from_env()
    return _config


def set_config(config: ServiceConfig) -> None:
    """Set the global configuration instance.

    Useful for testing or when loading configuration from files.
    """
    global _config  # noqa: PLW0603  # Global config pattern for application configuration
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance.

    The next call to get_config() will recreate from environment.
    """
    global _config  # noqa: PLW0603  # Global config pattern for application configuration
    _config = None
