"""Configuration classes for circuit breaker behavior."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import CircuitBreakerConfigurationError


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class ServiceType(Enum):
    """Service types for preset configurations."""

    DATABASE = "database"
    HTTP_API = "http_api"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"
    ANALYSIS = "analysis"
    SCRAPING = "scraping"
    QUEUE = "queue"
    CACHE = "cache"


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    fallback_calls: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list[dict[str, Any]] = field(default_factory=list)
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure threshold configuration
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open before closing
    timeout: float = 60.0  # Seconds to wait before trying half-open

    # Time window for counting failures
    failure_window: float = 60.0  # Time window for failure counting

    # What exceptions should trigger the circuit breaker
    expected_exceptions: tuple[type[Exception], ...] = (Exception,)

    # Optional fallback function
    fallback: Callable[[], Any] | None = None

    # Monitoring hooks
    on_open: Callable[[str], None] | None = None
    on_close: Callable[[str], None] | None = None
    on_half_open: Callable[[str], None] | None = None

    # Domain-specific configuration
    domain: str | None = None  # Domain this configuration applies to
    service_type: ServiceType | None = None  # Service type for preset selection

    # Performance monitoring
    track_response_time: bool = True  # Track response times for analysis
    response_time_threshold: float | None = None  # Threshold for slow responses

    # Advanced configuration
    half_open_max_calls: int = 3  # Max calls in half-open state
    recovery_delay_multiplier: float = 1.0  # Multiplier for recovery delay
    enable_jitter: bool = True  # Add jitter to timeouts

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.failure_threshold < 1:
            raise CircuitBreakerConfigurationError(
                "failure_threshold must be at least 1",
                config_key="failure_threshold",
                config_value=self.failure_threshold,
            )

        if self.success_threshold < 1:
            raise CircuitBreakerConfigurationError(
                "success_threshold must be at least 1",
                config_key="success_threshold",
                config_value=self.success_threshold,
            )

        if self.timeout <= 0:
            raise CircuitBreakerConfigurationError(
                "timeout must be positive",
                config_key="timeout",
                config_value=self.timeout,
            )

        if self.failure_window <= 0:
            raise CircuitBreakerConfigurationError(
                "failure_window must be positive",
                config_key="failure_window",
                config_value=self.failure_window,
            )


class ServicePresets:
    """Predefined configurations for common service types."""

    @staticmethod
    def database() -> CircuitBreakerConfig:
        """Configuration for database connections."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0,
            failure_window=60.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.DATABASE,
            response_time_threshold=5.0,
            half_open_max_calls=1,
        )

    @staticmethod
    def http_api() -> CircuitBreakerConfig:
        """Configuration for HTTP API calls."""
        return CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=60.0,
            failure_window=120.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.HTTP_API,
            response_time_threshold=10.0,
            half_open_max_calls=2,
        )

    @staticmethod
    def file_system() -> CircuitBreakerConfig:
        """Configuration for file system operations."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=15.0,
            failure_window=30.0,
            expected_exceptions=(OSError, IOError),
            service_type=ServiceType.FILE_SYSTEM,
            response_time_threshold=2.0,
            half_open_max_calls=1,
        )

    @staticmethod
    def external_service() -> CircuitBreakerConfig:
        """Configuration for external service calls."""
        return CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout=120.0,
            failure_window=300.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.EXTERNAL_SERVICE,
            response_time_threshold=30.0,
            half_open_max_calls=3,
            recovery_delay_multiplier=1.5,
        )

    @staticmethod
    def analysis() -> CircuitBreakerConfig:
        """Configuration for analysis operations."""
        return CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=45.0,
            failure_window=90.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.ANALYSIS,
            response_time_threshold=60.0,
            half_open_max_calls=1,
        )

    @staticmethod
    def scraping() -> CircuitBreakerConfig:
        """Configuration for web scraping operations."""
        return CircuitBreakerConfig(
            failure_threshold=8,
            success_threshold=4,
            timeout=180.0,
            failure_window=600.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.SCRAPING,
            response_time_threshold=45.0,
            half_open_max_calls=2,
            recovery_delay_multiplier=2.0,
            enable_jitter=True,
        )

    @staticmethod
    def queue() -> CircuitBreakerConfig:
        """Configuration for queue operations."""
        return CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0,
            failure_window=60.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.QUEUE,
            response_time_threshold=5.0,
            half_open_max_calls=2,
        )

    @staticmethod
    def cache() -> CircuitBreakerConfig:
        """Configuration for cache operations."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=10.0,
            failure_window=30.0,
            expected_exceptions=(Exception,),
            service_type=ServiceType.CACHE,
            response_time_threshold=1.0,
            half_open_max_calls=1,
        )

    @classmethod
    def get_preset(cls, service_type: ServiceType) -> CircuitBreakerConfig:
        """Get preset configuration for a service type.

        Args:
            service_type: Type of service

        Returns:
            Circuit breaker configuration

        Raises:
            CircuitBreakerConfigurationError: If service type is not supported
        """
        preset_map = {
            ServiceType.DATABASE: cls.database,
            ServiceType.HTTP_API: cls.http_api,
            ServiceType.FILE_SYSTEM: cls.file_system,
            ServiceType.EXTERNAL_SERVICE: cls.external_service,
            ServiceType.ANALYSIS: cls.analysis,
            ServiceType.SCRAPING: cls.scraping,
            ServiceType.QUEUE: cls.queue,
            ServiceType.CACHE: cls.cache,
        }

        if service_type not in preset_map:
            raise CircuitBreakerConfigurationError(
                f"No preset available for service type: {service_type}",
                config_key="service_type",
                config_value=service_type,
            )

        return preset_map[service_type]()


@dataclass
class DomainConfig:
    """Domain-specific circuit breaker configuration."""

    domain: str
    config: CircuitBreakerConfig
    created_at: float = field(default_factory=lambda: __import__("time").time())
    last_updated: float = field(default_factory=lambda: __import__("time").time())

    def update_config(self, config: CircuitBreakerConfig) -> None:
        """Update configuration for this domain.

        Args:
            config: New configuration
        """
        self.config = config
        self.last_updated = __import__("time").time()


class ConfigurationManager:
    """Manages circuit breaker configurations across domains."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._domain_configs: dict[str, DomainConfig] = {}
        self._default_config = CircuitBreakerConfig()

    def set_default_config(self, config: CircuitBreakerConfig) -> None:
        """Set the default configuration.

        Args:
            config: Default configuration
        """
        self._default_config = config

    def get_default_config(self) -> CircuitBreakerConfig:
        """Get the default configuration.

        Returns:
            Default configuration
        """
        return self._default_config

    def set_domain_config(self, domain: str, config: CircuitBreakerConfig) -> None:
        """Set configuration for a specific domain.

        Args:
            domain: Domain name
            config: Configuration for the domain
        """
        if domain in self._domain_configs:
            self._domain_configs[domain].update_config(config)
        else:
            self._domain_configs[domain] = DomainConfig(domain=domain, config=config)

    def get_domain_config(self, domain: str) -> CircuitBreakerConfig:
        """Get configuration for a domain.

        Args:
            domain: Domain name

        Returns:
            Configuration for the domain or default if not found
        """
        if domain in self._domain_configs:
            return self._domain_configs[domain].config
        return self._default_config

    def remove_domain_config(self, domain: str) -> bool:
        """Remove configuration for a domain.

        Args:
            domain: Domain name

        Returns:
            True if domain was removed, False if not found
        """
        if domain in self._domain_configs:
            del self._domain_configs[domain]
            return True
        return False

    def list_domains(self) -> list[str]:
        """List all configured domains.

        Returns:
            List of domain names
        """
        return list(self._domain_configs.keys())

    def get_all_configs(self) -> dict[str, CircuitBreakerConfig]:
        """Get all domain configurations.

        Returns:
            Dictionary mapping domains to their configurations
        """
        return {domain: domain_config.config for domain, domain_config in self._domain_configs.items()}
