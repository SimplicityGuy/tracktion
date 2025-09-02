"""Configuration for the cataloging service."""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "tracktion"
    user: str = "tracktion"
    password: str = "tracktion"

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RabbitMQConfig:
    """RabbitMQ configuration."""

    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    exchange: str = "file_events"
    queue: str = "cataloging.file.events"

    @property
    def url(self) -> str:
        """Get RabbitMQ URL."""
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"


@dataclass
class ServiceConfig:
    """Service configuration."""

    soft_delete_enabled: bool = True
    cleanup_interval_days: int = 30
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    """Main configuration class."""

    database: DatabaseConfig
    rabbitmq: RabbitMQConfig
    service: ServiceConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            database=DatabaseConfig(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "tracktion"),
                user=os.getenv("DB_USER", "tracktion"),
                password=os.getenv("DB_PASSWORD", "tracktion"),
            ),
            rabbitmq=RabbitMQConfig(
                host=os.getenv("RABBITMQ_HOST", "localhost"),
                port=int(os.getenv("RABBITMQ_PORT", "5672")),
                username=os.getenv("RABBITMQ_USERNAME", "guest"),
                password=os.getenv("RABBITMQ_PASSWORD", "guest"),
                exchange=os.getenv("RABBITMQ_EXCHANGE", "file_events"),
                queue=os.getenv("RABBITMQ_QUEUE", "cataloging.file.events"),
            ),
            service=ServiceConfig(
                soft_delete_enabled=os.getenv("SOFT_DELETE_ENABLED", "true").lower() == "true",
                cleanup_interval_days=int(os.getenv("CLEANUP_INTERVAL_DAYS", "30")),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                host=os.getenv("SERVICE_HOST", "0.0.0.0"),
                port=int(os.getenv("SERVICE_PORT", "8000")),
            ),
        )


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config  # noqa: PLW0603 - Necessary for singleton pattern
    if _config is None:
        _config = Config.from_env()
    return _config
