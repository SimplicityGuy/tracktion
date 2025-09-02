"""Configuration for the cataloging service."""

import os
from dataclasses import dataclass
from typing import Any


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


class ConfigSingleton:
    """Singleton wrapper for Config."""

    _instance: Config | None = None
    _singleton_instance: "ConfigSingleton | None" = None

    def __new__(cls) -> "ConfigSingleton":
        """Get the singleton Config instance."""
        if cls._singleton_instance is None:
            cls._singleton_instance = super().__new__(cls)
            cls._instance = Config.from_env()
        return cls._singleton_instance

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped config instance."""
        if self._instance is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._instance, name)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._singleton_instance = None


def get_config() -> "ConfigSingleton":
    """Get the singleton configuration instance."""
    return ConfigSingleton()
