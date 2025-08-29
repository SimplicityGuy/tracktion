"""Configuration management for File Rename Service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = "File Rename Service"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development, staging, production)")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # Database settings
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/file_rename_service",
        description="PostgreSQL connection URL",
    )
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Maximum overflow connections")
    database_echo: bool = Field(default=False, description="Echo SQL queries")

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_pool_size: int = Field(default=10, description="Redis connection pool size")

    # RabbitMQ settings
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="RabbitMQ connection URL",
    )
    rabbitmq_exchange: str = Field(default="file_rename", description="RabbitMQ exchange name")
    rabbitmq_prefetch_count: int = Field(default=10, description="RabbitMQ prefetch count")

    # CORS settings
    cors_allow_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: list[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_allow_headers: list[str] = Field(default=["*"], description="Allowed CORS headers")

    # Security settings
    secret_key: str = Field(
        default="development-secret-key-change-in-production",
        description="Secret key for security",
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: list[str] = Field(default=[], description="List of valid API keys")

    # ML Model settings
    model_cache_dir: str = Field(default="/tmp/models", description="Directory to cache ML models")
    model_max_size_mb: int = Field(default=100, description="Maximum model size in MB")

    # Service discovery settings
    consul_url: str | None = Field(default=None, description="Consul URL for service registry")
    service_name: str = Field(default="file-rename-service", description="Service name for registration")
    service_id: str = Field(default="file-rename-service-1", description="Service instance ID")
    service_tags: list[str] = Field(default=["rename", "ml", "pattern"], description="Service tags")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Create a single instance for import
settings = get_settings()
