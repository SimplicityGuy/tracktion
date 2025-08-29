"""Async configuration for analysis service."""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = "postgresql+asyncpg://user:password@localhost/tracktion"


@dataclass
class Neo4jConfig:
    """Neo4j configuration."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"


@dataclass
class RedisConfig:
    """Redis configuration."""

    url: str = "redis://localhost:6379/0"


@dataclass
class RabbitMQConfig:
    """RabbitMQ configuration."""

    url: str = "amqp://guest:guest@localhost:5672/"


@dataclass
class AsyncServiceConfig:
    """Async service configuration."""

    database: DatabaseConfig
    neo4j: Neo4jConfig
    redis: RedisConfig
    rabbitmq: RabbitMQConfig


def get_config() -> AsyncServiceConfig:
    """Get configuration from environment variables."""
    return AsyncServiceConfig(
        database=DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/tracktion")
        ),
        neo4j=Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        ),
        redis=RedisConfig(url=os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        rabbitmq=RabbitMQConfig(url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")),
    )
