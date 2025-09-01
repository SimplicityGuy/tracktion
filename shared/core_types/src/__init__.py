"""Core types package for Tracktion - shared data models and database connections."""

__version__ = "0.1.0"

# Sync database operations
# Async database operations
from .async_database import (
    AsyncDatabaseManager,
    async_db_manager,
    get_async_db_session,
    get_async_neo4j_driver,
    get_async_redis_client,
)
from .database import (
    Base,
    DatabaseManager,
    db_manager,
    get_db_session,
    get_neo4j_driver,
)

# Models
from .models import Metadata, Recording, RenameProposal, Tracklist

__all__ = [
    # Async
    "AsyncDatabaseManager",
    # Sync
    "Base",
    # Sync
    "DatabaseManager",
    # Models
    "Metadata",
    # Models
    "Recording",
    # Models
    "RenameProposal",
    # Models
    "Tracklist",
    # Async
    "async_db_manager",
    # Sync
    "db_manager",
    # Async
    "get_async_db_session",
    # Async
    "get_async_neo4j_driver",
    # Async
    "get_async_redis_client",
    # Sync
    "get_db_session",
    # Sync
    "get_neo4j_driver",
]
