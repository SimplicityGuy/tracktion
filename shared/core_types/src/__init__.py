"""Core types package for Tracktion - shared data models and database connections."""

__version__ = "0.1.0"

# Sync database operations
from .database import Base, DatabaseManager, db_manager, get_db_session, get_neo4j_driver

# Async database operations
from .async_database import (
    AsyncDatabaseManager,
    async_db_manager,
    get_async_db_session,
    get_async_neo4j_driver,
    get_async_redis_client,
)

# Models
from .models import Recording, Metadata, Tracklist, RenameProposal

__all__ = [
    # Sync
    "Base",
    "DatabaseManager",
    "db_manager",
    "get_db_session",
    "get_neo4j_driver",
    # Async
    "AsyncDatabaseManager",
    "async_db_manager",
    "get_async_db_session",
    "get_async_neo4j_driver",
    "get_async_redis_client",
    # Models
    "Recording",
    "Metadata",
    "Tracklist",
    "RenameProposal",
]
