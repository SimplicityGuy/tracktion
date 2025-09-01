"""Pytest configuration and fixtures."""

import os

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import models to ensure they're registered
from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import Base

# Load environment variables
load_dotenv()


@pytest.fixture(scope="function")
def test_db_session():
    """Create a test database session for PostgreSQL.

    Creates a clean database for each test and tears it down after.
    """
    # Use test database URL
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://tracktion_user:changeme@localhost:5432/test_tracktion",
    )

    # Create engine and tables
    engine = create_engine(test_db_url)

    # Create UUID extension
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session
    session_local = sessionmaker(bind=engine)
    session = session_local()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def neo4j_test_session():
    """Create a Neo4j test session with cleanup.

    Clears the database before and after each test.
    """
    neo4j_uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "changeme")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Clean database before test
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    yield driver

    # Clean database after test
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    driver.close()


@pytest.fixture(scope="function")
def db_manager(monkeypatch):
    """Create a test DatabaseManager instance.

    Uses test database URLs and mocks environment variables.
    """
    # Mock environment variables for testing
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://tracktion_user:changeme@localhost:5432/test_tracktion",
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "changeme")

    # Create and return manager
    manager = DatabaseManager()

    yield manager

    # Cleanup
    manager.close()


@pytest.fixture
def sample_recording_data():
    """Provide sample recording data for tests."""
    return {
        "file_path": "/music/test_set.mp3",
        "file_name": "test_set.mp3",
        "sha256_hash": "abc123def456",
        "xxh128_hash": "xyz789",
    }


@pytest.fixture
def sample_metadata_items():
    """Provide sample metadata items for tests."""
    return [
        {"key": "bpm", "value": "128"},
        {"key": "key", "value": "A minor"},
        {"key": "mood", "value": "energetic"},
        {"key": "genre", "value": "techno"},
    ]


@pytest.fixture
def sample_tracks():
    """Provide sample track data for tests."""
    return [
        {"title": "Opening Track", "artist": "DJ One", "start_time": "00:00:00"},
        {"title": "Peak Time", "artist": "DJ Two", "start_time": "00:05:30"},
        {"title": "Breakdown", "artist": "DJ Three", "start_time": "00:11:45"},
        {"title": "Build Up", "artist": "DJ Four", "start_time": "00:18:20"},
        {"title": "Closing", "artist": "DJ Five", "start_time": "00:25:00"},
    ]
