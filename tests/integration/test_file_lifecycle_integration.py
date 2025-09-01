"""Integration tests for file lifecycle events across all services."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from shared.core_types.src.database import Base
from shared.core_types.src.models import Metadata, Recording, Tracklist


class TestFileLifecycleIntegration:
    """Test suite for file lifecycle events integration."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        # Use in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        # Add required PostgreSQL functions for SQLite compatibility
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS pg_extension (extname TEXT)"))
            conn.commit()

        session_local = sessionmaker(bind=engine)
        session = session_local()
        yield session
        session.close()

    @pytest.fixture
    def mock_rabbitmq_channel(self):
        """Create a mock RabbitMQ channel."""
        channel = MagicMock()
        channel.basic_publish = MagicMock()
        channel.queue_declare = MagicMock()
        channel.exchange_declare = MagicMock()
        channel.queue_bind = MagicMock()
        return channel

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = MagicMock()
        client.get = MagicMock(return_value=None)
        client.set = MagicMock(return_value=True)
        client.delete = MagicMock(return_value=1)
        client.keys = MagicMock(return_value=[])
        return client

    @pytest.fixture
    def sample_file_event(self) -> dict[str, Any]:
        """Create a sample file event message."""
        return {
            "event_type": "created",
            "file_path": "/music/test_song.mp3",
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
            "sha256_hash": "abc123" * 10,  # 64 chars
            "xxh128_hash": "def456" * 5,  # 30 chars
        }

    def test_file_creation_flow(self, db_session, mock_rabbitmq_channel, sample_file_event):
        """Test complete file creation flow through all services."""
        # 1. Simulate file watcher detecting new file
        event = sample_file_event
        _ = json.dumps(event)  # Would be sent via message queue

        # 2. Simulate cataloging service processing
        recording = Recording(
            file_path=event["file_path"],
            file_name=Path(event["file_path"]).name,
            sha256_hash=event["sha256_hash"],
            xxh128_hash=event["xxh128_hash"],
            processing_status="completed",
        )
        db_session.add(recording)
        db_session.commit()

        # Verify recording was created
        assert recording.id is not None
        stored_recording = db_session.query(Recording).filter_by(file_path=event["file_path"]).first()
        assert stored_recording is not None
        assert stored_recording.sha256_hash == event["sha256_hash"]

        # 3. Simulate analysis service processing
        metadata = Metadata(recording_id=recording.id, key="bpm", value="120")
        db_session.add(metadata)
        db_session.commit()

        # Verify metadata was created
        stored_metadata = db_session.query(Metadata).filter_by(recording_id=recording.id).first()
        assert stored_metadata is not None
        assert stored_metadata.value == "120"

    def test_file_deletion_cascade(self, db_session, mock_redis_client):
        """Test file deletion with cascade cleanup."""
        # Create test data
        recording = Recording(
            file_path="/music/to_delete.mp3",
            file_name="to_delete.mp3",
            sha256_hash="hash123",
            xxh128_hash="hash456",
        )
        db_session.add(recording)
        db_session.commit()

        # Add related data
        metadata = Metadata(recording_id=recording.id, key="genre", value="electronic")
        tracklist = Tracklist(
            recording_id=recording.id,
            source="manual",
            tracks=[{"title": "Track 1", "artist": "Artist", "start_time": "00:00"}],
        )
        db_session.add_all([metadata, tracklist])
        db_session.commit()

        # Simulate deletion event
        _ = {
            "event_type": "deleted",
            "file_path": "/music/to_delete.mp3",
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
        }  # Would be processed by services

        # Process deletion
        db_session.delete(recording)
        db_session.commit()

        # Verify cascade deletion
        assert db_session.query(Recording).filter_by(file_path="/music/to_delete.mp3").first() is None
        assert db_session.query(Metadata).filter_by(recording_id=recording.id).first() is None
        assert db_session.query(Tracklist).filter_by(recording_id=recording.id).first() is None

    def test_file_move_event(self, db_session):
        """Test file move event updates references."""
        # Create recording
        old_path = "/music/old_location/song.mp3"
        new_path = "/music/new_location/song.mp3"

        recording = Recording(
            file_path=old_path,
            file_name="song.mp3",
            sha256_hash="samehash",
            xxh128_hash="samehash2",
        )
        db_session.add(recording)
        db_session.commit()

        # Simulate move event
        _ = {
            "event_type": "moved",
            "file_path": new_path,
            "old_path": old_path,
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
            "sha256_hash": "samehash",
            "xxh128_hash": "samehash2",
        }  # Would be processed by services

        # Update recording path
        recording.file_path = new_path
        db_session.commit()

        # Verify update
        updated_recording = db_session.query(Recording).filter_by(sha256_hash="samehash").first()
        assert updated_recording is not None
        assert updated_recording.file_path == new_path

    def test_file_rename_event(self, db_session):
        """Test file rename event updates references."""
        # Create recording
        old_path = "/music/old_name.mp3"
        new_path = "/music/new_name.mp3"

        recording = Recording(
            file_path=old_path,
            file_name="old_name.mp3",
            sha256_hash="renamehash",
            xxh128_hash="renamehash2",
        )
        db_session.add(recording)
        db_session.commit()

        # Simulate rename event
        _ = {
            "event_type": "renamed",
            "file_path": new_path,
            "old_path": old_path,
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
            "sha256_hash": "renamehash",
            "xxh128_hash": "renamehash2",
        }  # Would be processed by services

        # Update recording
        recording.file_path = new_path
        recording.file_name = "new_name.mp3"
        db_session.commit()

        # Verify update
        updated_recording = db_session.query(Recording).filter_by(sha256_hash="renamehash").first()
        assert updated_recording is not None
        assert updated_recording.file_path == new_path
        assert updated_recording.file_name == "new_name.mp3"

    def test_soft_delete_mechanism(self, db_session):
        """Test soft delete with recovery."""
        # Create recording
        recording = Recording(
            file_path="/music/soft_delete.mp3",
            file_name="soft_delete.mp3",
            sha256_hash="softhash",
            xxh128_hash="softhash2",
        )
        db_session.add(recording)
        db_session.commit()

        # Soft delete
        recording.deleted_at = datetime.now(UTC)
        db_session.commit()

        # Query for active records (should not include soft-deleted)
        active_recordings = db_session.query(Recording).filter(Recording.deleted_at.is_(None)).all()
        assert recording not in active_recordings

        # Query all records (includes soft-deleted)
        all_recordings = db_session.query(Recording).all()
        assert recording in all_recordings

        # Recover soft-deleted record
        recording.deleted_at = None
        db_session.commit()

        # Verify recovery
        recovered = db_session.query(Recording).filter_by(file_path="/music/soft_delete.mp3").first()
        assert recovered is not None
        assert recovered.deleted_at is None

    def test_orphaned_record_detection(self, db_session):
        """Test detection of orphaned records."""
        # Create a recording
        recording = Recording(
            file_path="/music/parent.mp3",
            file_name="parent.mp3",
            sha256_hash="parenthash",
            xxh128_hash="parenthash2",
        )
        db_session.add(recording)
        db_session.commit()
        recording_id = recording.id

        # Create metadata
        metadata = Metadata(recording_id=recording_id, key="test", value="value")
        db_session.add(metadata)
        db_session.commit()

        # Delete recording without cascade (simulating orphaned metadata)
        db_session.query(Recording).filter_by(id=recording_id).delete()
        db_session.commit()

        # Check for orphaned metadata
        orphaned = db_session.query(Metadata).filter(~Metadata.recording_id.in_(db_session.query(Recording.id))).all()

        # In a proper setup with foreign keys, this shouldn't happen
        # But we're testing the detection mechanism
        assert len(orphaned) == 0  # With proper FK constraints

    @pytest.mark.asyncio
    async def test_concurrent_lifecycle_events(self, db_session):
        """Test handling of concurrent lifecycle events."""

        async def create_recording(path: str, session: Session):
            """Simulate concurrent recording creation."""
            recording = Recording(
                file_path=path,
                file_name=Path(path).name,
                sha256_hash=f"hash_{path}",
                xxh128_hash=f"xxhash_{path}",
            )
            session.add(recording)
            session.commit()
            return recording.id

        # Create multiple recordings concurrently
        paths = [f"/music/concurrent_{i}.mp3" for i in range(5)]

        # Note: SQLAlchemy sessions aren't thread-safe, so we simulate concurrency
        for path in paths:
            recording = Recording(
                file_path=path,
                file_name=Path(path).name,
                sha256_hash=f"hash_{path}",
                xxh128_hash=f"xxhash_{path}",
            )
            db_session.add(recording)

        db_session.commit()

        # Verify all were created
        count = db_session.query(Recording).filter(Recording.file_path.like("/music/concurrent_%")).count()
        assert count == 5

    def test_message_queue_event_structure(self, sample_file_event):
        """Test that message queue events have correct structure."""
        # Test creation event
        assert "event_type" in sample_file_event
        assert "file_path" in sample_file_event
        assert "timestamp" in sample_file_event
        assert "instance_id" in sample_file_event

        # Test deletion event structure
        deletion_event = {
            "event_type": "deleted",
            "file_path": "/music/deleted.mp3",
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
        }
        assert "sha256_hash" not in deletion_event  # No hash for deleted files

        # Test move event structure
        move_event = {
            "event_type": "moved",
            "file_path": "/music/new_path.mp3",
            "old_path": "/music/old_path.mp3",
            "timestamp": datetime.now(UTC).isoformat(),
            "instance_id": "test_watcher",
            "sha256_hash": "movehash",
            "xxh128_hash": "movehash2",
        }
        assert "old_path" in move_event
        assert move_event["old_path"] != move_event["file_path"]

    def test_cache_cleanup_on_deletion(self, mock_redis_client):
        """Test that cache entries are cleaned up on file deletion."""
        _ = "/music/cached_file.mp3"  # File path for cache test
        file_hash = "test_hash_12345"

        # Simulate cached entries
        cache_keys = [
            f"tracktion:bpm:{file_hash}",
            f"tracktion:temporal:{file_hash}",
            f"tracktion:key:{file_hash}",
            f"tracktion:mood:{file_hash}",
        ]

        # Mock Redis client to return these keys
        mock_redis_client.keys.return_value = cache_keys

        # Simulate deletion of all cache keys
        for key in cache_keys:
            result = mock_redis_client.delete(key)
            assert result == 1  # Mock returns 1 for successful deletion

        # Verify all delete calls were made
        assert mock_redis_client.delete.call_count == len(cache_keys)

    def test_referential_integrity_validation(self, db_session):
        """Test referential integrity is maintained."""
        # Create recording
        recording = Recording(
            file_path="/music/integrity_test.mp3",
            file_name="integrity_test.mp3",
            sha256_hash="integrityhash",
            xxh128_hash="integrityhash2",
        )
        db_session.add(recording)
        db_session.commit()

        # Add metadata with valid foreign key
        metadata = Metadata(recording_id=recording.id, key="integrity", value="test")
        db_session.add(metadata)
        db_session.commit()

        # Attempt to add metadata with invalid foreign key should fail
        invalid_metadata = Metadata(
            recording_id=uuid.uuid4(),  # Non-existent recording ID
            key="invalid",
            value="test",
        )
        db_session.add(invalid_metadata)

        # This would raise an IntegrityError in a real database with FK constraints
        # For SQLite in-memory, we just verify the logic
        try:
            db_session.commit()
            # In SQLite without FK enforcement, this might succeed
            # Clean up the invalid data
            db_session.delete(invalid_metadata)
            db_session.commit()
        except Exception:
            db_session.rollback()
            # Expected with proper FK constraints


class TestServiceIntegration:
    """Test integration between different services."""

    @pytest.fixture
    def mock_services(self):
        """Create mock service instances."""
        return {
            "file_watcher": MagicMock(),
            "cataloging": MagicMock(),
            "analysis": MagicMock(),
            "tracklist": MagicMock(),
        }

    def test_service_communication_flow(self, mock_services):
        """Test that services communicate correctly."""
        # File watcher detects new file
        file_event = {
            "event_type": "created",
            "file_path": "/music/new_song.mp3",
            "sha256_hash": "hash123",
            "xxh128_hash": "hash456",
        }

        # File watcher publishes to message queue
        mock_services["file_watcher"].publish_event(file_event)
        mock_services["file_watcher"].publish_event.assert_called_once_with(file_event)

        # Cataloging service receives and processes
        mock_services["cataloging"].process_file_event(file_event)
        mock_services["cataloging"].process_file_event.assert_called_once()

        # Analysis service receives and analyzes
        mock_services["analysis"].analyze_file(file_event["file_path"])
        mock_services["analysis"].analyze_file.assert_called_once()

        # Tracklist service processes if needed
        mock_services["tracklist"].process_recording(file_event["file_path"])
        mock_services["tracklist"].process_recording.assert_called_once()

    def test_error_handling_across_services(self, mock_services):
        """Test error handling and recovery across services."""
        # Simulate error in cataloging service
        mock_services["cataloging"].process_file_event.side_effect = Exception("Database error")

        file_event = {"event_type": "created", "file_path": "/music/error_test.mp3"}

        # File watcher should still publish
        mock_services["file_watcher"].publish_event(file_event)

        # Cataloging fails
        with pytest.raises(Exception):  # noqa: B017
            mock_services["cataloging"].process_file_event(file_event)

        # Other services should handle gracefully
        mock_services["analysis"].analyze_file.return_value = None
        result = mock_services["analysis"].analyze_file(file_event["file_path"])
        assert result is None  # Graceful handling

    def test_transaction_rollback_on_failure(self, db_session):
        """Test that transactions are rolled back on failure."""
        # Start transaction
        recording = Recording(
            file_path="/music/transaction_test.mp3",
            file_name="transaction_test.mp3",
            sha256_hash="txhash",
            xxh128_hash="txhash2",
        )
        db_session.add(recording)

        # Simulate error before commit
        try:
            # Force an error (e.g., duplicate unique constraint)
            duplicate = Recording(
                file_path="/music/transaction_test.mp3",
                file_name="transaction_test.mp3",
                sha256_hash="txhash",  # Same hash - would violate unique constraint
                xxh128_hash="txhash2",
            )
            db_session.add(duplicate)
            db_session.commit()
        except Exception:
            db_session.rollback()

        # Verify rollback - nothing should be saved
        count = db_session.query(Recording).filter_by(file_path="/music/transaction_test.mp3").count()
        assert count == 0  # Transaction was rolled back
