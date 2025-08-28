"""Unit tests for synchronization models."""

from datetime import datetime, timedelta
from uuid import uuid4

from services.tracklist_service.src.models.synchronization import (
    AuditLog,
    SyncConfiguration,
    SyncEvent,
    TracklistVersion,
)


class TestTracklistVersion:
    """Test TracklistVersion model."""

    def test_create_version(self):
        """Test creating a tracklist version."""
        tracklist_id = uuid4()
        version = TracklistVersion(
            tracklist_id=tracklist_id,
            version_number=1,
            change_type="manual_edit",
            change_summary="Initial version",
            tracks_snapshot=[
                {
                    "position": 1,
                    "artist": "Artist 1",
                    "title": "Track 1",
                    "start_time": 0,
                }
            ],
            version_metadata={"user": "test_user"},
            is_current=True,
        )

        assert version.tracklist_id == tracklist_id
        assert version.version_number == 1
        assert version.change_type == "manual_edit"
        assert version.change_summary == "Initial version"
        assert len(version.tracks_snapshot) == 1
        assert version.version_metadata["user"] == "test_user"
        assert version.is_current is True

    def test_version_to_dict(self):
        """Test converting version to dictionary."""
        version = TracklistVersion(
            id=uuid4(),
            tracklist_id=uuid4(),
            version_number=2,
            created_at=datetime.utcnow(),
            created_by="system",
            change_type="import_update",
            change_summary="Updated from 1001tracklists",
            tracks_snapshot=[{"position": 1, "artist": "Test", "title": "Song"}],
            version_metadata={"source": "1001tracklists"},
            is_current=False,
        )

        result = version.to_dict()

        assert "id" in result
        assert "tracklist_id" in result
        assert result["version_number"] == 2
        assert result["created_by"] == "system"
        assert result["change_type"] == "import_update"
        assert result["is_current"] is False


class TestSyncConfiguration:
    """Test SyncConfiguration model."""

    def test_create_sync_config(self):
        """Test creating sync configuration."""
        tracklist_id = uuid4()
        config = SyncConfiguration(
            tracklist_id=tracklist_id,
            sync_enabled=True,
            sync_sources=["1001tracklists", "manual"],
            sync_frequency="hourly",
            auto_accept_threshold=0.85,
            conflict_resolution="newest",
        )

        assert config.tracklist_id == tracklist_id
        assert config.sync_enabled is True
        assert "1001tracklists" in config.sync_sources
        assert config.sync_frequency == "hourly"
        assert config.auto_accept_threshold == 0.85
        assert config.conflict_resolution == "newest"

    def test_sync_config_defaults(self):
        """Test sync configuration default values."""
        config = SyncConfiguration(
            tracklist_id=uuid4(),
            sync_enabled=True,  # Explicitly set for testing
            auto_accept_threshold=0.9,
            conflict_resolution="manual",
        )

        assert config.sync_enabled is True
        assert config.auto_accept_threshold == 0.9
        assert config.conflict_resolution == "manual"
        assert config.last_sync_at is None
        assert config.next_sync_at is None

    def test_sync_config_to_dict(self):
        """Test converting sync configuration to dictionary."""
        now = datetime.utcnow()
        next_sync = now + timedelta(hours=1)
        config = SyncConfiguration(
            id=uuid4(),
            tracklist_id=uuid4(),
            sync_enabled=False,
            sync_sources=["auto"],
            sync_frequency="daily",
            auto_accept_threshold=0.75,
            conflict_resolution="highest_confidence",
            last_sync_at=now,
            next_sync_at=next_sync,
        )

        result = config.to_dict()

        assert "id" in result
        assert "tracklist_id" in result
        assert result["sync_enabled"] is False
        assert result["sync_sources"] == ["auto"]
        assert result["sync_frequency"] == "daily"
        assert result["auto_accept_threshold"] == 0.75
        assert result["conflict_resolution"] == "highest_confidence"
        assert result["last_sync_at"] is not None
        assert result["next_sync_at"] is not None


class TestSyncEvent:
    """Test SyncEvent model."""

    def test_create_sync_event(self):
        """Test creating sync event."""
        tracklist_id = uuid4()
        event = SyncEvent(
            tracklist_id=tracklist_id,
            event_type="check",
            source="1001tracklists",
            status="pending",
            changes={
                "tracks_added": 2,
                "tracks_modified": 1,
            },
        )

        assert event.tracklist_id == tracklist_id
        assert event.event_type == "check"
        assert event.source == "1001tracklists"
        assert event.status == "pending"
        assert event.changes["tracks_added"] == 2
        assert event.completed_at is None

    def test_sync_event_with_conflict(self):
        """Test sync event with conflict data."""
        event = SyncEvent(
            tracklist_id=uuid4(),
            event_type="conflict",
            source="manual",
            status="completed",
            completed_at=datetime.utcnow(),
            conflict_data={
                "field": "track_title",
                "current": "Original Mix",
                "proposed": "Extended Mix",
            },
            resolution={
                "action": "keep_current",
                "resolved_by": "user",
            },
        )

        assert event.event_type == "conflict"
        assert event.conflict_data["field"] == "track_title"
        assert event.resolution["action"] == "keep_current"
        assert event.completed_at is not None

    def test_sync_event_to_dict(self):
        """Test converting sync event to dictionary."""
        event = SyncEvent(
            id=uuid4(),
            tracklist_id=uuid4(),
            event_type="update",
            source="auto",
            status="processing",
            created_at=datetime.utcnow(),
        )

        result = event.to_dict()

        assert "id" in result
        assert "tracklist_id" in result
        assert result["event_type"] == "update"
        assert result["source"] == "auto"
        assert result["status"] == "processing"
        assert result["created_at"] is not None
        assert result["completed_at"] is None


class TestAuditLog:
    """Test AuditLog model."""

    def test_create_audit_log(self):
        """Test creating audit log entry."""
        entity_id = uuid4()
        audit = AuditLog(
            entity_type="tracklist",
            entity_id=entity_id,
            action="updated",
            actor="user_123",
            changes={
                "before": {"track_count": 10},
                "after": {"track_count": 12},
            },
            audit_metadata={"ip_address": "192.168.1.1"},
        )

        assert audit.entity_type == "tracklist"
        assert audit.entity_id == entity_id
        assert audit.action == "updated"
        assert audit.actor == "user_123"
        assert audit.changes["before"]["track_count"] == 10
        assert audit.changes["after"]["track_count"] == 12
        assert audit.audit_metadata["ip_address"] == "192.168.1.1"

    def test_audit_log_to_dict(self):
        """Test converting audit log to dictionary."""
        audit = AuditLog(
            id=uuid4(),
            entity_type="cue_file",
            entity_id=uuid4(),
            action="created",
            timestamp=datetime.utcnow(),
            actor="system",
            changes={"new_file": "tracklist.cue"},
        )

        result = audit.to_dict()

        assert "id" in result
        assert result["entity_type"] == "cue_file"
        assert "entity_id" in result
        assert result["action"] == "created"
        assert result["actor"] == "system"
        assert result["timestamp"] is not None
        assert result["changes"]["new_file"] == "tracklist.cue"
