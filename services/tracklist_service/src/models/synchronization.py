"""Synchronization models for tracklist version management and sync operations."""

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import relationship

from services.tracklist_service.src.models.tracklist import Base


class TracklistVersion(Base):
    """Model for tracking tracklist version history."""

    __tablename__ = "tracklist_versions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tracklist_id = Column(PG_UUID(as_uuid=True), ForeignKey("tracklists.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255))  # User or system identifier
    change_type = Column(String(50), nullable=False)  # manual_edit, import_update, auto_sync
    change_summary = Column(Text, nullable=False)
    tracks_snapshot = Column(JSONB, nullable=False)  # Complete tracks data at this version
    version_metadata = Column(JSONB, default=dict)
    is_current = Column(Boolean, default=False, nullable=False)

    # Relationships
    tracklist = relationship("TracklistDB", back_populates="versions")

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary."""
        return {
            "id": str(self.id),
            "tracklist_id": str(self.tracklist_id),
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "change_type": self.change_type,
            "change_summary": self.change_summary,
            "tracks_snapshot": self.tracks_snapshot,
            "metadata": self.version_metadata,
            "is_current": self.is_current,
        }


class SyncConfiguration(Base):
    """Model for tracklist synchronization configuration."""

    __tablename__ = "sync_configurations"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tracklist_id = Column(PG_UUID(as_uuid=True), ForeignKey("tracklists.id"), unique=True, nullable=False)
    sync_enabled = Column(Boolean, default=True, nullable=False)
    sync_sources = Column(JSONB, default=list)  # ["1001tracklists", "manual", "auto"]
    sync_frequency = Column(String(20))  # realtime, hourly, daily, manual
    auto_accept_threshold = Column(Float, default=0.9, nullable=False)
    conflict_resolution = Column(String(20), default="manual", nullable=False)  # manual, newest, highest_confidence
    last_sync_at = Column(DateTime)
    next_sync_at = Column(DateTime)

    # Relationships
    tracklist = relationship("TracklistDB", back_populates="sync_configuration")

    def to_dict(self) -> Dict[str, Any]:
        """Convert sync configuration to dictionary."""
        return {
            "id": str(self.id),
            "tracklist_id": str(self.tracklist_id),
            "sync_enabled": self.sync_enabled,
            "sync_sources": self.sync_sources,
            "sync_frequency": self.sync_frequency,
            "auto_accept_threshold": self.auto_accept_threshold,
            "conflict_resolution": self.conflict_resolution,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "next_sync_at": self.next_sync_at.isoformat() if self.next_sync_at else None,
        }


class SyncEvent(Base):
    """Model for tracking synchronization events."""

    __tablename__ = "sync_events"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tracklist_id = Column(PG_UUID(as_uuid=True), ForeignKey("tracklists.id"), nullable=False)
    event_type = Column(String(20), nullable=False)  # check, update, conflict, resolved
    source = Column(String(50), nullable=False)  # 1001tracklists, manual, auto
    status = Column(String(20), nullable=False)  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    changes = Column(JSONB)  # Detailed change information
    conflict_data = Column(JSONB)  # Conflict details if any
    resolution = Column(JSONB)  # How conflict was resolved

    # Relationships
    tracklist = relationship("TracklistDB", back_populates="sync_events")

    def to_dict(self) -> Dict[str, Any]:
        """Convert sync event to dictionary."""
        return {
            "id": str(self.id),
            "tracklist_id": str(self.tracklist_id),
            "event_type": self.event_type,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "changes": self.changes,
            "conflict_data": self.conflict_data,
            "resolution": self.resolution,
        }


class AuditLog(Base):
    """Model for audit trail of all changes."""

    __tablename__ = "audit_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type = Column(String(50), nullable=False)  # tracklist, cue_file
    entity_id = Column(PG_UUID(as_uuid=True), nullable=False)
    action = Column(String(50), nullable=False)  # created, updated, deleted, synced
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    actor = Column(String(255), nullable=False)  # User or system identifier
    changes = Column(JSONB, nullable=False)  # Detailed change data
    audit_metadata = Column(JSONB, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        return {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "action": self.action,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actor": self.actor,
            "changes": self.changes,
            "metadata": self.audit_metadata,
        }
