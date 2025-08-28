"""Message schemas for synchronization events."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SyncEventType(str, Enum):
    """Types of synchronization events."""
    
    # Sync lifecycle events
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"
    SYNC_STATUS_UPDATE = "sync_status_update"
    
    # Conflict events
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    
    # Version events
    VERSION_CREATED = "version_created"
    VERSION_ROLLBACK = "version_rollback"
    VERSION_PRUNED = "version_pruned"
    
    # CUE regeneration events
    CUE_REGENERATION_TRIGGERED = "cue_regeneration_triggered"
    CUE_REGENERATION_COMPLETED = "cue_regeneration_completed"
    CUE_REGENERATION_FAILED = "cue_regeneration_failed"
    
    # Batch operations
    BATCH_SYNC_STARTED = "batch_sync_started"
    BATCH_SYNC_COMPLETED = "batch_sync_completed"
    BATCH_SYNC_FAILED = "batch_sync_failed"
    
    # Configuration events
    SYNC_CONFIG_UPDATED = "sync_config_updated"
    SYNC_SCHEDULED = "sync_scheduled"
    SYNC_CANCELLED = "sync_cancelled"


class BaseSyncMessage(BaseModel):
    """Base message schema for synchronization events."""
    
    message_id: UUID = Field(description="Unique message identifier")
    event_type: SyncEventType = Field(description="Type of sync event")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[UUID] = Field(None, description="Correlation ID for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseSyncMessage":
        """Create message from JSON string."""
        return cls.model_validate_json(json_str)


class SyncEventMessage(BaseSyncMessage):
    """Generic sync event message."""
    
    tracklist_id: UUID = Field(description="ID of the tracklist")
    source: str = Field(description="Sync source (1001tracklists, manual, etc)")
    actor: str = Field(default="system", description="Who triggered the event")


class SyncCompletedMessage(SyncEventMessage):
    """Message for sync completion events."""
    
    event_type: SyncEventType = Field(SyncEventType.SYNC_COMPLETED, frozen=True)
    changes_applied: int = Field(description="Number of changes applied")
    confidence: float = Field(description="Confidence score of changes")
    duration_seconds: float = Field(description="Duration of sync operation")


class SyncFailedMessage(SyncEventMessage):
    """Message for sync failure events."""
    
    event_type: SyncEventType = Field(SyncEventType.SYNC_FAILED, frozen=True)
    error_message: str = Field(description="Error message")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    will_retry: bool = Field(default=False, description="Whether retry is scheduled")


class ConflictDetectedMessage(BaseSyncMessage):
    """Message for conflict detection events."""
    
    event_type: SyncEventType = Field(SyncEventType.CONFLICT_DETECTED, frozen=True)
    tracklist_id: UUID = Field(description="ID of the tracklist")
    source: str = Field(description="Source of the conflict")
    conflicts: List[Dict[str, Any]] = Field(description="List of detected conflicts")
    conflict_count: int = Field(description="Number of conflicts")
    auto_resolvable: bool = Field(description="Whether conflicts can be auto-resolved")


class VersionCreatedMessage(BaseSyncMessage):
    """Message for version creation events."""
    
    event_type: SyncEventType = Field(SyncEventType.VERSION_CREATED, frozen=True)
    tracklist_id: UUID = Field(description="ID of the tracklist")
    version_id: UUID = Field(description="ID of the new version")
    version_number: int = Field(description="Version number")
    change_type: str = Field(description="Type of change")
    change_summary: str = Field(description="Summary of changes")
    created_by: str = Field(description="Who created the version")


class CueRegenerationTriggeredMessage(BaseSyncMessage):
    """Message for CUE regeneration trigger events."""
    
    event_type: SyncEventType = Field(SyncEventType.CUE_REGENERATION_TRIGGERED, frozen=True)
    tracklist_id: UUID = Field(description="ID of the tracklist")
    trigger: str = Field(description="What triggered regeneration")
    priority: str = Field(description="Regeneration priority")
    cue_formats: List[str] = Field(description="CUE formats to regenerate")
    job_count: int = Field(description="Number of regeneration jobs")
    actor: str = Field(default="system", description="Who triggered regeneration")


class BatchSyncMessage(BaseSyncMessage):
    """Message for batch synchronization operations."""
    
    event_type: SyncEventType = Field(description="Batch event type")
    tracklist_ids: List[UUID] = Field(description="List of tracklist IDs")
    tracklist_count: int = Field(description="Number of tracklists")
    source: str = Field(description="Sync source")
    operation: str = Field(description="Batch operation type")
    actor: str = Field(default="system", description="Who triggered the batch")
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    
    class Config:
        """Pydantic configuration."""
        
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class SyncStatusUpdateMessage(BaseSyncMessage):
    """Message for sync status updates."""
    
    event_type: SyncEventType = Field(SyncEventType.SYNC_STATUS_UPDATE, frozen=True)
    tracklist_id: UUID = Field(description="ID of the tracklist")
    status: str = Field(description="Current status")
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message")


class SyncConfigurationMessage(BaseSyncMessage):
    """Message for sync configuration changes."""
    
    event_type: SyncEventType = Field(SyncEventType.SYNC_CONFIG_UPDATED, frozen=True)
    tracklist_id: UUID = Field(description="ID of the tracklist")
    config_changes: Dict[str, Any] = Field(description="Configuration changes")
    actor: str = Field(default="user", description="Who changed the configuration")


# Consumer message handlers schemas
class SyncTriggerRequest(BaseModel):
    """Request to trigger synchronization."""
    
    tracklist_id: UUID = Field(description="ID of the tracklist")
    source: str = Field(default="all", description="Sync source")
    force: bool = Field(default=False, description="Force sync even if recently synced")
    priority: int = Field(default=5, description="Priority (1-10)")
    actor: str = Field(default="system", description="Who triggered the sync")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConflictResolutionRequest(BaseModel):
    """Request to resolve conflicts."""
    
    tracklist_id: UUID = Field(description="ID of the tracklist")
    conflict_resolutions: List[Dict[str, Any]] = Field(description="Conflict resolutions")
    auto_resolve: bool = Field(default=False, description="Auto-resolve if possible")
    actor: str = Field(default="user", description="Who is resolving conflicts")


class VersionRollbackRequest(BaseModel):
    """Request to rollback to a version."""
    
    tracklist_id: UUID = Field(description="ID of the tracklist")
    version_id: UUID = Field(description="ID of version to rollback to")
    create_backup: bool = Field(default=True, description="Create backup before rollback")
    actor: str = Field(default="user", description="Who requested rollback")
    reason: Optional[str] = Field(None, description="Reason for rollback")


class BatchSyncRequest(BaseModel):
    """Request for batch synchronization."""
    
    tracklist_ids: List[UUID] = Field(description="List of tracklist IDs")
    source: str = Field(default="all", description="Sync source")
    parallel: bool = Field(default=True, description="Process in parallel")
    max_parallel: int = Field(default=5, description="Maximum parallel operations")
    continue_on_error: bool = Field(default=True, description="Continue if some fail")
    priority: int = Field(default=3, description="Priority (1-10)")
    actor: str = Field(default="system", description="Who triggered batch sync")