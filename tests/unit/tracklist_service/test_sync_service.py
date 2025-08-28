"""Unit tests for main synchronization service."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import SyncConfiguration, SyncEvent
from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.sync_service import (
    SynchronizationService,
    SyncFrequency,
    SyncSource,
    SyncStatus,
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_tracklists_sync():
    """Create a mock 1001tracklists sync service."""
    service = MagicMock()
    service.check_for_updates = AsyncMock()
    service.apply_updates = AsyncMock(return_value=(True, None))
    return service


@pytest.fixture
def mock_conflict_service():
    """Create a mock conflict resolution service."""
    service = MagicMock()
    service.detect_conflicts = AsyncMock(return_value=[])
    service.prepare_conflict_ui_data = AsyncMock()
    service.auto_resolve_conflicts = AsyncMock(return_value=[])
    service.resolve_conflicts = AsyncMock(return_value=(True, None))
    return service


@pytest.fixture
def mock_cue_service():
    """Create a mock CUE regeneration service."""
    service = MagicMock()
    service.handle_tracklist_change = AsyncMock()
    return service


@pytest.fixture
def mock_version_service():
    """Create a mock version service."""
    service = MagicMock()
    return service


@pytest.fixture
def mock_audit_service():
    """Create a mock audit service."""
    service = MagicMock()
    service.log_tracklist_change = AsyncMock()
    return service


@pytest.fixture
def sync_service(
    mock_session,
    mock_tracklists_sync,
    mock_conflict_service,
    mock_cue_service,
    mock_version_service,
    mock_audit_service,
):
    """Create synchronization service instance."""
    return SynchronizationService(
        session=mock_session,
        tracklists_sync_service=mock_tracklists_sync,
        conflict_service=mock_conflict_service,
        cue_service=mock_cue_service,
        version_service=mock_version_service,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def sample_sync_config():
    """Create a sample sync configuration."""
    config = MagicMock(spec=SyncConfiguration)
    config.tracklist_id = uuid4()
    config.sync_enabled = True
    config.sync_frequency = SyncFrequency.DAILY.value
    config.sync_source = SyncSource.ALL.value
    config.auto_accept_threshold = 0.8
    config.auto_resolve_conflicts = False
    config.last_sync_at = None
    return config


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracklist = MagicMock(spec=TracklistDB)
    tracklist.id = uuid4()
    tracklist.tracks = []
    return tracklist


class TestSynchronizationService:
    """Test SynchronizationService methods."""

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_no_updates(
        self, sync_service, mock_session, mock_tracklists_sync, sample_sync_config
    ):
        """Test manual sync when no updates are available."""
        tracklist_id = uuid4()
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        # No updates available
        mock_tracklists_sync.check_for_updates.return_value = None
        
        result = await sync_service.trigger_manual_sync(tracklist_id)
        
        assert result["status"] == SyncStatus.COMPLETED.value
        assert result["message"] == "No updates available"

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_with_updates(
        self,
        sync_service,
        mock_session,
        mock_tracklists_sync,
        mock_conflict_service,
        sample_sync_config,
        sample_tracklist,
    ):
        """Test manual sync when updates are available."""
        tracklist_id = sample_tracklist.id
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        mock_session.get.return_value = sample_tracklist
        
        # Updates available
        updates = {
            "has_updates": True,
            "changes": {
                "total_changes": 3,
                "tracks_added": [],
                "tracks_removed": [],
                "tracks_modified": [1, 2, 3],
            },
            "confidence": 0.9,
        }
        mock_tracklists_sync.check_for_updates.return_value = updates
        
        # No conflicts
        mock_conflict_service.detect_conflicts.return_value = []
        
        result = await sync_service.trigger_manual_sync(tracklist_id)
        
        assert result["status"] == SyncStatus.COMPLETED.value
        assert result["changes_applied"] == 3
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_with_conflicts(
        self,
        sync_service,
        mock_session,
        mock_tracklists_sync,
        mock_conflict_service,
        sample_sync_config,
        sample_tracklist,
    ):
        """Test manual sync when conflicts are detected."""
        tracklist_id = sample_tracklist.id
        
        # Disable auto-resolve
        sample_sync_config.auto_resolve_conflicts = False
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        mock_session.get.return_value = sample_tracklist
        
        # Updates with conflicts
        updates = {
            "has_updates": True,
            "changes": {"total_changes": 1},
            "confidence": 0.5,
        }
        mock_tracklists_sync.check_for_updates.return_value = updates
        
        # Conflicts detected
        conflicts = [
            {"id": "conflict1", "type": "track_modified", "severity": "high"}
        ]
        mock_conflict_service.detect_conflicts.return_value = conflicts
        
        # Mock UI data preparation
        ui_data = {"conflicts": conflicts, "total_conflicts": 1}
        mock_conflict_service.prepare_conflict_ui_data.return_value = ui_data
        
        result = await sync_service.trigger_manual_sync(tracklist_id)
        
        assert result["status"] == SyncStatus.CONFLICT.value
        assert "conflicts" in result

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_auto_resolve(
        self,
        sync_service,
        mock_session,
        mock_tracklists_sync,
        mock_conflict_service,
        sample_sync_config,
        sample_tracklist,
    ):
        """Test manual sync with automatic conflict resolution."""
        tracklist_id = sample_tracklist.id
        
        # Enable auto-resolve
        sample_sync_config.auto_resolve_conflicts = True
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        mock_session.get.return_value = sample_tracklist
        
        # Updates with conflicts
        updates = {
            "has_updates": True,
            "changes": {"total_changes": 1},
            "confidence": 0.9,
        }
        mock_tracklists_sync.check_for_updates.return_value = updates
        
        # Conflicts detected
        conflicts = [{"id": "conflict1", "auto_resolvable": True}]
        mock_conflict_service.detect_conflicts.return_value = conflicts
        
        # Auto-resolution
        resolutions = [{"conflict_id": "conflict1", "strategy": "use_proposed"}]
        mock_conflict_service.auto_resolve_conflicts.return_value = resolutions
        
        result = await sync_service.trigger_manual_sync(tracklist_id)
        
        assert result["status"] == SyncStatus.COMPLETED.value
        mock_conflict_service.resolve_conflicts.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_already_in_progress(self, sync_service):
        """Test manual sync when sync is already in progress."""
        tracklist_id = uuid4()
        
        # Mark as already syncing
        sync_service.active_syncs.add(tracklist_id)
        
        result = await sync_service.trigger_manual_sync(tracklist_id)
        
        assert result["status"] == SyncStatus.FAILED.value
        assert "already in progress" in result["error"]
        
        # Cleanup
        sync_service.active_syncs.discard(tracklist_id)

    @pytest.mark.asyncio
    async def test_trigger_manual_sync_recently_synced(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test manual sync when recently synced."""
        tracklist_id = uuid4()
        
        # Set recent sync time
        sample_sync_config.last_sync_at = datetime.utcnow() - timedelta(minutes=2)
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        result = await sync_service.trigger_manual_sync(tracklist_id, force=False)
        
        assert result["status"] == SyncStatus.COMPLETED.value
        assert "Recently synced" in result["message"]

    @pytest.mark.asyncio
    async def test_schedule_sync(self, sync_service, mock_session, sample_sync_config):
        """Test scheduling automatic synchronization."""
        tracklist_id = uuid4()
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        result = await sync_service.schedule_sync(
            tracklist_id, SyncFrequency.DAILY, SyncSource.ALL
        )
        
        assert result["status"] == "scheduled"
        assert result["frequency"] == SyncFrequency.DAILY.value
        assert tracklist_id in sync_service.scheduled_tasks
        
        # Cleanup
        sync_service.scheduled_tasks[tracklist_id].cancel()

    @pytest.mark.asyncio
    async def test_cancel_scheduled_sync(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test canceling scheduled synchronization."""
        tracklist_id = uuid4()
        
        # Create a scheduled task
        task = asyncio.create_task(asyncio.sleep(10))
        sync_service.scheduled_tasks[tracklist_id] = task
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        result = await sync_service.cancel_scheduled_sync(tracklist_id)
        
        assert result["status"] == "cancelled"
        assert tracklist_id not in sync_service.scheduled_tasks
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_handle_sync_failure_with_retry(self, sync_service):
        """Test handling sync failure with retry."""
        tracklist_id = uuid4()
        
        result = await sync_service.handle_sync_failure(
            tracklist_id, "Test error", retry_count=0
        )
        
        assert result["status"] == "retry_scheduled"
        assert result["retry_count"] == 1
        assert "retry_in" in result

    @pytest.mark.asyncio
    async def test_handle_sync_failure_max_retries(
        self, sync_service, mock_audit_service
    ):
        """Test handling sync failure when max retries reached."""
        tracklist_id = uuid4()
        
        result = await sync_service.handle_sync_failure(
            tracklist_id, "Test error", retry_count=3
        )
        
        assert result["status"] == "failed"
        assert result["retry_count"] == 3
        mock_audit_service.log_tracklist_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sync_status(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test getting sync status."""
        tracklist_id = uuid4()
        
        # Create a sync event
        sync_event = MagicMock(spec=SyncEvent)
        sync_event.event_type = "sync"
        sync_event.status = "completed"
        sync_event.created_at = datetime.utcnow()
        sync_event.completed_at = datetime.utcnow()
        
        # Mock database responses
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = sample_sync_config
        
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = sync_event
        
        mock_session.execute.side_effect = [mock_result1, mock_result2]
        
        result = await sync_service.get_sync_status(tracklist_id)
        
        assert result["tracklist_id"] == str(tracklist_id)
        assert result["sync_enabled"] == sample_sync_config.sync_enabled
        assert result["latest_event"] is not None

    @pytest.mark.asyncio
    async def test_update_sync_configuration(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test updating sync configuration."""
        tracklist_id = uuid4()
        
        # Mock database responses
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        updates = {
            "sync_enabled": False,
            "auto_accept_threshold": 0.9,
        }
        
        result = await sync_service.update_sync_configuration(tracklist_id, updates)
        
        assert result["status"] == "updated"
        assert sample_sync_config.sync_enabled is False
        assert sample_sync_config.auto_accept_threshold == 0.9

    @pytest.mark.asyncio
    async def test_coordinate_multi_source_sync(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test coordinating sync from multiple sources."""
        tracklist_id = uuid4()
        
        # Mock trigger_manual_sync to return success
        async def mock_trigger(tid, source, force, actor):
            return {
                "status": SyncStatus.COMPLETED.value,
                "changes_applied": 2,
                "tracklist_id": str(tid),
            }
        
        sync_service.trigger_manual_sync = mock_trigger
        
        sources = [SyncSource.ONETHOUSANDONE, SyncSource.MANUAL]
        
        result = await sync_service.coordinate_multi_source_sync(tracklist_id, sources)
        
        assert result["status"] == SyncStatus.COMPLETED.value
        assert len(result["sources_processed"]) == 2
        assert result["total_changes"] == 4  # 2 changes per source

    @pytest.mark.asyncio
    async def test_get_or_create_sync_config_existing(
        self, sync_service, mock_session, sample_sync_config
    ):
        """Test getting existing sync configuration."""
        tracklist_id = uuid4()
        
        # Mock existing config
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_config
        mock_session.execute.return_value = mock_result
        
        config = await sync_service._get_or_create_sync_config(tracklist_id)
        
        assert config == sample_sync_config
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_sync_config_new(self, sync_service, mock_session):
        """Test creating new sync configuration."""
        tracklist_id = uuid4()
        
        # Mock no existing config
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        config = await sync_service._get_or_create_sync_config(tracklist_id)
        
        assert config.tracklist_id == tracklist_id
        assert config.sync_enabled is False
        mock_session.add.assert_called_once()