"""Unit tests for batch synchronization service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import SyncConfiguration, SyncEvent
from services.tracklist_service.src.services.batch_sync_service import (
    BatchProgress,
    BatchStatus,
    BatchStrategy,
    BatchSyncService,
)
from services.tracklist_service.src.services.sync_service import SyncSource, SyncStatus


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_sync_service():
    """Create a mock synchronization service."""
    service = MagicMock()
    service.trigger_manual_sync = AsyncMock()
    return service


@pytest.fixture
def mock_event_publisher():
    """Create a mock event publisher."""
    publisher = MagicMock()
    publisher.publish_batch_sync = AsyncMock()
    return publisher


@pytest.fixture
def batch_sync_service(mock_session, mock_sync_service, mock_event_publisher):
    """Create batch sync service instance."""
    return BatchSyncService(
        session=mock_session,
        sync_service=mock_sync_service,
        event_publisher=mock_event_publisher,
    )


@pytest.fixture
def sample_tracklist_ids():
    """Create sample tracklist IDs."""
    return [uuid4() for _ in range(5)]


class TestBatchSyncService:
    """Test BatchSyncService methods."""

    @pytest.mark.asyncio
    async def test_batch_sync_parallel_success(self, batch_sync_service, mock_sync_service, sample_tracklist_ids):
        """Test successful parallel batch synchronization."""
        # Mock successful syncs
        mock_sync_service.trigger_manual_sync.return_value = {
            "status": SyncStatus.COMPLETED.value,
            "changes_applied": 3,
        }

        result = await batch_sync_service.batch_sync_tracklists(
            tracklist_ids=sample_tracklist_ids,
            source=SyncSource.ALL,
            strategy=BatchStrategy.PARALLEL,
            max_parallel=3,
        )

        assert result.status == BatchStatus.COMPLETED
        assert result.progress.successful == 5
        assert result.progress.failed == 0
        assert len(result.results) == 5

    @pytest.mark.asyncio
    async def test_batch_sync_sequential(self, batch_sync_service, mock_sync_service, sample_tracklist_ids):
        """Test sequential batch synchronization."""
        # Mock successful syncs
        mock_sync_service.trigger_manual_sync.return_value = {
            "status": SyncStatus.COMPLETED.value,
        }

        result = await batch_sync_service.batch_sync_tracklists(
            tracklist_ids=sample_tracklist_ids,
            source=SyncSource.ALL,
            strategy=BatchStrategy.SEQUENTIAL,
        )

        assert result.status == BatchStatus.COMPLETED
        assert result.progress.successful == 5
        # Verify sequential calls
        assert mock_sync_service.trigger_manual_sync.call_count == 5

    @pytest.mark.asyncio
    async def test_batch_sync_partial_failure(self, batch_sync_service, mock_sync_service, sample_tracklist_ids):
        """Test batch sync with partial failures."""

        # Mock mixed results
        async def sync_side_effect(tracklist_id, **kwargs):
            # Fail every other sync
            if sample_tracklist_ids.index(tracklist_id) % 2 == 0:
                return {"status": SyncStatus.COMPLETED.value}
            else:
                return {"status": SyncStatus.FAILED.value, "error": "Test error"}

        mock_sync_service.trigger_manual_sync.side_effect = sync_side_effect

        result = await batch_sync_service.batch_sync_tracklists(
            tracklist_ids=sample_tracklist_ids,
            source=SyncSource.ALL,
            strategy=BatchStrategy.PARALLEL,
            continue_on_error=True,
        )

        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.progress.successful == 3
        assert result.progress.failed == 2

    @pytest.mark.asyncio
    async def test_batch_sync_stop_on_error(self, batch_sync_service, mock_sync_service, sample_tracklist_ids):
        """Test batch sync stops on error when configured."""
        # Mock first success, then failure
        mock_sync_service.trigger_manual_sync.side_effect = [
            {"status": SyncStatus.COMPLETED.value},
            Exception("Test error"),
        ]

        with pytest.raises(Exception):
            await batch_sync_service.batch_sync_tracklists(
                tracklist_ids=sample_tracklist_ids[:2],
                source=SyncSource.ALL,
                strategy=BatchStrategy.SEQUENTIAL,
                continue_on_error=False,
            )

    @pytest.mark.asyncio
    async def test_batch_sync_adaptive_strategy(self, batch_sync_service, mock_sync_service, sample_tracklist_ids):
        """Test adaptive batch synchronization strategy."""
        # Mock successful syncs
        mock_sync_service.trigger_manual_sync.return_value = {
            "status": SyncStatus.COMPLETED.value,
        }

        # Mock system load
        with patch.object(batch_sync_service, "_get_system_load", return_value=0.3):
            result = await batch_sync_service.batch_sync_tracklists(
                tracklist_ids=sample_tracklist_ids,
                source=SyncSource.ALL,
                strategy=BatchStrategy.ADAPTIVE,
            )

        assert result.status == BatchStatus.COMPLETED
        assert result.progress.successful == 5

    @pytest.mark.asyncio
    async def test_batch_sync_priority_based(
        self, batch_sync_service, mock_sync_service, mock_session, sample_tracklist_ids
    ):
        """Test priority-based batch synchronization."""
        # Mock sync configurations with different priorities
        configs = []
        for i, tid in enumerate(sample_tracklist_ids):
            config = MagicMock(spec=SyncConfiguration)
            config.sync_frequency = ["realtime", "hourly", "daily", "weekly", "manual"][i]
            configs.append(config)

        mock_session.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=config)) for config in configs
        ]

        # Mock successful syncs
        mock_sync_service.trigger_manual_sync.return_value = {
            "status": SyncStatus.COMPLETED.value,
        }

        result = await batch_sync_service.batch_sync_tracklists(
            tracklist_ids=sample_tracklist_ids,
            source=SyncSource.ALL,
            strategy=BatchStrategy.PRIORITY_BASED,
        )

        assert result.status == BatchStatus.COMPLETED
        assert result.progress.successful == 5

    @pytest.mark.asyncio
    async def test_batch_progress_tracking(self, batch_sync_service):
        """Test batch progress tracking."""
        progress = BatchProgress(total=10)

        assert progress.progress_percentage == 0

        progress.completed = 5
        progress.successful = 3
        progress.failed = 2

        assert progress.progress_percentage == 50

        progress.start_time = datetime.utcnow()
        assert progress.duration is not None

    @pytest.mark.asyncio
    async def test_apply_priority_order(self, batch_sync_service):
        """Test applying priority order to tracklist IDs."""
        ids = [uuid4() for _ in range(5)]
        priority_order = [ids[3], ids[1], ids[4]]

        reordered = batch_sync_service._apply_priority_order(ids, priority_order)

        # Priority items should come first
        assert reordered[0] == ids[3]
        assert reordered[1] == ids[1]
        assert reordered[2] == ids[4]
        # Remaining items follow
        assert ids[0] in reordered[3:]
        assert ids[2] in reordered[3:]

    @pytest.mark.asyncio
    async def test_get_system_load(self, batch_sync_service):
        """Test system load calculation."""
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.percent = 60.0

                load = await batch_sync_service._get_system_load()

                # (50 * 0.7 + 60 * 0.3) / 100 = 0.53
                assert 0.5 <= load <= 0.6

    @pytest.mark.asyncio
    async def test_get_batch_status(self, batch_sync_service):
        """Test getting batch operation status."""
        batch_id = uuid4()
        progress = BatchProgress(
            total=10,
            completed=5,
            successful=3,
            failed=2,
            start_time=datetime.utcnow(),
        )

        batch_sync_service.active_batches[batch_id] = progress

        status = await batch_sync_service.get_batch_status(batch_id)

        assert status is not None
        assert status["batch_id"] == str(batch_id)
        assert status["status"] == "processing"
        assert status["progress"] == 50
        assert status["total"] == 10
        assert status["completed"] == 5

    @pytest.mark.asyncio
    async def test_cancel_batch(self, batch_sync_service):
        """Test cancelling a batch operation."""
        batch_id = uuid4()
        batch_sync_service.active_batches[batch_id] = BatchProgress()

        result = await batch_sync_service.cancel_batch(batch_id)

        assert result is True

        # Non-existent batch
        result = await batch_sync_service.cancel_batch(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_aggregate_batch_conflicts(self, batch_sync_service, mock_session, sample_tracklist_ids):
        """Test aggregating conflicts across tracklists."""
        # Create mock sync events with conflicts
        events = []
        for i, tid in enumerate(sample_tracklist_ids[:2]):
            event = MagicMock(spec=SyncEvent)
            event.conflict_data = {
                "conflicts": [
                    {"type": "track_modified"},
                    {"type": "track_added"},
                ]
            }
            events.append(event)

        # Mock no conflicts for remaining tracklists
        for _ in sample_tracklist_ids[2:]:
            events.append(None)

        mock_session.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=event)) for event in events
        ]

        result = await batch_sync_service.aggregate_batch_conflicts(sample_tracklist_ids)

        assert result["total_conflicts"] == 4  # 2 conflicts per tracklist * 2 tracklists
        assert len(result["affected_tracklists"]) == 2
        assert result["conflict_types"]["track_modified"] == 2
        assert result["conflict_types"]["track_added"] == 2
