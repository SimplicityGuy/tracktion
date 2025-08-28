"""Unit tests for CUE regeneration service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.cue_file import CueFileDB
from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.cue_regeneration_service import (
    CueRegenerationService,
    RegenerationPriority,
    RegenerationTrigger,
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_cue_service():
    """Create a mock CUE service."""
    service = MagicMock()
    service.generate_cue_content = AsyncMock(return_value="GENERATED CUE CONTENT")
    return service


@pytest.fixture
def mock_audit_service():
    """Create a mock audit service."""
    service = MagicMock()
    service.log_tracklist_change = AsyncMock()
    service.log_cue_file_change = AsyncMock()
    return service


@pytest.fixture
def regeneration_service(mock_session, mock_cue_service, mock_audit_service):
    """Create CUE regeneration service instance."""
    return CueRegenerationService(
        session=mock_session,
        cue_service=mock_cue_service,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def sample_cue_files():
    """Create sample CUE files."""
    cue1 = MagicMock(spec=CueFileDB)
    cue1.id = uuid4()
    cue1.format = "standard"
    cue1.is_active = True
    
    cue2 = MagicMock(spec=CueFileDB)
    cue2.id = uuid4()
    cue2.format = "enhanced"
    cue2.is_active = True
    
    return [cue1, cue2]


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracklist = MagicMock(spec=TracklistDB)
    tracklist.id = uuid4()
    tracklist.tracks = []
    return tracklist


class TestCueRegenerationService:
    """Test CueRegenerationService methods."""

    @pytest.mark.asyncio
    async def test_handle_tracklist_change_with_active_cues(
        self, regeneration_service, mock_session, sample_cue_files
    ):
        """Test handling tracklist change with active CUE files."""
        tracklist_id = uuid4()
        
        # Mock active CUE files
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_cue_files
        mock_session.execute.return_value = mock_result
        
        result = await regeneration_service.handle_tracklist_change(
            tracklist_id=tracklist_id,
            change_type="manual",
            actor="user",
        )
        
        assert result["status"] == "queued"
        assert result["jobs_queued"] == 2
        assert len(regeneration_service.regeneration_queue) == 2

    @pytest.mark.asyncio
    async def test_handle_tracklist_change_no_active_cues(
        self, regeneration_service, mock_session
    ):
        """Test handling tracklist change with no active CUE files."""
        tracklist_id = uuid4()
        
        # Mock no active CUE files
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        result = await regeneration_service.handle_tracklist_change(
            tracklist_id=tracklist_id,
            change_type="sync",
        )
        
        assert result["status"] == "skipped"
        assert result["reason"] == "no_active_cue_files"

    def test_map_change_to_trigger(self, regeneration_service):
        """Test mapping change types to triggers."""
        assert regeneration_service._map_change_to_trigger("manual") == RegenerationTrigger.MANUAL_EDIT
        assert regeneration_service._map_change_to_trigger("sync") == RegenerationTrigger.SYNC_UPDATE
        assert regeneration_service._map_change_to_trigger("rollback") == RegenerationTrigger.VERSION_ROLLBACK
        assert regeneration_service._map_change_to_trigger("unknown") == RegenerationTrigger.FORCED

    def test_determine_priority(self, regeneration_service):
        """Test priority determination."""
        # Manual edits get high priority
        priority = regeneration_service._determine_priority(
            RegenerationTrigger.MANUAL_EDIT
        )
        assert priority == RegenerationPriority.HIGH
        
        # Version rollbacks are critical
        priority = regeneration_service._determine_priority(
            RegenerationTrigger.VERSION_ROLLBACK
        )
        assert priority == RegenerationPriority.CRITICAL
        
        # Large changes get higher priority
        priority = regeneration_service._determine_priority(
            RegenerationTrigger.SYNC_UPDATE,
            {"tracks_affected": 15}
        )
        assert priority == RegenerationPriority.HIGH
        
        # Small changes get normal priority
        priority = regeneration_service._determine_priority(
            RegenerationTrigger.SYNC_UPDATE,
            {"tracks_affected": 3}
        )
        assert priority == RegenerationPriority.NORMAL

    @pytest.mark.asyncio
    async def test_queue_regeneration(self, regeneration_service):
        """Test queuing a regeneration job."""
        tracklist_id = uuid4()
        cue_file_id = uuid4()
        
        job = await regeneration_service._queue_regeneration(
            tracklist_id=tracklist_id,
            cue_file_id=cue_file_id,
            format="standard",
            trigger=RegenerationTrigger.MANUAL_EDIT,
            priority=RegenerationPriority.HIGH,
            actor="user",
        )
        
        assert job["tracklist_id"] == str(tracklist_id)
        assert job["cue_file_id"] == str(cue_file_id)
        assert job["priority"] == RegenerationPriority.HIGH.value
        assert job["status"] == "pending"
        assert len(regeneration_service.regeneration_queue) == 1

    def test_sort_queue(self, regeneration_service):
        """Test queue sorting by priority."""
        # Add jobs with different priorities
        regeneration_service.regeneration_queue = [
            {"priority": RegenerationPriority.LOW.value, "queued_at": "2025-08-27T10:00:00"},
            {"priority": RegenerationPriority.CRITICAL.value, "queued_at": "2025-08-27T10:01:00"},
            {"priority": RegenerationPriority.HIGH.value, "queued_at": "2025-08-27T10:02:00"},
            {"priority": RegenerationPriority.NORMAL.value, "queued_at": "2025-08-27T10:03:00"},
        ]
        
        regeneration_service._sort_queue()
        
        # Critical should be first
        assert regeneration_service.regeneration_queue[0]["priority"] == RegenerationPriority.CRITICAL.value
        assert regeneration_service.regeneration_queue[1]["priority"] == RegenerationPriority.HIGH.value
        assert regeneration_service.regeneration_queue[2]["priority"] == RegenerationPriority.NORMAL.value
        assert regeneration_service.regeneration_queue[3]["priority"] == RegenerationPriority.LOW.value

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, regeneration_service):
        """Test cache invalidation."""
        tracklist_id = uuid4()
        
        await regeneration_service._invalidate_cache(tracklist_id)
        
        assert tracklist_id in regeneration_service.cache_invalidation_set

    @pytest.mark.asyncio
    async def test_process_regeneration_queue(
        self, regeneration_service, mock_session, sample_tracklist
    ):
        """Test processing the regeneration queue."""
        tracklist_id = uuid4()
        cue_file_id = uuid4()
        
        # Create a mock CUE file
        mock_cue_file = MagicMock(spec=CueFileDB)
        mock_cue_file.id = cue_file_id
        mock_cue_file.format = "standard"
        
        # Add a job to the queue
        job = {
            "job_id": "test-job-1",
            "tracklist_id": str(tracklist_id),
            "cue_file_id": str(cue_file_id),
            "format": "standard",
            "trigger": RegenerationTrigger.MANUAL_EDIT.value,
            "priority": RegenerationPriority.HIGH.value,
            "actor": "user",
            "queued_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        regeneration_service.regeneration_queue.append(job)
        
        # Mock database responses
        mock_session.get.side_effect = [sample_tracklist, mock_cue_file]
        
        # Process the queue
        processed = await regeneration_service.process_regeneration_queue(max_jobs=1)
        
        assert len(processed) == 1
        assert processed[0]["status"] == "completed"
        assert len(regeneration_service.regeneration_queue) == 0

    @pytest.mark.asyncio
    async def test_batch_regenerate(self, regeneration_service, mock_session):
        """Test batch regeneration."""
        tracklist_ids = [uuid4(), uuid4(), uuid4()]
        
        # Mock active CUE files for each tracklist
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        result = await regeneration_service.batch_regenerate(
            tracklist_ids=tracklist_ids,
            format="standard",
            actor="system",
        )
        
        assert result["total"] == 3
        assert len(result["details"]) == 3

    @pytest.mark.asyncio
    async def test_get_regeneration_status_specific_tracklist(self, regeneration_service):
        """Test getting regeneration status for a specific tracklist."""
        tracklist_id = uuid4()
        other_id = uuid4()
        
        # Add jobs to queue
        regeneration_service.regeneration_queue = [
            {"tracklist_id": str(tracklist_id), "priority": "high"},
            {"tracklist_id": str(other_id), "priority": "normal"},
            {"tracklist_id": str(tracklist_id), "priority": "normal"},
        ]
        
        # Add to cache invalidation
        regeneration_service.cache_invalidation_set.add(tracklist_id)
        
        status = await regeneration_service.get_regeneration_status(tracklist_id)
        
        assert status["tracklist_id"] == str(tracklist_id)
        assert status["queued_jobs"] == 2
        assert status["cache_invalidated"] is True

    @pytest.mark.asyncio
    async def test_get_regeneration_status_overall(self, regeneration_service):
        """Test getting overall regeneration status."""
        # Add jobs to queue
        regeneration_service.regeneration_queue = [
            {"priority": RegenerationPriority.HIGH.value},
            {"priority": RegenerationPriority.NORMAL.value},
            {"priority": RegenerationPriority.HIGH.value},
        ]
        
        status = await regeneration_service.get_regeneration_status()
        
        assert status["total_queued"] == 3
        assert status["by_priority"][RegenerationPriority.HIGH.value] == 2
        assert status["by_priority"][RegenerationPriority.NORMAL.value] == 1

    @pytest.mark.asyncio
    async def test_cancel_regeneration_by_job_id(self, regeneration_service):
        """Test canceling a specific regeneration job."""
        job_id = "test-job-1"
        
        # Add jobs to queue
        regeneration_service.regeneration_queue = [
            {"job_id": job_id, "tracklist_id": "123"},
            {"job_id": "test-job-2", "tracklist_id": "456"},
        ]
        
        result = await regeneration_service.cancel_regeneration(job_id=job_id)
        
        assert result["cancelled_count"] == 1
        assert len(regeneration_service.regeneration_queue) == 1
        assert regeneration_service.regeneration_queue[0]["job_id"] == "test-job-2"

    @pytest.mark.asyncio
    async def test_cancel_regeneration_by_tracklist(self, regeneration_service):
        """Test canceling all regeneration jobs for a tracklist."""
        tracklist_id = uuid4()
        
        # Add jobs to queue
        regeneration_service.regeneration_queue = [
            {"job_id": "1", "tracklist_id": str(tracklist_id)},
            {"job_id": "2", "tracklist_id": "other"},
            {"job_id": "3", "tracklist_id": str(tracklist_id)},
        ]
        
        result = await regeneration_service.cancel_regeneration(tracklist_id=tracklist_id)
        
        assert result["cancelled_count"] == 2
        assert len(regeneration_service.regeneration_queue) == 1
        assert regeneration_service.regeneration_queue[0]["tracklist_id"] == "other"