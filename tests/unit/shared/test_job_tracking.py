"""Tests for job tracking system."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import Job
from shared.core_types.src.repositories import JobRepository, JobStatus


class TestJobModel:
    """Test Job model."""

    def test_job_creation(self):
        """Test creating a job model."""
        job_id = uuid4()
        correlation_id = uuid4()

        job = Job(
            id=job_id,
            job_type="test_job",
            status="created",
            service_name="test_service",
            correlation_id=correlation_id,
            context={"test": "context"},
            created_at=datetime.now(UTC),
        )

        assert job.id == job_id
        assert job.job_type == "test_job"
        assert job.status == "created"
        assert job.service_name == "test_service"
        assert job.correlation_id == correlation_id
        assert job.context == {"test": "context"}
        assert job.parent_job_id is None
        assert job.result is None
        assert job.error_message is None
        assert job.progress in [0, None]  # May be 0 or None depending on DB defaults
        assert job.total_items is None
        assert job.started_at is None
        assert job.completed_at is None

    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        job_id = uuid4()
        correlation_id = uuid4()
        now = datetime.now(UTC)

        job = Job(
            id=job_id,
            job_type="test_job",
            status="running",
            service_name="test_service",
            correlation_id=correlation_id,
            context={"test": "context"},
            result={"output": "data"},
            progress=50,
            total_items=100,
            started_at=now,
            created_at=now,
        )

        job_dict = job.to_dict()

        assert job_dict["id"] == str(job_id)
        assert job_dict["job_type"] == "test_job"
        assert job_dict["status"] == "running"
        assert job_dict["service_name"] == "test_service"
        assert job_dict["correlation_id"] == str(correlation_id)
        assert job_dict["context"] == {"test": "context"}
        assert job_dict["result"] == {"output": "data"}
        assert job_dict["progress"] == 50
        assert job_dict["total_items"] == 100
        assert job_dict["started_at"] == now.isoformat()
        assert job_dict["created_at"] == now.isoformat()
        assert job_dict["parent_job_id"] is None
        assert job_dict["completed_at"] is None


class TestJobRepository:
    """Test JobRepository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def job_repo(self, mock_db_manager, mock_session):
        """Create job repository with mocked dependencies."""
        # Setup context manager properly
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_db_manager.get_db_session.return_value = context_manager
        return JobRepository(mock_db_manager)

    def test_create_job(self, job_repo, mock_session):
        """Test creating a job."""
        # Mock the session behavior
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()
            job.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_job

        # Create job
        job = job_repo.create(
            job_type="analysis",
            service_name="test_service",
            context={"file": "test.mp3"},
        )

        # Verify
        assert job is not None
        assert job.job_type == "analysis"
        assert job.status == JobStatus.CREATED.value
        assert job.service_name == "test_service"
        assert job.context == {"file": "test.mp3"}
        assert job.correlation_id is not None

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_get_job_by_id(self, job_repo, mock_session):
        """Test getting job by ID."""
        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        result = job_repo.get_by_id(job_id)

        assert result == mock_job
        mock_session.query.assert_called_once_with(Job)
        mock_query.filter.assert_called_once()

    def test_get_jobs_by_correlation_id(self, job_repo, mock_session):
        """Test getting jobs by correlation ID."""
        correlation_id = uuid4()
        mock_jobs = [Mock(spec=Job), Mock(spec=Job)]

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_jobs
        mock_session.query.return_value = mock_query

        result = job_repo.get_by_correlation_id(correlation_id)

        assert result == mock_jobs
        assert len(result) == 2
        mock_session.query.assert_called_once_with(Job)
        mock_query.filter.assert_called_once()

    def test_get_jobs_by_status(self, job_repo, mock_session):
        """Test getting jobs by status."""
        mock_jobs = [Mock(spec=Job), Mock(spec=Job)]

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_jobs
        mock_session.query.return_value = mock_query

        result = job_repo.get_by_status(JobStatus.RUNNING, "test_service")

        assert result == mock_jobs
        assert len(result) == 2
        mock_session.query.assert_called_once_with(Job)
        # Should have two filter calls - one for status, one for service_name
        assert mock_query.filter.call_count == 2

    def test_update_job_status_to_running(self, job_repo, mock_session):
        """Test updating job status to running."""
        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.status = JobStatus.CREATED.value
        mock_job.started_at = None

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        result = job_repo.update_status(job_id, JobStatus.RUNNING)

        assert result == mock_job
        assert mock_job.status == JobStatus.RUNNING.value
        assert mock_job.started_at is not None
        assert mock_job.updated_at is not None
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_job_status_to_completed(self, job_repo, mock_session):
        """Test updating job status to completed."""
        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.status = JobStatus.RUNNING.value
        mock_job.completed_at = None

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        result = job_repo.update_status(
            job_id,
            JobStatus.COMPLETED,
            result={"output": "success"},
        )

        assert result == mock_job
        assert mock_job.status == JobStatus.COMPLETED.value
        assert mock_job.completed_at is not None
        assert mock_job.result == {"output": "success"}
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_job_status_to_failed(self, job_repo, mock_session):
        """Test updating job status to failed."""
        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.status = JobStatus.RUNNING.value
        mock_job.completed_at = None

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        result = job_repo.update_status(
            job_id,
            JobStatus.FAILED,
            error_message="Test error",
        )

        assert result == mock_job
        assert mock_job.status == JobStatus.FAILED.value
        assert mock_job.completed_at is not None
        assert mock_job.error_message == "Test error"
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_job_progress(self, job_repo, mock_session):
        """Test updating job progress."""
        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.progress = 0
        mock_job.total_items = None

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        result = job_repo.update_progress(job_id, 50, 100)

        assert result == mock_job
        assert mock_job.progress == 50
        assert mock_job.total_items == 100
        assert mock_job.updated_at is not None
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_create_child_job(self, job_repo, mock_session):
        """Test creating a child job."""
        parent_job_id = uuid4()
        correlation_id = uuid4()

        mock_parent_job = Mock(spec=Job)
        mock_parent_job.id = parent_job_id
        mock_parent_job.correlation_id = correlation_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_parent_job
        mock_session.query.return_value = mock_query

        # Mock the child job creation
        created_child = None

        def capture_job(job):
            nonlocal created_child
            created_child = job
            job.id = uuid4()
            job.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_job

        # Create child job
        child_job = job_repo.create_child_job(
            parent_job_id=parent_job_id,
            job_type="child_task",
            service_name="child_service",
            context={"step": 1},
        )

        # Verify
        assert child_job is not None
        assert child_job.job_type == "child_task"
        assert child_job.status == JobStatus.CREATED.value
        assert child_job.service_name == "child_service"
        assert child_job.context == {"step": 1}
        assert child_job.correlation_id == correlation_id
        assert child_job.parent_job_id == parent_job_id

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_get_child_jobs(self, job_repo, mock_session):
        """Test getting child jobs."""
        parent_job_id = uuid4()
        mock_child_jobs = [Mock(spec=Job), Mock(spec=Job)]

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_child_jobs
        mock_session.query.return_value = mock_query

        result = job_repo.get_child_jobs(parent_job_id)

        assert result == mock_child_jobs
        assert len(result) == 2
        mock_session.query.assert_called_once_with(Job)
        mock_query.filter.assert_called_once()

    def test_cleanup_old_jobs(self, job_repo, mock_session):
        """Test cleaning up old jobs."""
        old_jobs = [Mock(spec=Job), Mock(spec=Job), Mock(spec=Job)]

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = old_jobs
        mock_session.query.return_value = mock_query

        count = job_repo.cleanup_old_jobs(days=30)

        assert count == 3
        assert mock_session.delete.call_count == 3
        mock_session.query.assert_called_once_with(Job)
        mock_query.filter.assert_called_once()

    def test_update_nonexistent_job(self, job_repo, mock_session):
        """Test updating a job that doesn't exist."""
        job_id = uuid4()

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = job_repo.update_status(job_id, JobStatus.COMPLETED)

        assert result is None
        mock_session.flush.assert_not_called()
        mock_session.refresh.assert_not_called()

    def test_create_child_job_with_nonexistent_parent(self, job_repo, mock_session):
        """Test creating child job when parent doesn't exist."""
        parent_job_id = uuid4()

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = job_repo.create_child_job(
            parent_job_id=parent_job_id,
            job_type="child_task",
        )

        assert result is None
        mock_session.add.assert_not_called()
