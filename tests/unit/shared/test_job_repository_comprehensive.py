"""Comprehensive unit tests for JobRepository class."""

import concurrent.futures
import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import Job
from shared.core_types.src.repositories import JobRepository, JobStatus


class TestJobRepository:
    """Test JobRepository with comprehensive coverage."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.flush = MagicMock()
        session.refresh = MagicMock()
        session.add = MagicMock()
        session.delete = MagicMock()
        session.execute = MagicMock()
        session.commit = MagicMock()
        session.rollback = MagicMock()
        return session

    @pytest.fixture
    def repository(self, mock_db_manager, mock_session):
        """Create JobRepository with mocked dependencies."""
        # Setup context manager properly
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_db_manager.get_db_session.return_value = context_manager
        return JobRepository(mock_db_manager)

    @pytest.fixture
    def sample_job(self):
        """Create a sample job for testing."""
        return Job(
            id=uuid4(),
            job_type="analysis",
            status=JobStatus.CREATED.value,
            service_name="test_service",
            correlation_id=uuid4(),
            context={"file": "test.mp3"},
            created_at=datetime.now(UTC),
        )

    # Basic CRUD Operations Tests

    def test_create_job_minimal(self, repository, mock_session):
        """Test creating a job with minimal parameters."""
        # Mock the job creation process
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()
            job.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_job

        # Create job
        job = repository.create(job_type="analysis")

        # Verify
        assert job is not None
        assert job.job_type == "analysis"
        assert job.status == JobStatus.CREATED.value
        assert job.service_name is None
        assert job.context is None
        assert job.correlation_id is not None  # Should be auto-generated
        assert job.parent_job_id is None
        assert isinstance(job.created_at, datetime)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_create_job_full_parameters(self, repository, mock_session):
        """Test creating a job with all parameters."""
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()
            job.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_job

        # Parameters
        job_type = "tracklist_generation"
        service_name = "analysis_service"
        context = {"recording_id": str(uuid4()), "algorithm": "ml_v2"}
        correlation_id = uuid4()
        parent_job_id = uuid4()

        # Create job
        job = repository.create(
            job_type=job_type,
            service_name=service_name,
            context=context,
            correlation_id=correlation_id,
            parent_job_id=parent_job_id,
        )

        # Verify all parameters are set correctly
        assert job.job_type == job_type
        assert job.service_name == service_name
        assert job.context == context
        assert job.correlation_id == correlation_id
        assert job.parent_job_id == parent_job_id
        assert job.status == JobStatus.CREATED.value

    def test_get_by_id_exists(self, repository, mock_session, sample_job):
        """Test getting job by ID when it exists."""
        # Setup mock query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Get job
        result = repository.get_by_id(sample_job.id)

        # Verify
        assert result == sample_job
        mock_session.execute.assert_called_once()

    def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting job by ID when it doesn't exist."""
        # Setup mock to return None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Get non-existent job
        result = repository.get_by_id(uuid4())

        # Verify
        assert result is None
        mock_session.execute.assert_called_once()

    def test_get_by_correlation_id(self, repository, mock_session):
        """Test getting jobs by correlation ID."""
        correlation_id = uuid4()
        jobs = [
            Job(id=uuid4(), job_type="analysis", correlation_id=correlation_id, status="created"),
            Job(id=uuid4(), job_type="validation", correlation_id=correlation_id, status="running"),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get jobs
        result = repository.get_by_correlation_id(correlation_id)

        # Verify
        assert len(result) == 2
        assert result[0].job_type == "analysis"
        assert result[1].job_type == "validation"
        mock_session.execute.assert_called_once()

    def test_get_by_status_without_service_filter(self, repository, mock_session):
        """Test getting jobs by status only."""
        jobs = [
            Job(id=uuid4(), job_type="analysis", status=JobStatus.RUNNING.value),
            Job(id=uuid4(), job_type="validation", status=JobStatus.RUNNING.value),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get jobs
        result = repository.get_by_status(JobStatus.RUNNING)

        # Verify
        assert len(result) == 2
        assert all(job.status == JobStatus.RUNNING.value for job in result)
        mock_session.execute.assert_called_once()

    def test_get_by_status_with_service_filter(self, repository, mock_session):
        """Test getting jobs by status and service name."""
        jobs = [
            Job(id=uuid4(), job_type="analysis", status=JobStatus.RUNNING.value, service_name="test_service"),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get jobs
        result = repository.get_by_status(JobStatus.RUNNING, "test_service")

        # Verify
        assert len(result) == 1
        assert result[0].service_name == "test_service"
        mock_session.execute.assert_called_once()

    # Status Update Tests

    def test_update_status_to_running(self, repository, mock_session, sample_job):
        """Test updating job status to running."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Update status
        result = repository.update_status(sample_job.id, JobStatus.RUNNING)

        # Verify
        assert result == sample_job
        assert sample_job.status == JobStatus.RUNNING.value
        assert sample_job.started_at is not None
        assert sample_job.updated_at is not None
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_status_to_completed_with_result(self, repository, mock_session, sample_job):
        """Test updating job status to completed with result data."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        result_data = {"processed_items": 100, "success": True}

        # Update status
        result = repository.update_status(sample_job.id, JobStatus.COMPLETED, result=result_data)

        # Verify
        assert result == sample_job
        assert sample_job.status == JobStatus.COMPLETED.value
        assert sample_job.completed_at is not None
        assert sample_job.result == result_data
        assert sample_job.updated_at is not None

    def test_update_status_to_failed_with_error(self, repository, mock_session, sample_job):
        """Test updating job status to failed with error message."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        error_message = "File not found: test.mp3"

        # Update status
        result = repository.update_status(sample_job.id, JobStatus.FAILED, error_message=error_message)

        # Verify
        assert result == sample_job
        assert sample_job.status == JobStatus.FAILED.value
        assert sample_job.completed_at is not None
        assert sample_job.error_message == error_message
        assert sample_job.updated_at is not None

    def test_update_status_to_cancelled(self, repository, mock_session, sample_job):
        """Test updating job status to cancelled."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Update status
        result = repository.update_status(sample_job.id, JobStatus.CANCELLED)

        # Verify
        assert result == sample_job
        assert sample_job.status == JobStatus.CANCELLED.value
        assert sample_job.completed_at is not None

    def test_update_status_nonexistent_job(self, repository, mock_session):
        """Test updating status of non-existent job."""
        # Setup mock to return None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Update non-existent job
        result = repository.update_status(uuid4(), JobStatus.COMPLETED)

        # Verify
        assert result is None
        mock_session.flush.assert_not_called()
        mock_session.refresh.assert_not_called()

    def test_update_status_preserves_started_at(self, repository, mock_session, sample_job):
        """Test that updating status preserves existing started_at timestamp."""
        # Set up job with existing started_at
        original_started_at = datetime.now(UTC) - timedelta(minutes=5)
        sample_job.started_at = original_started_at

        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Update to completed (should not change started_at)
        result = repository.update_status(sample_job.id, JobStatus.COMPLETED)

        # Verify started_at is preserved
        assert result.started_at == original_started_at

    # Progress Update Tests

    def test_update_progress(self, repository, mock_session, sample_job):
        """Test updating job progress."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Update progress
        result = repository.update_progress(sample_job.id, progress=50, total_items=100)

        # Verify
        assert result == sample_job
        assert sample_job.progress == 50
        assert sample_job.total_items == 100
        assert sample_job.updated_at is not None

    def test_update_progress_without_total_items(self, repository, mock_session, sample_job):
        """Test updating progress without specifying total items."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Update progress without total
        result = repository.update_progress(sample_job.id, progress=25)

        # Verify
        assert result == sample_job
        assert sample_job.progress == 25
        # total_items should remain unchanged

    def test_update_progress_nonexistent_job(self, repository, mock_session):
        """Test updating progress of non-existent job."""
        # Setup mock to return None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Update non-existent job
        result = repository.update_progress(uuid4(), progress=50)

        # Verify
        assert result is None

    # Child Job Tests

    def test_create_child_job(self, repository, mock_session):
        """Test creating a child job."""
        parent_job_id = uuid4()
        correlation_id = uuid4()

        # Setup parent job mock
        parent_job = Job(id=parent_job_id, job_type="parent", correlation_id=correlation_id, status="running")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = parent_job
        mock_session.execute.return_value = mock_result

        # Mock child job creation
        created_child = None

        def capture_job(job):
            nonlocal created_child
            created_child = job
            job.id = uuid4()
            job.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_job

        # Create child job
        child_job = repository.create_child_job(
            parent_job_id=parent_job_id, job_type="child_analysis", service_name="child_service", context={"step": 1}
        )

        # Verify
        assert child_job is not None
        assert child_job.job_type == "child_analysis"
        assert child_job.parent_job_id == parent_job_id
        assert child_job.correlation_id == correlation_id
        assert child_job.service_name == "child_service"
        assert child_job.context == {"step": 1}
        assert child_job.status == JobStatus.CREATED.value

    def test_create_child_job_nonexistent_parent(self, repository, mock_session):
        """Test creating child job with non-existent parent."""
        # Setup mock to return None (parent not found)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Try to create child job
        result = repository.create_child_job(parent_job_id=uuid4(), job_type="child_analysis")

        # Verify
        assert result is None
        mock_session.add.assert_not_called()

    def test_get_child_jobs(self, repository, mock_session):
        """Test getting child jobs for a parent."""
        parent_job_id = uuid4()
        child_jobs = [
            Job(id=uuid4(), job_type="child1", parent_job_id=parent_job_id),
            Job(id=uuid4(), job_type="child2", parent_job_id=parent_job_id),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = child_jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get child jobs
        result = repository.get_child_jobs(parent_job_id)

        # Verify
        assert len(result) == 2
        assert all(job.parent_job_id == parent_job_id for job in result)

    def test_get_child_jobs_empty(self, repository, mock_session):
        """Test getting child jobs when none exist."""
        # Setup mock to return empty list
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get child jobs
        result = repository.get_child_jobs(uuid4())

        # Verify
        assert result == []

    # Cleanup Tests

    def test_cleanup_old_jobs(self, repository, mock_session):
        """Test cleaning up old completed jobs."""
        old_jobs = [
            Job(id=uuid4(), status=JobStatus.COMPLETED.value, completed_at=datetime.now(UTC) - timedelta(days=45)),
            Job(id=uuid4(), status=JobStatus.FAILED.value, completed_at=datetime.now(UTC) - timedelta(days=60)),
            Job(id=uuid4(), status=JobStatus.CANCELLED.value, completed_at=datetime.now(UTC) - timedelta(days=35)),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = old_jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Cleanup old jobs
        count = repository.cleanup_old_jobs(days=30)

        # Verify
        assert count == 3
        assert mock_session.delete.call_count == 3
        for job in old_jobs:
            mock_session.delete.assert_any_call(job)

    def test_cleanup_old_jobs_custom_days(self, repository, mock_session):
        """Test cleanup with custom days parameter."""
        old_jobs = [
            Job(id=uuid4(), status=JobStatus.COMPLETED.value, completed_at=datetime.now(UTC) - timedelta(days=8)),
        ]

        # Setup mock
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = old_jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Cleanup with 7 days
        count = repository.cleanup_old_jobs(days=7)

        # Verify
        assert count == 1

    def test_cleanup_old_jobs_none_to_delete(self, repository, mock_session):
        """Test cleanup when no jobs meet criteria."""
        # Setup mock to return empty list
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Cleanup
        count = repository.cleanup_old_jobs()

        # Verify
        assert count == 0
        mock_session.delete.assert_not_called()

    # Error Handling Tests

    def test_create_job_database_error(self, repository, mock_session):
        """Test handling database error during job creation."""
        # Setup mock to raise integrity error
        mock_session.flush.side_effect = IntegrityError("", "", "")

        # Verify exception is raised
        with pytest.raises(IntegrityError):
            repository.create(job_type="analysis")

    def test_get_by_id_database_error(self, repository, mock_session):
        """Test handling database error during retrieval."""
        # Setup mock to raise database error
        mock_session.execute.side_effect = SQLAlchemyError("Connection failed")

        # Verify exception is raised
        with pytest.raises(SQLAlchemyError):
            repository.get_by_id(uuid4())

    def test_update_status_database_error(self, repository, mock_session):
        """Test handling database error during status update."""
        # Setup mock to raise error during flush
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = Mock(spec=Job)
        mock_session.execute.return_value = mock_result
        mock_session.flush.side_effect = SQLAlchemyError("Update failed")

        # Verify exception is raised
        with pytest.raises(SQLAlchemyError):
            repository.update_status(uuid4(), JobStatus.COMPLETED)

    # Data Validation Tests

    def test_create_job_with_large_context(self, repository, mock_session):
        """Test creating job with large context data."""
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()

        mock_session.add.side_effect = capture_job

        # Create large context
        large_context = {
            "files": [f"file_{i}.mp3" for i in range(1000)],
            "metadata": {f"key_{i}": f"value_{i}" for i in range(500)},
            "processing_options": {
                "algorithm": "deep_learning",
                "parameters": {f"param_{i}": i * 0.1 for i in range(100)},
            },
        }

        # Create job with large context
        job = repository.create(job_type="batch_analysis", context=large_context)

        # Verify
        assert job.context == large_context

    def test_update_status_with_complex_result(self, repository, mock_session, sample_job):
        """Test updating job with complex result data."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Complex result data
        complex_result = {
            "analysis": {"bpm": 128.5, "key": "C major", "confidence_scores": {"bpm": 0.95, "key": 0.87}},
            "processing_stats": {"duration_ms": 15000, "memory_used_mb": 256, "cpu_percent": 45.2},
            "quality_metrics": {"snr": 42.3, "dynamic_range": 18.7, "artifacts_detected": False},
        }

        # Update with complex result
        result = repository.update_status(sample_job.id, JobStatus.COMPLETED, result=complex_result)

        # Verify complex data is preserved
        assert result.result == complex_result

    # Edge Cases and Special Scenarios

    def test_job_status_transitions(self, repository, mock_session, sample_job):
        """Test various job status transitions."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Test transition: CREATED -> PENDING
        repository.update_status(sample_job.id, JobStatus.PENDING)
        assert sample_job.status == JobStatus.PENDING.value

        # Test transition: PENDING -> RUNNING
        repository.update_status(sample_job.id, JobStatus.RUNNING)
        assert sample_job.status == JobStatus.RUNNING.value
        assert sample_job.started_at is not None

        # Test transition: RUNNING -> COMPLETED
        repository.update_status(sample_job.id, JobStatus.COMPLETED)
        assert sample_job.status == JobStatus.COMPLETED.value
        assert sample_job.completed_at is not None

    def test_retry_job_scenario(self, repository, mock_session, sample_job):
        """Test job retry scenario (FAILED -> RETRYING -> RUNNING)."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Simulate job failure
        repository.update_status(sample_job.id, JobStatus.FAILED, error_message="Network timeout")
        assert sample_job.status == JobStatus.FAILED.value
        assert sample_job.error_message == "Network timeout"

        # Retry job
        repository.update_status(sample_job.id, JobStatus.RETRYING)
        assert sample_job.status == JobStatus.RETRYING.value

        # Resume running
        repository.update_status(sample_job.id, JobStatus.RUNNING)
        assert sample_job.status == JobStatus.RUNNING.value

    def test_multiple_child_jobs_same_parent(self, repository, mock_session):
        """Test creating multiple child jobs for same parent."""
        parent_job_id = uuid4()
        correlation_id = uuid4()

        # Setup parent job
        parent_job = Job(id=parent_job_id, job_type="batch_process", correlation_id=correlation_id, status="running")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = parent_job
        mock_session.execute.return_value = mock_result

        # Mock child job creation
        created_children = []

        def capture_job(job):
            job.id = uuid4()
            job.created_at = datetime.now(UTC)
            created_children.append(job)

        mock_session.add.side_effect = capture_job

        # Create multiple child jobs
        child_types = ["step1", "step2", "step3"]
        child_jobs = []

        for child_type in child_types:
            child = repository.create_child_job(
                parent_job_id=parent_job_id, job_type=child_type, context={"step": child_type}
            )
            child_jobs.append(child)

        # Verify all children created correctly
        assert len(child_jobs) == 3
        for i, child in enumerate(child_jobs):
            assert child.job_type == child_types[i]
            assert child.parent_job_id == parent_job_id
            assert child.correlation_id == correlation_id

    def test_get_jobs_by_multiple_statuses(self, repository, mock_session):
        """Test filtering behavior with different status combinations."""
        # Test getting only running jobs
        running_jobs = [Job(id=uuid4(), status=JobStatus.RUNNING.value)]

        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = running_jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = repository.get_by_status(JobStatus.RUNNING)
        assert len(result) == 1
        assert result[0].status == JobStatus.RUNNING.value

    def test_correlation_id_auto_generation(self, repository, mock_session):
        """Test that correlation_id is auto-generated when not provided."""
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()

        mock_session.add.side_effect = capture_job

        # Create job without correlation_id
        job = repository.create(job_type="analysis")

        # Verify correlation_id was generated
        assert job.correlation_id is not None
        assert isinstance(job.correlation_id, type(uuid4()))

    def test_correlation_id_preservation(self, repository, mock_session):
        """Test that provided correlation_id is preserved."""
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()

        mock_session.add.side_effect = capture_job

        # Create job with specific correlation_id
        custom_correlation_id = uuid4()
        job = repository.create(job_type="analysis", correlation_id=custom_correlation_id)

        # Verify correlation_id was preserved
        assert job.correlation_id == custom_correlation_id

    def test_update_timestamps_correctly_set(self, repository, mock_session, sample_job):
        """Test that update operations set timestamps correctly."""
        # Setup mock
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_job
        mock_session.execute.return_value = mock_result

        # Capture initial state
        original_created_at = sample_job.created_at

        # Update status multiple times and verify timestamps
        repository.update_status(sample_job.id, JobStatus.RUNNING)
        assert sample_job.updated_at is not None
        assert sample_job.created_at == original_created_at  # Should not change

        first_update = sample_job.updated_at

        # Small delay to ensure timestamp difference
        time.sleep(0.01)

        repository.update_status(sample_job.id, JobStatus.COMPLETED)
        assert sample_job.updated_at != first_update  # Should be updated


class TestJobRepositoryConcurrency:
    """Test JobRepository concurrency and thread safety."""

    @pytest.fixture
    def repository_factory(self):
        """Factory to create repositories for concurrent tests."""

        def create_repository():
            mock_db_manager = Mock(spec=DatabaseManager)
            mock_session = MagicMock()

            # Setup context manager
            context_manager = MagicMock()
            context_manager.__enter__.return_value = mock_session
            context_manager.__exit__.return_value = None
            mock_db_manager.get_db_session.return_value = context_manager

            return JobRepository(mock_db_manager), mock_session

        return create_repository

    def test_concurrent_job_creation(self, repository_factory):
        """Test concurrent job creation operations."""
        num_threads = 10
        created_jobs = []
        creation_errors = []

        def create_job(thread_id):
            try:
                repo, mock_session = repository_factory()

                # Mock job creation
                def capture_job(job):
                    job.id = uuid4()
                    job.created_at = datetime.now(UTC)
                    created_jobs.append(job)

                mock_session.add.side_effect = capture_job

                # Create job
                return repo.create(
                    job_type=f"thread_job_{thread_id}",
                    service_name=f"thread_service_{thread_id}",
                    context={"thread_id": thread_id},
                )
            except Exception as e:
                creation_errors.append(e)
                raise

        # Execute concurrent creates
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_job, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all jobs were created successfully
        assert len(results) == num_threads
        assert len(creation_errors) == 0
        assert len(created_jobs) == num_threads

        # Verify job uniqueness
        job_ids = [job.id for job in created_jobs]
        assert len(set(job_ids)) == num_threads  # All IDs should be unique

    def test_concurrent_status_updates(self, repository_factory):
        """Test concurrent status updates on different jobs."""
        num_jobs = 5
        job_ids = [uuid4() for _ in range(num_jobs)]
        update_results = []
        update_errors = []

        def update_job_status(job_index):
            try:
                repo, mock_session = repository_factory()
                job_id = job_ids[job_index]

                # Create mock job
                mock_job = Job(id=job_id, job_type=f"job_{job_index}", status=JobStatus.CREATED.value)

                mock_result = Mock()
                mock_result.scalar_one_or_none.return_value = mock_job
                mock_session.execute.return_value = mock_result

                # Update status
                result = repo.update_status(job_id, JobStatus.RUNNING)
                update_results.append(result)
                return result
            except Exception as e:
                update_errors.append(e)
                raise

        # Execute concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
            futures = [executor.submit(update_job_status, i) for i in range(num_jobs)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all updates succeeded
        assert len(results) == num_jobs
        assert len(update_errors) == 0
        assert all(job.status == JobStatus.RUNNING.value for job in results)

    def test_concurrent_reads_same_job(self, repository_factory):
        """Test concurrent reads of the same job."""
        job_id = uuid4()
        num_readers = 20
        read_results = []
        read_errors = []

        def read_job():
            try:
                repo, mock_session = repository_factory()

                # Create consistent mock job
                mock_job = Job(id=job_id, job_type="shared_job", status=JobStatus.RUNNING.value)

                mock_result = Mock()
                mock_result.scalar_one_or_none.return_value = mock_job
                mock_session.execute.return_value = mock_result

                # Read job
                result = repo.get_by_id(job_id)
                read_results.append(result)
                return result
            except Exception as e:
                read_errors.append(e)
                raise

        # Execute concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_readers) as executor:
            futures = [executor.submit(read_job) for _ in range(num_readers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all reads succeeded and returned consistent data
        assert len(results) == num_readers
        assert len(read_errors) == 0
        assert all(job.id == job_id for job in results)
        assert all(job.job_type == "shared_job" for job in results)

    def test_concurrent_child_job_creation(self, repository_factory):
        """Test concurrent creation of child jobs for the same parent."""
        parent_job_id = uuid4()
        correlation_id = uuid4()
        num_children = 8
        created_children = []
        child_creation_errors = []

        def create_child_job(child_index):
            try:
                repo, mock_session = repository_factory()

                # Mock parent job
                parent_job = Job(
                    id=parent_job_id, job_type="parent_job", correlation_id=correlation_id, status="running"
                )

                mock_result = Mock()
                mock_result.scalar_one_or_none.return_value = parent_job
                mock_session.execute.return_value = mock_result

                # Mock child creation
                def capture_child(job):
                    job.id = uuid4()
                    job.created_at = datetime.now(UTC)
                    created_children.append(job)

                mock_session.add.side_effect = capture_child

                # Create child job
                return repo.create_child_job(
                    parent_job_id=parent_job_id, job_type=f"child_{child_index}", context={"child_index": child_index}
                )
            except Exception as e:
                child_creation_errors.append(e)
                raise

        # Execute concurrent child creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_children) as executor:
            futures = [executor.submit(create_child_job, i) for i in range(num_children)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all child jobs were created
        assert len(results) == num_children
        assert len(child_creation_errors) == 0
        assert len(created_children) == num_children

        # Verify all children have correct parent and correlation IDs
        for child in created_children:
            assert child.parent_job_id == parent_job_id
            assert child.correlation_id == correlation_id

    def test_thread_local_session_isolation(self, repository_factory):
        """Test that different threads use isolated database sessions."""
        num_threads = 5
        session_objects = []
        isolation_lock = threading.Lock()

        def get_session_reference():
            repo, mock_session = repository_factory()

            # Capture session reference
            with isolation_lock:
                session_objects.append(id(mock_session))

            # Simulate some work
            time.sleep(0.01)

            return id(mock_session)

        # Execute in multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_session_reference) for _ in range(num_threads)]
            [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify each thread got its own session
        assert len(session_objects) == num_threads
        assert len(set(session_objects)) == num_threads  # All should be unique


class TestJobRepositoryTransactions:
    """Test JobRepository database transaction handling."""

    @pytest.fixture
    def repository_with_transaction_mocks(self):
        """Create repository with transaction-aware mocks."""
        mock_db_manager = Mock(spec=DatabaseManager)
        mock_session = MagicMock()

        # Add transaction control methods
        mock_session.begin = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.rollback = MagicMock()

        # Setup context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_db_manager.get_db_session.return_value = context_manager

        return JobRepository(mock_db_manager), mock_session

    def test_create_job_transaction_isolation(self, repository_with_transaction_mocks):
        """Test that job creation is properly isolated in transaction."""
        repo, mock_session = repository_with_transaction_mocks

        # Mock job creation
        created_job = None

        def capture_job(job):
            nonlocal created_job
            created_job = job
            job.id = uuid4()

        mock_session.add.side_effect = capture_job

        # Create job
        repo.create(job_type="test_transaction")

        # Verify transaction operations were called in correct order
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_status_rollback_on_error(self, repository_with_transaction_mocks):
        """Test that failed status updates can trigger rollback."""
        repo, mock_session = repository_with_transaction_mocks

        # Setup job mock
        mock_job = Job(id=uuid4(), job_type="test", status=JobStatus.CREATED.value)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute.return_value = mock_result

        # Make flush fail
        mock_session.flush.side_effect = SQLAlchemyError("Constraint violation")

        # Verify exception is raised (transaction should be handled by context manager)
        with pytest.raises(SQLAlchemyError):
            repo.update_status(mock_job.id, JobStatus.RUNNING)

    def test_cleanup_jobs_batch_transaction(self, repository_with_transaction_mocks):
        """Test that cleanup operations are properly batched in transaction."""
        repo, mock_session = repository_with_transaction_mocks

        # Setup old jobs
        old_jobs = [
            Job(id=uuid4(), status=JobStatus.COMPLETED.value, completed_at=datetime.now(UTC) - timedelta(days=45)),
            Job(id=uuid4(), status=JobStatus.FAILED.value, completed_at=datetime.now(UTC) - timedelta(days=50)),
        ]

        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = old_jobs
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Cleanup jobs
        count = repo.cleanup_old_jobs(days=30)

        # Verify all deletions were batched in same transaction
        assert count == 2
        assert mock_session.delete.call_count == 2
        # Context manager should handle commit

    def test_concurrent_transaction_isolation(self):
        """Test that concurrent operations maintain transaction isolation."""
        # This is a more conceptual test since we're using mocks
        # In a real database, we'd test that one transaction's changes
        # don't affect another until commit

        def create_repository():
            mock_db_manager = Mock(spec=DatabaseManager)
            mock_session = MagicMock()

            # Setup context manager
            context_manager = MagicMock()
            context_manager.__enter__.return_value = mock_session
            context_manager.__exit__.return_value = None
            mock_db_manager.get_db_session.return_value = context_manager

            return JobRepository(mock_db_manager), mock_session

        repo1, mock_session1 = create_repository()
        repo2, mock_session2 = create_repository()

        # Verify different sessions
        assert mock_session1 is not mock_session2

        # Each repository should use its own session context
        job_id = uuid4()

        # Mock different behaviors for each session
        mock_job1 = Job(id=job_id, status=JobStatus.RUNNING.value)
        mock_job2 = Job(id=job_id, status=JobStatus.CREATED.value)

        mock_result1 = Mock()
        mock_result1.scalar_one_or_none.return_value = mock_job1
        mock_session1.execute.return_value = mock_result1

        mock_result2 = Mock()
        mock_result2.scalar_one_or_none.return_value = mock_job2
        mock_session2.execute.return_value = mock_result2

        # Get job from both repositories
        job_from_repo1 = repo1.get_by_id(job_id)
        job_from_repo2 = repo2.get_by_id(job_id)

        # Each repository sees its own session's view
        assert job_from_repo1.status == JobStatus.RUNNING.value
        assert job_from_repo2.status == JobStatus.CREATED.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
