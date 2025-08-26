"""Unit tests for parallel processing engine."""

import asyncio
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, UTC

import pytest

from services.tracklist_service.src.workers.parallel_processor import (
    ParallelProcessor,
    WorkerPool,
    WorkerState,
    LoadMetrics,
    BatchResult,
)
from services.tracklist_service.src.queue.batch_queue import Job, JobPriority


@pytest.fixture
def mock_job():
    """Create a mock job."""
    return Job(
        id="job-123",
        batch_id="batch-456",
        url="http://1001tracklists.com/track1",
        priority=JobPriority.NORMAL,
        user_id="user123",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_jobs():
    """Create multiple mock jobs."""
    jobs = []
    for i in range(5):
        jobs.append(
            Job(
                id=f"job-{i}",
                batch_id="batch-123",
                url=f"http://1001tracklists.com/track{i}",
                priority=JobPriority.NORMAL,
                user_id="user123",
                created_at=datetime.now(UTC),
            )
        )
    return jobs


class TestLoadMetrics:
    """Test LoadMetrics class."""
    
    @patch("services.tracklist_service.src.workers.parallel_processor.psutil")
    def test_update_metrics(self, mock_psutil):
        """Test updating system metrics."""
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        
        metrics = LoadMetrics()
        metrics.update()
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.0
        assert metrics.last_update is not None
    
    def test_get_load_factor(self):
        """Test load factor calculation."""
        metrics = LoadMetrics()
        metrics.cpu_percent = 50.0
        metrics.memory_percent = 60.0
        metrics.queue_depth = 30
        
        load_factor = metrics.get_load_factor()
        
        # (50/100 * 0.4) + (60/100 * 0.3) + (30/100 * 0.3) = 0.47
        assert 0.45 <= load_factor <= 0.50
    
    def test_get_load_factor_max_queue(self):
        """Test load factor with maxed queue."""
        metrics = LoadMetrics()
        metrics.cpu_percent = 80.0
        metrics.memory_percent = 70.0
        metrics.queue_depth = 200  # Over 100, should cap at 1.0
        
        load_factor = metrics.get_load_factor()
        
        # (80/100 * 0.4) + (70/100 * 0.3) + (1.0 * 0.3) = 0.83
        assert 0.80 <= load_factor <= 0.85


class TestWorkerPool:
    """Test WorkerPool class."""
    
    def test_initialization(self):
        """Test worker pool initialization."""
        pool = WorkerPool(min_workers=2, max_workers=8)
        
        assert pool.min_workers == 2
        assert pool.max_workers == 8
        assert pool.current_workers == 2
        assert pool.state == WorkerState.IDLE
        assert pool.executor is None
    
    def test_start_stop(self):
        """Test starting and stopping worker pool."""
        pool = WorkerPool(min_workers=2, max_workers=4)
        
        pool.start()
        assert pool.executor is not None
        assert pool.state == WorkerState.IDLE
        
        pool.stop()
        assert pool.executor is None
        assert pool.state == WorkerState.STOPPING
        assert pool.get_active_count() == 0
    
    def test_scale_up(self):
        """Test scaling up workers."""
        pool = WorkerPool(min_workers=2, max_workers=8)
        pool.start()
        
        pool.scale(5)
        assert pool.current_workers == 5
    
    def test_scale_down(self):
        """Test scaling down workers."""
        pool = WorkerPool(min_workers=2, max_workers=8)
        pool.current_workers = 6
        pool.start()
        
        pool.scale(3)
        assert pool.current_workers == 3
    
    def test_scale_bounds(self):
        """Test scaling respects min/max bounds."""
        pool = WorkerPool(min_workers=2, max_workers=8)
        
        # Try to scale below minimum
        pool.scale(1)
        assert pool.current_workers == 2
        
        # Try to scale above maximum
        pool.scale(10)
        assert pool.current_workers == 8
    
    def test_job_tracking(self):
        """Test active job tracking."""
        pool = WorkerPool()
        
        pool.add_job("job1")
        pool.add_job("job2")
        assert pool.get_active_count() == 2
        
        pool.remove_job("job1")
        assert pool.get_active_count() == 1
        
        pool.remove_job("job2")
        assert pool.get_active_count() == 0


class TestParallelProcessor:
    """Test ParallelProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = ParallelProcessor(min_workers=2, max_workers=10)
        
        assert processor.worker_pool.min_workers == 2
        assert processor.worker_pool.max_workers == 10
        assert processor.default_domain_limit == 10
        assert "1001tracklists.com" in processor.domain_configs
    
    @pytest.mark.asyncio
    async def test_process_batch_empty(self):
        """Test processing empty batch."""
        processor = ParallelProcessor()
        
        result = await processor.process_batch([], lambda x: x)
        
        assert isinstance(result, BatchResult)
        assert result.total_jobs == 0
        assert result.successful == 0
        assert result.failed == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, mock_jobs):
        """Test successful batch processing."""
        processor = ParallelProcessor()
        
        # Mock successful processor function
        def mock_processor(job):
            return {"data": f"processed_{job.id}"}
        
        with patch.object(processor.worker_pool, 'executor') as mock_executor:
            # Create mock futures
            futures = []
            for job in mock_jobs:
                future = Mock()
                future.result.return_value = {
                    "success": True,
                    "job_id": job.id,
                    "result": {"data": f"processed_{job.id}"}
                }
                futures.append(future)
            
            # Mock submit to return futures
            mock_executor.submit.side_effect = futures
            
            # Mock as_completed
            with patch("services.tracklist_service.src.workers.parallel_processor.as_completed") as mock_completed:
                mock_completed.return_value = futures
                
                result = await processor.process_batch(mock_jobs, mock_processor)
        
        assert result.total_jobs == 5
        assert result.successful == 5
        assert result.failed == 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self, mock_jobs):
        """Test batch processing with some failures."""
        processor = ParallelProcessor()
        
        def mock_processor(job):
            if "2" in job.id:
                raise Exception("Processing error")
            return {"data": f"processed_{job.id}"}
        
        with patch.object(processor.worker_pool, 'executor') as mock_executor:
            # Create mixed success/failure futures
            futures = []
            for i, job in enumerate(mock_jobs):
                future = Mock()
                if i == 2:  # Make third job fail
                    future.result.return_value = {
                        "success": False,
                        "job_id": job.id,
                        "error": "Processing error"
                    }
                else:
                    future.result.return_value = {
                        "success": True,
                        "job_id": job.id,
                        "result": {"data": f"processed_{job.id}"}
                    }
                futures.append(future)
            
            mock_executor.submit.side_effect = futures
            
            with patch("services.tracklist_service.src.workers.parallel_processor.as_completed") as mock_completed:
                mock_completed.return_value = futures
                
                result = await processor.process_batch(mock_jobs, mock_processor)
        
        assert result.successful == 4
        assert result.failed == 1
        assert len(result.errors) == 1
    
    def test_group_jobs_by_domain(self, mock_jobs):
        """Test grouping jobs by domain."""
        processor = ParallelProcessor()
        
        # Add job from different domain
        mock_jobs.append(
            Job(
                id="job-other",
                batch_id="batch-123",
                url="http://example.com/track",
                priority=JobPriority.NORMAL,
                user_id="user123",
                created_at=datetime.now(UTC),
            )
        )
        
        grouped = processor._group_jobs_by_domain(mock_jobs)
        
        assert "1001tracklists.com" in grouped
        assert "example.com" in grouped
        assert len(grouped["1001tracklists.com"]) == 5
        assert len(grouped["example.com"]) == 1
    
    def test_extract_domain(self):
        """Test domain extraction from URL."""
        processor = ParallelProcessor()
        
        assert processor._extract_domain("http://1001tracklists.com/track") == "1001tracklists.com"
        assert processor._extract_domain("https://www.example.com/page") == "www.example.com"
        assert processor._extract_domain("invalid-url") == "unknown"
    
    @pytest.mark.asyncio
    async def test_apply_rate_limit(self):
        """Test rate limiting application."""
        processor = ParallelProcessor()
        
        # First request should succeed immediately
        result = await processor._apply_rate_limit("1001tracklists.com")
        assert result is True
        
        # Rapid requests should be rate limited
        start = time.time()
        results = []
        for _ in range(3):
            result = await processor._apply_rate_limit("1001tracklists.com")
            results.append(result)
        elapsed = time.time() - start
        
        # Should have some delay due to rate limiting
        assert elapsed > 0.1
    
    @patch("services.tracklist_service.src.workers.parallel_processor.psutil")
    def test_adjust_worker_count_low_load(self, mock_psutil):
        """Test worker adjustment for low load."""
        processor = ParallelProcessor(min_workers=2, max_workers=10)
        processor.worker_pool.current_workers = 5  # Start with different value
        
        metrics = LoadMetrics()
        metrics.cpu_percent = 20.0
        metrics.memory_percent = 30.0
        metrics.queue_depth = 5
        
        with patch.object(processor.worker_pool, 'scale') as mock_scale:
            processor.adjust_worker_count(metrics)
            mock_scale.assert_called_with(2)  # Should use minimum
    
    @patch("services.tracklist_service.src.workers.parallel_processor.psutil")
    def test_adjust_worker_count_high_load(self, mock_psutil):
        """Test worker adjustment for high load."""
        processor = ParallelProcessor(min_workers=2, max_workers=10)
        
        metrics = LoadMetrics()
        metrics.cpu_percent = 80.0
        metrics.memory_percent = 75.0
        metrics.queue_depth = 100
        
        with patch.object(processor.worker_pool, 'scale') as mock_scale:
            processor.adjust_worker_count(metrics)
            mock_scale.assert_called_with(10)  # Should use maximum
    
    @patch("services.tracklist_service.src.workers.parallel_processor.psutil")
    def test_adjust_worker_count_high_errors(self, mock_psutil):
        """Test worker reduction with high error rate."""
        processor = ParallelProcessor(min_workers=2, max_workers=10)
        processor.worker_pool.current_workers = 8
        
        metrics = LoadMetrics()
        metrics.cpu_percent = 60.0
        metrics.memory_percent = 60.0
        metrics.queue_depth = 50
        metrics.error_rate = 0.3  # High error rate
        
        with patch.object(processor.worker_pool, 'scale') as mock_scale:
            processor.adjust_worker_count(metrics)
            # Should reduce workers due to errors
            called_with = mock_scale.call_args[0][0]
            assert called_with < 8
    
    @patch("services.tracklist_service.src.workers.parallel_processor.psutil")
    def test_get_metrics(self, mock_psutil):
        """Test getting current metrics."""
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(percent=55.0)
        
        processor = ParallelProcessor()
        processor.metrics.queue_depth = 10
        processor.metrics.avg_processing_time = 2.5
        processor.metrics.error_rate = 0.05
        
        metrics = processor.get_metrics()
        
        assert metrics["cpu_percent"] == 45.0
        assert metrics["memory_percent"] == 55.0
        assert metrics["queue_depth"] == 10
        assert metrics["avg_processing_time"] == 2.5
        assert metrics["error_rate"] == 0.05
        assert metrics["worker_count"] == processor.worker_pool.current_workers
        assert metrics["worker_state"] == "idle"
    
    def test_shutdown(self):
        """Test processor shutdown."""
        processor = ParallelProcessor()
        processor.worker_pool.start()
        
        with patch.object(processor.worker_pool, 'stop') as mock_stop:
            processor.shutdown()
            mock_stop.assert_called_once()
    
    def test_process_job_wrapper_success(self, mock_job):
        """Test successful job processing wrapper."""
        processor = ParallelProcessor()
        
        def mock_processor(job):
            return {"result": "success"}
        
        result = processor._process_job_wrapper(mock_job, mock_processor)
        
        assert result["success"] is True
        assert result["job_id"] == mock_job.id
        assert result["result"] == {"result": "success"}
    
    def test_process_job_wrapper_failure(self, mock_job):
        """Test failed job processing wrapper."""
        processor = ParallelProcessor()
        
        def mock_processor(job):
            raise ValueError("Processing failed")
        
        result = processor._process_job_wrapper(mock_job, mock_processor)
        
        assert result["success"] is False
        assert result["job_id"] == mock_job.id
        assert "Processing failed" in result["error"]