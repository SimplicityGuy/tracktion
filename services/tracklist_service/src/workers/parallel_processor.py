"""Parallel processing engine for batch tracklist operations."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from threading import Lock, Semaphore
import psutil

from services.tracklist_service.src.queue.batch_queue import Job

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker state enumeration."""

    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"


class LoadMetrics:
    """System load metrics for worker scaling."""

    def __init__(self) -> None:
        """Initialize load metrics."""
        self.cpu_percent: float = 0.0
        self.memory_percent: float = 0.0
        self.active_jobs: int = 0
        self.queue_depth: int = 0
        self.avg_processing_time: float = 0.0
        self.error_rate: float = 0.0
        self.last_update: datetime = datetime.now(UTC)

    def update(self) -> None:
        """Update system metrics."""
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        self.memory_percent = psutil.virtual_memory().percent
        self.last_update = datetime.now(UTC)

    def get_load_factor(self) -> float:
        """Calculate overall load factor (0.0 to 1.0)."""
        cpu_factor = self.cpu_percent / 100.0
        mem_factor = self.memory_percent / 100.0
        queue_factor = min(self.queue_depth / 100.0, 1.0)
        
        # Weighted average
        return (cpu_factor * 0.4 + mem_factor * 0.3 + queue_factor * 0.3)


@dataclass
class BatchResult:
    """Result of batch processing."""

    batch_id: str
    total_jobs: int
    successful: int
    failed: int
    skipped: int
    processing_time: float
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class WorkerPool:
    """Pool of workers for parallel processing."""

    def __init__(self, min_workers: int = 1, max_workers: int = 10):
        """Initialize worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
        """
        self.min_workers = max(1, min_workers)
        self.max_workers = max(self.min_workers, max_workers)
        self.current_workers = self.min_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.state = WorkerState.IDLE
        self._lock = Lock()
        self._active_jobs: Set[str] = set()
        
    def start(self) -> None:
        """Start the worker pool."""
        with self._lock:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.current_workers,
                    thread_name_prefix="tracklist_worker"
                )
                self.state = WorkerState.IDLE
                logger.info(f"Worker pool started with {self.current_workers} workers")
    
    def stop(self) -> None:
        """Stop the worker pool."""
        with self._lock:
            if self.executor:
                self.state = WorkerState.STOPPING
                self.executor.shutdown(wait=True)
                self.executor = None
                self._active_jobs.clear()
                logger.info("Worker pool stopped")
    
    def scale(self, target_workers: int) -> None:
        """Scale the worker pool to target size.
        
        Args:
            target_workers: Target number of workers
        """
        target = max(self.min_workers, min(target_workers, self.max_workers))
        
        if target != self.current_workers:
            with self._lock:
                old_count = self.current_workers
                self.current_workers = target
                
                # Recreate executor with new size
                if self.executor:
                    self.executor.shutdown(wait=False)
                    self.executor = ThreadPoolExecutor(
                        max_workers=self.current_workers,
                        thread_name_prefix="tracklist_worker"
                    )
                
                logger.info(f"Scaled worker pool from {old_count} to {self.current_workers} workers")
    
    def get_active_count(self) -> int:
        """Get number of active jobs."""
        return len(self._active_jobs)
    
    def add_job(self, job_id: str) -> None:
        """Mark job as active."""
        self._active_jobs.add(job_id)
    
    def remove_job(self, job_id: str) -> None:
        """Mark job as complete."""
        self._active_jobs.discard(job_id)


class ParallelProcessor:
    """Parallel processing engine for batch operations."""

    def __init__(self, min_workers: int = 1, max_workers: int = 10):
        """Initialize parallel processor.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
        """
        self.worker_pool = WorkerPool(min_workers, max_workers)
        self.metrics = LoadMetrics()
        self.domain_limits: Dict[str, Semaphore] = {}
        self.domain_last_request: Dict[str, float] = {}
        self._domain_lock = Lock()
        
        # Default domain limits (requests per second)
        self.default_domain_limit = 10
        self.domain_configs = {
            "1001tracklists.com": 5,
            "www.1001tracklists.com": 5,
        }
        
        # Initialize semaphores for each domain
        for domain, limit in self.domain_configs.items():
            self.domain_limits[domain] = Semaphore(limit)
    
    async def process_batch(self, jobs: List[Job], processor: Callable) -> BatchResult:
        """Process multiple jobs in parallel with rate limiting.
        
        Args:
            jobs: List of jobs to process
            processor: Function to process each job
            
        Returns:
            Batch processing result
        """
        if not jobs:
            return BatchResult(
                batch_id="",
                total_jobs=0,
                successful=0,
                failed=0,
                skipped=0,
                processing_time=0.0
            )
        
        batch_id = jobs[0].batch_id if jobs else ""
        start_time = time.time()
        
        # Start worker pool
        self.worker_pool.start()
        
        # Update metrics
        self.metrics.queue_depth = len(jobs)
        self.metrics.update()
        
        # Adjust workers based on load
        self.adjust_worker_count(self.metrics)
        
        # Group jobs by domain for better rate limiting
        jobs_by_domain = self._group_jobs_by_domain(jobs)
        
        # Process jobs
        results = []
        errors = []
        successful = 0
        failed = 0
        skipped = 0
        
        # Create futures for all jobs
        futures = {}
        executor = self.worker_pool.executor
        
        for domain, domain_jobs in jobs_by_domain.items():
            for job in domain_jobs:
                # Apply rate limiting
                if await self._apply_rate_limit(domain):
                    self.worker_pool.add_job(job.id)
                    future = executor.submit(self._process_job_wrapper, job, processor)
                    futures[future] = job
                else:
                    skipped += 1
                    logger.warning(f"Skipped job {job.id} due to rate limit")
        
        # Collect results
        for future in as_completed(futures):
            job = futures[future]
            self.worker_pool.remove_job(job.id)
            
            try:
                result = future.result(timeout=30)
                if result.get("success"):
                    successful += 1
                    results.append(result)
                else:
                    failed += 1
                    errors.append({
                        "job_id": job.id,
                        "error": result.get("error", "Unknown error")
                    })
            except Exception as e:
                failed += 1
                errors.append({
                    "job_id": job.id,
                    "error": str(e)
                })
                logger.error(f"Job {job.id} failed: {e}")
        
        # Calculate metrics
        processing_time = time.time() - start_time
        self.metrics.error_rate = failed / len(jobs) if jobs else 0.0
        self.metrics.avg_processing_time = processing_time / len(jobs) if jobs else 0.0
        
        return BatchResult(
            batch_id=batch_id,
            total_jobs=len(jobs),
            successful=successful,
            failed=failed,
            skipped=skipped,
            processing_time=processing_time,
            results=results,
            errors=errors
        )
    
    def _process_job_wrapper(self, job: Job, processor: Callable) -> Dict[str, Any]:
        """Wrapper to process a single job.
        
        Args:
            job: Job to process
            processor: Processing function
            
        Returns:
            Processing result
        """
        try:
            # Process the job
            result = processor(job)
            return {
                "success": True,
                "job_id": job.id,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}")
            return {
                "success": False,
                "job_id": job.id,
                "error": str(e)
            }
    
    def _group_jobs_by_domain(self, jobs: List[Job]) -> Dict[str, List[Job]]:
        """Group jobs by domain for rate limiting.
        
        Args:
            jobs: List of jobs
            
        Returns:
            Jobs grouped by domain
        """
        grouped: Dict[str, List[Job]] = {}
        
        for job in jobs:
            # Extract domain from URL
            domain = self._extract_domain(job.url)
            if domain not in grouped:
                grouped[domain] = []
            grouped[domain].append(job)
        
        return grouped
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain name
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    
    async def _apply_rate_limit(self, domain: str) -> bool:
        """Apply rate limiting for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            True if request allowed, False otherwise
        """
        # Get or create semaphore for domain
        if domain not in self.domain_limits:
            with self._domain_lock:
                if domain not in self.domain_limits:
                    limit = self.domain_configs.get(domain, self.default_domain_limit)
                    self.domain_limits[domain] = Semaphore(limit)
        
        # Try to acquire permit
        semaphore = self.domain_limits[domain]
        
        # Check time since last request
        current_time = time.time()
        last_request = self.domain_last_request.get(domain, 0)
        time_since_last = current_time - last_request
        
        # Calculate minimum delay between requests
        limit = self.domain_configs.get(domain, self.default_domain_limit)
        min_delay = 1.0 / limit  # seconds between requests
        
        if time_since_last < min_delay:
            # Need to wait
            await asyncio.sleep(min_delay - time_since_last)
        
        # Update last request time
        self.domain_last_request[domain] = time.time()
        
        return semaphore.acquire(blocking=False)
    
    def adjust_worker_count(self, load_metrics: LoadMetrics) -> None:
        """Dynamically scale workers based on load.
        
        Args:
            load_metrics: Current load metrics
        """
        load_factor = load_metrics.get_load_factor()
        
        # Determine target worker count based on load
        if load_factor < 0.3:
            # Low load - use minimum workers
            target = self.worker_pool.min_workers
        elif load_factor < 0.7:
            # Medium load - scale proportionally
            range_size = self.worker_pool.max_workers - self.worker_pool.min_workers
            target = self.worker_pool.min_workers + int(range_size * load_factor)
        else:
            # High load - use maximum workers
            target = self.worker_pool.max_workers
        
        # Consider error rate
        if load_metrics.error_rate > 0.2:
            # High error rate - reduce workers
            target = max(self.worker_pool.min_workers, target - 2)
        
        # Apply scaling
        if target != self.worker_pool.current_workers:
            self.worker_pool.scale(target)
            logger.info(f"Adjusted worker count to {target} (load: {load_factor:.2f})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics.
        
        Returns:
            Current metrics
        """
        self.metrics.update()
        self.metrics.active_jobs = self.worker_pool.get_active_count()
        
        return {
            "cpu_percent": self.metrics.cpu_percent,
            "memory_percent": self.metrics.memory_percent,
            "active_jobs": self.metrics.active_jobs,
            "queue_depth": self.metrics.queue_depth,
            "avg_processing_time": self.metrics.avg_processing_time,
            "error_rate": self.metrics.error_rate,
            "worker_count": self.worker_pool.current_workers,
            "worker_state": self.worker_pool.state.value,
            "last_update": self.metrics.last_update.isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self.worker_pool.stop()
        logger.info("Parallel processor shutdown complete")