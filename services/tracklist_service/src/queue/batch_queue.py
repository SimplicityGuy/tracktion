"""Batch job queue management for tracklist processing."""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib

import pika
from redis import Redis
from croniter import croniter  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""

    IMMEDIATE = 0
    NORMAL = 5
    LOW = 10


@dataclass
class Job:
    """Represents a single job in the batch."""

    id: str
    batch_id: str
    url: str
    priority: JobPriority
    user_id: str
    created_at: datetime
    status: str = "pending"
    retry_count: int = 0
    error: Optional[str] = None


class BatchJobQueue:
    """Manages batch job queuing with RabbitMQ."""

    def __init__(
        self,
        rabbitmq_host: str = "localhost",
        rabbitmq_port: int = 5672,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """Initialize batch job queue.

        Args:
            rabbitmq_host: RabbitMQ host address
            rabbitmq_port: RabbitMQ port number
            redis_host: Redis host for deduplication
            redis_port: Redis port number
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Setup RabbitMQ connection
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self._setup_rabbitmq()

    def _setup_rabbitmq(self) -> None:
        """Setup RabbitMQ connection and declare queues."""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.rabbitmq_host, port=self.rabbitmq_port)
            )
            self.channel = self.connection.channel()

            # Declare priority queues
            for priority in JobPriority:
                queue_name = f"batch_jobs_{priority.name.lower()}"
                self.channel.queue_declare(
                    queue=queue_name,
                    durable=True,
                    arguments={
                        "x-max-priority": 10,
                        "x-message-ttl": 86400000,  # 24 hours
                    },
                )

            # Declare dead letter queue
            self.channel.queue_declare(
                queue="batch_jobs_dlq",
                durable=True,
                arguments={"x-message-ttl": 604800000},  # 7 days
            )

            logger.info("RabbitMQ queues initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            raise

    def enqueue_batch(self, urls: List[str], priority: str = "normal", user_id: str = "") -> str:
        """Enqueue a batch of URLs for processing.

        Args:
            urls: List of URLs to process
            priority: Job priority level
            user_id: User ID submitting the batch

        Returns:
            Batch ID for tracking
        """
        batch_id = str(uuid.uuid4())
        job_priority = JobPriority[priority.upper()]

        # Deduplicate URLs within batch
        unique_urls = list(set(urls))

        jobs = []
        for url in unique_urls:
            job = Job(
                id=str(uuid.uuid4()),
                batch_id=batch_id,
                url=url,
                priority=job_priority,
                user_id=user_id,
                created_at=datetime.now(UTC),
            )
            jobs.append(job)

        # Deduplicate against existing jobs
        deduplicated_jobs = self.deduplicate_jobs(jobs)

        # Store batch metadata in Redis
        batch_metadata = {
            "batch_id": batch_id,
            "user_id": user_id,
            "total_jobs": len(deduplicated_jobs),
            "priority": priority,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "queued",
        }
        # Convert to proper types for Redis
        redis_data: Dict[str, Any] = {k: str(v) for k, v in batch_metadata.items()}
        self.redis.hset(f"batch:{batch_id}", mapping=redis_data)  # type: ignore[arg-type]
        self.redis.expire(f"batch:{batch_id}", 86400)  # 24 hour TTL

        # Enqueue jobs
        queue_name = f"batch_jobs_{job_priority.name.lower()}"
        for job in deduplicated_jobs:
            message = json.dumps(asdict(job), default=str)
            if self.channel:
                self.channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        priority=job_priority.value,
                    ),
                )

            # Track job in Redis
            job_dict = asdict(job)
            # Convert to proper types for Redis
            redis_job_data: Dict[str, Any] = {k: str(v) for k, v in job_dict.items()}
            self.redis.hset(f"job:{job.id}", mapping=redis_job_data)  # type: ignore[arg-type]
            self.redis.sadd(f"batch:{batch_id}:jobs", job.id)

        logger.info(f"Batch {batch_id} enqueued with {len(deduplicated_jobs)} jobs (priority: {priority})")

        return batch_id

    def deduplicate_jobs(self, jobs: List[Job]) -> List[Job]:
        """Remove duplicate jobs based on URL hash.

        Args:
            jobs: List of jobs to deduplicate

        Returns:
            List of unique jobs
        """
        deduplicated = []
        seen_hashes = set()

        for job in jobs:
            # Create hash of URL for deduplication
            url_hash = hashlib.sha256(job.url.encode()).hexdigest()

            # Check if URL is already being processed
            existing_job_id = self.redis.get(f"url_hash:{url_hash}")

            if existing_job_id:
                # Check if existing job is still active
                existing_job = self.redis.hgetall(f"job:{existing_job_id}")
                if existing_job:
                    status = existing_job.get("status")
                    if status and status in ["pending", "processing"]:
                        logger.debug(f"Skipping duplicate URL: {job.url}")
                        continue

            # Mark URL as being processed
            self.redis.setex(f"url_hash:{url_hash}", 3600, job.id)

            if url_hash not in seen_hashes:
                deduplicated.append(job)
                seen_hashes.add(url_hash)

        logger.info(f"Deduplicated {len(jobs)} jobs to {len(deduplicated)}")
        return deduplicated

    def schedule_batch(self, urls: List[str], cron_expression: str, user_id: str = "") -> str:
        """Schedule a batch for execution based on cron expression.

        Args:
            urls: List of URLs to process
            cron_expression: Cron expression for scheduling
            user_id: User ID scheduling the batch

        Returns:
            Schedule ID for tracking
        """
        schedule_id = str(uuid.uuid4())

        # Validate cron expression
        if not croniter.is_valid(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression}")

        # Calculate next execution time
        cron = croniter(cron_expression, datetime.now(UTC))
        next_run = cron.get_next(datetime)

        # Store schedule in Redis
        schedule_data = {
            "schedule_id": schedule_id,
            "urls": json.dumps(urls),
            "cron_expression": cron_expression,
            "user_id": user_id,
            "next_run": next_run.isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "active": "true",
        }

        # Convert to proper types for Redis
        redis_schedule_data: Dict[str, Any] = {k: str(v) for k, v in schedule_data.items()}
        self.redis.hset(f"schedule:{schedule_id}", mapping=redis_schedule_data)  # type: ignore[arg-type]
        self.redis.zadd("scheduled_batches", {schedule_id: next_run.timestamp()})

        logger.info(f"Batch scheduled with ID {schedule_id}, next run: {next_run}")

        return schedule_id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch job.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status information
        """
        batch_meta = self.redis.hgetall(f"batch:{batch_id}")
        if not batch_meta:
            return {"error": "Batch not found"}

        # Get job statuses
        job_ids = self.redis.smembers(f"batch:{batch_id}:jobs")
        jobs_status = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}

        if isinstance(job_ids, set):
            for job_id in job_ids:
                job_data = self.redis.hgetall(f"job:{job_id}")
                if job_data:
                    # Convert bytes to strings if needed
                    if isinstance(job_data, dict):
                        # Handle both bytes and string keys
                        status_value = None
                        if b"status" in job_data:
                            status_value = job_data[b"status"]
                        elif "status" in job_data:
                            status_value = job_data["status"]
                        else:
                            status_value = "pending"

                        if isinstance(status_value, bytes):
                            status_value = status_value.decode()
                        status = str(status_value)
                    else:
                        status = "pending"
                    jobs_status[status] = jobs_status.get(status, 0) + 1

            if isinstance(batch_meta, dict):
                batch_meta["jobs_status"] = str(jobs_status)
                batch_meta["progress_percentage"] = str(
                    (jobs_status["completed"] / len(job_ids)) * 100 if job_ids else 0
                )

        return dict(batch_meta)

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch job.

        Args:
            batch_id: Batch identifier

        Returns:
            Success status
        """
        batch_meta = self.redis.hgetall(f"batch:{batch_id}")
        if not batch_meta:
            return False

        # Update batch status
        self.redis.hset(f"batch:{batch_id}", "status", "cancelled")

        # Cancel pending jobs
        job_ids = self.redis.smembers(f"batch:{batch_id}:jobs")
        for job_id in job_ids:
            job_data = self.redis.hgetall(f"job:{job_id}")
            if job_data and job_data.get("status") == "pending":
                self.redis.hset(f"job:{job_id}", "status", "cancelled")

        logger.info(f"Batch {batch_id} cancelled")
        return True

    def close(self) -> None:
        """Close connections."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        self.redis.close()
        logger.info("Connections closed")
