"""Storage backend for feedback data."""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from redis import asyncio as aioredis

from services.file_rename_service.app.feedback.models import Feedback, FeedbackBatch, LearningMetrics

logger = logging.getLogger(__name__)


class FeedbackStorage:
    """Storage backend for feedback and learning metrics."""

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        cache_ttl_seconds: int = 3600,
    ):
        """Initialize storage backend.

        Args:
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis connection URL
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl_seconds
        self._pg_pool: asyncpg.Pool | None = None
        self._redis_client: aioredis.Redis | None = None
        self._last_retrain_key = "feedback:last_retrain"
        self._feedback_count_key = "feedback:count"
        self._metrics_cache_key = "feedback:metrics"

    async def initialize(self) -> None:
        """Initialize database connections."""
        # Create PostgreSQL connection pool
        self._pg_pool = await asyncpg.create_pool(
            self.postgres_dsn,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )

        # Create Redis client
        self._redis_client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Create tables if not exist
        await self._create_tables()

        logger.info("Feedback storage initialized")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")
        async with self._pg_pool.acquire() as conn:
            # Create feedback table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY,
                    proposal_id VARCHAR(255) NOT NULL,
                    original_filename TEXT NOT NULL,
                    proposed_filename TEXT NOT NULL,
                    user_action VARCHAR(50) NOT NULL,
                    user_filename TEXT,
                    confidence_score FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    processing_time_ms FLOAT,
                    context_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_feedback_proposal (proposal_id),
                    INDEX idx_feedback_timestamp (timestamp),
                    INDEX idx_feedback_model (model_version)
                )
            """
            )

            # Create feedback batches table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_batches (
                    batch_id UUID PRIMARY KEY,
                    feedbacks JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    processed_at TIMESTAMP,
                    error TEXT,
                    INDEX idx_batch_created (created_at),
                    INDEX idx_batch_processed (processed)
                )
            """
            )

            # Create learning metrics table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id SERIAL PRIMARY KEY,
                    model_version VARCHAR(100) NOT NULL,
                    total_feedback INTEGER DEFAULT 0,
                    approval_rate FLOAT DEFAULT 0.0,
                    rejection_rate FLOAT DEFAULT 0.0,
                    modification_rate FLOAT DEFAULT 0.0,
                    accuracy_trend JSONB,
                    last_retrained TIMESTAMP,
                    next_retrain_at TIMESTAMP,
                    performance_metrics JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_metrics_model (model_version),
                    INDEX idx_metrics_updated (updated_at)
                )
            """
            )

    async def store_feedback(self, feedback: Feedback) -> None:
        """Store feedback in database.

        Args:
            feedback: Feedback object to store
        """
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pg_pool.acquire() as conn, conn.transaction():
            try:
                # Insert feedback into database
                await conn.execute(
                    """
                        INSERT INTO feedback (
                            id, proposal_id, original_filename, proposed_filename,
                            user_action, user_filename, confidence_score, timestamp,
                            model_version, processing_time_ms, context_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                    feedback.id,
                    feedback.proposal_id,
                    feedback.original_filename,
                    feedback.proposed_filename,
                    feedback.user_action.value,
                    feedback.user_filename,
                    feedback.confidence_score,
                    feedback.timestamp,
                    feedback.model_version,
                    feedback.processing_time_ms,
                    json.dumps(feedback.context_metadata),
                )

                # Update cache only after successful DB insert
                if self._redis_client is not None:
                    await self._redis_client.incr(self._feedback_count_key)
                    # Invalidate metrics cache
                    await self._redis_client.delete(self._metrics_cache_key)

            except Exception as e:
                logger.error(f"Failed to store feedback {feedback.id}: {e}")
                # Transaction will be automatically rolled back
                raise

    async def store_batch(self, batch: FeedbackBatch) -> None:
        """Store feedback batch.

        Args:
            batch: Feedback batch to store
        """
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pg_pool.acquire() as conn, conn.transaction():
            try:
                await conn.execute(
                    """
                        INSERT INTO feedback_batches (
                            batch_id, feedbacks, created_at, processed, processed_at, error
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                    batch.batch_id,
                    json.dumps([f.dict() for f in batch.feedbacks]),
                    batch.created_at,
                    batch.processed,
                    batch.processed_at,
                    batch.error,
                )
            except Exception as e:
                logger.error(f"Failed to store batch {batch.batch_id}: {e}")
                # Transaction will be automatically rolled back
                raise

    async def update_batch(self, batch: FeedbackBatch) -> None:
        """Update feedback batch.

        Args:
            batch: Feedback batch to update
        """
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pg_pool.acquire() as conn, conn.transaction():
            try:
                await conn.execute(
                    """
                        UPDATE feedback_batches
                        SET processed = $2, processed_at = $3, error = $4
                        WHERE batch_id = $1
                        """,
                    batch.batch_id,
                    batch.processed,
                    batch.processed_at,
                    batch.error,
                )
            except Exception as e:
                logger.error(f"Failed to update batch {batch.batch_id}: {e}")
                # Transaction will be automatically rolled back
                raise

    async def get_batch_for_learning(self, batch_size: int = 100, processed: bool = False) -> list[FeedbackBatch]:
        """Get batch of feedback data for learning.

        Args:
            batch_size: Maximum number of batches to return
            processed: Filter by processed status

        Returns:
            List of feedback batches
        """
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self._pg_pool.acquire() as conn, conn.transaction(isolation="repeatable_read"):
                rows = await conn.fetch(
                    """
                        SELECT * FROM feedback_batches
                        WHERE processed = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                    processed,
                    batch_size,
                )

            batches = []
            for row in rows:
                batch_dict = dict(row)
                # Parse feedbacks JSON
                feedbacks_data = json.loads(batch_dict["feedbacks"])
                batch_dict["feedbacks"] = [Feedback(**f) for f in feedbacks_data]
                batches.append(FeedbackBatch(**batch_dict))

            return batches

        except Exception as e:
            logger.error(f"Failed to get batch for learning: {e}")
            return []

    async def get_feedbacks(
        self,
        model_version: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[Feedback]:
        """Get feedbacks with optional filtering.

        Args:
            model_version: Filter by model version
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results

        Returns:
            List of feedback objects
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []
        param_count = 0

        if model_version:
            param_count += 1
            query += f" AND model_version = ${param_count}"
            params.append(model_version)

        if start_date:
            param_count += 1
            query += f" AND timestamp >= ${param_count}"
            params.append(start_date)

        if end_date:
            param_count += 1
            query += f" AND timestamp <= ${param_count}"
            params.append(end_date)

        param_count += 1
        query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
        params.append(limit)

        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")
        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        feedbacks = []
        for row in rows:
            feedback_dict = dict(row)
            feedback_dict["context_metadata"] = json.loads(feedback_dict.get("context_metadata", "{}"))
            feedbacks.append(Feedback(**feedback_dict))

        return feedbacks

    async def update_learning_metrics(self, stats: dict) -> None:
        """Update learning metrics.

        Args:
            stats: Statistics dictionary
        """
        model_versions = stats.get("model_versions", [])

        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pg_pool.acquire() as conn, conn.transaction():
            try:
                for model_version in model_versions:
                    # Check if metrics exist for this model
                    existing = await conn.fetchrow(
                        "SELECT id FROM learning_metrics WHERE model_version = $1",
                        model_version,
                    )

                    if existing:
                        # Update existing metrics
                        await conn.execute(
                            """
                                UPDATE learning_metrics
                                SET total_feedback = total_feedback + $2,
                                    approval_rate = $3,
                                    rejection_rate = $4,
                                    modification_rate = $5,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE model_version = $1
                                """,
                            model_version,
                            stats["total"],
                            stats["approval_rate"],
                            stats["rejection_rate"],
                            stats["modification_rate"],
                        )
                    else:
                        # Insert new metrics
                        await conn.execute(
                            """
                                INSERT INTO learning_metrics (
                                    model_version, total_feedback, approval_rate,
                                    rejection_rate, modification_rate
                                ) VALUES ($1, $2, $3, $4, $5)
                                """,
                            model_version,
                            stats["total"],
                            stats["approval_rate"],
                            stats["rejection_rate"],
                            stats["modification_rate"],
                        )

                # Invalidate metrics cache only after all DB operations succeed
                if self._redis_client is not None:
                    await self._redis_client.delete(self._metrics_cache_key)

            except Exception as e:
                logger.error(f"Failed to update learning metrics: {e}")
                # Transaction will be automatically rolled back
                raise

    async def get_feedback_count_since_retrain(self) -> int:
        """Get feedback count since last retrain.

        Returns:
            Number of feedbacks since last retrain
        """
        if self._redis_client is None:
            raise RuntimeError("Redis client not initialized")
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        # Try to get from cache first
        count = await self._redis_client.get(self._feedback_count_key)
        if count:
            return int(count)

        try:
            # Get from database with transaction isolation
            last_retrain_str = await self._redis_client.get(self._last_retrain_key)

            async with self._pg_pool.acquire() as conn, conn.transaction(isolation="repeatable_read"):
                if last_retrain_str:
                    last_retrain = datetime.fromisoformat(last_retrain_str)
                    result = await conn.fetchval(
                        "SELECT COUNT(*) FROM feedback WHERE timestamp > $1",
                        last_retrain,
                    )
                else:
                    result = await conn.fetchval("SELECT COUNT(*) FROM feedback")

            # Update cache only after successful DB query
            count_value = int(result) if result is not None else 0
            await self._redis_client.set(self._feedback_count_key, str(count_value), ex=self.cache_ttl)

            return count_value

        except Exception as e:
            logger.error(f"Failed to get feedback count since retrain: {e}")
            # Return 0 as safe fallback if we can't get an accurate count
            return 0

    async def mark_retrain_triggered(self) -> None:
        """Mark that retraining has been triggered."""
        now = datetime.now(UTC)

        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pg_pool.acquire() as conn, conn.transaction():
            try:
                # Update database first
                await conn.execute(
                    """
                        UPDATE learning_metrics
                        SET last_retrained = $1, next_retrain_at = $2
                        WHERE id = (SELECT id FROM learning_metrics ORDER BY updated_at DESC LIMIT 1)
                        """,
                    now,
                    now + timedelta(hours=24),  # Next retrain in 24 hours
                )

                # Update Redis cache only after successful DB update
                if self._redis_client is not None:
                    await self._redis_client.set(self._last_retrain_key, now.isoformat())
                    await self._redis_client.set(self._feedback_count_key, "0")

            except Exception as e:
                logger.error(f"Failed to mark retrain as triggered: {e}")
                # Transaction will be automatically rolled back
                raise

    async def get_learning_metrics(self, model_version: str | None = None) -> LearningMetrics | None:
        """Get learning metrics.

        Args:
            model_version: Model version to get metrics for

        Returns:
            Learning metrics or None
        """
        if self._redis_client is None:
            raise RuntimeError("Redis client not initialized")
        if self._pg_pool is None:
            raise RuntimeError("Database pool not initialized")

        # Try cache first
        cache_key = f"{self._metrics_cache_key}:{model_version or 'latest'}"
        cached = await self._redis_client.get(cache_key)
        if cached:
            return LearningMetrics(**json.loads(cached))

        try:
            # Get from database with consistent read
            async with self._pg_pool.acquire() as conn, conn.transaction(isolation="repeatable_read"):
                if model_version:
                    row = await conn.fetchrow(
                        "SELECT * FROM learning_metrics WHERE model_version = $1",
                        model_version,
                    )
                else:
                    row = await conn.fetchrow("SELECT * FROM learning_metrics ORDER BY updated_at DESC LIMIT 1")

            if not row:
                return None

            metrics_dict = dict(row)
            metrics_dict["accuracy_trend"] = json.loads(metrics_dict.get("accuracy_trend", "[]"))
            metrics_dict["performance_metrics"] = json.loads(metrics_dict.get("performance_metrics", "{}"))

            metrics = LearningMetrics(**metrics_dict)

            # Cache result only after successful data processing
            await self._redis_client.set(cache_key, metrics.json(), ex=self.cache_ttl)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get learning metrics for model {model_version}: {e}")
            return None

    async def close(self) -> None:
        """Close database connections."""
        if self._pg_pool:
            await self._pg_pool.close()

        if self._redis_client:
            await self._redis_client.close()

        logger.info("Feedback storage closed")
