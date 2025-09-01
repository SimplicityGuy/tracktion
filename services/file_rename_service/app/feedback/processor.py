"""Feedback processor for handling user feedback and triggering learning updates."""

import asyncio
import contextlib
import logging
import os
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

import psutil

from services.file_rename_service.app.feedback.models import (
    Feedback,
    FeedbackAction,
    FeedbackBatch,
)
from services.file_rename_service.app.feedback.storage import FeedbackStorage

logger = logging.getLogger(__name__)


class BackpressureStrategy(Enum):
    """Strategy for handling backpressure when queue is full."""

    DROP_OLDEST = "drop_oldest"
    REJECT_NEW = "reject_new"


class BackpressureError(Exception):
    """Raised when backpressure limit is exceeded and strategy is REJECT_NEW."""


class FeedbackProcessor:
    """Process and aggregate feedback for model learning."""

    def __init__(
        self,
        storage: FeedbackStorage,
        batch_size: int = 100,
        batch_timeout_seconds: int = 300,
        retrain_threshold: int = 1000,
        max_pending_size: int = 10000,
        backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
        memory_warning_threshold_mb: float = 100.0,
    ):
        """Initialize feedback processor.

        Args:
            storage: Feedback storage backend
            batch_size: Minimum batch size for processing
            batch_timeout_seconds: Max time to wait before processing partial batch
            retrain_threshold: Feedback count threshold for triggering retraining
            max_pending_size: Maximum number of pending feedback items
            backpressure_strategy: Strategy when pending queue is full
            memory_warning_threshold_mb: Memory usage warning threshold in MB
        """
        self.storage = storage
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.retrain_threshold = retrain_threshold
        self.max_pending_size = max_pending_size
        self.backpressure_strategy = backpressure_strategy
        self.memory_warning_threshold_mb = memory_warning_threshold_mb

        self._pending_feedback: list[Feedback] = []
        self._last_batch_time = datetime.now(UTC)
        self._total_processed = 0
        self._dropped_items = 0
        self._rejected_items = 0

        # Locks for thread safety
        self._processing_lock = asyncio.Lock()
        self._counter_lock = asyncio.Lock()  # Lock for counter updates
        self._list_lock = asyncio.Lock()  # Lock for pending feedback list

        # Resource monitoring
        self._shutdown_event = asyncio.Event()
        self._monitoring_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        """Start resource monitoring task."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_resources())
            logger.info("Resource monitoring started")

    async def _monitor_resources(self) -> None:
        """Monitor resource usage and log warnings."""
        while not self._shutdown_event.is_set():
            try:
                # Get memory usage
                memory_mb = self._get_memory_usage_mb()

                # Check if memory usage exceeds threshold
                if memory_mb > self.memory_warning_threshold_mb:
                    async with self._list_lock:
                        pending_count = len(self._pending_feedback)
                    logger.warning(
                        f"High memory usage detected: {memory_mb:.1f}MB "
                        f"(threshold: {self.memory_warning_threshold_mb:.1f}MB), "
                        f"Pending items: {pending_count}"
                    )

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            # Get memory info for current process
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / 1024 / 1024)
        except Exception:
            return 0.0

    async def _handle_backpressure(self, new_feedback: Feedback) -> bool:
        """Handle backpressure when pending queue is full.

        Args:
            new_feedback: New feedback item to add

        Returns:
            True if feedback was accepted, False if rejected/dropped

        Raises:
            BackpressureError: If strategy is REJECT_NEW and queue is full
        """
        async with self._list_lock:
            current_size = len(self._pending_feedback)

            if current_size < self.max_pending_size:
                # Queue has space
                self._pending_feedback.append(new_feedback)
                return True

            # Queue is full, apply backpressure strategy
            if self.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
                # Drop oldest item and add new one
                dropped_item = self._pending_feedback.pop(0)
                self._pending_feedback.append(new_feedback)

                # Update metrics
                async with self._counter_lock:
                    self._dropped_items += 1
                    dropped_count = self._dropped_items

                logger.warning(
                    f"Dropped oldest feedback item (ID: {dropped_item.id}) "
                    f"due to backpressure. Total dropped: {dropped_count}"
                )
                return True

            if self.backpressure_strategy == BackpressureStrategy.REJECT_NEW:
                # Reject new item
                async with self._counter_lock:
                    self._rejected_items += 1
                    rejected_count = self._rejected_items

                logger.warning(
                    f"Rejected new feedback item (ID: {new_feedback.id}) "
                    f"due to backpressure. Total rejected: {rejected_count}"
                )
                raise BackpressureError(
                    f"Feedback queue is full ({current_size}/{self.max_pending_size}). Cannot accept new feedback."
                )

        return False

    async def submit_feedback(
        self,
        proposal_id: str,
        original_filename: str,
        proposed_filename: str,
        user_action: FeedbackAction,
        confidence_score: float,
        model_version: str,
        user_filename: str | None = None,
        context_metadata: dict | None = None,
    ) -> Feedback:
        """Submit new feedback for processing.

        Args:
            proposal_id: ID of the rename proposal
            original_filename: Original file name
            proposed_filename: System proposed name
            user_action: User's action (approved/rejected/modified)
            confidence_score: Model confidence for proposal
            model_version: Version of model that generated proposal
            user_filename: User-provided filename if modified
            context_metadata: Additional context information

        Returns:
            Created feedback object

        Raises:
            ValueError: If validation fails
        """
        start_time = datetime.now(UTC)

        # Create feedback object
        feedback = Feedback(
            id=str(uuid4()),
            proposal_id=proposal_id,
            original_filename=original_filename,
            proposed_filename=proposed_filename,
            user_action=user_action,
            user_filename=user_filename,
            confidence_score=confidence_score,
            timestamp=datetime.now(UTC),
            model_version=model_version,
            context_metadata=context_metadata or {},
        )

        # Calculate processing time
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
        feedback.processing_time_ms = processing_time

        # Store feedback
        await self.storage.store_feedback(feedback)

        # Add to pending batch with backpressure handling
        feedback_accepted = await self._handle_backpressure(feedback)
        if not feedback_accepted:
            # This shouldn't happen since _handle_backpressure either accepts or raises
            logger.error(f"Unexpected backpressure handling result for feedback {feedback.id}")
            return feedback

        # Check if batch should be processed (under processing lock)
        async with self._processing_lock:
            should_process = await self._should_process_batch()
            if should_process:
                await self._process_batch()

        # Log feedback submission
        logger.info(
            f"Feedback submitted: {feedback.id} - Action: {user_action}, Processing time: {processing_time:.2f}ms"
        )

        # Check if retraining should be triggered
        if await self._should_trigger_retrain():
            await self._trigger_retrain()

        return feedback

    async def _should_process_batch(self) -> bool:
        """Check if pending batch should be processed.

        Returns:
            True if batch should be processed
        """
        # Thread-safe access to pending feedback list
        async with self._list_lock:
            pending_count = len(self._pending_feedback)
            has_pending = bool(self._pending_feedback)

        # Check batch size
        if pending_count >= self.batch_size:
            return True

        # Check timeout
        time_since_last = (datetime.now(UTC) - self._last_batch_time).total_seconds()
        return has_pending and time_since_last >= self.batch_timeout_seconds

    async def _process_batch(self) -> None:
        """Process pending feedback batch."""
        # Thread-safe check and copy of pending feedback
        async with self._list_lock:
            if not self._pending_feedback:
                return

            # Create a copy of pending feedback for processing
            batch_feedback = self._pending_feedback.copy()

        try:
            # Create batch
            batch = FeedbackBatch(
                feedbacks=batch_feedback,
                batch_id=str(uuid4()),
                created_at=datetime.now(UTC),
            )

            # Store batch
            await self.storage.store_batch(batch)

            # Process feedback for learning
            await self._apply_learning_updates(batch)

            # Mark batch as processed
            batch.mark_processed()
            await self.storage.update_batch(batch)

            # Thread-safe update of counters and list clearing
            async with self._counter_lock:
                self._total_processed += len(batch_feedback)
                current_total = self._total_processed

            async with self._list_lock:
                # Only clear items that were actually processed
                self._pending_feedback = self._pending_feedback[len(batch_feedback) :]
                self._last_batch_time = datetime.now(UTC)

            logger.info(
                f"Processed feedback batch {batch.batch_id} with "
                f"{len(batch.feedbacks)} items. Total processed: {current_total}"
            )

        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
            # Keep feedback for retry
            raise

    async def _apply_learning_updates(self, batch: FeedbackBatch) -> None:
        """Apply learning updates from feedback batch.

        Args:
            batch: Feedback batch to process
        """
        # Calculate batch statistics
        stats = self._calculate_batch_stats(batch.feedbacks)

        # Store learning metrics
        await self.storage.update_learning_metrics(stats)

        # Log learning update
        logger.info(
            f"Learning update applied - Approval: {stats['approval_rate']:.2%}, "
            f"Rejection: {stats['rejection_rate']:.2%}, "
            f"Modification: {stats['modification_rate']:.2%}"
        )

    def _calculate_batch_stats(self, feedbacks: list[Feedback]) -> dict:
        """Calculate statistics for feedback batch.

        Args:
            feedbacks: List of feedback items

        Returns:
            Dictionary of statistics
        """
        total = len(feedbacks)
        if total == 0:
            return {
                "total": 0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
                "modification_rate": 0.0,
                "avg_confidence": 0.0,
            }

        approved = sum(1 for f in feedbacks if f.user_action == FeedbackAction.APPROVED)
        rejected = sum(1 for f in feedbacks if f.user_action == FeedbackAction.REJECTED)
        modified = sum(1 for f in feedbacks if f.user_action == FeedbackAction.MODIFIED)
        avg_confidence = sum(f.confidence_score for f in feedbacks) / total

        return {
            "total": total,
            "approval_rate": approved / total,
            "rejection_rate": rejected / total,
            "modification_rate": modified / total,
            "avg_confidence": avg_confidence,
            "model_versions": list({f.model_version for f in feedbacks}),
        }

    async def _should_trigger_retrain(self) -> bool:
        """Check if model retraining should be triggered.

        Returns:
            True if retraining should be triggered
        """
        # Get total feedback count since last retrain
        feedback_count = await self.storage.get_feedback_count_since_retrain()

        # Check against threshold
        return bool(feedback_count >= self.retrain_threshold)

    async def _trigger_retrain(self) -> None:
        """Trigger model retraining."""
        try:
            # Thread-safe access to total processed count
            async with self._counter_lock:
                total_processed = self._total_processed

            logger.info(f"Triggering model retraining. Total feedback: {total_processed}")

            # Mark retrain triggered in storage
            await self.storage.mark_retrain_triggered()

            # Here we would trigger the actual retraining process
            # This could be via a message queue, API call, or direct invocation
            # For now, we'll just log it
            logger.info("Model retraining triggered successfully")

        except Exception as e:
            logger.error(f"Failed to trigger model retraining: {e}")
            raise

    async def get_feedback_stats(
        self,
        model_version: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """Get feedback statistics.

        Args:
            model_version: Filter by model version
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Dictionary of statistics
        """
        feedbacks = await self.storage.get_feedbacks(
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
        )

        return self._calculate_batch_stats(feedbacks)

    async def get_resource_stats(self) -> dict:
        """Get resource usage and backpressure statistics.

        Returns:
            Dictionary containing resource statistics
        """
        async with self._list_lock:
            pending_count = len(self._pending_feedback)
            queue_utilization = pending_count / self.max_pending_size

        async with self._counter_lock:
            dropped_count = self._dropped_items
            rejected_count = self._rejected_items
            total_processed = self._total_processed

        memory_mb = self._get_memory_usage_mb()

        return {
            "pending_feedback_count": pending_count,
            "max_pending_size": self.max_pending_size,
            "queue_utilization": queue_utilization,
            "queue_utilization_percent": queue_utilization * 100,
            "total_processed": total_processed,
            "dropped_items": dropped_count,
            "rejected_items": rejected_count,
            "memory_usage_mb": memory_mb,
            "memory_threshold_mb": self.memory_warning_threshold_mb,
            "backpressure_strategy": self.backpressure_strategy.value,
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
        }

    async def force_batch_processing(self) -> None:
        """Force processing of pending feedback batch."""
        async with self._processing_lock:
            # Check if there's pending feedback to process
            async with self._list_lock:
                has_pending = bool(self._pending_feedback)

            if has_pending:
                await self._process_batch()
                logger.info("Forced batch processing completed")

    async def cleanup(self) -> None:
        """Cleanup processor resources and shutdown gracefully."""
        logger.info("Starting FeedbackProcessor cleanup...")

        try:
            # Signal shutdown to monitoring task
            self._shutdown_event.set()

            # Cancel and wait for monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitoring_task
                logger.info("Resource monitoring stopped")

            # Process any remaining feedback
            await self.force_batch_processing()

            # Clear pending feedback list
            async with self._list_lock:
                pending_count = len(self._pending_feedback)
                self._pending_feedback.clear()
                if pending_count > 0:
                    logger.warning(f"Cleared {pending_count} pending feedback items during shutdown")

            # Close storage connection
            if hasattr(self.storage, "close"):
                await self.storage.close()
                logger.info("Storage connection closed")

            # Log final statistics
            stats = await self.get_resource_stats()
            logger.info(
                f"FeedbackProcessor cleanup completed. "
                f"Total processed: {stats['total_processed']}, "
                f"Dropped: {stats['dropped_items']}, "
                f"Rejected: {stats['rejected_items']}"
            )

        except Exception as e:
            logger.error(f"Error during FeedbackProcessor cleanup: {e}")
            raise
