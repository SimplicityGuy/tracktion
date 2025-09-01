"""Scheduled tasks for quota management."""

import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .manager import QuotaManager
from .models import QuotaType

logger = logging.getLogger(__name__)


class QuotaScheduler:
    """Manages scheduled quota reset tasks."""

    def __init__(self, quota_manager: QuotaManager):
        """Initialize quota scheduler.

        Args:
            quota_manager: QuotaManager instance for executing resets
        """
        self.quota_manager = quota_manager
        self.scheduler = AsyncIOScheduler()
        self._running = False

    async def start(self) -> None:
        """Start the scheduler and add quota reset jobs."""
        if self._running:
            logger.warning("Quota scheduler is already running")
            return

        # Daily quota reset at midnight UTC
        self.scheduler.add_job(
            func=self._daily_quota_reset,
            trigger=CronTrigger(hour=0, minute=0, second=0),
            id="daily_quota_reset",
            name="Daily Quota Reset",
            replace_existing=True,
        )

        # Monthly quota reset at midnight on the 1st of each month UTC
        self.scheduler.add_job(
            func=self._monthly_quota_reset,
            trigger=CronTrigger(day=1, hour=0, minute=0, second=0),
            id="monthly_quota_reset",
            name="Monthly Quota Reset",
            replace_existing=True,
        )

        # Quota cleanup job - remove old alert records
        self.scheduler.add_job(
            func=self._cleanup_old_records,
            trigger=CronTrigger(hour=2, minute=0, second=0),  # Daily at 2 AM UTC
            id="quota_cleanup",
            name="Quota Cleanup",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        logger.info("Quota scheduler started with daily and monthly reset jobs")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            logger.warning("Quota scheduler is not running")
            return

        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Quota scheduler stopped")

    async def trigger_manual_reset(self, quota_type: QuotaType) -> int:
        """Manually trigger a quota reset.

        Args:
            quota_type: Type of quota to reset

        Returns:
            Number of users whose quotas were reset
        """
        logger.info(f"Manual {quota_type.value} quota reset triggered")
        return await self.quota_manager.reset_quotas(quota_type)

    async def _daily_quota_reset(self) -> None:
        """Scheduled daily quota reset job."""
        try:
            reset_count = await self.quota_manager.reset_quotas(QuotaType.DAILY)
            logger.info(f"Daily quota reset completed for {reset_count} users")
        except Exception as e:
            logger.error(f"Daily quota reset failed: {e}")

    async def _monthly_quota_reset(self) -> None:
        """Scheduled monthly quota reset job."""
        try:
            reset_count = await self.quota_manager.reset_quotas(QuotaType.MONTHLY)
            logger.info(f"Monthly quota reset completed for {reset_count} users")
        except Exception as e:
            logger.error(f"Monthly quota reset failed: {e}")

    async def _cleanup_old_records(self) -> None:
        """Clean up old quota-related records."""
        try:
            # This would clean up old alert records, upgrade records, etc.
            # For now, we rely on Redis TTL expiration
            logger.debug("Quota cleanup job executed (relying on Redis TTL)")
        except Exception as e:
            logger.error(f"Quota cleanup failed: {e}")

    def get_job_status(self) -> dict[str, Any]:
        """Get status of scheduled jobs.

        Returns:
            Dictionary with job status information
        """
        if not self._running:
            return {"status": "stopped", "jobs": []}

        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run,
                    "trigger": str(job.trigger),
                }
            )

        return {
            "status": "running",
            "scheduler_state": "running" if self.scheduler.running else "stopped",
            "jobs": jobs,
        }
