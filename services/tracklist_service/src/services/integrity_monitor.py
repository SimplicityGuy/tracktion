"""Service for monitoring database integrity and orphaned records."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional
from sqlalchemy.orm import Session

from ..utils.integrity_validator import IntegrityValidator

logger = logging.getLogger(__name__)


class IntegrityMonitor:
    """Monitors database integrity and alerts on orphaned records."""

    def __init__(
        self, session_factory: Callable[[], Session], check_interval_minutes: int = 60, auto_clean: bool = False
    ):
        """Initialize the integrity monitor.

        Args:
            session_factory: Factory function to create database sessions
            check_interval_minutes: How often to check for orphaned records (default: 60)
            auto_clean: Whether to automatically clean orphaned records (default: False)
        """
        self.session_factory = session_factory
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.auto_clean = auto_clean
        self.last_check: Optional[datetime] = None
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the integrity monitoring service."""
        if self.is_running:
            logger.warning("Integrity monitor is already running")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Started integrity monitor (interval: {self.check_interval.total_seconds() / 60} minutes, "
            f"auto_clean: {self.auto_clean})"
        )

    async def stop(self) -> None:
        """Stop the integrity monitoring service."""
        if not self.is_running:
            return

        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped integrity monitor")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._perform_check()
                await asyncio.sleep(self.check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in integrity monitor loop: {e}")
                # Continue monitoring even if there's an error
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def _perform_check(self) -> None:
        """Perform an integrity check."""
        session = self.session_factory()
        try:
            validator = IntegrityValidator(session)
            self.last_check = datetime.utcnow()

            # Check for orphaned records
            orphaned = validator.check_orphaned_records()
            total_orphaned = sum(orphaned.values())

            if total_orphaned > 0:
                logger.warning(f"Integrity check found {total_orphaned} orphaned records: {orphaned}")

                if self.auto_clean:
                    logger.info("Auto-cleaning orphaned records")
                    cleaned = validator.clean_orphaned_records(dry_run=False)
                    total_cleaned = sum(cleaned.values())
                    logger.info(f"Auto-cleaned {total_cleaned} orphaned records: {cleaned}")
                else:
                    logger.info("Auto-clean is disabled. Run integrity validation script to clean manually.")
            else:
                logger.debug("Integrity check passed - no orphaned records found")

            # Validate foreign keys
            fk_issues = [
                (name, table)
                for name, table, is_cascade in validator.validate_foreign_keys()
                if not is_cascade and table in ["metadata", "tracklists", "rename_proposals"]
            ]

            if fk_issues:
                logger.error(f"Found {len(fk_issues)} foreign key constraints without CASCADE: {fk_issues}")

            # Validate indexes
            missing_indexes = [(name, table) for name, table, exists in validator.validate_indexes() if not exists]

            if missing_indexes:
                logger.warning(f"Found {len(missing_indexes)} missing indexes: {missing_indexes}")

        except Exception as e:
            logger.error(f"Error performing integrity check: {e}")
        finally:
            session.close()

    async def check_now(self) -> Dict[str, Any]:
        """Perform an immediate integrity check.

        Returns:
            Dictionary with check results
        """
        session = self.session_factory()
        try:
            validator = IntegrityValidator(session)
            results = validator.run_full_validation()
            self.last_check = datetime.utcnow()
            return results
        finally:
            session.close()

    def get_status(self) -> Dict[str, Any]:
        """Get the current monitor status.

        Returns:
            Dictionary with monitor status information
        """
        return {
            "is_running": self.is_running,
            "check_interval_minutes": self.check_interval.total_seconds() / 60,
            "auto_clean": self.auto_clean,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "next_check": (
                (self.last_check + self.check_interval).isoformat() if self.last_check and self.is_running else None
            ),
        }


# Singleton instance
_monitor_instance: Optional[IntegrityMonitor] = None


def get_integrity_monitor(
    session_factory: Optional[Callable[[], Session]] = None, check_interval_minutes: int = 60, auto_clean: bool = False
) -> IntegrityMonitor:
    """Get or create the singleton integrity monitor instance.

    Args:
        session_factory: Factory function to create database sessions
        check_interval_minutes: How often to check for orphaned records
        auto_clean: Whether to automatically clean orphaned records

    Returns:
        The integrity monitor instance
    """
    global _monitor_instance

    if _monitor_instance is None:
        if session_factory is None:
            raise ValueError("session_factory is required to create monitor instance")
        _monitor_instance = IntegrityMonitor(session_factory, check_interval_minutes, auto_clean)

    return _monitor_instance
