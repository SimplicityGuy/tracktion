"""Main synchronization service orchestrating all sync operations."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.models.synchronization import (
    SyncConfiguration,
    SyncEvent,
)
from services.tracklist_service.src.services.tracklists_sync_service import TracklistsSyncService
from services.tracklist_service.src.services.conflict_resolution_service import ConflictResolutionService
from services.tracklist_service.src.services.cue_regeneration_service import CueRegenerationService
from services.tracklist_service.src.services.version_service import VersionService
from services.tracklist_service.src.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class SyncSource(Enum):
    """Available synchronization sources."""
    
    ALL = "all"
    ONETHOUSANDONE = "1001tracklists"
    MANUAL = "manual"
    API = "api"


class SyncStatus(Enum):
    """Synchronization status."""
    
    IDLE = "idle"
    CHECKING = "checking"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncFrequency(Enum):
    """Synchronization frequency options."""
    
    MANUAL = "manual"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    REALTIME = "realtime"


class SynchronizationService:
    """Main service orchestrating all synchronization operations."""

    def __init__(
        self,
        session: AsyncSession,
        tracklists_sync_service: Optional[TracklistsSyncService] = None,
        conflict_service: Optional[ConflictResolutionService] = None,
        cue_service: Optional[CueRegenerationService] = None,
        version_service: Optional[VersionService] = None,
        audit_service: Optional[AuditService] = None,
    ):
        """Initialize synchronization service.

        Args:
            session: Database session
            tracklists_sync_service: Service for 1001tracklists sync
            conflict_service: Service for conflict resolution
            cue_service: Service for CUE regeneration
            version_service: Service for version management
            audit_service: Service for audit logging
        """
        self.session = session
        self.tracklists_sync = tracklists_sync_service or TracklistsSyncService(session)
        self.conflict_service = conflict_service or ConflictResolutionService(session)
        self.cue_service = cue_service or CueRegenerationService(session)
        self.version_service = version_service or VersionService(session)
        self.audit_service = audit_service or AuditService(session)
        
        # Track active sync operations to prevent concurrent syncs
        self.active_syncs: Set[UUID] = set()
        
        # Scheduled sync tasks
        self.scheduled_tasks: Dict[UUID, asyncio.Task] = {}

    async def trigger_manual_sync(
        self,
        tracklist_id: UUID,
        source: SyncSource = SyncSource.ALL,
        force: bool = False,
        actor: str = "user",
    ) -> Dict[str, Any]:
        """Trigger manual synchronization for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            source: Source to sync from
            force: Force sync even if recently synced
            actor: Who triggered the sync

        Returns:
            Sync result with status and details
        """
        # Check if sync is already in progress
        if tracklist_id in self.active_syncs:
            return {
                "status": SyncStatus.FAILED.value,
                "error": "Sync already in progress for this tracklist",
                "tracklist_id": str(tracklist_id),
            }

        try:
            # Mark as active
            self.active_syncs.add(tracklist_id)
            
            # Get sync configuration
            config = await self._get_or_create_sync_config(tracklist_id)
            
            # Check if force sync is needed
            if not force and config.last_sync_at:
                time_since_sync = datetime.utcnow() - config.last_sync_at
                if time_since_sync < timedelta(minutes=5):
                    return {
                        "status": SyncStatus.COMPLETED.value,
                        "message": "Recently synced, skipping",
                        "last_sync": config.last_sync_at.isoformat(),
                        "tracklist_id": str(tracklist_id),
                    }

            # Create sync event
            sync_event = await self._create_sync_event(
                tracklist_id=tracklist_id,
                source=source.value,
                actor=actor,
            )

            # Perform sync based on source
            if source in [SyncSource.ALL, SyncSource.ONETHOUSANDONE]:
                result = await self._sync_from_1001tracklists(tracklist_id, sync_event, config)
            else:
                result = {
                    "status": SyncStatus.COMPLETED.value,
                    "message": f"Sync source {source.value} not implemented yet",
                }

            # Update sync event
            sync_event.status = "completed" if result.get("status") == SyncStatus.COMPLETED.value else "failed"
            sync_event.completed_at = datetime.utcnow()
            sync_event.result_data = result
            
            # Update config
            config.last_sync_at = datetime.utcnow()
            
            await self.session.commit()
            
            return result

        except Exception as e:
            logger.error(f"Failed to sync tracklist {tracklist_id}: {e}")
            return {
                "status": SyncStatus.FAILED.value,
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }
        finally:
            # Remove from active syncs
            self.active_syncs.discard(tracklist_id)

    async def _sync_from_1001tracklists(
        self,
        tracklist_id: UUID,
        sync_event: SyncEvent,
        config: SyncConfiguration,
    ) -> Dict[str, Any]:
        """Sync from 1001tracklists.

        Args:
            tracklist_id: ID of the tracklist
            sync_event: Current sync event
            config: Sync configuration

        Returns:
            Sync result
        """
        try:
            # Check for updates
            updates = await self.tracklists_sync.check_for_updates(tracklist_id)
            
            if not updates:
                return {
                    "status": SyncStatus.COMPLETED.value,
                    "message": "No updates available",
                    "tracklist_id": str(tracklist_id),
                }

            # Check if there are conflicts
            tracklist = await self.session.get(TracklistDB, tracklist_id)
            current_state = {"tracks": len(tracklist.tracks) if tracklist else 0}
            
            conflicts = await self.conflict_service.detect_conflicts(
                tracklist_id=tracklist_id,
                current_state=current_state,
                proposed_changes=updates["changes"],
            )

            if conflicts and not config.auto_resolve_conflicts:
                # Queue for manual resolution
                sync_event.status = "conflict"
                sync_event.conflict_data = {
                    "conflicts": conflicts,
                    "proposed_changes": updates["changes"],
                }
                
                # Prepare UI data
                ui_data = await self.conflict_service.prepare_conflict_ui_data(
                    tracklist_id=tracklist_id,
                    conflicts=conflicts,
                    current_state=current_state,
                    proposed_changes=updates["changes"],
                )
                
                return {
                    "status": SyncStatus.CONFLICT.value,
                    "conflicts": ui_data,
                    "tracklist_id": str(tracklist_id),
                }

            # Auto-resolve if configured
            if conflicts and config.auto_resolve_conflicts:
                resolutions = await self.conflict_service.auto_resolve_conflicts(
                    tracklist_id=tracklist_id,
                    conflicts=conflicts,
                    proposed_changes=updates["changes"],
                )
                
                if resolutions:
                    success, error = await self.conflict_service.resolve_conflicts(
                        tracklist_id=tracklist_id,
                        resolutions=resolutions,
                        actor="system",
                    )
                    
                    if not success:
                        return {
                            "status": SyncStatus.FAILED.value,
                            "error": f"Failed to auto-resolve conflicts: {error}",
                            "tracklist_id": str(tracklist_id),
                        }

            # Apply updates
            success, error = await self.tracklists_sync.apply_updates(
                tracklist_id=tracklist_id,
                updates=updates,
                auto=True,
            )

            if not success:
                return {
                    "status": SyncStatus.FAILED.value,
                    "error": error,
                    "tracklist_id": str(tracklist_id),
                }

            # Trigger CUE regeneration
            await self.cue_service.handle_tracklist_change(
                tracklist_id=tracklist_id,
                change_type="sync",
                change_details=updates["changes"],
                actor="system",
            )

            return {
                "status": SyncStatus.COMPLETED.value,
                "message": "Successfully synchronized",
                "changes_applied": updates["changes"]["total_changes"],
                "confidence": updates["confidence"],
                "tracklist_id": str(tracklist_id),
            }

        except Exception as e:
            logger.error(f"Failed to sync from 1001tracklists: {e}")
            return {
                "status": SyncStatus.FAILED.value,
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    async def schedule_sync(
        self,
        tracklist_id: UUID,
        frequency: SyncFrequency,
        source: SyncSource = SyncSource.ALL,
    ) -> Dict[str, Any]:
        """Schedule automatic synchronization for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            frequency: Sync frequency
            source: Source to sync from

        Returns:
            Scheduling result
        """
        try:
            # Get or create sync configuration
            config = await self._get_or_create_sync_config(tracklist_id)
            
            # Update configuration
            config.sync_enabled = True
            config.sync_frequency = frequency.value
            config.sync_source = source.value
            
            await self.session.commit()
            
            # Cancel existing scheduled task if any
            if tracklist_id in self.scheduled_tasks:
                self.scheduled_tasks[tracklist_id].cancel()
                del self.scheduled_tasks[tracklist_id]
            
            # Schedule new task if not manual
            if frequency != SyncFrequency.MANUAL:
                task = asyncio.create_task(
                    self._scheduled_sync_loop(tracklist_id, frequency, source)
                )
                self.scheduled_tasks[tracklist_id] = task
            
            # Log the scheduling
            await self.audit_service.log_tracklist_change(
                tracklist_id=tracklist_id,
                action="sync_scheduled",
                actor="system",
                metadata={
                    "frequency": frequency.value,
                    "source": source.value,
                },
            )
            
            return {
                "status": "scheduled",
                "tracklist_id": str(tracklist_id),
                "frequency": frequency.value,
                "source": source.value,
            }

        except Exception as e:
            logger.error(f"Failed to schedule sync for {tracklist_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    async def _scheduled_sync_loop(
        self,
        tracklist_id: UUID,
        frequency: SyncFrequency,
        source: SyncSource,
    ) -> None:
        """Background loop for scheduled synchronization.

        Args:
            tracklist_id: ID of the tracklist
            frequency: Sync frequency
            source: Source to sync from
        """
        intervals = {
            SyncFrequency.HOURLY: 3600,
            SyncFrequency.DAILY: 86400,
            SyncFrequency.WEEKLY: 604800,
            SyncFrequency.REALTIME: 300,  # Every 5 minutes
        }
        
        interval = intervals.get(frequency, 3600)
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Check if sync is still enabled
                config = await self._get_or_create_sync_config(tracklist_id)
                if not config.sync_enabled:
                    break
                
                # Trigger sync
                await self.trigger_manual_sync(
                    tracklist_id=tracklist_id,
                    source=source,
                    force=False,
                    actor="scheduler",
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduled sync for {tracklist_id}: {e}")
                # Continue the loop despite errors

    async def cancel_scheduled_sync(self, tracklist_id: UUID) -> Dict[str, Any]:
        """Cancel scheduled synchronization for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Cancellation result
        """
        try:
            # Update configuration
            config = await self._get_or_create_sync_config(tracklist_id)
            config.sync_enabled = False
            config.sync_frequency = SyncFrequency.MANUAL.value
            
            await self.session.commit()
            
            # Cancel scheduled task
            if tracklist_id in self.scheduled_tasks:
                self.scheduled_tasks[tracklist_id].cancel()
                del self.scheduled_tasks[tracklist_id]
            
            return {
                "status": "cancelled",
                "tracklist_id": str(tracklist_id),
            }

        except Exception as e:
            logger.error(f"Failed to cancel scheduled sync: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    async def handle_sync_failure(
        self,
        tracklist_id: UUID,
        error: str,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Handle synchronization failures with retry logic.

        Args:
            tracklist_id: ID of the tracklist
            error: Error message
            retry_count: Current retry attempt

        Returns:
            Handling result
        """
        max_retries = 3
        
        if retry_count < max_retries:
            # Schedule retry with exponential backoff
            delay = 2 ** retry_count * 60  # 1, 2, 4 minutes
            
            logger.info(f"Scheduling retry {retry_count + 1} for {tracklist_id} in {delay} seconds")
            
            asyncio.create_task(self._retry_sync(tracklist_id, retry_count + 1, delay))
            
            return {
                "status": "retry_scheduled",
                "tracklist_id": str(tracklist_id),
                "retry_count": retry_count + 1,
                "retry_in": delay,
            }
        else:
            # Max retries reached, log failure
            await self.audit_service.log_tracklist_change(
                tracklist_id=tracklist_id,
                action="sync_failed",
                actor="system",
                metadata={
                    "error": error,
                    "retry_count": retry_count,
                },
            )
            
            return {
                "status": "failed",
                "tracklist_id": str(tracklist_id),
                "error": error,
                "retry_count": retry_count,
            }

    async def _retry_sync(
        self,
        tracklist_id: UUID,
        retry_count: int,
        delay: int,
    ) -> None:
        """Retry synchronization after delay.

        Args:
            tracklist_id: ID of the tracklist
            retry_count: Current retry attempt
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        
        result = await self.trigger_manual_sync(
            tracklist_id=tracklist_id,
            source=SyncSource.ALL,
            force=True,
            actor="retry",
        )
        
        if result["status"] == SyncStatus.FAILED.value:
            await self.handle_sync_failure(
                tracklist_id=tracklist_id,
                error=result.get("error", "Unknown error"),
                retry_count=retry_count,
            )

    async def get_sync_status(
        self,
        tracklist_id: UUID,
    ) -> Dict[str, Any]:
        """Get current synchronization status for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Sync status information
        """
        try:
            # Get sync configuration
            config = await self._get_or_create_sync_config(tracklist_id)
            
            # Get latest sync event
            query = select(SyncEvent).where(
                SyncEvent.tracklist_id == tracklist_id
            ).order_by(SyncEvent.created_at.desc()).limit(1)
            
            result = await self.session.execute(query)
            latest_event = result.scalar_one_or_none()
            
            # Check if currently syncing
            is_syncing = tracklist_id in self.active_syncs
            is_scheduled = tracklist_id in self.scheduled_tasks
            
            return {
                "tracklist_id": str(tracklist_id),
                "is_syncing": is_syncing,
                "is_scheduled": is_scheduled,
                "sync_enabled": config.sync_enabled,
                "sync_frequency": config.sync_frequency,
                "sync_source": config.sync_source,
                "last_sync_at": config.last_sync_at.isoformat() if config.last_sync_at else None,
                "auto_accept_threshold": config.auto_accept_threshold,
                "auto_resolve_conflicts": config.auto_resolve_conflicts,
                "latest_event": {
                    "event_type": latest_event.event_type,
                    "status": latest_event.status,
                    "created_at": latest_event.created_at.isoformat(),
                    "completed_at": latest_event.completed_at.isoformat() if latest_event.completed_at else None,
                } if latest_event else None,
            }

        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    async def update_sync_configuration(
        self,
        tracklist_id: UUID,
        config_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update synchronization configuration for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            config_updates: Configuration updates to apply

        Returns:
            Updated configuration
        """
        try:
            config = await self._get_or_create_sync_config(tracklist_id)
            
            # Update allowed fields
            allowed_fields = [
                "sync_enabled",
                "sync_frequency", 
                "sync_source",
                "auto_accept_threshold",
                "auto_resolve_conflicts",
                "conflict_resolution",
            ]
            
            for field in allowed_fields:
                if field in config_updates:
                    setattr(config, field, config_updates[field])
            
            await self.session.commit()
            
            # Handle frequency changes
            if "sync_frequency" in config_updates:
                frequency = SyncFrequency(config_updates["sync_frequency"])
                source = SyncSource(config.sync_source or "all")
                
                if config.sync_enabled:
                    await self.schedule_sync(tracklist_id, frequency, source)
                else:
                    await self.cancel_scheduled_sync(tracklist_id)
            
            return {
                "status": "updated",
                "tracklist_id": str(tracklist_id),
                "configuration": {
                    "sync_enabled": config.sync_enabled,
                    "sync_frequency": config.sync_frequency,
                    "sync_source": config.sync_source,
                    "auto_accept_threshold": config.auto_accept_threshold,
                    "auto_resolve_conflicts": config.auto_resolve_conflicts,
                },
            }

        except Exception as e:
            logger.error(f"Failed to update sync configuration: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    async def coordinate_multi_source_sync(
        self,
        tracklist_id: UUID,
        sources: List[SyncSource],
    ) -> Dict[str, Any]:
        """Coordinate synchronization from multiple sources.

        Args:
            tracklist_id: ID of the tracklist
            sources: List of sources to sync from

        Returns:
            Aggregated sync results
        """
        results = {
            "tracklist_id": str(tracklist_id),
            "sources_processed": [],
            "total_changes": 0,
            "conflicts": [],
            "status": SyncStatus.COMPLETED.value,
        }
        
        for source in sources:
            try:
                result = await self.trigger_manual_sync(
                    tracklist_id=tracklist_id,
                    source=source,
                    force=False,
                    actor="multi_source",
                )
                
                results["sources_processed"].append({
                    "source": source.value,
                    "status": result.get("status"),
                    "changes": result.get("changes_applied", 0),
                })
                
                if result.get("changes_applied"):
                    results["total_changes"] += result["changes_applied"]
                
                if result.get("conflicts"):
                    results["conflicts"].extend(result["conflicts"])
                
                if result["status"] == SyncStatus.FAILED.value:
                    results["status"] = SyncStatus.FAILED.value
                
            except Exception as e:
                logger.error(f"Failed to sync from {source.value}: {e}")
                results["sources_processed"].append({
                    "source": source.value,
                    "status": "error",
                    "error": str(e),
                })
        
        return results

    async def _get_or_create_sync_config(
        self,
        tracklist_id: UUID,
    ) -> SyncConfiguration:
        """Get or create sync configuration for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Sync configuration
        """
        query = select(SyncConfiguration).where(
            SyncConfiguration.tracklist_id == tracklist_id
        )
        result = await self.session.execute(query)
        config = result.scalar_one_or_none()
        
        if not config:
            config = SyncConfiguration(
                tracklist_id=tracklist_id,
                sync_enabled=False,
                sync_frequency=SyncFrequency.MANUAL.value,
                auto_accept_threshold=0.8,
                auto_resolve_conflicts=False,
                conflict_resolution="manual",
            )
            self.session.add(config)
            await self.session.flush()
        
        return config

    async def _create_sync_event(
        self,
        tracklist_id: UUID,
        source: str,
        actor: str,
    ) -> SyncEvent:
        """Create a new sync event.

        Args:
            tracklist_id: ID of the tracklist
            source: Sync source
            actor: Who triggered the sync

        Returns:
            Created sync event
        """
        event = SyncEvent(
            tracklist_id=tracklist_id,
            event_type="sync",
            source=source,
            status="processing",
            created_at=datetime.utcnow(),
            metadata={"actor": actor},
        )
        self.session.add(event)
        await self.session.flush()
        
        return event