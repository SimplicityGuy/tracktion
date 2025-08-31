"""1001tracklists synchronization service for checking and applying updates."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from services.tracklist_service.src.models.tracklist import TracklistDB, TrackEntry
from services.tracklist_service.src.models.synchronization import SyncConfiguration, SyncEvent
from services.tracklist_service.src.services.import_service import ImportService
from services.tracklist_service.src.services.version_service import VersionService
from services.tracklist_service.src.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class TracklistsSyncService:
    """Service for synchronizing tracklists with 1001tracklists."""

    def __init__(
        self,
        session: AsyncSession,
        import_service: Optional[ImportService] = None,
        version_service: Optional[VersionService] = None,
        audit_service: Optional[AuditService] = None,
    ):
        """Initialize sync service.

        Args:
            session: Database session
            import_service: Service for importing from 1001tracklists
            version_service: Service for version management
            audit_service: Service for audit logging
        """
        self.session = session
        self.import_service = import_service or ImportService()
        self.version_service = version_service or VersionService(session)
        self.audit_service = audit_service or AuditService(session)
        self.auto_accept_threshold = 0.9

    async def check_for_updates(self, tracklist_id: UUID) -> Optional[Dict[str, Any]]:
        """Check if a tracklist has updates from 1001tracklists.

        Args:
            tracklist_id: ID of the tracklist to check

        Returns:
            Update information if available, None otherwise
        """
        # Get the tracklist
        tracklist = await self.session.get(TracklistDB, tracklist_id)
        if not tracklist:
            logger.warning(f"Tracklist {tracklist_id} not found")
            return None

        # Only check 1001tracklists imports
        if tracklist.source != "1001tracklists":
            return None

        # Get the URL from tracklist metadata or stored data
        # For now, we'll need to enhance the model to store the original URL
        url = getattr(tracklist, "import_url", None)
        if not url:
            # Try to extract from metadata if stored
            logger.warning(f"No import URL found for tracklist {tracklist_id}")
            return None

        try:
            # Fetch latest version from 1001tracklists
            scraped_tracklist = self.import_service.fetch_tracklist_from_1001(url, force_refresh=True)

            # Transform scraped tracks to TrackEntry format
            latest_tracks = self.import_service.transform_to_track_entries(scraped_tracklist.tracks)

            # Compare with current version
            current_tracks = [TrackEntry.from_dict(t) for t in tracklist.tracks]
            changes = self._compare_tracklists(current_tracks, latest_tracks)

            if changes["has_changes"]:
                # Calculate confidence score
                confidence = self._calculate_change_confidence(changes)

                return {
                    "has_updates": True,
                    "changes": changes,
                    "confidence": confidence,
                    "source_url": url,
                    "fetched_at": datetime.utcnow(),
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check updates for tracklist {tracklist_id}: {e}")
            return None

    def _compare_tracklists(self, current: List[TrackEntry], latest: List[TrackEntry]) -> Dict[str, Any]:
        """Compare two tracklist versions to find differences.

        Args:
            current: Current tracklist tracks
            latest: Latest tracklist tracks

        Returns:
            Dictionary describing the changes
        """
        changes: Dict[str, Any] = {
            "has_changes": False,
            "tracks_added": [],
            "tracks_removed": [],
            "tracks_modified": [],
            "total_changes": 0,
        }

        # Create position-based lookups
        current_by_pos = {track.position: track for track in current}
        latest_by_pos = {track.position: track for track in latest}

        # Find added tracks
        for pos, track in latest_by_pos.items():
            if pos not in current_by_pos:
                changes["tracks_added"].append(track.to_dict())

        # Find removed tracks
        for pos, track in current_by_pos.items():
            if pos not in latest_by_pos:
                changes["tracks_removed"].append(track.to_dict())

        # Find modified tracks
        for pos in set(current_by_pos.keys()) & set(latest_by_pos.keys()):
            current_track = current_by_pos[pos]
            latest_track = latest_by_pos[pos]

            if self._track_differs(current_track, latest_track):
                changes["tracks_modified"].append(
                    {
                        "position": pos,
                        "old": current_track.to_dict(),
                        "new": latest_track.to_dict(),
                    }
                )

        # Update summary
        changes["total_changes"] = (
            len(changes["tracks_added"]) + len(changes["tracks_removed"]) + len(changes["tracks_modified"])
        )
        changes["has_changes"] = changes["total_changes"] > 0

        return changes

    def _track_differs(self, track1: TrackEntry, track2: TrackEntry) -> bool:
        """Check if two tracks are different.

        Args:
            track1: First track
            track2: Second track

        Returns:
            True if tracks differ, False otherwise
        """
        # Compare key fields
        fields_to_compare = ["artist", "title", "remix", "label", "start_time", "end_time"]

        for field in fields_to_compare:
            if getattr(track1, field) != getattr(track2, field):
                return True

        return False

    def _calculate_change_confidence(self, changes: Dict[str, Any]) -> float:
        """Calculate confidence score for changes.

        Args:
            changes: Dictionary of changes

        Returns:
            Confidence score between 0 and 1
        """
        if not changes["has_changes"]:
            return 1.0

        total = len(changes["tracks_added"]) + len(changes["tracks_removed"]) + len(changes["tracks_modified"])

        # Start with base confidence
        confidence = 0.8

        # Reduce confidence for major changes
        if len(changes["tracks_removed"]) > 3:
            confidence -= 0.2

        if len(changes["tracks_added"]) > 5:
            confidence -= 0.1

        # Modifications are generally safer
        if len(changes["tracks_modified"]) == total:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    async def apply_updates(
        self, tracklist_id: UUID, updates: Dict[str, Any], auto: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Apply updates from 1001tracklists to a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            updates: Update information from check_for_updates
            auto: Whether this is an automatic update

        Returns:
            Tuple of (success, error_message)
        """
        # Get sync configuration
        config_query = select(SyncConfiguration).where(SyncConfiguration.tracklist_id == tracklist_id)
        result = await self.session.execute(config_query)
        sync_config = result.scalar_one_or_none()

        if not sync_config:
            # Create default config
            sync_config = SyncConfiguration(
                tracklist_id=tracklist_id,
                sync_enabled=True,
                auto_accept_threshold=self.auto_accept_threshold,
                conflict_resolution="manual",
            )
            self.session.add(sync_config)

        # Check if auto-accept is allowed
        if auto and updates["confidence"] < sync_config.auto_accept_threshold:
            # Queue for manual review
            await self._queue_for_review(tracklist_id, updates)
            return False, "Changes queued for manual review (confidence too low)"

        # Create sync event
        sync_event = SyncEvent(
            tracklist_id=tracklist_id,
            event_type="update",
            source="1001tracklists",
            status="processing",
            created_at=datetime.utcnow(),
            changes=updates["changes"],
        )
        self.session.add(sync_event)

        try:
            # Get the tracklist
            tracklist = await self.session.get(TracklistDB, tracklist_id)
            if not tracklist:
                sync_event.status = "failed"  # type: ignore[assignment]
                await self.session.commit()
                return False, "Tracklist not found"

            # Create a new version before applying changes
            await self.version_service.create_version(
                tracklist_id=tracklist_id,
                change_type="1001tracklists_sync",
                change_summary=f"Auto-sync from 1001tracklists with {updates['changes']['total_changes']} changes",
                created_by="system",
            )

            # Apply the changes
            new_tracks = self._apply_changes_to_tracks(tracklist.tracks, updates["changes"])
            tracklist.tracks = new_tracks
            tracklist.updated_at = datetime.utcnow()

            # Log to audit
            await self.audit_service.log_tracklist_change(
                tracklist_id=tracklist_id,
                action="synced",
                actor="system",
                before={"track_count": len(tracklist.tracks)},
                after={"track_count": len(new_tracks)},
                metadata={"source": "1001tracklists", "confidence": updates["confidence"]},
            )

            # Update sync configuration
            sync_config.last_sync_at = datetime.utcnow()  # type: ignore[assignment]

            # Mark sync event as completed
            sync_event.status = "completed"  # type: ignore[assignment]
            sync_event.completed_at = datetime.utcnow()  # type: ignore[assignment]

            await self.session.commit()

            return True, None

        except Exception as e:
            logger.error(f"Failed to apply updates for tracklist {tracklist_id}: {e}")
            sync_event.status = "failed"  # type: ignore[assignment]
            await self.session.commit()
            return False, str(e)

    def _apply_changes_to_tracks(
        self, current_tracks: List[Dict[str, Any]], changes: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply changes to the current tracks list.

        Args:
            current_tracks: Current tracks data
            changes: Changes to apply

        Returns:
            Updated tracks list
        """
        # Create a working copy
        tracks_by_pos = {t["position"]: t for t in current_tracks}

        # Remove tracks
        for track in changes["tracks_removed"]:
            tracks_by_pos.pop(track["position"], None)

        # Add tracks
        for track in changes["tracks_added"]:
            tracks_by_pos[track["position"]] = track

        # Modify tracks
        for modification in changes["tracks_modified"]:
            tracks_by_pos[modification["position"]] = modification["new"]

        # Return as sorted list
        return [tracks_by_pos[pos] for pos in sorted(tracks_by_pos.keys())]

    async def _queue_for_review(self, tracklist_id: UUID, updates: Dict[str, Any]) -> None:
        """Queue changes for manual review.

        Args:
            tracklist_id: ID of the tracklist
            updates: Update information
        """
        # Create a sync event with conflict status
        sync_event = SyncEvent(
            tracklist_id=tracklist_id,
            event_type="conflict",
            source="1001tracklists",
            status="pending",
            created_at=datetime.utcnow(),
            changes=updates["changes"],
            conflict_data={
                "confidence": updates["confidence"],
                "reason": "confidence_below_threshold",
            },
        )
        self.session.add(sync_event)
        await self.session.commit()

        # Log to audit
        await self.audit_service.log_tracklist_change(
            tracklist_id=tracklist_id,
            action="review_queued",
            actor="system",
            metadata={
                "confidence": updates["confidence"],
                "changes": updates["changes"]["total_changes"],
            },
        )
