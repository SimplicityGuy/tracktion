"""Conflict resolution service for handling synchronization conflicts."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from services.tracklist_service.src.models.synchronization import SyncEvent, SyncConfiguration
from services.tracklist_service.src.models.tracklist import TracklistDB, TrackEntry
from services.tracklist_service.src.services.version_service import VersionService
from services.tracklist_service.src.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    """Available conflict resolution strategies."""
    
    KEEP_CURRENT = "keep_current"
    USE_PROPOSED = "use_proposed"
    MANUAL_EDIT = "manual_edit"
    MERGE = "merge"
    AUTO_RESOLVE = "auto_resolve"


class ConflictType(Enum):
    """Types of conflicts that can occur."""
    
    TRACK_ADDED = "track_added"
    TRACK_REMOVED = "track_removed"
    TRACK_MODIFIED = "track_modified"
    METADATA_CHANGE = "metadata_change"
    MAJOR_RESTRUCTURE = "major_restructure"


class ConflictResolutionService:
    """Service for resolving conflicts during synchronization."""

    def __init__(
        self,
        session: AsyncSession,
        version_service: Optional[VersionService] = None,
        audit_service: Optional[AuditService] = None,
    ):
        """Initialize conflict resolution service.

        Args:
            session: Database session
            version_service: Service for version management
            audit_service: Service for audit logging
        """
        self.session = session
        self.version_service = version_service or VersionService(session)
        self.audit_service = audit_service or AuditService(session)
        self.confidence_threshold = 0.8

    async def detect_conflicts(
        self, 
        tracklist_id: UUID,
        current_state: Dict[str, Any],
        proposed_changes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between current state and proposed changes.

        Args:
            tracklist_id: ID of the tracklist
            current_state: Current state of the tracklist
            proposed_changes: Proposed changes from synchronization

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for major structural changes
        if self._is_major_restructure(proposed_changes):
            conflicts.append({
                "id": str(uuid4()),
                "type": ConflictType.MAJOR_RESTRUCTURE.value,
                "severity": "high",
                "description": "Major structural changes detected",
                "details": {
                    "tracks_added": len(proposed_changes.get("tracks_added", [])),
                    "tracks_removed": len(proposed_changes.get("tracks_removed", [])),
                    "tracks_modified": len(proposed_changes.get("tracks_modified", [])),
                },
                "auto_resolvable": False,
                "recommended_strategy": ResolutionStrategy.MANUAL_EDIT.value,
            })

        # Check individual track conflicts
        for track_mod in proposed_changes.get("tracks_modified", []):
            conflict_detail = self._analyze_track_modification(track_mod)
            if conflict_detail:
                conflicts.append({
                    "id": str(uuid4()),
                    "type": ConflictType.TRACK_MODIFIED.value,
                    "severity": conflict_detail["severity"],
                    "description": f"Track {track_mod['position']} has conflicting changes",
                    "details": conflict_detail,
                    "auto_resolvable": conflict_detail["confidence"] > self.confidence_threshold,
                    "recommended_strategy": self._recommend_strategy(conflict_detail),
                })

        # Check removed tracks that might be important
        for track_removed in proposed_changes.get("tracks_removed", []):
            if self._is_critical_track(track_removed):
                conflicts.append({
                    "id": str(uuid4()),
                    "type": ConflictType.TRACK_REMOVED.value,
                    "severity": "medium",
                    "description": f"Important track {track_removed['position']} would be removed",
                    "details": track_removed,
                    "auto_resolvable": False,
                    "recommended_strategy": ResolutionStrategy.KEEP_CURRENT.value,
                })

        return conflicts

    def _is_major_restructure(self, changes: Dict[str, Any]) -> bool:
        """Determine if changes constitute a major restructure.

        Args:
            changes: Proposed changes

        Returns:
            True if major restructure detected
        """
        added = len(changes.get("tracks_added", []))
        removed = len(changes.get("tracks_removed", []))
        modified = len(changes.get("tracks_modified", []))
        
        # Major restructure if many tracks affected
        total_changes = added + removed + modified
        return total_changes > 10 or removed > 5 or (added > 5 and removed > 2)

    def _analyze_track_modification(self, track_mod: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a track modification for conflicts.

        Args:
            track_mod: Track modification details

        Returns:
            Conflict details if conflict detected
        """
        old = track_mod.get("old", {})
        new = track_mod.get("new", {})
        
        # Calculate field-level changes
        changed_fields = []
        for field in ["artist", "title", "remix", "label"]:
            if old.get(field) != new.get(field):
                changed_fields.append({
                    "field": field,
                    "old_value": old.get(field),
                    "new_value": new.get(field),
                    "confidence": self._calculate_field_confidence(field, old.get(field), new.get(field)),
                })

        if not changed_fields:
            return None

        # Determine severity
        severity = "low"
        if len(changed_fields) > 2:
            severity = "high"
        elif any(f["field"] in ["artist", "title"] for f in changed_fields):
            severity = "medium"

        # Calculate overall confidence
        confidence = sum(f["confidence"] for f in changed_fields) / len(changed_fields)

        return {
            "position": track_mod["position"],
            "changed_fields": changed_fields,
            "severity": severity,
            "confidence": confidence,
        }

    def _calculate_field_confidence(self, field: str, old_value: Any, new_value: Any) -> float:
        """Calculate confidence for a field change.

        Args:
            field: Field name
            old_value: Current value
            new_value: Proposed value

        Returns:
            Confidence score between 0 and 1
        """
        if not old_value and new_value:
            # Adding missing data is usually good
            return 0.9
        
        if field in ["artist", "title"]:
            # Major fields need more caution
            if old_value and new_value:
                # Check if it's likely a correction or major change
                if isinstance(old_value, str) and isinstance(new_value, str):
                    old_lower = old_value.lower()
                    new_lower = new_value.lower()
                    
                    # Small changes like adding (Extended) are likely correct
                    if old_lower in new_lower or new_lower in old_lower:
                        return 0.85
                    
                    # Complete changes need manual review
                    return 0.3
        
        # Minor fields can be more confidently updated
        return 0.7

    def _is_critical_track(self, track: Dict[str, Any]) -> bool:
        """Determine if a track is critical and shouldn't be removed automatically.

        Args:
            track: Track data

        Returns:
            True if track is critical
        """
        # Tracks with specific markers or at key positions might be critical
        position = track.get("position", 0)
        title = track.get("title", "").lower()
        
        # First and last tracks are often important
        if position == 1:
            return True
            
        # Intro/outro tracks are important
        if any(marker in title for marker in ["intro", "outro", "opening", "closing"]):
            return True
            
        return False

    def _recommend_strategy(self, conflict_detail: Dict[str, Any]) -> str:
        """Recommend a resolution strategy based on conflict details.

        Args:
            conflict_detail: Details about the conflict

        Returns:
            Recommended strategy
        """
        confidence = conflict_detail.get("confidence", 0)
        severity = conflict_detail.get("severity", "low")
        
        if confidence > 0.85:
            return ResolutionStrategy.USE_PROPOSED.value
        elif confidence > 0.7 and severity == "low":
            return ResolutionStrategy.USE_PROPOSED.value
        elif confidence < 0.4:
            return ResolutionStrategy.KEEP_CURRENT.value
        else:
            return ResolutionStrategy.MANUAL_EDIT.value

    async def prepare_conflict_ui_data(
        self,
        tracklist_id: UUID,
        conflicts: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        proposed_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare conflict data for UI presentation.

        Args:
            tracklist_id: ID of the tracklist
            conflicts: List of detected conflicts
            current_state: Current state
            proposed_changes: Proposed changes

        Returns:
            UI-ready conflict data
        """
        ui_data = {
            "tracklist_id": str(tracklist_id),
            "total_conflicts": len(conflicts),
            "auto_resolvable_count": sum(1 for c in conflicts if c.get("auto_resolvable", False)),
            "conflicts": [],
            "resolution_options": [s.value for s in ResolutionStrategy],
            "summary": {
                "tracks_to_add": len(proposed_changes.get("tracks_added", [])),
                "tracks_to_remove": len(proposed_changes.get("tracks_removed", [])),
                "tracks_to_modify": len(proposed_changes.get("tracks_modified", [])),
            },
        }

        # Format each conflict for UI
        for conflict in conflicts:
            ui_conflict = {
                "id": conflict["id"],
                "type": conflict["type"],
                "severity": conflict["severity"],
                "description": conflict["description"],
                "auto_resolvable": conflict.get("auto_resolvable", False),
                "recommended_strategy": conflict["recommended_strategy"],
                "details": self._format_conflict_details(conflict, current_state, proposed_changes),
            }
            ui_data["conflicts"].append(ui_conflict)

        return ui_data

    def _format_conflict_details(
        self,
        conflict: Dict[str, Any],
        current_state: Dict[str, Any],
        proposed_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format conflict details for UI display.

        Args:
            conflict: Conflict information
            current_state: Current state
            proposed_changes: Proposed changes

        Returns:
            Formatted conflict details
        """
        details = conflict.get("details", {})
        
        if conflict["type"] == ConflictType.TRACK_MODIFIED.value:
            # Format track modification conflicts
            position = details.get("position")
            return {
                "position": position,
                "fields": [
                    {
                        "name": field["field"],
                        "current_value": field["old_value"],
                        "proposed_value": field["new_value"],
                        "confidence": field["confidence"],
                    }
                    for field in details.get("changed_fields", [])
                ],
            }
        elif conflict["type"] == ConflictType.MAJOR_RESTRUCTURE.value:
            # Format major restructure
            return {
                "impact": details,
                "recommendation": "Manual review recommended due to major structural changes",
            }
        else:
            return details

    async def resolve_conflicts(
        self,
        tracklist_id: UUID,
        resolutions: List[Dict[str, Any]],
        actor: str = "user",
    ) -> Tuple[bool, Optional[str]]:
        """Apply conflict resolutions.

        Args:
            tracklist_id: ID of the tracklist
            resolutions: List of resolutions to apply
            actor: User or system applying resolutions

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get the tracklist
            tracklist = await self.session.get(TracklistDB, tracklist_id)
            if not tracklist:
                return False, "Tracklist not found"

            # Create a version before applying resolutions
            await self.version_service.create_version(
                tracklist_id=tracklist_id,
                change_type="conflict_resolution",
                change_summary=f"Resolved {len(resolutions)} conflicts",
                created_by=actor,
            )

            # Apply each resolution
            for resolution in resolutions:
                await self._apply_single_resolution(tracklist, resolution)

            # Log the resolution
            await self.audit_service.log_tracklist_change(
                tracklist_id=tracklist_id,
                action="conflicts_resolved",
                actor=actor,
                metadata={
                    "resolution_count": len(resolutions),
                    "strategies_used": list(set(r.get("strategy") for r in resolutions)),
                },
            )

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to resolve conflicts for tracklist {tracklist_id}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def _apply_single_resolution(
        self,
        tracklist: TracklistDB,
        resolution: Dict[str, Any],
    ) -> None:
        """Apply a single conflict resolution.

        Args:
            tracklist: Tracklist to update
            resolution: Resolution to apply
        """
        strategy = resolution.get("strategy")
        conflict_id = resolution.get("conflict_id")
        
        if strategy == ResolutionStrategy.KEEP_CURRENT.value:
            # No changes needed
            logger.info(f"Keeping current state for conflict {conflict_id}")
        elif strategy == ResolutionStrategy.USE_PROPOSED.value:
            # Apply the proposed change
            proposed_data = resolution.get("proposed_data", {})
            await self._apply_proposed_changes(tracklist, proposed_data)
        elif strategy == ResolutionStrategy.MANUAL_EDIT.value:
            # Apply manual edits
            manual_data = resolution.get("manual_data", {})
            await self._apply_manual_edits(tracklist, manual_data)
        elif strategy == ResolutionStrategy.MERGE.value:
            # Merge changes
            merge_data = resolution.get("merge_data", {})
            await self._apply_merge(tracklist, merge_data)

    async def _apply_proposed_changes(
        self,
        tracklist: TracklistDB,
        proposed_data: Dict[str, Any],
    ) -> None:
        """Apply proposed changes to tracklist.

        Args:
            tracklist: Tracklist to update
            proposed_data: Proposed changes to apply
        """
        # This would update the tracklist with the proposed changes
        # Implementation depends on the exact structure of your tracklist
        if "tracks" in proposed_data:
            tracklist.tracks = proposed_data["tracks"]
        
        tracklist.updated_at = datetime.utcnow()

    async def _apply_manual_edits(
        self,
        tracklist: TracklistDB,
        manual_data: Dict[str, Any],
    ) -> None:
        """Apply manual edits to tracklist.

        Args:
            tracklist: Tracklist to update
            manual_data: Manual edits to apply
        """
        # Apply user's manual edits
        if "tracks" in manual_data:
            tracklist.tracks = manual_data["tracks"]
        
        tracklist.updated_at = datetime.utcnow()

    async def _apply_merge(
        self,
        tracklist: TracklistDB,
        merge_data: Dict[str, Any],
    ) -> None:
        """Apply merged changes to tracklist.

        Args:
            tracklist: Tracklist to update
            merge_data: Merged data to apply
        """
        # Apply merged changes - combination of current and proposed
        if "tracks" in merge_data:
            tracklist.tracks = merge_data["tracks"]
        
        tracklist.updated_at = datetime.utcnow()

    async def auto_resolve_conflicts(
        self,
        tracklist_id: UUID,
        conflicts: List[Dict[str, Any]],
        proposed_changes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Automatically resolve conflicts where possible.

        Args:
            tracklist_id: ID of the tracklist
            conflicts: List of conflicts
            proposed_changes: Proposed changes

        Returns:
            List of resolutions
        """
        resolutions = []
        
        for conflict in conflicts:
            if conflict.get("auto_resolvable", False):
                resolution = {
                    "conflict_id": conflict["id"],
                    "strategy": conflict["recommended_strategy"],
                    "automated": True,
                    "confidence": conflict.get("details", {}).get("confidence", 0),
                }
                
                # Add proposed data if using proposed strategy
                if resolution["strategy"] == ResolutionStrategy.USE_PROPOSED.value:
                    resolution["proposed_data"] = self._extract_proposed_data(
                        conflict,
                        proposed_changes
                    )
                
                resolutions.append(resolution)
        
        return resolutions

    def _extract_proposed_data(
        self,
        conflict: Dict[str, Any],
        proposed_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant proposed data for a conflict.

        Args:
            conflict: Conflict information
            proposed_changes: All proposed changes

        Returns:
            Relevant proposed data
        """
        if conflict["type"] == ConflictType.TRACK_MODIFIED.value:
            position = conflict["details"].get("position")
            # Find the specific modification
            for mod in proposed_changes.get("tracks_modified", []):
                if mod.get("position") == position:
                    return {"tracks": [mod["new"]]}
        
        return {}