"""Version management service for tracklist versioning."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import TracklistVersion
from services.tracklist_service.src.models.tracklist import TracklistDB


class VersionService:
    """Service for managing tracklist versions."""

    def __init__(self, session: AsyncSession):
        """Initialize version service.

        Args:
            session: Database session
        """
        self.session = session

    async def create_version(
        self,
        tracklist_id: UUID,
        change_type: str,
        change_summary: str,
        created_by: str | None = None,
        tracks_snapshot: list[dict[str, Any]] | None = None,
    ) -> TracklistVersion:
        """Create a new version of a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            change_type: Type of change (manual_edit, import_update, auto_sync)
            change_summary: Human-readable change description
            created_by: User or system identifier
            tracks_snapshot: Optional explicit tracks data (if not provided, uses current)

        Returns:
            Created version
        """
        # Get current version number
        current_version = await self.get_latest_version(tracklist_id)
        new_version_number = (current_version.version_number + 1) if current_version else 1

        # Get tracklist tracks if not provided
        if tracks_snapshot is None:
            tracklist = await self.session.get(TracklistDB, tracklist_id)
            if not tracklist:
                raise ValueError(f"Tracklist {tracklist_id} not found")
            tracks_snapshot = tracklist.tracks

        # Mark previous version as not current
        if current_version:
            current_version.is_current = False
            self.session.add(current_version)

        # Create new version
        version = TracklistVersion(
            tracklist_id=tracklist_id,
            version_number=new_version_number,
            created_at=datetime.now(UTC),
            created_by=created_by or "system",
            change_type=change_type,
            change_summary=change_summary,
            tracks_snapshot=tracks_snapshot,
            version_metadata={"timestamp": datetime.now(UTC).isoformat()},
            is_current=True,
        )

        self.session.add(version)
        await self.session.commit()
        await self.session.refresh(version)

        return version

    async def get_latest_version(self, tracklist_id: UUID) -> TracklistVersion | None:
        """Get the latest version of a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Latest version or None
        """
        query = select(TracklistVersion).where(
            and_(
                TracklistVersion.tracklist_id == tracklist_id,
                TracklistVersion.is_current.is_(True),
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy returns model but typed as Any

    async def get_version(self, tracklist_id: UUID, version_number: int) -> TracklistVersion | None:
        """Get a specific version of a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            version_number: Version number to retrieve

        Returns:
            Version or None
        """
        query = select(TracklistVersion).where(
            and_(
                TracklistVersion.tracklist_id == tracklist_id,
                TracklistVersion.version_number == version_number,
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy returns model but typed as Any

    async def list_versions(self, tracklist_id: UUID, limit: int = 50, offset: int = 0) -> list[TracklistVersion]:
        """List versions for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            limit: Maximum number of versions to return
            offset: Number of versions to skip

        Returns:
            List of versions
        """
        query = (
            select(TracklistVersion)
            .where(TracklistVersion.tracklist_id == tracklist_id)
            .order_by(TracklistVersion.version_number.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def rollback_to_version(self, tracklist_id: UUID, version_number: int) -> TracklistDB:
        """Rollback tracklist to a specific version.

        Args:
            tracklist_id: ID of the tracklist
            version_number: Version to rollback to

        Returns:
            Updated tracklist
        """
        # Get the target version
        version = await self.get_version(tracklist_id, version_number)
        if not version:
            raise ValueError(f"Version {version_number} not found for tracklist {tracklist_id}")

        # Get the tracklist
        tracklist = await self.session.get(TracklistDB, tracklist_id)
        if not tracklist:
            raise ValueError(f"Tracklist {tracklist_id} not found")

        # Create a new version for the rollback
        await self.create_version(
            tracklist_id=tracklist_id,
            change_type="rollback",
            change_summary=f"Rolled back to version {version_number}",
            tracks_snapshot=version.tracks_snapshot,
        )

        # Update the tracklist with the rolled back tracks
        tracklist.tracks = version.tracks_snapshot
        tracklist.updated_at = datetime.now(UTC)

        self.session.add(tracklist)
        await self.session.commit()
        await self.session.refresh(tracklist)

        return tracklist  # type: ignore[no-any-return]  # SQLAlchemy model after refresh typed as Any

    async def get_version_diff(self, tracklist_id: UUID, version1: int, version2: int) -> dict[str, Any]:
        """Get differences between two versions.

        Args:
            tracklist_id: ID of the tracklist
            version1: First version number
            version2: Second version number

        Returns:
            Dictionary with differences
        """
        v1 = await self.get_version(tracklist_id, version1)
        v2 = await self.get_version(tracklist_id, version2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        # Compare tracks
        tracks1 = v1.tracks_snapshot
        tracks2 = v2.tracks_snapshot

        added = []
        removed = []
        modified = []

        # Create position-based lookup
        tracks1_by_pos = {t.get("position"): t for t in tracks1}
        tracks2_by_pos = {t.get("position"): t for t in tracks2}

        # Find added and modified
        for pos, track in tracks2_by_pos.items():
            if pos not in tracks1_by_pos:
                added.append(track)
            elif track != tracks1_by_pos[pos]:
                modified.append({"position": pos, "old": tracks1_by_pos[pos], "new": track})

        # Find removed
        for pos, track in tracks1_by_pos.items():
            if pos not in tracks2_by_pos:
                removed.append(track)

        return {
            "version1": version1,
            "version2": version2,
            "added": added,
            "removed": removed,
            "modified": modified,
            "total_changes": len(added) + len(removed) + len(modified),
        }

    async def prune_old_versions(self, tracklist_id: UUID, keep_count: int = 50, keep_days: int = 90) -> int:
        """Prune old versions based on retention policy.

        Args:
            tracklist_id: ID of the tracklist
            keep_count: Number of recent versions to keep
            keep_days: Number of days to keep versions

        Returns:
            Number of versions deleted
        """

        cutoff_date = datetime.now(UTC) - timedelta(days=keep_days)

        # Get versions to keep by count
        keep_versions_query = (
            select(TracklistVersion.id)
            .where(TracklistVersion.tracklist_id == tracklist_id)
            .order_by(TracklistVersion.version_number.desc())
            .limit(keep_count)
        )
        keep_result = await self.session.execute(keep_versions_query)
        keep_ids = [row[0] for row in keep_result]

        # Delete old versions
        delete_query = select(TracklistVersion).where(
            and_(
                TracklistVersion.tracklist_id == tracklist_id,
                TracklistVersion.created_at < cutoff_date,
                TracklistVersion.id.notin_(keep_ids) if keep_ids else True,
                TracklistVersion.is_current.is_(False),
            )
        )

        result = await self.session.execute(delete_query)
        versions_to_delete = result.scalars().all()

        for version in versions_to_delete:
            await self.session.delete(version)

        await self.session.commit()

        return len(versions_to_delete)

    async def get_version_by_id(self, version_id: UUID) -> TracklistVersion | None:
        """Get a version by its UUID.

        Args:
            version_id: ID of the version

        Returns:
            Version or None
        """
        query = select(TracklistVersion).where(TracklistVersion.id == version_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy returns model but typed as Any

    async def rollback_to_version_by_id(self, tracklist_id: UUID, version_id: UUID) -> TracklistDB:
        """Rollback tracklist to a specific version using version ID.

        Args:
            tracklist_id: ID of the tracklist
            version_id: ID of version to rollback to

        Returns:
            Updated tracklist
        """
        # Get the target version by ID
        target_version = await self.get_version_by_id(version_id)
        if not target_version:
            raise ValueError(f"Version {version_id} not found")

        # Verify the version belongs to the correct tracklist
        if target_version.tracklist_id != tracklist_id:
            raise ValueError(f"Version {version_id} does not belong to tracklist {tracklist_id}")

        # Use the existing rollback method with the version number
        return await self.rollback_to_version(tracklist_id, target_version.version_number)
