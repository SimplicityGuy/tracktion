"""Audit service for tracking all system changes and activities."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import AuditLog


class AuditService:
    """Service for managing audit logs of all system changes."""

    def __init__(self, session: AsyncSession):
        """Initialize audit service.

        Args:
            session: Database session
        """
        self.session = session

    async def log_change(
        self,
        entity_type: str,
        entity_id: UUID,
        action: str,
        actor: str,
        changes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Create an audit log entry for a change.

        Args:
            entity_type: Type of entity (tracklist, cue_file, etc.)
            entity_id: ID of the entity
            action: Action performed (created, updated, deleted, synced)
            actor: User or system identifier
            changes: Details of changes made
            metadata: Additional context information

        Returns:
            Created audit log entry
        """
        audit_log = AuditLog(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            timestamp=datetime.utcnow(),
            actor=actor,
            changes=changes,
            audit_metadata=metadata or {},
        )

        self.session.add(audit_log)
        await self.session.commit()
        await self.session.refresh(audit_log)

        return audit_log

    async def log_tracklist_change(
        self,
        tracklist_id: UUID,
        action: str,
        actor: str,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log a tracklist-specific change.

        Args:
            tracklist_id: ID of the tracklist
            action: Action performed
            actor: User or system identifier
            before: State before change
            after: State after change
            metadata: Additional context

        Returns:
            Created audit log entry
        """
        changes = {}
        if before is not None:
            changes["before"] = before
        if after is not None:
            changes["after"] = after

        return await self.log_change(
            entity_type="tracklist",
            entity_id=tracklist_id,
            action=action,
            actor=actor,
            changes=changes,
            metadata=metadata,
        )

    async def log_cue_file_change(
        self,
        cue_file_id: UUID,
        action: str,
        actor: str,
        changes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log a CUE file-specific change.

        Args:
            cue_file_id: ID of the CUE file
            action: Action performed
            actor: User or system identifier
            changes: Details of changes
            metadata: Additional context

        Returns:
            Created audit log entry
        """
        return await self.log_change(
            entity_type="cue_file",
            entity_id=cue_file_id,
            action=action,
            actor=actor,
            changes=changes,
            metadata=metadata,
        )

    async def query_audit_logs(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[UUID] = None,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """Query audit logs with filters.

        Args:
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            action: Filter by action
            actor: Filter by actor
            date_from: Start date for filtering
            date_to: End date for filtering
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of audit log entries
        """
        query = select(AuditLog)

        conditions = []
        if entity_type:
            conditions.append(AuditLog.entity_type == entity_type)
        if entity_id:
            conditions.append(AuditLog.entity_id == entity_id)
        if action:
            conditions.append(AuditLog.action == action)
        if actor:
            conditions.append(AuditLog.actor == actor)
        if date_from:
            conditions.append(AuditLog.timestamp >= date_from)
        if date_to:
            conditions.append(AuditLog.timestamp <= date_to)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_entity_history(self, entity_type: str, entity_id: UUID, limit: int = 50) -> List[AuditLog]:
        """Get the complete history for a specific entity.

        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            limit: Maximum number of entries

        Returns:
            List of audit log entries for the entity
        """
        return await self.query_audit_logs(entity_type=entity_type, entity_id=entity_id, limit=limit)

    async def get_actor_activity(
        self, actor: str, date_from: Optional[datetime] = None, limit: int = 100
    ) -> List[AuditLog]:
        """Get all activity by a specific actor.

        Args:
            actor: User or system identifier
            date_from: Start date for filtering
            limit: Maximum number of entries

        Returns:
            List of audit log entries by the actor
        """
        return await self.query_audit_logs(actor=actor, date_from=date_from, limit=limit)

    async def apply_retention_policy(self, retention_days: int = 365) -> int:
        """Apply retention policy to delete old audit logs.

        Args:
            retention_days: Number of days to retain audit logs

        Returns:
            Number of logs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        # Get logs to delete
        query = select(AuditLog).where(AuditLog.timestamp < cutoff_date)
        result = await self.session.execute(query)
        logs_to_delete = result.scalars().all()

        # Delete old logs
        for log in logs_to_delete:
            await self.session.delete(log)

        await self.session.commit()

        return len(logs_to_delete)

    async def get_audit_statistics(
        self, date_from: Optional[datetime] = None, date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get statistics about audit logs.

        Args:
            date_from: Start date for statistics
            date_to: End date for statistics

        Returns:
            Dictionary with audit statistics
        """
        from sqlalchemy import func

        # Build base query
        query = select(
            AuditLog.entity_type,
            AuditLog.action,
            func.count(AuditLog.id).label("count"),
        )

        # Apply date filters
        conditions = []
        if date_from:
            conditions.append(AuditLog.timestamp >= date_from)
        if date_to:
            conditions.append(AuditLog.timestamp <= date_to)

        if conditions:
            query = query.where(and_(*conditions))

        # Group by entity type and action
        query = query.group_by(AuditLog.entity_type, AuditLog.action)

        result = await self.session.execute(query)
        rows = result.all()

        # Organize statistics
        stats: Dict[str, Any] = {
            "total_entries": sum(row.count for row in rows),
            "by_entity_type": {},
            "by_action": {},
            "detailed": [],
        }

        for row in rows:
            entity_type = row.entity_type
            action = row.action
            count = row.count

            # Aggregate by entity type
            if entity_type not in stats["by_entity_type"]:
                stats["by_entity_type"][entity_type] = 0
            stats["by_entity_type"][entity_type] += count

            # Aggregate by action
            if action not in stats["by_action"]:
                stats["by_action"][action] = 0
            stats["by_action"][action] += count

            # Detailed breakdown
            stats["detailed"].append({"entity_type": entity_type, "action": action, "count": count})

        return stats

    async def enrich_audit_entry(self, audit_log: AuditLog, enrichment_data: Dict[str, Any]) -> AuditLog:
        """Enrich an existing audit log entry with additional data.

        Args:
            audit_log: Existing audit log entry
            enrichment_data: Additional data to add

        Returns:
            Updated audit log entry
        """
        if audit_log.audit_metadata is None:
            audit_log.audit_metadata = {}

        audit_log.audit_metadata.update(enrichment_data)

        self.session.add(audit_log)
        await self.session.commit()
        await self.session.refresh(audit_log)

        return audit_log
