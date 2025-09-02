"""Repository for RenameProposal database operations."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import and_, func, select

from .database import DatabaseManager
from .models import RenameProposal

logger = logging.getLogger(__name__)


class RenameProposalRepository:
    """Repository for managing RenameProposal entities."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create(self, **kwargs: Any) -> RenameProposal:
        """Create a new rename proposal.

        Args:
            **kwargs: RenameProposal attributes

        Returns:
            Created RenameProposal instance

        Raises:
            SQLAlchemyError: If database operation fails
        """
        with self.db.get_db_session() as session:
            proposal = RenameProposal(**kwargs)
            session.add(proposal)
            session.flush()
            session.refresh(proposal)
            logger.info(f"Created rename proposal: {proposal.id}")
            return proposal

    def get(self, proposal_id: UUID) -> RenameProposal | None:
        """Get a rename proposal by ID.

        Args:
            proposal_id: UUID of the proposal

        Returns:
            RenameProposal instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.id == proposal_id)
            result = session.execute(stmt)
            return cast("RenameProposal | None", result.scalar_one_or_none())

    def get_by_recording(self, recording_id: UUID, status: str | None = None) -> list[RenameProposal]:
        """Get all proposals for a recording.

        Args:
            recording_id: UUID of the recording
            status: Optional status filter

        Returns:
            List of RenameProposal instances
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.recording_id == recording_id)

            if status:
                stmt = stmt.where(RenameProposal.status == status)

            stmt = stmt.order_by(RenameProposal.created_at.desc())
            result = session.execute(stmt)
            return cast("list[RenameProposal]", result.scalars().all())

    def get_pending_proposals(self, limit: int | None = None) -> list[RenameProposal]:
        """Get all pending rename proposals.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of pending RenameProposal instances
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.status == "pending")
            stmt = stmt.order_by(RenameProposal.created_at)

            if limit:
                stmt = stmt.limit(limit)

            result = session.execute(stmt)
            return cast("list[RenameProposal]", result.scalars().all())

    def get_by_status(self, status: str, limit: int | None = None) -> list[RenameProposal]:
        """Get proposals by status.

        Args:
            status: Status to filter by
            limit: Optional limit on number of results

        Returns:
            List of RenameProposal instances
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.status == status)
            stmt = stmt.order_by(RenameProposal.created_at.desc())

            if limit:
                stmt = stmt.limit(limit)

            result = session.execute(stmt)
            return cast("list[RenameProposal]", result.scalars().all())

    def update(self, proposal_id: UUID, **kwargs: Any) -> RenameProposal | None:
        """Update a rename proposal.

        Args:
            proposal_id: UUID of the proposal
            **kwargs: Attributes to update

        Returns:
            Updated RenameProposal instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.id == proposal_id)
            result = session.execute(stmt)
            proposal = cast("RenameProposal | None", result.scalar_one_or_none())

            if not proposal:
                return None

            for key, value in kwargs.items():
                if hasattr(proposal, key):
                    setattr(proposal, key, value)

            proposal.updated_at = datetime.now(UTC)
            session.flush()
            session.refresh(proposal)

            logger.info(f"Updated rename proposal: {proposal_id}")
            return proposal

    def update_status(self, proposal_id: UUID, status: str) -> bool:
        """Update the status of a rename proposal.

        Args:
            proposal_id: UUID of the proposal
            status: New status value

        Returns:
            True if updated successfully
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.id == proposal_id)
            result = session.execute(stmt)
            proposal = cast("RenameProposal | None", result.scalar_one_or_none())

            if not proposal:
                return False

            proposal.status = status
            proposal.updated_at = datetime.now(UTC)
            session.flush()

            logger.info(f"Updated status of proposal {proposal_id} to {status}")
            return True

    def batch_update_status(self, proposal_ids: list[UUID], status: str) -> int:
        """Update status for multiple proposals.

        Args:
            proposal_ids: List of proposal UUIDs
            status: New status value

        Returns:
            Number of proposals updated
        """
        with self.db.get_db_session() as session:
            count = 0
            for proposal_id in proposal_ids:
                stmt = select(RenameProposal).where(RenameProposal.id == proposal_id)
                result = session.execute(stmt)
                proposal = cast("RenameProposal | None", result.scalar_one_or_none())
                if proposal:
                    proposal.status = status
                    proposal.updated_at = datetime.now(UTC)
                    count += 1

            session.flush()
            logger.info(f"Updated status of {count} proposals to {status}")
            return count

    def delete(self, proposal_id: UUID) -> bool:
        """Delete a rename proposal.

        Args:
            proposal_id: UUID of the proposal

        Returns:
            True if deleted successfully
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(RenameProposal.id == proposal_id)
            result = session.execute(stmt)
            proposal = cast("RenameProposal | None", result.scalar_one_or_none())

            if not proposal:
                return False

            session.delete(proposal)
            logger.info(f"Deleted rename proposal: {proposal_id}")
            return True

    def find_conflicts(self, proposed_path: str, recording_id: UUID) -> list[RenameProposal]:
        """Find proposals with conflicting paths.

        Args:
            proposed_path: Path to check for conflicts
            recording_id: Recording ID to exclude from conflict check

        Returns:
            List of conflicting proposals
        """
        with self.db.get_db_session() as session:
            stmt = select(RenameProposal).where(
                and_(
                    RenameProposal.full_proposed_path == proposed_path,
                    RenameProposal.recording_id != recording_id,
                    RenameProposal.status.in_(["pending", "approved"]),
                )
            )

            result = session.execute(stmt)
            return cast("list[RenameProposal]", result.scalars().all())

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about rename proposals.

        Returns:
            Dictionary with statistics
        """
        with self.db.get_db_session() as session:
            # Count by status
            stmt = select(RenameProposal.status, func.count(RenameProposal.id)).group_by(RenameProposal.status)
            result = session.execute(stmt)
            status_counts: dict[str, int] = dict(result.all())

            # Average confidence scores
            stmt = select(func.avg(RenameProposal.confidence_score)).where(RenameProposal.confidence_score.isnot(None))
            result = session.execute(stmt)
            avg_confidence = result.scalar() or 0.0

            # Count proposals with conflicts
            stmt = select(func.count(RenameProposal.id)).where(RenameProposal.conflicts.isnot(None))
            result = session.execute(stmt)
            with_conflicts = result.scalar() or 0

            # Count proposals with warnings
            stmt = select(func.count(RenameProposal.id)).where(RenameProposal.warnings.isnot(None))
            result = session.execute(stmt)
            with_warnings = result.scalar() or 0

            return {
                "total": sum(status_counts.values()),
                "by_status": status_counts,
                "average_confidence": float(avg_confidence),
                "with_conflicts": with_conflicts,
                "with_warnings": with_warnings,
            }

    def cleanup_old_proposals(self, days: int = 30) -> int:
        """Clean up old rejected or applied proposals.

        Args:
            days: Number of days to keep proposals

        Returns:
            Number of proposals deleted
        """
        with self.db.get_db_session() as session:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            stmt = select(RenameProposal).where(
                and_(
                    RenameProposal.status.in_(["rejected", "applied"]),
                    RenameProposal.updated_at < cutoff_date,
                )
            )
            result = session.execute(stmt)
            proposals = cast("list[RenameProposal]", result.scalars().all())

            count = len(proposals)

            for proposal in proposals:
                session.delete(proposal)

            session.flush()
            logger.info(f"Cleaned up {count} old proposals")
            return count
