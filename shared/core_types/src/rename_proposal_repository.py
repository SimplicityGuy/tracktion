"""Repository for RenameProposal database operations."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func

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
            return session.query(RenameProposal).filter(RenameProposal.id == proposal_id).first()  # type: ignore[no-any-return] # SQLAlchemy query methods return Any but we know this returns RenameProposal | None

    def get_by_recording(self, recording_id: UUID, status: str | None = None) -> list[RenameProposal]:
        """Get all proposals for a recording.

        Args:
            recording_id: UUID of the recording
            status: Optional status filter

        Returns:
            List of RenameProposal instances
        """
        with self.db.get_db_session() as session:
            query = session.query(RenameProposal).filter(RenameProposal.recording_id == recording_id)

            if status:
                query = query.filter(RenameProposal.status == status)

            query = query.order_by(RenameProposal.created_at.desc())
            return query.all()  # type: ignore[no-any-return] # SQLAlchemy query methods return Any but we know this returns list[RenameProposal]

    def get_pending_proposals(self, limit: int | None = None) -> list[RenameProposal]:
        """Get all pending rename proposals.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of pending RenameProposal instances
        """
        with self.db.get_db_session() as session:
            query = session.query(RenameProposal).filter(RenameProposal.status == "pending")
            query = query.order_by(RenameProposal.created_at)

            if limit:
                query = query.limit(limit)

            return query.all()  # type: ignore[no-any-return] # SQLAlchemy query methods return Any but we know this returns list[RenameProposal]

    def get_by_status(self, status: str, limit: int | None = None) -> list[RenameProposal]:
        """Get proposals by status.

        Args:
            status: Status to filter by
            limit: Optional limit on number of results

        Returns:
            List of RenameProposal instances
        """
        with self.db.get_db_session() as session:
            query = session.query(RenameProposal).filter(RenameProposal.status == status)
            query = query.order_by(RenameProposal.created_at.desc())

            if limit:
                query = query.limit(limit)

            return query.all()  # type: ignore[no-any-return] # SQLAlchemy query methods return Any but we know this returns list[RenameProposal]

    def update(self, proposal_id: UUID, **kwargs: Any) -> RenameProposal | None:
        """Update a rename proposal.

        Args:
            proposal_id: UUID of the proposal
            **kwargs: Attributes to update

        Returns:
            Updated RenameProposal instance or None if not found
        """
        with self.db.get_db_session() as session:
            proposal = session.query(RenameProposal).filter(RenameProposal.id == proposal_id).first()

            if not proposal:
                return None

            for key, value in kwargs.items():
                if hasattr(proposal, key):
                    setattr(proposal, key, value)

            proposal.updated_at = datetime.now(UTC)
            session.flush()
            session.refresh(proposal)

            logger.info(f"Updated rename proposal: {proposal_id}")
            return proposal  # type: ignore[no-any-return] # RenameProposal object from database refresh, known to be RenameProposal type

    def update_status(self, proposal_id: UUID, status: str) -> bool:
        """Update the status of a rename proposal.

        Args:
            proposal_id: UUID of the proposal
            status: New status value

        Returns:
            True if updated successfully
        """
        with self.db.get_db_session() as session:
            proposal = session.query(RenameProposal).filter(RenameProposal.id == proposal_id).first()

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
                proposal = session.query(RenameProposal).filter(RenameProposal.id == proposal_id).first()
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
            proposal = session.query(RenameProposal).filter(RenameProposal.id == proposal_id).first()

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
            query = session.query(RenameProposal).filter(
                and_(
                    RenameProposal.full_proposed_path == proposed_path,
                    RenameProposal.recording_id != recording_id,
                    RenameProposal.status.in_(["pending", "approved"]),
                )
            )

            return query.all()  # type: ignore[no-any-return] # SQLAlchemy query methods return Any but we know this returns list[RenameProposal]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about rename proposals.

        Returns:
            Dictionary with statistics
        """
        with self.db.get_db_session() as session:
            # Count by status
            query = session.query(RenameProposal.status, func.count(RenameProposal.id)).group_by(RenameProposal.status)
            status_counts: dict[str, int] = dict(query)

            # Average confidence scores
            avg_confidence = (
                session.query(func.avg(RenameProposal.confidence_score))
                .filter(RenameProposal.confidence_score.isnot(None))
                .scalar()
                or 0.0
            )

            # Count proposals with conflicts
            with_conflicts = (
                session.query(func.count(RenameProposal.id)).filter(RenameProposal.conflicts.isnot(None)).scalar() or 0
            )

            # Count proposals with warnings
            with_warnings = (
                session.query(func.count(RenameProposal.id)).filter(RenameProposal.warnings.isnot(None)).scalar() or 0
            )

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

            proposals = (
                session.query(RenameProposal)
                .filter(
                    and_(
                        RenameProposal.status.in_(["rejected", "applied"]),
                        RenameProposal.updated_at < cutoff_date,
                    )
                )
                .all()
            )

            count = len(proposals)

            for proposal in proposals:
                session.delete(proposal)

            session.flush()
            logger.info(f"Cleaned up {count} old proposals")
            return count
