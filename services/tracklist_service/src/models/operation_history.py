"""
Operation history models for admin operations tracking.

This module provides database models and repository for storing
and retrieving admin operation history.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Index, String, Text, func, select
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import (  # type: ignore[attr-defined]  # SQLAlchemy 2.0 features; project uses 2.0.43 but type stubs are 1.4.x
    Mapped,
    mapped_column,
)

from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import Base


class OperationHistory(Base):
    """Database model for admin operation history."""

    __tablename__ = "admin_operation_history"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    operation_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    admin_user: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    affected_resources: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Create composite index for common queries
    __table_args__ = (
        Index("ix_admin_operation_history_user_type", "admin_user", "operation_type"),
        Index("ix_admin_operation_history_timestamp_type", "timestamp", "operation_type"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert operation history to dictionary."""
        return {
            "id": str(self.id),
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "admin_user": self.admin_user,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "affected_resources": self.affected_resources,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class OperationHistoryRepository:
    """Repository for admin operation history."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    def create_operation(
        self,
        operation_id: str,
        operation_type: str,
        admin_user: str,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        affected_resources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationHistory:
        """
        Create a new operation history record.

        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            admin_user: Username of admin who performed operation
            details: Operation details
            success: Whether operation was successful
            error_message: Error message if operation failed
            affected_resources: List of affected resource IDs
            metadata: Additional metadata

        Returns:
            Created operation history record
        """
        with self.db_manager.get_db_session() as session:
            operation = OperationHistory(
                operation_id=operation_id,
                operation_type=operation_type,
                timestamp=datetime.now(UTC),
                admin_user=admin_user,
                details=details or {},
                success=success,
                error_message=error_message,
                affected_resources=affected_resources,
                metadata=metadata,
            )
            session.add(operation)
            session.flush()
            session.refresh(operation)
            return operation

    def get_by_operation_id(self, operation_id: str) -> OperationHistory | None:
        """
        Get operation by operation ID.

        Args:
            operation_id: Operation ID

        Returns:
            Operation history record or None
        """
        with self.db_manager.get_db_session() as session:
            stmt = select(OperationHistory).where(OperationHistory.operation_id == operation_id)
            result: OperationHistory | None = session.execute(stmt).scalar_one_or_none()
            return result

    def get_by_user(
        self,
        admin_user: str,
        limit: int = 50,
        operation_type: str | None = None,
    ) -> list[OperationHistory]:
        """
        Get operations by admin user.

        Args:
            admin_user: Admin username
            limit: Maximum number of records to return
            operation_type: Optional filter by operation type

        Returns:
            List of operation history records
        """
        with self.db_manager.get_db_session() as session:
            stmt = select(OperationHistory).where(OperationHistory.admin_user == admin_user)

            if operation_type:
                stmt = stmt.where(OperationHistory.operation_type == operation_type)

            # Handle potential None values in timestamp ordering
            timestamp_attr = getattr(OperationHistory, "timestamp", None)
            if timestamp_attr is not None and hasattr(timestamp_attr, "desc"):
                stmt = stmt.order_by(timestamp_attr.desc()).limit(limit)
            else:
                stmt = stmt.limit(limit)

            result = session.execute(stmt)
            return list(result.scalars().all())

    def get_recent_operations(
        self,
        limit: int = 50,
        operation_type: str | None = None,
        admin_user: str | None = None,
        success_only: bool | None = None,
    ) -> list[OperationHistory]:
        """
        Get recent operations with optional filters.

        Args:
            limit: Maximum number of records to return
            operation_type: Optional filter by operation type
            admin_user: Optional filter by admin user
            success_only: Optional filter for successful operations only

        Returns:
            List of operation history records
        """
        with self.db_manager.get_db_session() as session:
            stmt = select(OperationHistory)

            if operation_type:
                stmt = stmt.where(OperationHistory.operation_type == operation_type)

            if admin_user:
                stmt = stmt.where(OperationHistory.admin_user == admin_user)

            if success_only is not None:
                stmt = stmt.where(OperationHistory.success == success_only)

            # Handle potential None values in timestamp ordering
            timestamp_attr = getattr(OperationHistory, "timestamp", None)
            if timestamp_attr is not None and hasattr(timestamp_attr, "desc"):
                stmt = stmt.order_by(timestamp_attr.desc()).limit(limit)
            else:
                stmt = stmt.limit(limit)

            result = session.execute(stmt)
            return list(result.scalars().all())

    def get_operations_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        operation_type: str | None = None,
    ) -> list[OperationHistory]:
        """
        Get operations within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            operation_type: Optional filter by operation type

        Returns:
            List of operation history records
        """
        with self.db_manager.get_db_session() as session:
            # Validate datetime parameters to avoid None comparison errors
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date cannot be None")

            # Get timestamp column safely to avoid None attribute access
            timestamp_col = getattr(OperationHistory, "timestamp", None)
            if timestamp_col is None:
                raise RuntimeError("OperationHistory.timestamp column not available")

            stmt = select(OperationHistory).where(
                timestamp_col >= start_date,
                timestamp_col <= end_date,
            )

            if operation_type:
                stmt = stmt.where(OperationHistory.operation_type == operation_type)

            # Handle potential None values in timestamp ordering
            if hasattr(timestamp_col, "desc"):
                stmt = stmt.order_by(timestamp_col.desc())
            # If desc() is not available, the ordering will be omitted

            result = session.execute(stmt)
            return list(result.scalars().all())

    def update_operation_status(
        self,
        operation_id: str,
        success: bool,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationHistory | None:
        """
        Update operation status.

        Args:
            operation_id: Operation ID
            success: Whether operation was successful
            error_message: Error message if operation failed
            metadata: Additional metadata to merge

        Returns:
            Updated operation or None if not found
        """
        with self.db_manager.get_db_session() as session:
            stmt = select(OperationHistory).where(OperationHistory.operation_id == operation_id)
            operation = session.execute(stmt).scalar_one_or_none()

            if not operation:
                return None

            operation.success = success
            if error_message:
                operation.error_message = error_message

            if metadata:
                current_metadata = operation.metadata or {}
                current_metadata.update(metadata)
                operation.metadata = current_metadata

            session.flush()
            session.refresh(operation)
            result: OperationHistory = operation
            return result

    def get_operation_statistics(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get operation statistics for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            Statistics dictionary
        """
        with self.db_manager.get_db_session() as session:
            cutoff_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            # Get all operations in the date range - validate cutoff_date is not None
            if cutoff_date is None:
                raise ValueError("cutoff_date cannot be None")
            # Get timestamp column safely to avoid None attribute access
            timestamp_col = getattr(OperationHistory, "timestamp", None)
            if timestamp_col is None:
                raise RuntimeError("OperationHistory.timestamp column not available")
            stmt = select(OperationHistory).where(timestamp_col >= cutoff_date)
            operations = list(session.execute(stmt).scalars().all())

            # Calculate statistics
            total_operations = len(operations)
            successful_operations = sum(1 for op in operations if op.success)
            failed_operations = total_operations - successful_operations

            # Group by operation type
            operations_by_type: dict[str, int] = {}
            for op in operations:
                operations_by_type[op.operation_type] = operations_by_type.get(op.operation_type, 0) + 1

            # Group by user
            operations_by_user: dict[str, int] = {}
            for op in operations:
                operations_by_user[op.admin_user] = operations_by_user.get(op.admin_user, 0) + 1

            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                "operations_by_type": operations_by_type,
                "operations_by_user": operations_by_user,
                "date_range": {
                    "start": cutoff_date.isoformat(),
                    "end": datetime.now(UTC).isoformat(),
                    "days": days,
                },
            }

    def cleanup_old_operations(self, days: int = 90) -> int:
        """
        Delete operation history older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted records
        """
        with self.db_manager.get_db_session() as session:
            cutoff_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            # Validate cutoff_date before comparison to avoid None comparison errors
            if cutoff_date is None:
                raise ValueError("cutoff_date cannot be None")
            # Get timestamp column safely to avoid None attribute access
            timestamp_col = getattr(OperationHistory, "timestamp", None)
            if timestamp_col is None:
                raise RuntimeError("OperationHistory.timestamp column not available")
            stmt = select(OperationHistory).where(timestamp_col < cutoff_date)
            old_operations = list(session.execute(stmt).scalars().all())

            for operation in old_operations:
                session.delete(operation)

            return len(old_operations)
