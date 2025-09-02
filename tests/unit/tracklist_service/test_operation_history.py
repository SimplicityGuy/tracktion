"""Tests for operation history storage and retrieval."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.operation_history import (
    OperationHistory,
    OperationHistoryRepository,
)
from shared.core_types.src.database import DatabaseManager


class TestOperationHistoryModel:
    """Test OperationHistory model."""

    def test_operation_history_creation(self):
        """Test creating an operation history record."""
        operation = OperationHistory(
            id=uuid4(),
            operation_id="test-operation-123",
            operation_type="selector_update",
            timestamp=datetime.now(UTC),
            admin_user="test_admin",
            details={"field": "value"},
            success=True,
            error_message=None,
            affected_resources=["resource1", "resource2"],
            metadata={"key": "value"},
            created_at=datetime.now(UTC),
        )

        assert operation.operation_id == "test-operation-123"
        assert operation.operation_type == "selector_update"
        assert operation.admin_user == "test_admin"
        assert operation.success is True
        assert operation.details == {"field": "value"}
        assert operation.affected_resources == ["resource1", "resource2"]

    def test_operation_history_to_dict(self):
        """Test converting operation history to dictionary."""
        now = datetime.now(UTC)
        operation_id = uuid4()

        operation = OperationHistory(
            id=operation_id,
            operation_id="test-operation-456",
            operation_type="manual_correction",
            timestamp=now,
            admin_user="admin_user",
            details={"correction": "data"},
            success=False,
            error_message="Test error",
            affected_resources=None,
            metadata=None,
            created_at=now,
        )

        result = operation.to_dict()

        assert result["id"] == str(operation_id)
        assert result["operation_id"] == "test-operation-456"
        assert result["operation_type"] == "manual_correction"
        assert result["timestamp"] == now.isoformat()
        assert result["admin_user"] == "admin_user"
        assert result["details"] == {"correction": "data"}
        assert result["success"] is False
        assert result["error_message"] == "Test error"
        assert result["affected_resources"] is None
        assert result["created_at"] == now.isoformat()


class TestOperationHistoryRepository:
    """Test OperationHistoryRepository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def operation_repo(self, mock_db_manager, mock_session):
        """Create operation history repository with mocked dependencies."""
        # Setup context manager properly
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_db_manager.get_db_session.return_value = context_manager
        return OperationHistoryRepository(mock_db_manager)

    def test_create_operation(self, operation_repo, mock_session):
        """Test creating an operation history record."""
        # Mock the session behavior
        created_operation = None

        def capture_operation(op):
            nonlocal created_operation
            created_operation = op
            op.id = uuid4()
            op.created_at = datetime.now(UTC)

        mock_session.add.side_effect = capture_operation

        # Create operation
        operation = operation_repo.create_operation(
            operation_id="test-op-789",
            operation_type="parser_rollback",
            admin_user="super_admin",
            details={"version": "1.0.0"},
            success=True,
        )

        # Verify
        assert operation is not None
        assert operation.operation_id == "test-op-789"
        assert operation.operation_type == "parser_rollback"
        assert operation.admin_user == "super_admin"
        assert operation.details == {"version": "1.0.0"}
        assert operation.success is True

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_get_by_operation_id(self, operation_repo, mock_session):
        """Test getting operation by operation ID."""
        # Mock operation
        mock_operation = Mock(spec=OperationHistory)
        mock_operation.operation_id = "test-op-123"

        # Mock query execution
        mock_stmt_result = Mock()
        mock_stmt_result.scalar_one_or_none.return_value = mock_operation
        mock_session.execute.return_value = mock_stmt_result

        # Get operation
        result = operation_repo.get_by_operation_id("test-op-123")

        assert result == mock_operation
        mock_session.execute.assert_called_once()

    def test_get_by_user(self, operation_repo, mock_session):
        """Test getting operations by admin user."""
        # Mock operations
        mock_operations = [
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
        ]

        # Mock query execution
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_operations
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get operations
        result = operation_repo.get_by_user("test_admin", limit=10)

        assert result == mock_operations
        assert len(result) == 2
        mock_session.execute.assert_called_once()

    def test_get_recent_operations(self, operation_repo, mock_session):
        """Test getting recent operations with filters."""
        # Mock operations
        mock_operations = [
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
        ]

        # Mock query execution
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_operations
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get recent operations
        result = operation_repo.get_recent_operations(
            limit=20,
            operation_type="selector_update",
            admin_user="parser_admin",
            success_only=True,
        )

        assert result == mock_operations
        assert len(result) == 3
        mock_session.execute.assert_called_once()

    def test_get_operations_by_date_range(self, operation_repo, mock_session):
        """Test getting operations within date range."""
        # Mock operations
        mock_operations = [
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
        ]

        # Mock query execution
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_operations
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get operations by date range
        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)

        result = operation_repo.get_operations_by_date_range(
            start_date=start_date,
            end_date=end_date,
            operation_type="manual_correction",
        )

        assert result == mock_operations
        assert len(result) == 2
        mock_session.execute.assert_called_once()

    def test_update_operation_status(self, operation_repo, mock_session):
        """Test updating operation status."""
        # Mock operation
        mock_operation = Mock(spec=OperationHistory)
        mock_operation.operation_id = "test-op-999"
        mock_operation.success = True
        mock_operation.metadata = {"existing": "data"}

        # Mock query execution
        mock_stmt_result = Mock()
        mock_stmt_result.scalar_one_or_none.return_value = mock_operation
        mock_session.execute.return_value = mock_stmt_result

        # Update operation status
        result = operation_repo.update_operation_status(
            operation_id="test-op-999",
            success=False,
            error_message="Operation failed",
            metadata={"new": "info"},
        )

        assert result == mock_operation
        assert mock_operation.success is False
        assert mock_operation.error_message == "Operation failed"
        assert mock_operation.metadata == {"existing": "data", "new": "info"}

        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    def test_update_operation_status_not_found(self, operation_repo, mock_session):
        """Test updating non-existent operation."""
        # Mock query execution - no operation found
        mock_stmt_result = Mock()
        mock_stmt_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_stmt_result

        # Update operation status
        result = operation_repo.update_operation_status(
            operation_id="non-existent",
            success=False,
        )

        assert result is None
        mock_session.flush.assert_not_called()

    def test_get_operation_statistics(self, operation_repo, mock_session):
        """Test getting operation statistics."""
        # Mock operations with different types and users
        mock_operations = [
            Mock(spec=OperationHistory, success=True, operation_type="selector_update", admin_user="admin1"),
            Mock(spec=OperationHistory, success=True, operation_type="selector_update", admin_user="admin2"),
            Mock(spec=OperationHistory, success=False, operation_type="manual_correction", admin_user="admin1"),
            Mock(spec=OperationHistory, success=True, operation_type="parser_rollback", admin_user="admin2"),
        ]

        # Mock query execution
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_operations
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get statistics
        stats = operation_repo.get_operation_statistics(days=30)

        assert stats["total_operations"] == 4
        assert stats["successful_operations"] == 3
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 75.0
        assert stats["operations_by_type"]["selector_update"] == 2
        assert stats["operations_by_type"]["manual_correction"] == 1
        assert stats["operations_by_type"]["parser_rollback"] == 1
        assert stats["operations_by_user"]["admin1"] == 2
        assert stats["operations_by_user"]["admin2"] == 2
        assert stats["date_range"]["days"] == 30

    def test_cleanup_old_operations(self, operation_repo, mock_session):
        """Test cleaning up old operations."""
        # Mock old operations
        old_operations = [
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
            Mock(spec=OperationHistory),
        ]

        # Mock query execution
        mock_scalars = Mock()
        mock_scalars.all.return_value = old_operations
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Cleanup old operations
        count = operation_repo.cleanup_old_operations(days=90)

        assert count == 3
        assert mock_session.delete.call_count == 3
