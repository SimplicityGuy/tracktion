"""Unit tests for audit service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.synchronization import AuditLog
from services.tracklist_service.src.services.audit_service import AuditService


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def audit_service(mock_session):
    """Create audit service instance."""
    return AuditService(mock_session)


class TestAuditService:
    """Test AuditService methods."""

    @pytest.mark.asyncio
    async def test_log_change(self, audit_service, mock_session):
        """Test logging a change."""
        entity_id = uuid4()
        changes = {"field": "value"}
        metadata = {"ip": "192.168.1.1"}

        await audit_service.log_change(
            entity_type="tracklist",
            entity_id=entity_id,
            action="updated",
            actor="user123",
            changes=changes,
            metadata=metadata,
        )

        # Verify audit log was added
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

        # Check the audit log object
        audit_log = mock_session.add.call_args[0][0]
        assert isinstance(audit_log, AuditLog)
        assert audit_log.entity_type == "tracklist"
        assert audit_log.entity_id == entity_id
        assert audit_log.action == "updated"
        assert audit_log.actor == "user123"
        assert audit_log.changes == changes
        assert audit_log.audit_metadata == metadata

    @pytest.mark.asyncio
    async def test_log_tracklist_change(self, audit_service):
        """Test logging a tracklist-specific change."""
        tracklist_id = uuid4()
        before = {"tracks": 10}
        after = {"tracks": 12}

        # Mock the log_change method
        audit_service.log_change = AsyncMock(return_value=MagicMock())

        await audit_service.log_tracklist_change(
            tracklist_id=tracklist_id,
            action="tracks_added",
            actor="system",
            before=before,
            after=after,
            metadata={"source": "import"},
        )

        # Verify log_change was called correctly
        audit_service.log_change.assert_called_once_with(
            entity_type="tracklist",
            entity_id=tracklist_id,
            action="tracks_added",
            actor="system",
            changes={"before": before, "after": after},
            metadata={"source": "import"},
        )

    @pytest.mark.asyncio
    async def test_log_cue_file_change(self, audit_service):
        """Test logging a CUE file-specific change."""
        cue_file_id = uuid4()
        changes = {"format": "standard"}

        # Mock the log_change method
        audit_service.log_change = AsyncMock(return_value=MagicMock())

        await audit_service.log_cue_file_change(
            cue_file_id=cue_file_id,
            action="generated",
            actor="system",
            changes=changes,
        )

        # Verify log_change was called correctly
        audit_service.log_change.assert_called_once_with(
            entity_type="cue_file",
            entity_id=cue_file_id,
            action="generated",
            actor="system",
            changes=changes,
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_filters(self, audit_service, mock_session):
        """Test querying audit logs with various filters."""
        entity_id = uuid4()
        date_from = datetime.utcnow() - timedelta(days=7)
        date_to = datetime.utcnow()

        # Mock query result
        mock_logs = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_logs
        mock_session.execute.return_value = mock_result

        result = await audit_service.query_audit_logs(
            entity_type="tracklist",
            entity_id=entity_id,
            action="updated",
            actor="user123",
            date_from=date_from,
            date_to=date_to,
            limit=50,
            offset=10,
        )

        assert result == mock_logs
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_audit_logs_no_filters(self, audit_service, mock_session):
        """Test querying audit logs without filters."""
        mock_logs = [MagicMock(), MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_logs
        mock_session.execute.return_value = mock_result

        result = await audit_service.query_audit_logs()

        assert len(result) == 3
        assert result == mock_logs

    @pytest.mark.asyncio
    async def test_get_entity_history(self, audit_service):
        """Test getting entity history."""
        entity_id = uuid4()
        mock_logs = [MagicMock(), MagicMock()]

        # Mock query_audit_logs
        audit_service.query_audit_logs = AsyncMock(return_value=mock_logs)

        result = await audit_service.get_entity_history("tracklist", entity_id, limit=25)

        assert result == mock_logs
        audit_service.query_audit_logs.assert_called_once_with(entity_type="tracklist", entity_id=entity_id, limit=25)

    @pytest.mark.asyncio
    async def test_get_actor_activity(self, audit_service):
        """Test getting actor activity."""
        date_from = datetime.utcnow() - timedelta(hours=24)
        mock_logs = [MagicMock(), MagicMock()]

        # Mock query_audit_logs
        audit_service.query_audit_logs = AsyncMock(return_value=mock_logs)

        result = await audit_service.get_actor_activity("user123", date_from=date_from)

        assert result == mock_logs
        audit_service.query_audit_logs.assert_called_once_with(actor="user123", date_from=date_from, limit=100)

    @pytest.mark.asyncio
    async def test_apply_retention_policy(self, audit_service, mock_session):
        """Test applying retention policy to delete old logs."""
        # Create mock old logs
        old_logs = [
            MagicMock(id=uuid4()),
            MagicMock(id=uuid4()),
            MagicMock(id=uuid4()),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = old_logs
        mock_session.execute.return_value = mock_result

        count = await audit_service.apply_retention_policy(retention_days=30)

        assert count == 3
        assert mock_session.delete.call_count == 3
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_audit_statistics(self, audit_service, mock_session):
        """Test getting audit statistics."""
        # Mock query results
        mock_rows = [
            MagicMock(entity_type="tracklist", action="created", count=5),
            MagicMock(entity_type="tracklist", action="updated", count=10),
            MagicMock(entity_type="cue_file", action="generated", count=3),
        ]

        mock_result = MagicMock()
        mock_result.all.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        stats = await audit_service.get_audit_statistics()

        assert stats["total_entries"] == 18
        assert stats["by_entity_type"]["tracklist"] == 15
        assert stats["by_entity_type"]["cue_file"] == 3
        assert stats["by_action"]["created"] == 5
        assert stats["by_action"]["updated"] == 10
        assert stats["by_action"]["generated"] == 3
        assert len(stats["detailed"]) == 3

    @pytest.mark.asyncio
    async def test_get_audit_statistics_with_dates(self, audit_service, mock_session):
        """Test getting audit statistics with date filters."""
        date_from = datetime.utcnow() - timedelta(days=7)
        date_to = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        stats = await audit_service.get_audit_statistics(date_from=date_from, date_to=date_to)

        assert stats["total_entries"] == 0
        assert len(stats["by_entity_type"]) == 0
        assert len(stats["by_action"]) == 0

    @pytest.mark.asyncio
    async def test_enrich_audit_entry(self, audit_service, mock_session):
        """Test enriching an audit log entry."""
        audit_log = MagicMock(spec=AuditLog)
        audit_log.audit_metadata = {"existing": "data"}

        enrichment_data = {"new_field": "new_value", "extra": "info"}

        await audit_service.enrich_audit_entry(audit_log, enrichment_data)

        # Verify metadata was updated
        assert audit_log.audit_metadata["existing"] == "data"
        assert audit_log.audit_metadata["new_field"] == "new_value"
        assert audit_log.audit_metadata["extra"] == "info"

        mock_session.add.assert_called_once_with(audit_log)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_enrich_audit_entry_no_existing_metadata(self, audit_service, mock_session):
        """Test enriching an audit log entry with no existing metadata."""
        audit_log = MagicMock(spec=AuditLog)
        audit_log.audit_metadata = None

        enrichment_data = {"field": "value"}

        await audit_service.enrich_audit_entry(audit_log, enrichment_data)

        # Verify metadata was created and updated
        assert audit_log.audit_metadata == {"field": "value"}
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
