"""Unit tests for synchronization API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from services.tracklist_service.src.models.synchronization import SyncEvent


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_sync_service():
    """Create a mock synchronization service."""
    with patch("services.tracklist_service.src.api.sync_endpoints.SynchronizationService") as mock:
        service = mock.return_value
        service.trigger_manual_sync = AsyncMock()
        service.get_sync_status = AsyncMock()
        service.update_sync_configuration = AsyncMock()
        service.schedule_sync = AsyncMock()
        service.cancel_scheduled_sync = AsyncMock()
        yield service


@pytest.fixture
def mock_version_service():
    """Create a mock version service."""
    with patch("services.tracklist_service.src.api.sync_endpoints.VersionService") as mock:
        service = mock.return_value
        service.get_version_history = AsyncMock()
        service.get_version = AsyncMock()
        service.rollback_to_version = AsyncMock()
        service.compare_versions = AsyncMock()
        yield service


@pytest.fixture
def mock_conflict_service():
    """Create a mock conflict resolution service."""
    with patch("services.tracklist_service.src.api.sync_endpoints.ConflictResolutionService") as mock:
        service = mock.return_value
        service.resolve_conflicts = AsyncMock()
        yield service


@pytest.fixture
def mock_audit_service():
    """Create a mock audit service."""
    with patch("services.tracklist_service.src.api.sync_endpoints.AuditService") as mock:
        service = mock.return_value
        service.query_audit_logs = AsyncMock()
        yield service


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from fastapi import FastAPI

    from services.tracklist_service.src.api.sync_endpoints import router

    app = FastAPI()
    app.include_router(router)

    return TestClient(app)


class TestSyncControlEndpoints:
    """Test sync control endpoints."""

    @pytest.mark.asyncio
    async def test_trigger_sync_success(self, test_client, mock_sync_service, mock_db_session):
        """Test successful sync trigger."""
        tracklist_id = uuid4()

        mock_sync_service.trigger_manual_sync.return_value = {
            "status": "completed",
            "tracklist_id": str(tracklist_id),
            "changes_applied": 5,
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.post(
                f"/api/v1/tracklists/{tracklist_id}/sync",
                json={"source": "all", "force": False},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_trigger_sync_invalid_source(self, test_client, mock_db_session):
        """Test sync trigger with invalid source."""
        tracklist_id = uuid4()

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.post(
                f"/api/v1/tracklists/{tracklist_id}/sync",
                json={"source": "invalid_source"},
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid sync source" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_sync_status(self, test_client, mock_sync_service, mock_db_session):
        """Test getting sync status."""
        tracklist_id = uuid4()

        mock_sync_service.get_sync_status.return_value = {
            "tracklist_id": str(tracklist_id),
            "is_syncing": False,
            "sync_enabled": True,
            "last_sync_at": datetime.utcnow().isoformat(),
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(f"/api/v1/tracklists/{tracklist_id}/sync/status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["sync_enabled"] is True

    @pytest.mark.asyncio
    async def test_update_sync_config(self, test_client, mock_sync_service, mock_db_session):
        """Test updating sync configuration."""
        tracklist_id = uuid4()

        mock_sync_service.update_sync_configuration.return_value = {
            "status": "updated",
            "tracklist_id": str(tracklist_id),
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.put(
                f"/api/v1/tracklists/{tracklist_id}/sync/config",
                json={
                    "sync_enabled": True,
                    "sync_frequency": "daily",
                    "auto_accept_threshold": 0.9,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_schedule_sync(self, test_client, mock_sync_service, mock_db_session):
        """Test scheduling automatic sync."""
        tracklist_id = uuid4()

        mock_sync_service.schedule_sync.return_value = {
            "status": "scheduled",
            "tracklist_id": str(tracklist_id),
            "frequency": "daily",
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.post(f"/api/v1/tracklists/{tracklist_id}/sync/schedule?frequency=daily&source=all")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_cancel_scheduled_sync(self, test_client, mock_sync_service, mock_db_session):
        """Test canceling scheduled sync."""
        tracklist_id = uuid4()

        mock_sync_service.cancel_scheduled_sync.return_value = {
            "status": "cancelled",
            "tracklist_id": str(tracklist_id),
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.delete(f"/api/v1/tracklists/{tracklist_id}/sync/schedule")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "cancelled"


class TestVersionHistoryEndpoints:
    """Test version history endpoints."""

    @pytest.mark.asyncio
    async def test_get_version_history(self, test_client, mock_version_service, mock_db_session):
        """Test getting version history."""
        tracklist_id = uuid4()

        # Create mock versions
        mock_version = MagicMock()
        mock_version.id = uuid4()
        mock_version.version_number = 1
        mock_version.created_at = datetime.utcnow()
        mock_version.created_by = "user"
        mock_version.change_type = "manual"
        mock_version.change_summary = "Test change"
        mock_version.is_current = True

        mock_version_service.get_version_history.return_value = [mock_version]

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(f"/api/v1/tracklists/{tracklist_id}/versions?limit=20&offset=0")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["versions"]) == 1

    @pytest.mark.asyncio
    async def test_get_version_details(self, test_client, mock_version_service, mock_db_session):
        """Test getting version details."""
        tracklist_id = uuid4()
        version_id = uuid4()

        # Create mock version
        mock_version = MagicMock()
        mock_version.id = version_id
        mock_version.tracklist_id = tracklist_id
        mock_version.version_number = 1
        mock_version.created_at = datetime.utcnow()
        mock_version.created_by = "user"
        mock_version.change_type = "manual"
        mock_version.change_summary = "Test change"
        mock_version.is_current = True
        mock_version.content = {"tracks": []}
        mock_version.metadata = {}

        mock_version_service.get_version.return_value = mock_version

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(f"/api/v1/tracklists/{tracklist_id}/versions/{version_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["version_id"] == str(version_id)

    @pytest.mark.asyncio
    async def test_compare_versions(self, test_client, mock_version_service, mock_db_session):
        """Test comparing two versions."""
        tracklist_id = uuid4()
        version1_id = uuid4()
        version2_id = uuid4()

        # Create mock versions
        mock_v1 = MagicMock()
        mock_v1.id = version1_id
        mock_v1.tracklist_id = tracklist_id
        mock_v1.version_number = 1
        mock_v1.created_at = datetime.utcnow()

        mock_v2 = MagicMock()
        mock_v2.id = version2_id
        mock_v2.tracklist_id = tracklist_id
        mock_v2.version_number = 2
        mock_v2.created_at = datetime.utcnow()

        mock_version_service.get_version.side_effect = [mock_v1, mock_v2]
        mock_version_service.compare_versions.return_value = {
            "tracks_added": 1,
            "tracks_removed": 0,
            "tracks_modified": 2,
        }

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(
                f"/api/v1/tracklists/{tracklist_id}/versions/compare?version1={version1_id}&version2={version2_id}"
            )

        assert response.status_code == status.HTTP_200_OK
        assert "differences" in response.json()


class TestConflictResolutionEndpoints:
    """Test conflict resolution endpoints."""

    @pytest.mark.asyncio
    async def test_get_pending_conflicts(self, test_client, mock_db_session):
        """Test getting pending conflicts."""
        tracklist_id = uuid4()

        # Create mock sync event with conflict
        mock_event = MagicMock(spec=SyncEvent)
        mock_event.id = uuid4()
        mock_event.created_at = datetime.utcnow()
        mock_event.source = "1001tracklists"
        mock_event.conflict_data = {"conflicts": [{"id": "conflict1", "type": "track_modified"}]}

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_event]
        mock_db_session.execute.return_value = mock_result

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(f"/api/v1/tracklists/{tracklist_id}/conflicts")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["pending_conflicts"]) == 1

    @pytest.mark.asyncio
    async def test_resolve_conflicts(self, test_client, mock_conflict_service, mock_db_session):
        """Test resolving conflicts."""
        tracklist_id = uuid4()

        mock_conflict_service.resolve_conflicts.return_value = (True, None)

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.post(
                f"/api/v1/tracklists/{tracklist_id}/conflicts/resolve",
                json={
                    "resolutions": [
                        {
                            "conflict_id": "conflict1",
                            "strategy": "use_proposed",
                        }
                    ]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "resolved"


class TestAuditTrailEndpoint:
    """Test audit trail endpoint."""

    @pytest.mark.asyncio
    async def test_get_audit_trail(self, test_client, mock_audit_service, mock_db_session):
        """Test getting audit trail."""
        tracklist_id = uuid4()

        # Create mock audit log
        mock_log = MagicMock()
        mock_log.id = uuid4()
        mock_log.action = "updated"
        mock_log.actor = "user"
        mock_log.timestamp = datetime.utcnow()
        mock_log.changes = {"tracks": "modified"}
        mock_log.audit_metadata = {}

        mock_audit_service.query_audit_logs.return_value = [mock_log]

        with patch("services.tracklist_service.src.api.sync_endpoints.get_db", return_value=mock_db_session):
            response = test_client.get(f"/api/v1/tracklists/{tracklist_id}/audit?limit=50&offset=0")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["audit_logs"]) == 1
