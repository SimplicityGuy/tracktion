"""Unit tests for integrity monitoring functionality."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from services.tracklist_service.src.services.integrity_monitor import IntegrityMonitor, get_integrity_monitor


@pytest.fixture
def mock_session_factory():
    """Create a mock session factory."""
    factory = Mock()
    session = Mock()
    factory.return_value = session
    return factory


@pytest.fixture
def monitor(mock_session_factory):
    """Create an IntegrityMonitor instance."""
    return IntegrityMonitor(mock_session_factory, check_interval_minutes=60, auto_clean=False)


@pytest.mark.asyncio
class TestIntegrityMonitor:
    """Test suite for IntegrityMonitor."""

    async def test_start_monitor(self, monitor):
        """Test starting the integrity monitor."""
        assert monitor.is_running is False

        await monitor.start()

        assert monitor.is_running is True
        assert monitor._monitor_task is not None

        # Clean up
        await monitor.stop()

    async def test_start_monitor_already_running(self, monitor):
        """Test starting monitor when already running."""
        await monitor.start()

        with patch("services.tracklist_service.src.services.integrity_monitor.logger") as mock_logger:
            await monitor.start()
            mock_logger.warning.assert_called_once()

        # Clean up
        await monitor.stop()

    async def test_stop_monitor(self, monitor):
        """Test stopping the integrity monitor."""
        await monitor.start()
        assert monitor.is_running is True

        await monitor.stop()

        assert monitor.is_running is False

    async def test_stop_monitor_not_running(self, monitor):
        """Test stopping monitor when not running."""
        assert monitor.is_running is False

        # Should not raise any errors
        await monitor.stop()

        assert monitor.is_running is False

    async def test_perform_check_no_orphans(self, monitor, mock_session_factory):
        """Test performing check with no orphaned records."""
        session = mock_session_factory.return_value

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.return_value = {"metadata": 0, "tracklists": 0, "rename_proposals": 0}
            validator.validate_foreign_keys.return_value = []
            validator.validate_indexes.return_value = []

            await monitor._perform_check()

            validator.check_orphaned_records.assert_called_once()
            assert monitor.last_check is not None
            session.close.assert_called_once()

    async def test_perform_check_with_orphans_no_auto_clean(self, monitor, mock_session_factory):
        """Test performing check with orphaned records but auto-clean disabled."""
        _ = mock_session_factory.return_value
        monitor.auto_clean = False

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.return_value = {"metadata": 5, "tracklists": 2, "rename_proposals": 0}
            validator.validate_foreign_keys.return_value = []
            validator.validate_indexes.return_value = []

            with patch("services.tracklist_service.src.services.integrity_monitor.logger") as mock_logger:
                await monitor._perform_check()

                # Should log warning about orphaned records
                assert any("orphaned records" in str(call) for call in mock_logger.warning.call_args_list)
                # Should not clean since auto_clean is False
                validator.clean_orphaned_records.assert_not_called()

    async def test_perform_check_with_orphans_auto_clean(self, monitor, mock_session_factory):
        """Test performing check with orphaned records and auto-clean enabled."""
        _ = mock_session_factory.return_value
        monitor.auto_clean = True

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.return_value = {"metadata": 5, "tracklists": 2, "rename_proposals": 0}
            validator.clean_orphaned_records.return_value = {"metadata": 5, "tracklists": 2, "rename_proposals": 0}
            validator.validate_foreign_keys.return_value = []
            validator.validate_indexes.return_value = []

            await monitor._perform_check()

            # Should clean since auto_clean is True
            validator.clean_orphaned_records.assert_called_once_with(dry_run=False)

    async def test_perform_check_with_fk_issues(self, monitor, mock_session_factory):
        """Test performing check with foreign key issues."""
        _ = mock_session_factory.return_value

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.return_value = {"metadata": 0, "tracklists": 0, "rename_proposals": 0}
            validator.validate_foreign_keys.return_value = [
                ("metadata_recording_id_fkey", "metadata", False),  # No CASCADE
                ("tracklists_recording_id_fkey", "tracklists", True),
            ]
            validator.validate_indexes.return_value = []

            with patch("services.tracklist_service.src.services.integrity_monitor.logger") as mock_logger:
                await monitor._perform_check()

                # Should log error about FK constraints
                mock_logger.error.assert_called_once()

    async def test_perform_check_with_missing_indexes(self, monitor, mock_session_factory):
        """Test performing check with missing indexes."""
        _ = mock_session_factory.return_value

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.return_value = {"metadata": 0, "tracklists": 0, "rename_proposals": 0}
            validator.validate_foreign_keys.return_value = []
            validator.validate_indexes.return_value = [
                ("idx_metadata_recording_id", "metadata", False),  # Missing
            ]

            with patch("services.tracklist_service.src.services.integrity_monitor.logger") as mock_logger:
                await monitor._perform_check()

                # Should log warning about missing indexes
                mock_logger.warning.assert_called_once()

    async def test_perform_check_error_handling(self, monitor, mock_session_factory):
        """Test error handling during integrity check."""
        session = mock_session_factory.return_value

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.check_orphaned_records.side_effect = Exception("Database error")

            with patch("services.tracklist_service.src.services.integrity_monitor.logger") as mock_logger:
                await monitor._perform_check()

                # Should log error but not raise
                mock_logger.error.assert_called_once()
                # Session should still be closed
                session.close.assert_called_once()

    async def test_check_now(self, monitor, mock_session_factory):
        """Test performing immediate check."""
        session = mock_session_factory.return_value

        with patch("services.tracklist_service.src.services.integrity_monitor.IntegrityValidator") as MockValidator:
            validator = MockValidator.return_value
            validator.run_full_validation.return_value = {
                "orphaned_records": {"metadata": 0},
                "foreign_keys": [],
                "check_constraints": [],
                "indexes": [],
                "is_valid": True,
            }

            result = await monitor.check_now()

            assert result["is_valid"] is True
            assert monitor.last_check is not None
            session.close.assert_called_once()

    def test_get_status_not_running(self, monitor):
        """Test getting status when monitor is not running."""
        status = monitor.get_status()

        assert status["is_running"] is False
        assert status["check_interval_minutes"] == 60
        assert status["auto_clean"] is False
        assert status["last_check"] is None
        assert status["next_check"] is None

    def test_get_status_running(self, monitor):
        """Test getting status when monitor is running."""
        monitor.is_running = True
        monitor.last_check = datetime.utcnow()

        status = monitor.get_status()

        assert status["is_running"] is True
        assert status["check_interval_minutes"] == 60
        assert status["auto_clean"] is False
        assert status["last_check"] is not None
        assert status["next_check"] is not None


class TestIntegrityMonitorSingleton:
    """Test singleton behavior of integrity monitor."""

    def test_get_integrity_monitor_creates_instance(self):
        """Test that get_integrity_monitor creates new instance."""
        # Reset singleton
        import services.tracklist_service.src.services.integrity_monitor as monitor_module

        monitor_module._monitor_instance = None

        session_factory = Mock()
        monitor = get_integrity_monitor(session_factory=session_factory, check_interval_minutes=30, auto_clean=True)

        assert monitor is not None
        assert monitor.check_interval == timedelta(minutes=30)
        assert monitor.auto_clean is True

    def test_get_integrity_monitor_returns_existing(self):
        """Test that get_integrity_monitor returns existing instance."""
        # Reset singleton
        import services.tracklist_service.src.services.integrity_monitor as monitor_module

        monitor_module._monitor_instance = None

        session_factory = Mock()
        monitor1 = get_integrity_monitor(session_factory=session_factory)
        monitor2 = get_integrity_monitor()  # No args

        assert monitor1 is monitor2

    def test_get_integrity_monitor_requires_factory_for_new(self):
        """Test that session_factory is required for new instance."""
        # Reset singleton
        import services.tracklist_service.src.services.integrity_monitor as monitor_module

        monitor_module._monitor_instance = None

        with pytest.raises(ValueError, match="session_factory is required"):
            get_integrity_monitor()
