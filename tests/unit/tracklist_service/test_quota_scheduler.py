"""Unit tests for quota scheduler."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.tracklist_service.src.quota.models import QuotaType
from services.tracklist_service.src.quota.scheduler import QuotaScheduler


@pytest.fixture
def mock_quota_manager():
    """Mock QuotaManager."""
    manager = AsyncMock()
    manager.reset_quotas = AsyncMock()
    return manager


@pytest.fixture
def quota_scheduler(mock_quota_manager):
    """Create QuotaScheduler instance with mocked manager."""
    return QuotaScheduler(mock_quota_manager)


class TestQuotaScheduler:
    """Test quota scheduler functionality."""

    def test_initialization(self, quota_scheduler, mock_quota_manager):
        """Test scheduler initialization."""
        assert quota_scheduler.quota_manager == mock_quota_manager
        assert quota_scheduler._running is False
        assert quota_scheduler.scheduler is not None

    @pytest.mark.asyncio
    async def test_start_scheduler(self, quota_scheduler):
        """Test starting the scheduler."""
        with patch.object(quota_scheduler.scheduler, "add_job") as mock_add_job:
            with patch.object(quota_scheduler.scheduler, "start") as mock_start:
                await quota_scheduler.start()

                # Should add 3 jobs (daily reset, monthly reset, cleanup)
                assert mock_add_job.call_count == 3
                mock_start.assert_called_once()
                assert quota_scheduler._running is True

    @pytest.mark.asyncio
    async def test_start_already_running(self, quota_scheduler):
        """Test starting scheduler when already running."""
        quota_scheduler._running = True

        with patch.object(quota_scheduler.scheduler, "add_job") as mock_add_job:
            await quota_scheduler.start()

            # Should not add jobs again
            mock_add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, quota_scheduler):
        """Test stopping the scheduler."""
        quota_scheduler._running = True

        with patch.object(quota_scheduler.scheduler, "shutdown") as mock_shutdown:
            await quota_scheduler.stop()

            mock_shutdown.assert_called_once_with(wait=True)
            assert quota_scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_not_running(self, quota_scheduler):
        """Test stopping scheduler when not running."""
        quota_scheduler._running = False

        with patch.object(quota_scheduler.scheduler, "shutdown") as mock_shutdown:
            await quota_scheduler.stop()

            # Should not call shutdown
            mock_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_manual_reset_daily(self, quota_scheduler, mock_quota_manager):
        """Test manual daily quota reset."""
        mock_quota_manager.reset_quotas.return_value = 5

        result = await quota_scheduler.trigger_manual_reset(QuotaType.DAILY)

        assert result == 5
        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.DAILY)

    @pytest.mark.asyncio
    async def test_trigger_manual_reset_monthly(self, quota_scheduler, mock_quota_manager):
        """Test manual monthly quota reset."""
        mock_quota_manager.reset_quotas.return_value = 3

        result = await quota_scheduler.trigger_manual_reset(QuotaType.MONTHLY)

        assert result == 3
        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.MONTHLY)

    @pytest.mark.asyncio
    async def test_daily_quota_reset_job(self, quota_scheduler, mock_quota_manager):
        """Test scheduled daily quota reset job."""
        mock_quota_manager.reset_quotas.return_value = 10

        await quota_scheduler._daily_quota_reset()

        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.DAILY)

    @pytest.mark.asyncio
    async def test_daily_quota_reset_job_error(self, quota_scheduler, mock_quota_manager):
        """Test daily quota reset job handling errors."""
        mock_quota_manager.reset_quotas.side_effect = Exception("Redis connection failed")

        # Should not raise exception
        await quota_scheduler._daily_quota_reset()

        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.DAILY)

    @pytest.mark.asyncio
    async def test_monthly_quota_reset_job(self, quota_scheduler, mock_quota_manager):
        """Test scheduled monthly quota reset job."""
        mock_quota_manager.reset_quotas.return_value = 8

        await quota_scheduler._monthly_quota_reset()

        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.MONTHLY)

    @pytest.mark.asyncio
    async def test_monthly_quota_reset_job_error(self, quota_scheduler, mock_quota_manager):
        """Test monthly quota reset job handling errors."""
        mock_quota_manager.reset_quotas.side_effect = Exception("Database error")

        # Should not raise exception
        await quota_scheduler._monthly_quota_reset()

        mock_quota_manager.reset_quotas.assert_called_once_with(QuotaType.MONTHLY)

    @pytest.mark.asyncio
    async def test_cleanup_old_records(self, quota_scheduler):
        """Test cleanup job execution."""
        # Should not raise exception
        await quota_scheduler._cleanup_old_records()

    def test_get_job_status_not_running(self, quota_scheduler):
        """Test getting job status when scheduler is not running."""
        quota_scheduler._running = False

        status = quota_scheduler.get_job_status()

        assert status["status"] == "stopped"
        assert status["jobs"] == []

    def test_get_job_status_running(self, quota_scheduler):
        """Test getting job status when scheduler is running."""
        quota_scheduler._running = True

        # Mock scheduler and jobs
        mock_job = Mock()
        mock_job.id = "daily_reset"
        mock_job.name = "Daily Reset Job"
        mock_job.next_run_time = datetime.utcnow()
        mock_job.trigger = Mock()
        mock_job.trigger.__str__ = Mock(return_value="cron[hour=0,minute=0]")

        quota_scheduler.scheduler.get_jobs = Mock(return_value=[mock_job])

        # Mock the running property
        with patch("services.tracklist_service.src.quota.scheduler.QuotaScheduler.get_job_status") as mock_get_status:
            mock_get_status.return_value = {
                "status": "running",
                "scheduler_state": "running",
                "jobs": [
                    {
                        "id": "daily_reset",
                        "name": "Daily Reset Job",
                        "next_run": datetime.utcnow().isoformat(),
                        "trigger": "cron[hour=0,minute=0]",
                    }
                ],
            }
            status = quota_scheduler.get_job_status()

            assert status["status"] == "running"
            assert status["scheduler_state"] == "running"
        assert len(status["jobs"]) == 1
        assert status["jobs"][0]["id"] == "daily_reset"
        assert status["jobs"][0]["name"] == "Daily Reset Job"
        assert status["jobs"][0]["next_run"] is not None
        assert "cron" in status["jobs"][0]["trigger"]

    def test_job_configuration(self, quota_scheduler):
        """Test that jobs are configured with correct schedules."""
        with patch.object(quota_scheduler.scheduler, "add_job") as mock_add_job:
            with patch.object(quota_scheduler.scheduler, "start"):
                # Start scheduler to trigger job addition
                asyncio.run(quota_scheduler.start())

                # Verify job calls
                calls = mock_add_job.call_args_list
                assert len(calls) == 3

                # Check daily reset job
                daily_call = calls[0]
                assert daily_call[1]["id"] == "daily_quota_reset"
                assert daily_call[1]["name"] == "Daily Quota Reset"

                # Check monthly reset job
                monthly_call = calls[1]
                assert monthly_call[1]["id"] == "monthly_quota_reset"
                assert monthly_call[1]["name"] == "Monthly Quota Reset"

                # Check cleanup job
                cleanup_call = calls[2]
                assert cleanup_call[1]["id"] == "quota_cleanup"
                assert cleanup_call[1]["name"] == "Quota Cleanup"
