"""Tests for HTML structure monitoring."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import redis.asyncio as redis

from services.tracklist_service.src.monitoring.structure_monitor import (
    ChangeReport,
    ChangeType,
    StructuralChange,
    StructureMonitor,
)


@pytest.fixture
def sample_html_v1():
    """Sample HTML structure version 1."""
    return """
    <html>
    <head><title>1001tracklists</title></head>
    <body>
        <div class="search-results">
            <div class="result-item">
                <a class="result-link" href="/tracklist/1">Tracklist 1</a>
                <span class="result-date">2024-01-01</span>
                <span class="result-venue">Venue 1</span>
            </div>
            <div class="result-item">
                <a class="result-link" href="/tracklist/2">Tracklist 2</a>
                <span class="result-date">2024-01-02</span>
                <span class="result-venue">Venue 2</span>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_html_v2():
    """Sample HTML structure version 2 with changes."""
    return """
    <html>
    <head><title>1001tracklists</title></head>
    <body>
        <div class="search-results-new">  <!-- Changed class name -->
            <div class="result-item">
                <a class="result-link" href="/tracklist/1">Tracklist 1</a>
                <span class="result-date">2024-01-01</span>
                <span class="result-location">Venue 1</span>  <!-- Changed class -->
            </div>
            <div class="result-item">
                <a class="result-link" href="/tracklist/2">Tracklist 2</a>
                <span class="result-date">2024-01-02</span>
                <span class="result-location">Venue 2</span>  <!-- Changed class -->
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_tracklist_html():
    """Sample tracklist HTML."""
    return """
    <html>
    <body>
        <div class="tracklist-container">
            <div class="track-item">
                <span class="track-time">00:00</span>
                <span class="track-artist">Artist 1</span>
                <span class="track-title">Track 1</span>
                <span class="track-label">Label 1</span>
            </div>
            <div class="track-item">
                <span class="track-time">03:45</span>
                <span class="track-artist">Artist 2</span>
                <span class="track-title">Track 2</span>
                <span class="track-label">Label 2</span>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    mock = AsyncMock(spec=redis.Redis)
    # Configure return values for async methods
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.lpush = AsyncMock(return_value=1)
    mock.ltrim = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def monitor(redis_mock, tmp_path):
    """Create structure monitor with mocked Redis."""
    config_path = tmp_path / "selectors.json"
    return StructureMonitor(redis_client=redis_mock, config_path=config_path)


class TestStructureMonitor:
    """Test StructureMonitor class."""

    async def test_capture_structure_fingerprint(self, monitor, sample_html_v1):
        """Test capturing HTML structure fingerprint."""
        fingerprint = monitor.capture_structure_fingerprint(sample_html_v1, "search")

        assert fingerprint["page_type"] == "search"
        assert "timestamp" in fingerprint
        assert "checksum" in fingerprint
        assert "structure" in fingerprint
        assert "selectors" in fingerprint

        # Check structure analysis
        structure = fingerprint["structure"]
        assert "tag_counts" in structure
        assert structure["tag_counts"]["div"] == 3  # search-results + 2 result-items
        assert structure["tag_counts"]["a"] == 2
        assert structure["tag_counts"]["span"] == 4

        # Check selector analysis
        selectors = fingerprint["selectors"]
        assert "critical" in selectors
        assert ".search-results" in selectors["critical"]
        assert selectors["critical"][".search-results"]["exists"] is True
        assert selectors["critical"][".search-results"]["count"] == 1

    async def test_compare_structures_no_changes(self, monitor, sample_html_v1):
        """Test comparing identical structures."""
        fingerprint1 = monitor.capture_structure_fingerprint(sample_html_v1, "search")
        fingerprint2 = monitor.capture_structure_fingerprint(sample_html_v1, "search")

        report = monitor.compare_structures(fingerprint1, fingerprint2)

        assert report.page_type == "search"
        assert len(report.changes) == 0
        assert report.fingerprint_match_percentage == 100.0
        assert report.severity == "low"
        assert not report.has_breaking_changes

    async def test_compare_structures_with_changes(self, monitor, sample_html_v1, sample_html_v2):
        """Test comparing structures with changes."""
        fingerprint1 = monitor.capture_structure_fingerprint(sample_html_v1, "search")
        fingerprint2 = monitor.capture_structure_fingerprint(sample_html_v2, "search")

        report = monitor.compare_structures(fingerprint2, fingerprint1)

        assert report.page_type == "search"
        assert len(report.changes) > 0
        assert report.fingerprint_match_percentage < 100.0

        # Check for critical selector changes
        selector_changes = [c for c in report.changes if "selector:" in c.element_path]
        assert len(selector_changes) > 0

        # .search-results should be detected as removed
        search_results_change = next((c for c in selector_changes if ".search-results" in c.element_path), None)
        assert search_results_change is not None
        assert search_results_change.change_type == ChangeType.REMOVED

    async def test_store_and_get_baseline(self, monitor, redis_mock, sample_html_v1):
        """Test storing and retrieving baseline structure."""
        structure = monitor.capture_structure_fingerprint(sample_html_v1, "search")

        # Store baseline
        await monitor.store_baseline("search", structure)

        # Verify Redis calls
        redis_mock.set.assert_called_once()
        redis_mock.lpush.assert_called_once()
        redis_mock.ltrim.assert_called_once()

        # Test retrieval from cache
        cached = await monitor.get_baseline("search")
        assert cached == structure

        # Test retrieval from Redis
        monitor._baseline_cache.clear()
        redis_mock.get.return_value = json.dumps(structure)
        retrieved = await monitor.get_baseline("search")
        assert retrieved == structure

    async def test_check_for_changes_no_baseline(self, monitor, redis_mock, sample_html_v1):
        """Test checking for changes when no baseline exists."""
        redis_mock.get.return_value = None

        report = await monitor.check_for_changes(sample_html_v1, "search")

        assert report is None
        # Should store as new baseline
        redis_mock.set.assert_called_once()

    async def test_check_for_changes_with_baseline(self, monitor, redis_mock, sample_html_v1, sample_html_v2):
        """Test checking for changes with existing baseline."""
        # Set up baseline
        baseline = monitor.capture_structure_fingerprint(sample_html_v1, "search")
        redis_mock.get.return_value = json.dumps(baseline)

        # Check for changes
        report = await monitor.check_for_changes(sample_html_v2, "search")

        assert report is not None
        assert len(report.changes) > 0
        assert report.fingerprint_match_percentage < 100.0

    def test_change_report_to_dict(self, monitor):
        """Test ChangeReport serialization."""
        report = ChangeReport(
            page_type="search",
            timestamp=datetime.now(UTC),
            changes=[
                StructuralChange(
                    element_path="selector:.search-results",
                    change_type=ChangeType.REMOVED,
                    old_value="True",
                    new_value="False",
                    impact_score=0.8,
                )
            ],
            severity="critical",
            fingerprint_match_percentage=75.0,
            requires_manual_review=True,
        )

        report_dict = report.to_dict()

        assert report_dict["page_type"] == "search"
        assert report_dict["severity"] == "critical"
        assert report_dict["fingerprint_match_percentage"] == 75.0
        assert report_dict["requires_manual_review"] is True
        assert report_dict["has_breaking_changes"] is True
        assert len(report_dict["changes"]) == 1
        assert report_dict["changes"][0]["change_type"] == "removed"

    def test_default_selectors(self, tmp_path):
        """Test default selector configuration."""
        config_path = tmp_path / "nonexistent.json"
        monitor = StructureMonitor(config_path=config_path)

        selectors = monitor._selectors_config
        assert "search" in selectors
        assert "tracklist" in selectors
        assert "dj" in selectors

        # Check search selectors
        assert "critical" in selectors["search"]
        assert ".search-results" in selectors["search"]["critical"]
        assert ".result-item" in selectors["search"]["critical"]

    def test_load_custom_selectors(self, tmp_path):
        """Test loading custom selector configuration."""
        config_path = tmp_path / "selectors.json"
        custom_config = {"search": {"critical": [".custom-selector"], "important": [], "optional": []}}
        config_path.write_text(json.dumps(custom_config))

        monitor = StructureMonitor(config_path=config_path)

        assert monitor._selectors_config == custom_config
        assert ".custom-selector" in monitor._selectors_config["search"]["critical"]

    async def test_tracklist_fingerprint(self, monitor, sample_tracklist_html):
        """Test fingerprinting tracklist pages."""
        fingerprint = monitor.capture_structure_fingerprint(sample_tracklist_html, "tracklist")

        selectors = fingerprint["selectors"]
        assert "critical" in selectors
        assert ".tracklist-container" in selectors["critical"]
        assert selectors["critical"][".tracklist-container"]["count"] == 1
        assert ".track-item" in selectors["critical"]
        assert selectors["critical"][".track-item"]["count"] == 2
