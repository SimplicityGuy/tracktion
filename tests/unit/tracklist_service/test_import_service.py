"""
Unit tests for the import service.
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.tracklist import Tracklist
from services.tracklist_service.src.models.tracklist_models import (
    CuePoint,
)
from services.tracklist_service.src.models.tracklist_models import (
    Track as ScrapedTrack,
)
from services.tracklist_service.src.models.tracklist_models import (
    Tracklist as ScrapedTracklist,
)
from services.tracklist_service.src.services.import_service import ImportService


class TestImportService:
    """Test the import service functionality."""

    @pytest.fixture
    def import_service(self):
        """Create an import service instance with mocked Redis."""
        with patch("services.tracklist_service.src.services.import_service.redis.Redis"):
            service = ImportService(redis_client=MagicMock())
            service.redis_client = MagicMock()
            return service

    @pytest.fixture
    def sample_scraped_tracklist(self):
        """Create a sample scraped tracklist."""
        tracks = [
            ScrapedTrack(
                number=1,
                timestamp=CuePoint(track_number=1, timestamp_ms=0, formatted_time="00:00"),
                artist="Artist 1",
                title="Track 1",
                remix="Original Mix",
                label="Label 1",
                is_id=False,
            ),
            ScrapedTrack(
                number=2,
                timestamp=CuePoint(track_number=2, timestamp_ms=240000, formatted_time="04:00"),
                artist="Artist 2",
                title="Track 2",
                remix="Remix",
                is_id=False,
            ),
            ScrapedTrack(
                number=3,
                timestamp=CuePoint(track_number=3, timestamp_ms=480000, formatted_time="08:00"),
                artist="ID",
                title="ID",
                is_id=True,
            ),
        ]

        return ScrapedTracklist(
            url="https://1001tracklists.com/tracklist/test",
            dj_name="Test DJ",
            event_name="Test Event",
            tracks=tracks,
        )

    def test_transform_to_track_entries(self, import_service, sample_scraped_tracklist):
        """Test transforming scraped tracks to TrackEntry objects."""
        track_entries = import_service.transform_to_track_entries(sample_scraped_tracklist.tracks)

        assert len(track_entries) == 3

        # Check first track
        assert track_entries[0].position == 1
        assert track_entries[0].start_time == timedelta(0)
        assert track_entries[0].end_time == timedelta(minutes=4)
        assert track_entries[0].artist == "Artist 1"
        assert track_entries[0].title == "Track 1"
        assert track_entries[0].confidence == 0.9

        # Check second track
        assert track_entries[1].position == 2
        assert track_entries[1].start_time == timedelta(minutes=4)
        assert track_entries[1].end_time == timedelta(minutes=8)
        assert track_entries[1].confidence == 0.9

        # Check ID track has lower confidence
        assert track_entries[2].position == 3
        assert track_entries[2].confidence == 0.5
        assert track_entries[2].end_time is None  # Last track has no end time

    def test_parse_timestamp(self, import_service):
        """Test parsing various timestamp formats."""
        # Test with CuePoint with timestamp_ms
        cue = CuePoint(track_number=1, timestamp_ms=150000, formatted_time="02:30")
        result = import_service._parse_timestamp(cue)
        assert result == timedelta(milliseconds=150000)

        # Test with CuePoint with only formatted_time MM:SS
        cue2 = MagicMock()
        cue2.timestamp_ms = None
        cue2.formatted_time = "04:30"
        result2 = import_service._parse_timestamp(cue2)
        assert result2 == timedelta(minutes=4, seconds=30)

        # Test with HH:MM:SS format
        cue3 = MagicMock()
        cue3.timestamp_ms = None
        cue3.formatted_time = "01:15:45"
        result3 = import_service._parse_timestamp(cue3)
        assert result3 == timedelta(hours=1, minutes=15, seconds=45)

        # Test with None
        assert import_service._parse_timestamp(None) == timedelta(0)

    def test_calculate_confidence_score(self, import_service, sample_scraped_tracklist):
        """Test confidence score calculation."""
        track_entries = import_service.transform_to_track_entries(sample_scraped_tracklist.tracks)

        confidence = import_service._calculate_confidence_score(sample_scraped_tracklist, track_entries)

        # With 3 tracks, 3 with timestamps, 2 non-ID tracks
        # time_coverage = 3/3 = 1.0 (weight 0.4)
        # id_coverage = 2/3 = 0.67 (weight 0.4)
        # metadata_score depends on completeness (weight 0.2)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high

    def test_fetch_tracklist_from_cache(self, import_service):
        """Test fetching tracklist from cache."""
        url = "https://1001tracklists.com/tracklist/test"
        cached_data = '{"url": "test", "dj_name": "Cached DJ", "tracks": []}'

        import_service.redis_client.get.return_value = cached_data

        with patch.object(import_service, "scraper") as mock_scraper:
            # Should not call scraper when cache hit
            import_service._get_cached_tracklist(url)
            mock_scraper.scrape_tracklist.assert_not_called()

    def test_fetch_tracklist_force_refresh(self, import_service, sample_scraped_tracklist):
        """Test force refresh bypasses cache."""
        url = "https://1001tracklists.com/tracklist/test"

        import_service.scraper.scrape_tracklist = MagicMock(return_value=sample_scraped_tracklist)
        import_service.redis_client.get.return_value = '{"cached": "data"}'

        # Force refresh should bypass cache
        import_service.fetch_tracklist_from_1001(url, force_refresh=True)

        # Should not check cache
        import_service.redis_client.get.assert_not_called()
        # Should call scraper
        import_service.scraper.scrape_tracklist.assert_called_once_with(url)

    def test_import_tracklist_complete(self, import_service, sample_scraped_tracklist):
        """Test complete tracklist import process."""
        url = "https://1001tracklists.com/tracklist/test"
        audio_file_id = uuid4()

        import_service.scraper.scrape_tracklist = MagicMock(return_value=sample_scraped_tracklist)

        tracklist = import_service.import_tracklist(url, audio_file_id)

        assert isinstance(tracklist, Tracklist)
        assert tracklist.audio_file_id == audio_file_id
        assert tracklist.source == "1001tracklists"
        assert len(tracklist.tracks) == 3
        assert tracklist.confidence_score > 0

    def test_import_tracklist_error_handling(self, import_service):
        """Test error handling during import."""
        url = "https://1001tracklists.com/tracklist/test"
        audio_file_id = uuid4()

        import_service.scraper.scrape_tracklist = MagicMock(side_effect=Exception("Scraping failed"))

        with pytest.raises(ValueError) as exc_info:
            import_service.import_tracklist(url, audio_file_id)

        assert "Import failed" in str(exc_info.value)

    def test_cache_tracklist(self, import_service, sample_scraped_tracklist):
        """Test caching a tracklist."""
        url = "https://1001tracklists.com/tracklist/test"

        import_service._cache_tracklist(url, sample_scraped_tracklist)

        # Check Redis setex was called with correct parameters
        import_service.redis_client.setex.assert_called_once()
        call_args = import_service.redis_client.setex.call_args
        assert call_args[0][0] == f"tracklist:1001:{url}"
        assert call_args[0][1] == 3600  # TTL
        assert isinstance(call_args[0][2], str)  # JSON string
