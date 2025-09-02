"""
Unit tests for tracklist scraper.

Tests HTML parsing, track extraction, and metadata extraction
with mock HTML responses.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from services.tracklist_service.src.models.tracklist_models import CuePoint, Track, TransitionType
from services.tracklist_service.src.scraper.tracklist_scraper import TracklistScraper

# Sample HTML snippets for testing
SAMPLE_DJ_HTML = """
<div class="tlHead">
    <h1 class="marL10">Carl Cox</h1>
    <h2>Tomorrowland Belgium 2024</h2>
</div>
"""

SAMPLE_EVENT_HTML = """
<div class="eventInfo">
    <h2>Ultra Music Festival</h2>
    <span class="venue">Bayfront Park</span>
    <div class="when">
        <time datetime="2024-03-24">March 24, 2024</time>
    </div>
</div>
"""

SAMPLE_TRACK_HTML = """
<div class="tlpTog">
    <span class="cue">00:00</span>
    <span class="artist">Adam Beyer</span>
    <span class="track">Your Mind (Original Mix)</span>
    <span class="label">Drumcode</span>
    <span class="bpm">128</span>
    <span class="key">Am</span>
</div>
"""

SAMPLE_ID_TRACK_HTML = """
<div class="tlpTog">
    <span class="cue">05:30</span>
    <span class="artist">ID</span>
    <span class="track">ID</span>
</div>
"""

SAMPLE_REMIX_TRACK_HTML = """
<div class="tlpTog">
    <span class="cue">10:45</span>
    <span class="artist">Charlotte de Witte</span>
    <span class="track">Liquid Slow (Amelie Lens Remix)</span>
    <span class="label">KNTXT</span>
</div>
"""

SAMPLE_METADATA_HTML = """
<div class="setInfo">
    <span class="setType">DJ Set</span>
    <span class="duration">120 minutes</span>
    <span class="plays">45,230</span>
    <span class="favorites">1,250</span>
    <a class="external" href="https://soundcloud.com/example">SoundCloud</a>
    <a class="tag">Techno</a>
    <a class="tag">Peak Time</a>
</div>
"""

FULL_TRACKLIST_HTML = """
<html>
<head>
    <meta property="og:title" content="Carl Cox @ Tomorrowland Belgium 2024">
</head>
<body>
    <div class="tlHead">
        <h1 class="marL10">Carl Cox</h1>
        <h2>Tomorrowland Belgium 2024</h2>
    </div>
    <div class="eventInfo">
        <span class="venue">Boom</span>
        <div class="when">
            <time datetime="2024-07-20">July 20, 2024</time>
        </div>
    </div>
    <div class="tracklist">
        <div class="tlpTog">
            <span class="cue">00:00</span>
            <span class="artist">Carl Cox</span>
            <span class="track">The Revolution Continues</span>
            <span class="label">Intec</span>
        </div>
        <div class="tlpTog">
            <span class="cue">03:45</span>
            <span class="artist">Adam Beyer</span>
            <span class="track">Your Mind</span>
            <span class="label">Drumcode</span>
        </div>
        <div class="tlpTog">
            <span class="cue">07:30</span>
            <span class="artist">ID</span>
            <span class="track">ID</span>
        </div>
    </div>
    <div class="setInfo">
        <span class="setType">DJ Set</span>
        <span class="duration">90 minutes</span>
        <span class="plays">25,000</span>
        <a class="external" href="https://soundcloud.com/carlcox">SoundCloud</a>
        <a class="tag">Techno</a>
    </div>
</body>
</html>
"""


class TestTracklistScraper:
    """Test TracklistScraper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = TracklistScraper()

    def test_is_valid_url(self):
        """Test URL validation."""
        valid_urls = [
            "https://www.1001tracklists.com/tracklist/test",
            "https://1001tracklists.com/tracklist/test",
            "http://www.1001tracklists.com/tracklist/test",
        ]

        for url in valid_urls:
            assert self.scraper._is_valid_url(url) is True

        invalid_urls = [
            "https://example.com/tracklist",
            "https://google.com",
            "not-a-url",
        ]

        for url in invalid_urls:
            assert self.scraper._is_valid_url(url) is False

    def test_extract_dj_name(self):
        """Test DJ name extraction."""
        soup = BeautifulSoup(SAMPLE_DJ_HTML, "html.parser")
        dj_name = self.scraper._extract_dj_name(soup)
        assert dj_name == "Carl Cox"

    def test_extract_dj_name_from_meta(self):
        """Test DJ name extraction from meta tag."""
        html = '<meta property="og:title" content="Amelie Lens @ Awakenings 2024">'
        soup = BeautifulSoup(html, "html.parser")
        dj_name = self.scraper._extract_dj_name(soup)
        assert dj_name == "Amelie Lens"

    def test_extract_event_info(self):
        """Test event information extraction."""
        soup = BeautifulSoup(SAMPLE_EVENT_HTML, "html.parser")
        info = self.scraper._extract_event_info(soup)

        assert info["event_name"] == "Ultra Music Festival"
        assert info["venue"] == "Bayfront Park"
        assert info["date"] == date(2024, 3, 24)

    def test_parse_date(self):
        """Test date parsing with various formats."""
        test_cases = [
            ("2024-03-24", date(2024, 3, 24)),
            ("24-03-2024", date(2024, 3, 24)),
            ("2024/03/24", date(2024, 3, 24)),
            ("24/03/2024", date(2024, 3, 24)),
            ("March 24, 2024", date(2024, 3, 24)),
            ("24 March 2024", date(2024, 3, 24)),
        ]

        for date_str, expected in test_cases:
            result = self.scraper._parse_date(date_str)
            assert result == expected

    def test_parse_time_to_ms(self):
        """Test time string to milliseconds conversion."""
        test_cases = [
            ("00:00", 0),
            ("01:30", 90000),
            ("10:45", 645000),
            ("1:00:00", 3600000),
            ("2:30:45", 9045000),
        ]

        for time_str, expected_ms in test_cases:
            result = self.scraper._parse_time_to_ms(time_str)
            assert result == expected_ms

    def test_extract_tracks(self):
        """Test track extraction from HTML."""
        html = """
        <div class="tracklist">
            <div class="tlpTog">
                <span class="cue">00:00</span>
                <span class="artist">Artist 1</span>
                <span class="track">Track 1</span>
                <span class="label">Label 1</span>
            </div>
            <div class="tlpTog">
                <span class="cue">03:30</span>
                <span class="artist">Artist 2</span>
                <span class="track">Track 2</span>
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        tracks = self.scraper._extract_tracks(soup)

        assert len(tracks) == 2
        assert tracks[0].artist == "Artist 1"
        assert tracks[0].title == "Track 1"
        assert tracks[0].label == "Label 1"
        assert tracks[0].timestamp.formatted_time == "00:00"
        assert tracks[1].artist == "Artist 2"
        assert tracks[1].title == "Track 2"

    def test_parse_track_with_remix(self):
        """Test parsing track with remix information."""
        soup = BeautifulSoup(SAMPLE_REMIX_TRACK_HTML, "html.parser")
        element = soup.select_one("div.tlpTog")
        track = self.scraper._parse_track(element, 1)

        assert track.artist == "Charlotte de Witte"
        assert track.title == "Liquid Slow"
        assert track.remix == "Amelie Lens Remix"
        assert track.label == "KNTXT"

    def test_parse_id_track(self):
        """Test parsing ID/unknown track."""
        soup = BeautifulSoup(SAMPLE_ID_TRACK_HTML, "html.parser")
        element = soup.select_one("div.tlpTog")
        track = self.scraper._parse_track(element, 1)

        assert track.artist == "ID"
        assert track.title == "ID"
        assert track.is_id is True

    def test_is_id_track(self):
        """Test ID track detection."""
        test_cases = [
            ("ID", "ID", True),
            ("Unknown", "Track", True),
            ("Artist", "Unknown Title", True),
            ("???", "???", True),
            ("Unreleased", "Track", True),
            ("Normal Artist", "Normal Track", False),
        ]

        for artist, title, expected in test_cases:
            result = self.scraper._is_id_track(artist, title)
            assert result == expected

    def test_extract_remix(self):
        """Test remix extraction from title."""
        test_cases = [
            ("Track (Original Mix)", "Original Mix"),
            ("Track [Amelie Lens Remix]", "Amelie Lens Remix"),
            ("Track (Extended Edit)", "Extended Edit"),
            ("Track [VIP Mix]", "VIP Mix"),
            ("Track (Bootleg)", "Bootleg"),
            ("Normal Track", None),
        ]

        for title, expected_remix in test_cases:
            result = self.scraper._extract_remix(title)
            assert result == expected_remix

    def test_extract_metadata(self):
        """Test metadata extraction."""
        soup = BeautifulSoup(SAMPLE_METADATA_HTML, "html.parser")
        metadata = self.scraper._extract_metadata(soup)

        assert metadata.recording_type == "DJ Set"
        assert metadata.duration_minutes == 120
        assert metadata.play_count == 45230
        assert metadata.favorite_count == 1250
        assert metadata.soundcloud_url == "https://soundcloud.com/example"
        assert "Techno" in metadata.tags
        assert "Peak Time" in metadata.tags

    def test_infer_transitions(self):
        """Test transition inference from timestamps."""
        tracks = [
            Track(
                number=1,
                artist="A1",
                title="T1",
                timestamp=CuePoint(track_number=1, timestamp_ms=0, formatted_time="00:00"),
            ),
            Track(
                number=2,
                artist="A2",
                title="T2",
                timestamp=CuePoint(track_number=2, timestamp_ms=180000, formatted_time="03:00"),
            ),
            Track(
                number=3,
                artist="A3",
                title="T3",
                timestamp=CuePoint(track_number=3, timestamp_ms=360000, formatted_time="06:00"),
            ),
        ]

        transitions = self.scraper._infer_transitions(tracks)

        assert len(transitions) == 2
        assert transitions[0].from_track == 1
        assert transitions[0].to_track == 2
        assert transitions[0].transition_type == TransitionType.BLEND
        assert transitions[1].from_track == 2
        assert transitions[1].to_track == 3

    @patch.object(TracklistScraper, "_make_request")
    def test_scrape_tracklist_complete(self, mock_request):
        """Test complete tracklist scraping."""
        mock_response = MagicMock()
        mock_response.text = FULL_TRACKLIST_HTML
        mock_request.return_value = mock_response

        url = "https://www.1001tracklists.com/tracklist/test"
        tracklist = self.scraper.scrape_tracklist(url)

        assert tracklist.url == url
        assert tracklist.dj_name == "Carl Cox"
        assert tracklist.event_name == "Tomorrowland Belgium 2024"
        assert tracklist.venue == "Boom"
        assert tracklist.date == date(2024, 7, 20)

        assert len(tracklist.tracks) == 3
        assert tracklist.tracks[0].artist == "Carl Cox"
        assert tracklist.tracks[0].title == "The Revolution Continues"
        assert tracklist.tracks[1].artist == "Adam Beyer"
        assert tracklist.tracks[2].is_id is True

        assert tracklist.metadata is not None
        assert tracklist.metadata.recording_type == "DJ Set"
        assert tracklist.metadata.duration_minutes == 90

        assert tracklist.source_html_hash is not None

    def test_scrape_tracklist_invalid_url(self):
        """Test scraping with invalid URL."""
        with pytest.raises(ValueError) as exc_info:
            self.scraper.scrape_tracklist("https://example.com/test")
        assert "Invalid 1001tracklists.com URL" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = TracklistScraper()

    def test_empty_tracklist(self):
        """Test handling of empty tracklist."""
        html = """
        <html>
            <div class="tlHead">
                <h1>Test DJ</h1>
            </div>
            <div class="tracklist"></div>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        tracks = self.scraper._extract_tracks(soup)
        assert tracks == []

    def test_track_with_multiple_artists(self):
        """Test track with multiple artists."""
        html = """
        <div class="tlpTog">
            <span class="cue">00:00</span>
            <span class="artist">Adam Beyer & Cirez D</span>
            <span class="track">Interchange</span>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        element = soup.select_one("div.tlpTog")
        track = self.scraper._parse_track(element, 1)

        assert "Adam Beyer" in track.artist
        assert "Cirez D" in track.artist

    def test_track_with_unicode(self):
        """Test track with Unicode characters."""
        html = """
        <div class="tlpTog">
            <span class="artist">Âme</span>
            <span class="track">Für Dich</span>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        element = soup.select_one("div.tlpTog")
        track = self.scraper._parse_track(element, 1)

        assert track.artist == "Âme"
        assert track.title == "Für Dich"

    def test_malformed_timestamp(self):
        """Test handling of malformed timestamp."""
        test_cases = [
            "not-a-time",
            "12:60",  # Invalid seconds
            "25:00",  # Could be valid for long sets
            "",
            None,
        ]

        for time_str in test_cases[:-1]:
            result = self.scraper._parse_time_to_ms(time_str)
            assert result is None or isinstance(result, int)

    def test_missing_dj_name(self):
        """Test handling when DJ name is missing."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        dj_name = self.scraper._extract_dj_name(soup)
        assert dj_name == "Unknown DJ"

    def test_track_without_label(self):
        """Test track without label information."""
        html = """
        <div class="tlpTog">
            <span class="artist">Test Artist</span>
            <span class="track">Test Track</span>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        element = soup.select_one("div.tlpTog")
        track = self.scraper._parse_track(element, 1)

        assert track.artist == "Test Artist"
        assert track.title == "Test Track"
        assert track.label is None
