"""Unit tests for catalog search service."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.tracklist import TrackEntry
from services.tracklist_service.src.services.catalog_search_service import CatalogSearchService
from shared.core_types.src.models import Metadata, Recording


class TestCatalogSearchService:
    """Test catalog search service."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def catalog_service(self, mock_db_session):
        """Create catalog search service."""
        return CatalogSearchService(mock_db_session)

    def test_search_catalog_with_query(self, catalog_service, mock_db_session):
        """Test searching catalog with general query."""
        # Setup mock data
        recording_id = uuid4()
        mock_recording = MagicMock(spec=Recording)
        mock_recording.id = recording_id
        mock_recording.file_name = "Test Track.mp3"

        mock_metadata_artist = MagicMock(spec=Metadata)
        mock_metadata_artist.key = "artist"
        mock_metadata_artist.value = "Test Artist"

        mock_metadata_title = MagicMock(spec=Metadata)
        mock_metadata_title.key = "title"
        mock_metadata_title.value = "Test Track"

        mock_query = MagicMock()
        mock_query.filter.return_value.limit.return_value.all.return_value = [
            (mock_recording, mock_metadata_artist),
            (mock_recording, mock_metadata_title),
        ]
        mock_db_session.query.return_value.join.return_value = mock_query

        # Perform search
        results = catalog_service.search_catalog(query="test", limit=10)

        assert len(results) == 1
        assert results[0][0].id == recording_id
        assert results[0][1] > 0  # Has confidence score

    def test_search_catalog_with_artist_and_title(self, catalog_service, mock_db_session):
        """Test searching catalog with specific artist and title."""
        recording_id = uuid4()
        mock_recording = MagicMock(spec=Recording)
        mock_recording.id = recording_id

        mock_metadata_artist = MagicMock(spec=Metadata)
        mock_metadata_artist.key = "artist"
        mock_metadata_artist.value = "Specific Artist"

        mock_metadata_title = MagicMock(spec=Metadata)
        mock_metadata_title.key = "title"
        mock_metadata_title.value = "Specific Title"

        mock_query = MagicMock()
        mock_query.filter.return_value.limit.return_value.all.return_value = [
            (mock_recording, mock_metadata_artist),
            (mock_recording, mock_metadata_title),
        ]
        mock_db_session.query.return_value.join.return_value = mock_query

        results = catalog_service.search_catalog(
            artist="Specific Artist",
            title="Specific Title",
            limit=5,
        )

        assert len(results) == 1
        assert results[0][1] > 0.7  # High confidence for exact matches

    def test_search_catalog_no_results(self, catalog_service, mock_db_session):
        """Test searching catalog with no results."""
        mock_query = MagicMock()
        mock_query.filter.return_value.limit.return_value.all.return_value = []
        mock_db_session.query.return_value.join.return_value = mock_query

        results = catalog_service.search_catalog(query="nonexistent")

        assert len(results) == 0

    def test_search_catalog_no_params(self, catalog_service):
        """Test searching catalog without parameters."""
        results = catalog_service.search_catalog()
        assert results == []

    def test_match_track_to_catalog_success(self, catalog_service, mock_db_session):
        """Test matching a track to catalog."""
        recording_id = uuid4()
        track = TrackEntry(
            position=1,
            start_time=0,
            artist="Match Artist",
            title="Match Title",
        )

        # Mock successful search
        mock_recording = MagicMock(spec=Recording)
        mock_recording.id = recording_id

        with patch.object(
            catalog_service,
            "search_catalog",
            return_value=[(mock_recording, 0.85)],
        ):
            result = catalog_service.match_track_to_catalog(track, threshold=0.7)

        assert result is not None
        assert result[0] == recording_id
        assert result[1] == 0.85

    def test_match_track_to_catalog_below_threshold(self, catalog_service):
        """Test matching track with confidence below threshold."""
        track = TrackEntry(
            position=1,
            start_time=0,
            artist="Low Match Artist",
            title="Low Match Title",
        )

        mock_recording = MagicMock(spec=Recording)
        mock_recording.id = uuid4()

        with patch.object(
            catalog_service,
            "search_catalog",
            return_value=[(mock_recording, 0.5)],
        ):
            result = catalog_service.match_track_to_catalog(track, threshold=0.7)

        assert result is None

    def test_match_track_to_catalog_no_results(self, catalog_service):
        """Test matching track with no catalog results."""
        track = TrackEntry(
            position=1,
            start_time=0,
            artist="No Match Artist",
            title="No Match Title",
        )

        with patch.object(catalog_service, "search_catalog", return_value=[]):
            result = catalog_service.match_track_to_catalog(track)

        assert result is None

    def test_fuzzy_match_tracks(self, catalog_service):
        """Test fuzzy matching multiple tracks."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=0,
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=180,
                artist="Artist 2",
                title="Track 2",
                catalog_track_id=uuid4(),  # Already matched
            ),
        ]

        recording_id = uuid4()
        with patch.object(
            catalog_service,
            "match_track_to_catalog",
            return_value=(recording_id, 0.9),
        ):
            matched = catalog_service.fuzzy_match_tracks(tracks, threshold=0.7)

        assert len(matched) == 2
        assert matched[0].catalog_track_id == recording_id
        assert matched[0].confidence == 0.9
        assert matched[1].catalog_track_id is not None  # Already had ID

    def test_calculate_confidence_exact_match(self, catalog_service):
        """Test confidence calculation for exact match."""
        metadata = {
            "artist": "Exact Artist",
            "title": "Exact Title",
        }

        confidence = catalog_service._calculate_confidence(
            metadata,
            artist="Exact Artist",
            title="Exact Title",
        )

        assert confidence >= 0.8  # High confidence for exact match (40% + 40% = 80%)

    def test_calculate_confidence_partial_match(self, catalog_service):
        """Test confidence calculation for partial match."""
        metadata = {
            "artist": "Similar Artist",
            "title": "Different Title",
        }

        confidence = catalog_service._calculate_confidence(
            metadata,
            artist="Similar Artist",
            title="Something Else",
        )

        assert 0.3 < confidence < 0.7  # Medium confidence for partial match

    def test_calculate_confidence_with_extras(self, catalog_service):
        """Test confidence calculation with extra metadata."""
        metadata = {
            "artist": "Artist",
            "title": "Title",
            "bpm": "128",
            "key": "Am",
            "genre": "House",
            "year": "2024",
        }

        confidence = catalog_service._calculate_confidence(
            metadata,
            artist="Artist",
            title="Title",
        )

        # Should get bonus points for extra metadata
        assert confidence > 0.9

    def test_fuzzy_match_exact(self, catalog_service):
        """Test fuzzy matching with exact match."""
        score = catalog_service._fuzzy_match("Test Track", "Test Track")
        assert score == 1.0

    def test_fuzzy_match_contains(self, catalog_service):
        """Test fuzzy matching with contains match."""
        score = catalog_service._fuzzy_match("Test", "Test Track")
        assert score >= 0.9

    def test_fuzzy_match_similar(self, catalog_service):
        """Test fuzzy matching with similar strings."""
        score = catalog_service._fuzzy_match("Test Track", "Test Trak")
        assert 0.7 < score < 1.0

    def test_fuzzy_match_empty(self, catalog_service):
        """Test fuzzy matching with empty string."""
        score = catalog_service._fuzzy_match("", "Test")
        assert score == 0.0

        score = catalog_service._fuzzy_match("Test", "")
        assert score == 0.0

    def test_check_artist_variations(self, catalog_service):
        """Test artist name variation detection."""
        # feat. vs ft.
        assert catalog_service._check_artist_variations(
            "artist feat. other",
            "artist ft. other",
        )

        # and vs &
        assert catalog_service._check_artist_variations(
            "artist and other",
            "artist & other",
        )

        # No variation
        assert not catalog_service._check_artist_variations(
            "artist",
            "different artist",
        )

    def test_check_remix_variations(self, catalog_service):
        """Test remix variation detection."""
        # One has remix, other doesn't
        assert catalog_service._check_remix_variations(
            "track (remix)",
            "track",
        )

        # Both have remix info
        assert not catalog_service._check_remix_variations(
            "track (remix)",
            "track (another remix)",
        )

        # Neither has remix info
        assert not catalog_service._check_remix_variations(
            "track",
            "track",
        )

    def test_get_catalog_track_metadata(self, catalog_service, mock_db_session):
        """Test getting metadata for a catalog track."""
        track_id = uuid4()

        mock_metadata = [
            MagicMock(key="artist", value="Test Artist"),
            MagicMock(key="title", value="Test Title"),
            MagicMock(key="album", value="Test Album"),
        ]

        mock_db_session.query().filter().all.return_value = mock_metadata

        metadata = catalog_service.get_catalog_track_metadata(track_id)

        assert metadata["artist"] == "Test Artist"
        assert metadata["title"] == "Test Title"
        assert metadata["album"] == "Test Album"
