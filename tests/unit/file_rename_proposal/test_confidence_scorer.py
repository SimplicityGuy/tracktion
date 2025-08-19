"""Unit tests for confidence scorer."""

import pytest

from services.analysis_service.src.file_rename_proposal.confidence_scorer import ConfidenceScorer


class TestConfidenceScorer:
    """Test confidence scoring functionality."""

    @pytest.fixture
    def scorer(self):
        """Create a confidence scorer instance."""
        return ConfidenceScorer()

    def test_calculate_confidence_high_quality(self, scorer):
        """Test confidence calculation for high-quality metadata."""
        metadata = {
            "artist": "The Beatles",
            "title": "Hey Jude",
            "album": "Hey Jude",
            "date": "1968",
            "track": "1",
            "genre": "Rock",
        }

        score, components = scorer.calculate_confidence(
            metadata=metadata,
            original_filename="track01.mp3",
            proposed_filename="The Beatles - Hey Jude.mp3",
            conflicts=[],
            warnings=[],
            pattern_used="{artist} - {title}",
            source="id3v2",
        )

        assert score > 0.8  # Should have high confidence
        assert components["metadata_completeness"] > 0.8
        assert components["pattern_match"] == 1.0  # All placeholders filled
        assert components["conflict_absence"] == 1.0  # No conflicts
        assert components["source_reliability"] > 0.9  # ID3v2 is reliable

    def test_calculate_confidence_minimal_metadata(self, scorer):
        """Test confidence calculation with minimal metadata."""
        metadata = {"artist": "Unknown Artist", "title": "Track 1"}

        score, components = scorer.calculate_confidence(
            metadata=metadata,
            original_filename="audio_001.mp3",
            proposed_filename="Unknown Artist - Track 1.mp3",
            conflicts=[],
            warnings=["Filename matches temporary file pattern"],
            pattern_used="{artist} - {title}",
            source="filename",
        )

        assert score < 0.7  # Should have lower confidence
        assert components["metadata_completeness"] < 0.5
        assert components["source_reliability"] == 0.6  # Filename parsing is less reliable

    def test_calculate_confidence_with_conflicts(self, scorer):
        """Test confidence calculation with conflicts."""
        metadata = {"artist": "Artist", "title": "Title", "album": "Album"}

        score, components = scorer.calculate_confidence(
            metadata=metadata,
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            conflicts=["File already exists"],
            warnings=[],
            pattern_used="{artist} - {title}",
            source="id3",
        )

        assert components["conflict_absence"] == 0.0  # Conflicts result in zero score
        assert score < 0.7  # Overall score should be impacted

    def test_score_metadata_completeness(self, scorer):
        """Test metadata completeness scoring."""
        # Complete metadata
        complete = {
            "artist": "Artist",
            "title": "Title",
            "album": "Album",
            "date": "2024",
            "track": "1",
            "genre": "Rock",
        }
        score = scorer._score_metadata_completeness(complete)
        assert score > 0.8

        # Essential only
        essential = {"artist": "Artist", "title": "Title"}
        score = scorer._score_metadata_completeness(essential)
        assert 0.3 < score < 0.6

        # Empty metadata
        empty = {}
        score = scorer._score_metadata_completeness(empty)
        assert score == 0.0

    def test_score_pattern_match(self, scorer):
        """Test pattern matching score."""
        metadata = {"artist": "The Beatles", "title": "Hey Jude", "album": "Hey Jude"}

        # All placeholders filled
        score = scorer._score_pattern_match(metadata, "The Beatles - Hey Jude.mp3", "{artist} - {title}")
        assert score == 1.0

        # Partial placeholders filled
        score = scorer._score_pattern_match(metadata, "The Beatles - Hey Jude.mp3", "{artist} - {title} [{year}]")
        assert score < 1.0

        # No pattern used but matches
        score = scorer._score_pattern_match(metadata, "the beatles - hey jude.mp3", None)
        assert score > 0.7

    def test_score_filename_quality(self, scorer):
        """Test filename quality scoring."""
        # Improvement from generic name
        score = scorer._score_filename_quality("track01.mp3", "Artist - Title.mp3")
        assert score > 0.5

        # Improvement with better structure
        score = scorer._score_filename_quality("artist_title_album.mp3", "Artist - Title.mp3")
        assert score > 0.5

        # Making filename too long
        score = scorer._score_filename_quality(
            "song.mp3",
            "This Is A Very Long Artist Name - This Is An Extremely Long Song Title That Goes On Forever.mp3",
        )
        assert score <= 0.5  # Should be 0.5 or less

        # Removing too much information
        score = scorer._score_filename_quality("Artist - Album - 01 - Title (Remastered).mp3", "Title.mp3")
        assert score < 0.5

    def test_score_conflict_absence(self, scorer):
        """Test conflict absence scoring."""
        # No conflicts or warnings
        score = scorer._score_conflict_absence([], [])
        assert score == 1.0

        # With warnings
        score = scorer._score_conflict_absence([], ["Warning 1", "Warning 2"])
        assert score == 0.7  # 1.0 - (2 * 0.15)

        # With conflicts
        score = scorer._score_conflict_absence(["Conflict"], [])
        assert score == 0.0

        # Many warnings
        score = scorer._score_conflict_absence([], ["W1", "W2", "W3", "W4", "W5", "W6", "W7"])
        assert score == 0.0  # Capped at 0

    def test_score_source_reliability(self, scorer):
        """Test source reliability scoring."""
        assert scorer._score_source_reliability("id3v2") == 0.95
        assert scorer._score_source_reliability("vorbis") == 0.95
        assert scorer._score_source_reliability("filename") == 0.60
        assert scorer._score_source_reliability("inferred") == 0.40
        assert scorer._score_source_reliability("unknown") == 0.50
        assert scorer._score_source_reliability(None) == 0.50

    def test_score_consistency(self, scorer):
        """Test consistency scoring."""
        # Consistent metadata
        consistent = {"artist": "Artist", "albumartist": "Artist", "track": "5", "date": "2023"}
        score = scorer._score_consistency(consistent, "Artist - Title.mp3")
        assert score > 0.9

        # Inconsistent artist
        inconsistent_artist = {"artist": "Artist 1", "albumartist": "Artist 2"}
        score = scorer._score_consistency(inconsistent_artist, "Artist 1 - Title.mp3")
        assert score < 1.0

        # Invalid track number
        invalid_track = {"track": "999"}
        score = scorer._score_consistency(invalid_track, "Title.mp3")
        assert score < 0.9

        # Invalid year
        invalid_year = {"date": "1800-01-01"}
        score = scorer._score_consistency(invalid_year, "Title.mp3")
        assert score < 0.8

        # Very long filename
        metadata = {"artist": "A", "title": "B"}
        long_filename = "A" * 250 + ".mp3"
        score = scorer._score_consistency(metadata, long_filename)
        assert score < 0.9

    def test_adjust_confidence_for_context(self, scorer):
        """Test context-based confidence adjustment."""
        base_score = 0.8

        # Lossless format boost
        adjusted = scorer.adjust_confidence_for_context(base_score, "flac")
        assert adjusted > base_score

        # Compilation penalty
        adjusted = scorer.adjust_confidence_for_context(base_score, "mp3", is_compilation=True)
        assert adjusted < base_score

        # Multiple artists penalty
        adjusted = scorer.adjust_confidence_for_context(base_score, "mp3", has_multiple_artists=True)
        assert adjusted < base_score

        # Combined adjustments
        adjusted = scorer.adjust_confidence_for_context(
            base_score, "flac", is_compilation=True, has_multiple_artists=True
        )
        assert adjusted != base_score

    def test_get_confidence_category(self, scorer):
        """Test confidence categorization."""
        assert scorer.get_confidence_category(0.95) == "very_high"
        assert scorer.get_confidence_category(0.80) == "high"
        assert scorer.get_confidence_category(0.65) == "medium"
        assert scorer.get_confidence_category(0.45) == "low"
        assert scorer.get_confidence_category(0.25) == "very_low"

    def test_should_auto_approve(self, scorer):
        """Test auto-approval logic."""
        assert scorer.should_auto_approve(0.90) is True
        assert scorer.should_auto_approve(0.85) is True
        assert scorer.should_auto_approve(0.84) is False

        # Custom threshold
        assert scorer.should_auto_approve(0.70, threshold=0.70) is True
        assert scorer.should_auto_approve(0.69, threshold=0.70) is False

    def test_edge_cases(self, scorer):
        """Test edge cases and boundary conditions."""
        # Empty metadata
        score, _ = scorer.calculate_confidence(
            metadata={},
            original_filename="file.mp3",
            proposed_filename="file.mp3",
            conflicts=[],
            warnings=[],
            pattern_used=None,
            source=None,
        )
        assert 0 <= score <= 1

        # All None values in metadata
        metadata_with_nones = {"artist": None, "title": None, "album": None}
        score, _ = scorer.calculate_confidence(
            metadata=metadata_with_nones,
            original_filename="file.mp3",
            proposed_filename="file.mp3",
            conflicts=[],
            warnings=[],
            pattern_used="{artist} - {title}",
            source="id3",
        )
        assert 0 <= score <= 1

        # Very long lists
        many_warnings = ["Warning"] * 100
        score, components = scorer.calculate_confidence(
            metadata={"artist": "A", "title": "T"},
            original_filename="file.mp3",
            proposed_filename="A - T.mp3",
            conflicts=[],
            warnings=many_warnings,
            pattern_used=None,
            source="id3",
        )
        assert components["conflict_absence"] == 0.0
        assert 0 <= score <= 1
