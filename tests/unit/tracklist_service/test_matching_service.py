"""
Unit tests for the matching service.
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch
import os

import pytest

from services.tracklist_service.src.models.tracklist_models import (
    Tracklist as ScrapedTracklist,
    TracklistMetadata,
    CuePoint,
    Track
)
from services.tracklist_service.src.services.matching_service import MatchingService


class TestMatchingService:
    """Test the matching service functionality."""
    
    @pytest.fixture
    def matching_service(self):
        """Create a matching service instance."""
        return MatchingService()
    
    @pytest.fixture
    def sample_scraped_tracklist(self):
        """Create a sample scraped tracklist."""
        metadata = TracklistMetadata(
            duration_minutes=60,
            play_count=1000
        )
        
        tracks = [
            Track(
                number=1,
                timestamp=CuePoint(track_number=1, timestamp_ms=0, formatted_time="00:00"),
                artist="Artist 1",
                title="Track 1"
            ),
            Track(
                number=2,
                timestamp=CuePoint(track_number=2, timestamp_ms=3600000, formatted_time="60:00"),
                artist="Artist 2",
                title="Track 2"
            )
        ]
        
        return ScrapedTracklist(
            url="https://1001tracklists.com/test",
            dj_name="Test DJ",
            event_name="Test Festival",
            venue="Test Venue",
            date=date(2024, 1, 15),
            metadata=metadata,
            tracks=tracks
        )
    
    def test_fuzzy_match_exact(self, matching_service):
        """Test fuzzy matching with exact match."""
        score = matching_service._fuzzy_match("Test String", "Test String")
        assert score == 1.0
    
    def test_fuzzy_match_case_insensitive(self, matching_service):
        """Test fuzzy matching is case insensitive."""
        score = matching_service._fuzzy_match("Test String", "test string")
        assert score > 0.9
    
    def test_fuzzy_match_partial(self, matching_service):
        """Test fuzzy matching with partial match."""
        score = matching_service._fuzzy_match("Test", "Test String")
        assert score >= 0.7  # Substring match gets at least 0.7
    
    def test_fuzzy_match_empty(self, matching_service):
        """Test fuzzy matching with empty strings."""
        assert matching_service._fuzzy_match("", "Test") == 0.0
        assert matching_service._fuzzy_match("Test", "") == 0.0
        assert matching_service._fuzzy_match("", "") == 0.0
    
    def test_normalize_string(self, matching_service):
        """Test string normalization."""
        # Test removing DJ mix indicators
        normalized = matching_service._normalize_string("DJ Test @ Festival - Live Set")
        assert "dj test" in normalized
        assert "festival" in normalized
        assert "@" not in normalized
        assert "-" not in normalized
        
        # Test removing special characters
        normalized = matching_service._normalize_string("Test!@#$%^&*()")
        assert normalized == "test"
    
    def test_match_duration_exact(self, matching_service, sample_scraped_tracklist):
        """Test duration matching with exact match."""
        # Tracklist is 60 minutes
        score = matching_service._match_duration(sample_scraped_tracklist, 3600)  # 60 minutes
        assert score == 1.0
    
    def test_match_duration_close(self, matching_service, sample_scraped_tracklist):
        """Test duration matching with close match."""
        # Within 5% (3600 +/- 180 seconds)
        score = matching_service._match_duration(sample_scraped_tracklist, 3700)  # ~2.7% diff
        assert score >= 0.9
    
    def test_match_duration_moderate(self, matching_service, sample_scraped_tracklist):
        """Test duration matching with moderate difference."""
        # Within 10% (3600 +/- 360 seconds)
        score = matching_service._match_duration(sample_scraped_tracklist, 3900)  # ~8.3% diff
        assert score >= 0.7
    
    def test_match_duration_no_metadata(self, matching_service):
        """Test duration matching with no duration metadata."""
        tracklist = ScrapedTracklist(
            url="https://1001tracklists.com/test",
            dj_name="Test DJ",
            tracks=[]
        )
        score = matching_service._match_duration(tracklist, 3600)
        assert score == 0.5  # Neutral score when no duration info
    
    def test_match_date_exact(self, matching_service):
        """Test date matching with exact match."""
        date1 = date(2024, 1, 15)
        date2 = date(2024, 1, 15)
        score = matching_service._match_date(date1, date2)
        assert score == 1.0
    
    def test_match_date_close(self, matching_service):
        """Test date matching with close dates."""
        date1 = date(2024, 1, 15)
        date2 = date(2024, 1, 16)  # 1 day diff
        score = matching_service._match_date(date1, date2)
        assert score == 0.9
        
        date3 = date(2024, 1, 20)  # 5 days diff
        score = matching_service._match_date(date1, date3)
        assert score == 0.7
    
    def test_match_date_no_date(self, matching_service):
        """Test date matching with missing dates."""
        score = matching_service._match_date(None, date(2024, 1, 15))
        assert score == 0.5  # Neutral score
        
        score = matching_service._match_date(date(2024, 1, 15), None)
        assert score == 0.5
    
    def test_match_tracklist_to_audio_complete(
        self, 
        matching_service, 
        sample_scraped_tracklist
    ):
        """Test complete tracklist to audio matching."""
        audio_metadata = {
            'title': 'Test DJ - Test Festival 2024',
            'artist': 'Test DJ',
            'duration_seconds': 3600,
            'date': '2024-01-15',
            'album': 'Test Festival'
        }
        
        confidence, details = matching_service.match_tracklist_to_audio(
            sample_scraped_tracklist,
            audio_metadata
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high confidence
        assert 'title_score' in details
        assert 'artist_score' in details
        assert 'duration_score' in details
        assert 'date_score' in details
        assert 'event_score' in details
        assert details['artist_score'] == 1.0  # Exact match
        assert details['duration_score'] == 1.0  # Exact match
    
    def test_match_tracklist_to_audio_partial(
        self, 
        matching_service, 
        sample_scraped_tracklist
    ):
        """Test partial tracklist to audio matching."""
        audio_metadata = {
            'title': 'DJ Mix Recording',
            'artist': 'Different Artist',
            'duration_seconds': 4000,  # Different duration
        }
        
        confidence, details = matching_service.match_tracklist_to_audio(
            sample_scraped_tracklist,
            audio_metadata
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.5  # Should be low confidence
        assert details['artist_score'] < 0.5  # Poor match
    
    def test_validate_audio_file_exists(self, matching_service):
        """Test audio file validation with existing file."""
        with patch('os.path.exists', return_value=True):
            with patch('os.access', return_value=True):
                with patch('os.path.getsize', return_value=10_000_000):  # 10MB
                    result = matching_service.validate_audio_file('/path/to/audio.mp3')
                    assert result is True
    
    def test_validate_audio_file_not_exists(self, matching_service):
        """Test audio file validation with non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = matching_service.validate_audio_file('/path/to/nonexistent.mp3')
            assert result is False
    
    def test_validate_audio_file_not_readable(self, matching_service):
        """Test audio file validation with non-readable file."""
        with patch('os.path.exists', return_value=True):
            with patch('os.access', return_value=False):
                result = matching_service.validate_audio_file('/path/to/audio.mp3')
                assert result is False
    
    def test_validate_audio_file_too_small(self, matching_service):
        """Test audio file validation with file that's too small."""
        with patch('os.path.exists', return_value=True):
            with patch('os.access', return_value=True):
                with patch('os.path.getsize', return_value=500_000):  # 500KB - too small
                    result = matching_service.validate_audio_file('/path/to/audio.mp3')
                    assert result is False
    
    def test_calculate_weighted_confidence(self, matching_service):
        """Test weighted confidence calculation."""
        scores = [
            ('title', 0.8),
            ('artist', 0.9),
            ('duration', 1.0),
            ('date', 0.7),
            ('event', 0.6)
        ]
        
        confidence = matching_service._calculate_weighted_confidence(scores)
        
        # Manual calculation with default weights:
        # title: 0.8 * 0.3 = 0.24
        # artist: 0.9 * 0.25 = 0.225
        # duration: 1.0 * 0.25 = 0.25
        # date: 0.7 * 0.1 = 0.07
        # event: 0.6 * 0.1 = 0.06
        # Total: 0.845
        
        assert 0.8 <= confidence <= 0.9
        assert confidence == pytest.approx(0.845, rel=0.01)