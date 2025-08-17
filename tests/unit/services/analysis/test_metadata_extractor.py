"""Unit tests for metadata extraction module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "services" / "analysis_service" / "src"))

from metadata_extractor import (
    MetadataExtractor,
    InvalidAudioFileError,
    MetadataExtractionError
)


class TestMetadataExtractor:
    """Test cases for MetadataExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
    
    def test_init(self):
        """Test MetadataExtractor initialization."""
        assert self.extractor is not None
        assert len(self.extractor.SUPPORTED_FORMATS) > 0
        assert '.mp3' in self.extractor.SUPPORTED_FORMATS
        assert '.flac' in self.extractor.SUPPORTED_FORMATS
        assert '.wav' in self.extractor.SUPPORTED_FORMATS
        assert '.m4a' in self.extractor.SUPPORTED_FORMATS
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = self.extractor.get_supported_formats()
        assert isinstance(formats, set)
        assert '.mp3' in formats
        assert '.flac' in formats
        assert '.wav' in formats
        assert '.m4a' in formats
    
    def test_extract_file_not_found(self):
        """Test extraction with non-existent file."""
        with pytest.raises(InvalidAudioFileError) as exc_info:
            self.extractor.extract("/path/to/nonexistent/file.mp3")
        assert "File not found" in str(exc_info.value)
    
    def test_extract_directory_not_file(self):
        """Test extraction with directory instead of file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(InvalidAudioFileError) as exc_info:
                self.extractor.extract(tmpdir)
            assert "Not a file" in str(exc_info.value)
    
    def test_extract_unsupported_format(self):
        """Test extraction with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmpfile:
            with pytest.raises(InvalidAudioFileError) as exc_info:
                self.extractor.extract(tmpfile.name)
            assert "Unsupported format" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
    
    @patch('metadata_extractor.MP3')
    def test_extract_mp3_success(self, mock_mp3):
        """Test successful MP3 extraction."""
        # Mock MP3 file object
        mock_audio = MagicMock()
        mock_audio.tags = {
            'TIT2': MagicMock(text=['Test Title']),
            'TPE1': MagicMock(text=['Test Artist']),
            'TALB': MagicMock(text=['Test Album'])
        }
        mock_audio.info.length = 180.5
        mock_audio.info.bitrate = 320000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.version = 1
        mock_audio.info.layer = 3
        mock_mp3.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.mp3') as tmpfile:
            result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'mp3'
        assert result['title'] == 'Test Title'
        assert result['artist'] == 'Test Artist'
        assert result['album'] == 'Test Album'
        assert result['duration'] == '180.5'
        assert result['bitrate'] == '320000'
        assert result['sample_rate'] == '44100'
    
    @patch('metadata_extractor.FLAC')
    def test_extract_flac_success(self, mock_flac):
        """Test successful FLAC extraction."""
        # Mock FLAC file object
        mock_audio = MagicMock()
        mock_audio.tags = MagicMock()
        mock_audio.get.side_effect = lambda key, default: {
            'title': ['Test FLAC Title'],
            'artist': ['Test FLAC Artist'],
            'album': ['Test FLAC Album'],
            'date': ['2023'],
            'genre': ['Electronic'],
            'tracknumber': ['5'],
            'albumartist': ['Various Artists'],
            'comment': ['Test comment']
        }.get(key, default)
        
        mock_audio.info.length = 240.75
        mock_audio.info.sample_rate = 48000
        mock_audio.info.channels = 2
        mock_audio.info.bits_per_sample = 16
        mock_flac.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.flac') as tmpfile:
            result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'flac'
        assert result['title'] == 'Test FLAC Title'
        assert result['artist'] == 'Test FLAC Artist'
        assert result['album'] == 'Test FLAC Album'
        assert result['duration'] == '240.75'
        assert result['sample_rate'] == '48000'
    
    @patch('metadata_extractor.WAVE')
    def test_extract_wav_success(self, mock_wave):
        """Test successful WAV extraction."""
        # Mock WAV file object
        mock_audio = MagicMock()
        mock_audio.tags = {
            'TIT2': MagicMock(text=['Test WAV Title']),
            'TPE1': MagicMock(text=['Test WAV Artist'])
        }
        mock_audio.info.length = 120.0
        mock_audio.info.bitrate = 1411200
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.bits_per_sample = 16
        mock_wave.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpfile:
            result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'wav'
        assert result['title'] == 'Test WAV Title'
        assert result['artist'] == 'Test WAV Artist'
        assert result['duration'] == '120.0'
        assert result['bitrate'] == '1411200'
    
    @patch('metadata_extractor.MP4')
    def test_extract_m4a_success(self, mock_mp4):
        """Test successful M4A extraction."""
        # Mock MP4/M4A file object
        mock_audio = MagicMock()
        mock_audio.tags = {
            '\xa9nam': ['Test M4A Title'],
            '\xa9ART': ['Test M4A Artist'],
            '\xa9alb': ['Test M4A Album'],
            '\xa9day': ['2024'],
            '\xa9gen': ['Pop'],
            'trkn': [(3, 12)],  # Track 3 of 12
            'aART': ['Album Artist'],
            '\xa9cmt': ['Test comment']
        }
        mock_audio.info.length = 195.25
        mock_audio.info.bitrate = 256000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.codec = 'mp4a.40.2'
        mock_audio.info.codec_description = 'AAC LC'
        mock_mp4.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.m4a') as tmpfile:
            result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'm4a'
        assert result['title'] == 'Test M4A Title'
        assert result['artist'] == 'Test M4A Artist'
        assert result['album'] == 'Test M4A Album'
        assert result['track'] == '3'
        assert result['duration'] == '195.25'
    
    @patch('metadata_extractor.File')
    def test_extract_generic_fallback(self, mock_file):
        """Test generic extraction fallback."""
        # Mock generic file object
        mock_audio = MagicMock()
        mock_audio.tags = {
            'TIT2': MagicMock(text=['Generic Title']),
            'TPE1': MagicMock(text=['Generic Artist'])
        }
        mock_audio.info.length = 300.0
        mock_audio.info.bitrate = 192000
        mock_file.return_value = mock_audio
        
        # Patch specific handlers to None to force generic fallback
        with patch.object(self.extractor, '_format_handlers', {}):
            with tempfile.NamedTemporaryFile(suffix='.mp3') as tmpfile:
                result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'mp3'
        assert result['title'] == 'Generic Title'
        assert result['artist'] == 'Generic Artist'
    
    @patch('metadata_extractor.MP3')
    def test_extract_missing_metadata(self, mock_mp3):
        """Test extraction with missing metadata fields."""
        # Mock MP3 with minimal metadata
        mock_audio = MagicMock()
        mock_audio.tags = {}
        mock_audio.info.length = 60.0
        mock_audio.info.bitrate = 128000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.version = 1
        mock_audio.info.layer = 3
        mock_mp3.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.mp3') as tmpfile:
            result = self.extractor.extract(tmpfile.name)
        
        assert result['format'] == 'mp3'
        assert result['title'] is None
        assert result['artist'] is None
        assert result['album'] is None
        assert result['duration'] == '60.0'
    
    @patch('metadata_extractor.MP3')
    def test_extract_corrupted_file(self, mock_mp3):
        """Test extraction with corrupted file."""
        mock_mp3.side_effect = Exception("File corrupted")
        
        with tempfile.NamedTemporaryFile(suffix='.mp3') as tmpfile:
            with pytest.raises(MetadataExtractionError) as exc_info:
                self.extractor.extract(tmpfile.name)
            assert "Failed to extract metadata" in str(exc_info.value)
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert self.extractor._format_duration(None) is None
        assert self.extractor._format_duration(60.0) == '60.0'
        assert self.extractor._format_duration(123.456789) == '123.457'
        assert self.extractor._format_duration(0.1) == '0.1'
    
    def test_get_tag_value(self):
        """Test tag value extraction."""
        # Test with ID3 frame
        mock_frame = MagicMock()
        mock_frame.text = ['Test Value']
        tags = {'TIT2': mock_frame}
        result = self.extractor._get_tag_value(tags, ['TIT2'])
        assert result == 'Test Value'
        
        # Test with list value
        tags = {'title': ['List Value']}
        result = self.extractor._get_tag_value(tags, ['title'])
        assert result == 'List Value'
        
        # Test with string value
        tags = {'key': 'String Value'}
        result = self.extractor._get_tag_value(tags, ['key'])
        assert result == 'String Value'
        
        # Test with missing key
        tags = {'other': 'value'}
        result = self.extractor._get_tag_value(tags, ['missing'])
        assert result is None
    
    def test_get_mp4_tag(self):
        """Test MP4 tag extraction."""
        # Test with list value
        tags = {'\xa9nam': ['MP4 Title']}
        result = self.extractor._get_mp4_tag(tags, '\xa9nam')
        assert result == 'MP4 Title'
        
        # Test with direct value
        tags = {'key': 'Direct Value'}
        result = self.extractor._get_mp4_tag(tags, 'key')
        assert result == 'Direct Value'
        
        # Test with missing key
        tags = {'\xa9ART': ['Artist']}
        result = self.extractor._get_mp4_tag(tags, '\xa9nam')
        assert result is None