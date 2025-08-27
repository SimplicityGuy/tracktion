"""Unit tests for CUE data models."""

import pytest

from services.analysis_service.src.cue_handler.models import (
    CueSheet,
    CueTime,
    FileReference,
    InvalidTimeFormatError,
    Track,
)


class TestCueTime:
    """Test CueTime class."""

    def test_init_valid(self):
        """Test creating valid CueTime."""
        time = CueTime(5, 30, 45)
        assert time.minutes == 5
        assert time.seconds == 30
        assert time.frames == 45

    def test_init_invalid_seconds(self):
        """Test invalid seconds raises error."""
        with pytest.raises(InvalidTimeFormatError):
            CueTime(5, 60, 45)  # 60 seconds invalid

    def test_init_invalid_frames(self):
        """Test invalid frames raises error."""
        with pytest.raises(InvalidTimeFormatError):
            CueTime(5, 30, 75)  # 75 frames invalid (max 74)

    def test_from_string_valid(self):
        """Test parsing valid time string."""
        time = CueTime.from_string("05:30:45")
        assert time.minutes == 5
        assert time.seconds == 30
        assert time.frames == 45

        # Test with leading zeros
        time = CueTime.from_string("00:00:00")
        assert time.minutes == 0
        assert time.seconds == 0
        assert time.frames == 0

    def test_from_string_invalid_format(self):
        """Test parsing invalid time string."""
        # Now allows single digit minutes, so test with invalid format
        with pytest.raises(InvalidTimeFormatError):
            CueTime.from_string("05:30")  # Missing frames

        with pytest.raises(InvalidTimeFormatError):
            CueTime.from_string("05:30:75")  # Invalid frames (max 74)

        with pytest.raises(InvalidTimeFormatError):
            CueTime.from_string("05:60:00")  # Invalid seconds (max 59)

    def test_from_milliseconds(self):
        """Test creating from milliseconds."""
        # 1 second = 1000ms = 75 frames
        time = CueTime.from_milliseconds(1000)
        assert time.seconds == 1
        assert time.frames == 0

        # 1 frame = 13.333ms (1000/75)
        time = CueTime.from_milliseconds(13)
        assert time.frames == 0  # Rounds down

        time = CueTime.from_milliseconds(14)
        assert time.frames == 1

    def test_from_frames(self):
        """Test creating from total frames."""
        time = CueTime.from_frames(75)  # 1 second
        assert time.seconds == 1
        assert time.frames == 0

        time = CueTime.from_frames(75 * 60)  # 1 minute
        assert time.minutes == 1
        assert time.seconds == 0
        assert time.frames == 0

    def test_to_frames(self):
        """Test converting to total frames."""
        time = CueTime(1, 1, 1)
        expected = 1 * 75 * 60 + 1 * 75 + 1
        assert time.to_frames() == expected

    def test_to_milliseconds(self):
        """Test converting to milliseconds."""
        time = CueTime(0, 1, 0)  # 1 second
        assert time.to_milliseconds() == 1000

        # 75 frames = 1 second, but max frame is 74, so use 0,1,0 instead
        time = CueTime(0, 1, 0)  # 1 second
        assert time.to_milliseconds() == 1000

    def test_str_representation(self):
        """Test string formatting."""
        time = CueTime(5, 30, 45)
        assert str(time) == "05:30:45"

        time = CueTime(0, 0, 0)
        assert str(time) == "00:00:00"

    def test_comparison(self):
        """Test time comparisons."""
        time1 = CueTime(1, 0, 0)
        time2 = CueTime(2, 0, 0)
        time3 = CueTime(1, 0, 0)

        assert time1 < time2
        assert time1 <= time2
        assert time2 > time1
        assert time1 == time3
        assert time1 != time2


class TestTrack:
    """Test Track class."""

    def test_init_valid(self):
        """Test creating valid track."""
        track = Track(number=1, track_type="AUDIO")
        assert track.number == 1
        assert track.track_type == "AUDIO"
        assert track.indices == {}
        assert track.flags == []

    def test_init_invalid_number(self):
        """Test invalid track number."""
        with pytest.raises(ValueError):
            Track(number=0)  # Too low

        with pytest.raises(ValueError):
            Track(number=100)  # Too high

    def test_title_truncation(self):
        """Test title truncated to 80 chars."""
        long_title = "A" * 100
        track = Track(number=1, title=long_title)
        assert len(track.title) == 80

    def test_performer_truncation(self):
        """Test performer truncated to 80 chars."""
        long_performer = "B" * 100
        track = Track(number=1, performer=long_performer)
        assert len(track.performer) == 80

    def test_isrc_validation(self):
        """Test ISRC must be 12 characters."""
        with pytest.raises(ValueError):
            Track(number=1, isrc="SHORT")

        # Valid ISRC
        track = Track(number=1, isrc="USRC17607839")
        assert track.isrc == "USRC17607839"

    def test_get_start_time(self):
        """Test getting INDEX 01 time."""
        track = Track(number=1)
        assert track.get_start_time() is None

        start_time = CueTime(0, 0, 0)
        track.indices[1] = start_time
        assert track.get_start_time() == start_time

    def test_get_pregap_start(self):
        """Test getting INDEX 00 time."""
        track = Track(number=1)
        assert track.get_pregap_start() is None

        pregap_time = CueTime(0, 0, 0)
        track.indices[0] = pregap_time
        assert track.get_pregap_start() == pregap_time

    def test_validate(self):
        """Test track validation."""
        track = Track(number=1)

        # Missing INDEX 01
        errors = track.validate()
        assert len(errors) == 1
        assert "INDEX 01" in errors[0]

        # Add INDEX 01
        track.indices[1] = CueTime(0, 0, 0)
        errors = track.validate()
        assert len(errors) == 0

        # INDEX 00 after INDEX 01
        track.indices[0] = CueTime(1, 0, 0)
        errors = track.validate()
        assert len(errors) == 1
        assert "INDEX 00" in errors[0]

    def test_flags_validation(self):
        """Test FLAGS validation."""
        track = Track(number=1)
        track.indices[1] = CueTime(0, 0, 0)

        # Valid flags
        track.flags = ["DCP", "4CH", "PRE", "SCMS"]
        errors = track.validate()
        assert len(errors) == 0

        # Invalid flag
        track.flags.append("INVALID")
        errors = track.validate()
        assert len(errors) == 1
        assert "Invalid flag" in errors[0]

    def test_to_dict(self):
        """Test dictionary conversion."""
        track = Track(number=1, track_type="AUDIO", title="Test Track", performer="Test Artist")
        track.indices[1] = CueTime(0, 0, 0)

        data = track.to_dict()
        assert data["number"] == 1
        assert data["type"] == "AUDIO"
        assert data["title"] == "Test Track"
        assert data["performer"] == "Test Artist"
        assert "1" in data["indices"]


class TestFileReference:
    """Test FileReference class."""

    def test_init(self):
        """Test creating file reference."""
        file_ref = FileReference(filename="test.mp3", file_type="MP3")
        assert file_ref.filename == "test.mp3"
        assert file_ref.file_type == "MP3"
        assert file_ref.tracks == []

    def test_validate_no_tracks(self):
        """Test validation with no tracks."""
        file_ref = FileReference(filename="test.mp3", file_type="MP3")
        errors = file_ref.validate()
        assert len(errors) == 1
        assert "No tracks" in errors[0]

    def test_validate_track_numbering(self):
        """Test validation of track numbering."""
        file_ref = FileReference(filename="test.mp3", file_type="MP3")

        # Add non-sequential tracks
        track1 = Track(number=1)
        track1.indices[1] = CueTime(0, 0, 0)
        track3 = Track(number=3)  # Skip 2
        track3.indices[1] = CueTime(1, 0, 0)

        file_ref.tracks = [track1, track3]
        errors = file_ref.validate()

        assert any("Non-sequential" in e for e in errors)


class TestCueSheet:
    """Test CueSheet class."""

    def test_init(self):
        """Test creating CUE sheet."""
        sheet = CueSheet()
        assert sheet.title is None
        assert sheet.performer is None
        assert sheet.files == []
        assert sheet.rem_fields == {}

    def test_title_truncation(self):
        """Test title truncated to 80 chars."""
        sheet = CueSheet(title="A" * 100)
        assert len(sheet.title) == 80

    def test_get_all_tracks(self):
        """Test getting all tracks from all files."""
        sheet = CueSheet()

        file1 = FileReference("file1.mp3", "MP3")
        file1.tracks.append(Track(number=1))
        file1.tracks.append(Track(number=2))

        file2 = FileReference("file2.mp3", "MP3")
        file2.tracks.append(Track(number=3))

        sheet.files = [file1, file2]

        all_tracks = sheet.get_all_tracks()
        assert len(all_tracks) == 3
        assert all_tracks[0].number == 1
        assert all_tracks[2].number == 3

    def test_get_track_count(self):
        """Test getting total track count."""
        sheet = CueSheet()
        assert sheet.get_track_count() == 0

        file1 = FileReference("file1.mp3", "MP3")
        file1.tracks = [Track(number=1), Track(number=2)]
        sheet.files.append(file1)

        assert sheet.get_track_count() == 2

    def test_validate_no_files(self):
        """Test validation with no files."""
        sheet = CueSheet()
        errors = sheet.validate()
        assert len(errors) == 1
        assert "No FILE" in errors[0]

    def test_validate_duplicate_tracks(self):
        """Test validation of duplicate track numbers."""
        sheet = CueSheet()

        file1 = FileReference("file1.mp3", "MP3")
        track1 = Track(number=1)
        track1.indices[1] = CueTime(0, 0, 0)
        file1.tracks.append(track1)

        file2 = FileReference("file2.mp3", "MP3")
        track1_dup = Track(number=1)  # Duplicate number
        track1_dup.indices[1] = CueTime(1, 0, 0)
        file2.tracks.append(track1_dup)

        sheet.files = [file1, file2]
        errors = sheet.validate()

        assert any("Duplicate track number" in e for e in errors)

    def test_to_dict(self):
        """Test dictionary conversion."""
        sheet = CueSheet(title="Test Album", performer="Test Artist", catalog="1234567890123")

        file_ref = FileReference("test.mp3", "MP3")
        track = Track(number=1, title="Track 1")
        track.indices[1] = CueTime(0, 0, 0)
        file_ref.tracks.append(track)
        sheet.files.append(file_ref)

        data = sheet.to_dict()
        assert data["title"] == "Test Album"
        assert data["performer"] == "Test Artist"
        assert data["catalog"] == "1234567890123"
        assert len(data["files"]) == 1
        assert len(data["files"][0]["tracks"]) == 1
