"""Unit tests for time parsing and formatting utilities."""

import pytest
from datetime import timedelta

from services.tracklist_service.src.utils.time_utils import (
    parse_time_string,
    time_string_to_seconds,
    timedelta_to_milliseconds,
    milliseconds_to_timedelta,
    format_timedelta,
    parse_cue_time,
    seconds_to_cue_time,
    validate_time_range,
)


class TestParseTimeString:
    """Tests for parse_time_string function."""

    def test_parse_hhmmss_format(self):
        """Test parsing HH:MM:SS format."""
        assert parse_time_string("01:30:45") == timedelta(hours=1, minutes=30, seconds=45)
        assert parse_time_string("00:05:30") == timedelta(minutes=5, seconds=30)
        assert parse_time_string("23:59:59") == timedelta(hours=23, minutes=59, seconds=59)

    def test_parse_mmss_format(self):
        """Test parsing MM:SS format."""
        assert parse_time_string("5:30") == timedelta(minutes=5, seconds=30)
        assert parse_time_string("59:59") == timedelta(minutes=59, seconds=59)
        assert parse_time_string("0:30") == timedelta(seconds=30)

    def test_parse_decimal_minutes(self):
        """Test parsing decimal minutes format."""
        assert parse_time_string("5.5") == timedelta(minutes=5.5)
        assert parse_time_string("10.25") == timedelta(minutes=10.25)
        assert parse_time_string("0.5") == timedelta(seconds=30)

    def test_parse_integer_seconds(self):
        """Test parsing integer seconds."""
        assert parse_time_string("300") == timedelta(seconds=300)
        assert parse_time_string("90") == timedelta(seconds=90)
        assert parse_time_string("3600") == timedelta(hours=1)

    def test_parse_empty_or_invalid(self):
        """Test parsing empty or invalid strings."""
        assert parse_time_string("") == timedelta(0)
        assert parse_time_string("invalid") == timedelta(0)
        assert parse_time_string("1:2:3:4") == timedelta(0)


class TestTimeStringToSeconds:
    """Tests for time_string_to_seconds function."""

    def test_convert_to_seconds(self):
        """Test converting time strings to seconds."""
        assert time_string_to_seconds("1:30:00") == 5400
        assert time_string_to_seconds("5:30") == 330
        assert time_string_to_seconds("300") == 300
        assert time_string_to_seconds("5.5") == 330


class TestTimedeltaConversions:
    """Tests for timedelta conversion functions."""

    def test_timedelta_to_milliseconds(self):
        """Test converting timedelta to milliseconds."""
        assert timedelta_to_milliseconds(timedelta(seconds=1)) == 1000
        assert timedelta_to_milliseconds(timedelta(seconds=1.5)) == 1500
        assert timedelta_to_milliseconds(timedelta(minutes=1, seconds=30)) == 90000

    def test_milliseconds_to_timedelta(self):
        """Test converting milliseconds to timedelta."""
        assert milliseconds_to_timedelta(1000) == timedelta(seconds=1)
        assert milliseconds_to_timedelta(1500) == timedelta(seconds=1.5)
        assert milliseconds_to_timedelta(90000) == timedelta(minutes=1, seconds=30)


class TestFormatTimedelta:
    """Tests for format_timedelta function."""

    def test_format_hhmmss(self):
        """Test formatting to HH:MM:SS."""
        assert format_timedelta(timedelta(hours=1, minutes=30, seconds=15)) == "01:30:15"
        assert format_timedelta(timedelta(minutes=5, seconds=30)) == "00:05:30"
        assert format_timedelta(timedelta(seconds=45)) == "00:00:45"

    def test_format_mmss(self):
        """Test formatting to MM:SS."""
        assert format_timedelta(timedelta(minutes=5, seconds=30), format="MM:SS") == "05:30"
        assert format_timedelta(timedelta(hours=1, minutes=30), format="MM:SS") == "90:00"
        assert format_timedelta(timedelta(seconds=45), format="MM:SS") == "00:45"

    def test_format_seconds(self):
        """Test formatting to seconds."""
        assert format_timedelta(timedelta(minutes=5), format="seconds") == "300"
        assert format_timedelta(timedelta(hours=1), format="seconds") == "3600"


class TestCueTimeFormat:
    """Tests for CUE time format functions."""

    def test_parse_cue_time(self):
        """Test parsing CUE time format."""
        assert parse_cue_time("05:30:00") == 330.0
        assert parse_cue_time("01:30:45") == pytest.approx(90.6, 0.1)  # 45 frames = 0.6 seconds
        assert parse_cue_time("00:00:75") == 1.0  # 75 frames = 1 second

    def test_seconds_to_cue_time(self):
        """Test converting seconds to CUE time format."""
        assert seconds_to_cue_time(330.0) == "05:30:00"
        # 90.6 seconds = 1 minute 30 seconds and 44-45 frames (depending on rounding)
        result = seconds_to_cue_time(90.6)
        assert result in ["01:30:44", "01:30:45"]  # Allow for rounding differences
        assert seconds_to_cue_time(1.0) == "00:01:00"


class TestValidateTimeRange:
    """Tests for validate_time_range function."""

    def test_valid_range(self):
        """Test valid time ranges."""
        assert validate_time_range(timedelta(minutes=1), timedelta(minutes=3)) is True
        assert validate_time_range(timedelta(minutes=5), timedelta(minutes=10)) is True

    def test_invalid_range(self):
        """Test invalid time ranges."""
        # Negative start time
        assert validate_time_range(timedelta(seconds=-1), timedelta(minutes=1)) is False
        # End before start
        assert validate_time_range(timedelta(minutes=5), timedelta(minutes=3)) is False
        # Duration too short
        assert validate_time_range(timedelta(minutes=1), timedelta(minutes=1, seconds=10)) is False
        # Duration too long
        assert validate_time_range(timedelta(0), timedelta(minutes=30)) is False

    def test_custom_duration_limits(self):
        """Test with custom duration limits."""
        assert (
            validate_time_range(
                timedelta(0),
                timedelta(seconds=10),
                min_duration=timedelta(seconds=5),
                max_duration=timedelta(seconds=20),
            )
            is True
        )
        assert (
            validate_time_range(
                timedelta(0),
                timedelta(seconds=3),
                min_duration=timedelta(seconds=5),
                max_duration=timedelta(seconds=20),
            )
            is False
        )
