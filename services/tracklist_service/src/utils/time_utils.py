"""Time parsing and formatting utilities for tracklist service.

This module centralizes all time parsing and conversion logic to ensure consistency
and avoid code duplication across the tracklist service.
"""

import re
from datetime import timedelta
from typing import Optional, Union


def parse_time_string(time_str: str) -> timedelta:
    """Parse various time formats to timedelta.

    Args:
        time_str: Time string in various formats.

    Returns:
        timedelta object.

    Supported formats:
        - HH:MM:SS (e.g., "1:05:30")
        - MM:SS (e.g., "5:30")
        - H:MM:SS (e.g., "1:05:30")
        - M:SS (e.g., "5:30")
        - Decimal minutes (e.g., "5.5" for 5 minutes 30 seconds)
        - Integer seconds (e.g., "300" for 5 minutes)

    Examples:
        >>> parse_time_string("1:30:00")
        timedelta(hours=1, minutes=30)
        >>> parse_time_string("5:30")
        timedelta(minutes=5, seconds=30)
        >>> parse_time_string("5.5")
        timedelta(minutes=5, seconds=30)
    """
    if not time_str:
        return timedelta(0)

    time_str = time_str.strip()

    # Try decimal minutes format (e.g., "5.5")
    if "." in time_str and ":" not in time_str:
        try:
            minutes = float(time_str)
            return timedelta(minutes=minutes)
        except ValueError:
            pass

    # Try integer seconds format (just a number)
    if time_str.isdigit():
        return timedelta(seconds=int(time_str))

    # Try time format (MM:SS or HH:MM:SS)
    if ":" in time_str:
        parts = time_str.split(":")

        try:
            if len(parts) == 2:
                # MM:SS or M:SS format
                minutes = int(parts[0])
                seconds = int(parts[1])
                return timedelta(minutes=minutes, seconds=seconds)
            elif len(parts) == 3:
                # HH:MM:SS or H:MM:SS format
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except ValueError:
            pass

    # If all parsing attempts fail, return zero
    return timedelta(0)


def time_string_to_seconds(time_str: str) -> int:
    """Convert time string to total seconds.

    Args:
        time_str: Time string in various formats.

    Returns:
        Total seconds as integer.

    Examples:
        >>> time_string_to_seconds("1:30:00")
        5400
        >>> time_string_to_seconds("5:30")
        330
    """
    td = parse_time_string(time_str)
    return int(td.total_seconds())


def timedelta_to_milliseconds(td: timedelta) -> int:
    """Convert timedelta to milliseconds.

    Args:
        td: timedelta object.

    Returns:
        Total milliseconds as integer.

    Examples:
        >>> timedelta_to_milliseconds(timedelta(seconds=1.5))
        1500
    """
    return int(td.total_seconds() * 1000)


def milliseconds_to_timedelta(ms: Union[int, float]) -> timedelta:
    """Convert milliseconds to timedelta.

    Args:
        ms: Milliseconds as integer or float.

    Returns:
        timedelta object.

    Examples:
        >>> milliseconds_to_timedelta(1500)
        timedelta(seconds=1.5)
    """
    return timedelta(milliseconds=ms)


def format_timedelta(td: timedelta, format: str = "HH:MM:SS") -> str:
    """Format timedelta to string representation.

    Args:
        td: timedelta object.
        format: Output format ("HH:MM:SS", "MM:SS", or "seconds").

    Returns:
        Formatted time string.

    Examples:
        >>> format_timedelta(timedelta(hours=1, minutes=30, seconds=15))
        "01:30:15"
        >>> format_timedelta(timedelta(minutes=5, seconds=30), format="MM:SS")
        "05:30"
    """
    total_seconds = int(td.total_seconds())

    if format == "seconds":
        return str(total_seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if format == "MM:SS":
        # Convert hours to minutes if present
        total_minutes = hours * 60 + minutes
        return f"{total_minutes:02d}:{seconds:02d}"
    else:  # HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_cue_time(time_str: str) -> float:
    """Parse CUE file time format to seconds.

    Args:
        time_str: CUE time string (MM:SS:FF format where FF is frames).

    Returns:
        Time in seconds as float.

    Examples:
        >>> parse_cue_time("05:30:00")
        330.0
        >>> parse_cue_time("01:30:45")
        90.6  # 45 frames = 0.6 seconds
    """
    if not time_str:
        return 0.0

    # CUE format is MM:SS:FF where FF is frames (75 frames = 1 second)
    match = re.match(r"(\d+):(\d+):(\d+)", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frames = int(match.group(3))
        return minutes * 60 + seconds + frames / 75.0

    # Fallback to standard parsing
    td = parse_time_string(time_str)
    return td.total_seconds()


def seconds_to_cue_time(seconds: float) -> str:
    """Convert seconds to CUE file time format.

    Args:
        seconds: Time in seconds.

    Returns:
        CUE time string (MM:SS:FF format).

    Examples:
        >>> seconds_to_cue_time(330.0)
        "05:30:00"
        >>> seconds_to_cue_time(90.6)
        "01:30:45"
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    frames = int((seconds % 1) * 75)

    return f"{minutes:02d}:{remaining_seconds:02d}:{frames:02d}"


def validate_time_range(
    start_time: timedelta,
    end_time: Optional[timedelta],
    min_duration: timedelta = timedelta(seconds=30),
    max_duration: timedelta = timedelta(minutes=20),
) -> bool:
    """Validate that a time range meets duration requirements.

    Args:
        start_time: Start time of the range.
        end_time: End time of the range (optional).
        min_duration: Minimum allowed duration.
        max_duration: Maximum allowed duration.

    Returns:
        True if valid, False otherwise.
    """
    if start_time < timedelta(0):
        return False

    if end_time:
        if end_time <= start_time:
            return False

        duration = end_time - start_time
        return min_duration <= duration <= max_duration

    return True
