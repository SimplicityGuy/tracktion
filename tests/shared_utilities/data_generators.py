"""Test data generators for consistent test data across services."""

import hashlib
import random
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any


class TestDataGenerator:
    """Centralized test data generator for all services."""

    def __init__(self, seed: int = 42):
        """Initialize with a fixed seed for deterministic tests."""
        random.seed(seed)
        self._uuid_counter = 0

    def generate_uuid_string(self) -> str:
        """Generate a deterministic UUID string for tests."""
        self._uuid_counter += 1
        # Create deterministic UUID based on counter
        namespace = uuid.NAMESPACE_DNS
        name = f"test-{self._uuid_counter}"
        return str(uuid.uuid5(namespace, name))

    def generate_hash(self, content: str | None = None) -> str:
        """Generate a hash for testing purposes."""
        if content is None:
            content = f"test_content_{random.randint(1000, 9999)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def generate_timestamp(self, offset_minutes: int = 0) -> str:
        """Generate an ISO timestamp for testing."""
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        if offset_minutes:
            base_time += timedelta(minutes=offset_minutes)
        return base_time.isoformat()


# Global instance for consistent data generation
_generator = TestDataGenerator()


def generate_sample_metadata(count: int = 3) -> list[dict[str, Any]]:
    """Generate sample metadata for testing."""
    metadata_types = [
        {"key": "bpm", "value": f"{random.randint(120, 140)}"},
        {"key": "genre", "value": random.choice(["techno", "house", "trance", "ambient"])},
        {"key": "key", "value": random.choice(["A minor", "C major", "F# minor", "Bb major"])},
        {"key": "energy", "value": f"{random.uniform(0.1, 1.0):.2f}"},
        {"key": "danceability", "value": f"{random.uniform(0.1, 1.0):.2f}"},
        {"key": "valence", "value": f"{random.uniform(0.1, 1.0):.2f}"},
    ]

    return random.sample(metadata_types, min(count, len(metadata_types)))


def generate_track_data(count: int = 3) -> list[dict[str, Any]]:
    """Generate sample track data for testing."""
    artists = ["DJ One", "DJ Two", "DJ Three", "Producer X", "Artist Y"]
    titles = ["Opening Track", "Peak Time", "Breakdown", "Sunrise", "Midnight"]
    keys = ["A minor", "C major", "F# minor", "Bb major", "D minor"]

    tracks = []
    current_time = 0

    for i in range(count):
        duration = random.randint(180, 480)  # 3-8 minutes
        track = {
            "position": i + 1,
            "title": f"{random.choice(titles)} {i + 1}",
            "artist": random.choice(artists),
            "start_time": f"{current_time // 3600:02d}:{(current_time % 3600) // 60:02d}:{current_time % 60:02d}",
            "duration": duration,
            "bpm": random.randint(120, 140),
            "key": random.choice(keys),
        }
        tracks.append(track)
        current_time += duration

    return tracks


def generate_file_event(event_type: str = "created", file_path: str | None = None) -> dict[str, Any]:
    """Generate a file event for testing."""
    if file_path is None:
        file_path = f"/music/test_song_{random.randint(1, 100)}.mp3"

    return {
        "event_type": event_type,
        "file_path": file_path,
        "timestamp": generate_timestamp(),
        "instance_id": "test_watcher",
        "sha256_hash": generate_hash(file_path),
        "xxh128_hash": generate_hash(f"xxh_{file_path}")[:32],
    }


def generate_complex_track_data() -> dict[str, Any]:
    """Generate complex track data with nested metadata for JSONB testing."""
    return {
        "title": "Complex Track",
        "artist": "DJ Complex",
        "start_time": "00:00:00",
        "end_time": "00:05:30",
        "duration": 330,
        "bmp": 128,
        "key": "A minor",
        "genre": "techno",
        "metadata": {
            "energy": 0.8,
            "danceability": 0.9,
            "valence": 0.7,
            "acousticness": 0.1,
            "custom_field": "custom_value",
        },
        "transitions": [
            {"type": "fade_in", "duration": 8, "start_volume": 0.0, "end_volume": 1.0},
            {"type": "fade_out", "duration": 12, "start_volume": 1.0, "end_volume": 0.0},
        ],
        "analysis": {
            "tempo_changes": [{"time": "00:02:30", "bpm": 130}, {"time": "00:04:00", "bpm": 126}],
            "key_changes": [{"time": "00:03:15", "key": "C major"}],
            "structure": [
                {"section": "intro", "start": "00:00:00", "end": "00:01:00"},
                {"section": "buildup", "start": "00:01:00", "end": "00:03:00"},
                {"section": "drop", "start": "00:03:00", "end": "00:05:30"},
            ],
        },
    }


def generate_recording_data() -> dict[str, Any]:
    """Generate sample recording data."""
    file_path = f"/music/test_{random.randint(1, 100)}.mp3"
    return {
        "file_path": file_path,
        "file_name": file_path.split("/")[-1],
        "sha256_hash": generate_hash(file_path),
        "xxh128_hash": generate_hash(f"xxh_{file_path}")[:32],
    }


# Convenience functions using the global generator
def generate_uuid_string() -> str:
    """Generate a deterministic UUID string."""
    return _generator.generate_uuid_string()


def generate_hash(content: str | None = None) -> str:
    """Generate a hash for testing."""
    return _generator.generate_hash(content)


def generate_timestamp(offset_minutes: int = 0) -> str:
    """Generate an ISO timestamp."""
    return _generator.generate_timestamp(offset_minutes)
