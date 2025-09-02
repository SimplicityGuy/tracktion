"""Unit tests for cataloging service models without database connection."""

import uuid
from unittest.mock import Mock

from services.cataloging_service.src.models.base import Base
from services.cataloging_service.src.models.metadata import Metadata
from services.cataloging_service.src.models.recording import Recording
from services.cataloging_service.src.models.tracklist import Tracklist


class TestModelStructure:
    """Test model structure without database operations."""

    def test_recording_model_attributes(self):
        """Test Recording model has expected attributes."""

        # Check that the model has the expected mapped columns
        assert hasattr(Recording, "id")
        assert hasattr(Recording, "file_path")
        assert hasattr(Recording, "file_name")
        assert hasattr(Recording, "sha256_hash")
        assert hasattr(Recording, "xxh128_hash")
        assert hasattr(Recording, "created_at")
        assert hasattr(Recording, "metadata_items")
        assert hasattr(Recording, "tracklists")

        # Check tablename
        assert Recording.__tablename__ == "recordings"

    def test_metadata_model_attributes(self):
        """Test Metadata model has expected attributes."""
        # Check that the model has the expected mapped columns
        assert hasattr(Metadata, "id")
        assert hasattr(Metadata, "recording_id")
        assert hasattr(Metadata, "key")
        assert hasattr(Metadata, "value")
        assert hasattr(Metadata, "recording")

        # Check tablename
        assert Metadata.__tablename__ == "metadata"

    def test_tracklist_model_attributes(self):
        """Test Tracklist model has expected attributes."""
        # Check that the model has the expected mapped columns
        assert hasattr(Tracklist, "id")
        assert hasattr(Tracklist, "recording_id")
        assert hasattr(Tracklist, "source")
        assert hasattr(Tracklist, "cue_file_path")
        assert hasattr(Tracklist, "tracks")
        assert hasattr(Tracklist, "recording")

        # Check tablename
        assert Tracklist.__tablename__ == "tracklists"

    def test_recording_repr_method(self):
        """Test Recording __repr__ method."""
        # Mock the recording with necessary attributes
        recording = Mock(spec=Recording)
        recording.id = uuid.uuid4()
        recording.file_name = "test.mp3"

        # Call the actual __repr__ method
        repr_str = Recording.__repr__(recording)
        assert "Recording" in repr_str
        assert str(recording.id) in repr_str
        assert "test.mp3" in repr_str

    def test_metadata_repr_method(self):
        """Test Metadata __repr__ method."""
        # Mock the metadata with necessary attributes
        metadata = Mock(spec=Metadata)
        metadata.id = uuid.uuid4()
        metadata.key = "bpm"
        metadata.value = "128"

        # Call the actual __repr__ method
        repr_str = Metadata.__repr__(metadata)
        assert "Metadata" in repr_str
        assert str(metadata.id) in repr_str
        assert "bpm" in repr_str
        assert "128" in repr_str

    def test_tracklist_repr_method(self):
        """Test Tracklist __repr__ method."""
        # Mock the tracklist with necessary attributes
        tracklist = Mock(spec=Tracklist)
        tracklist.id = uuid.uuid4()
        tracklist.source = "manual"
        tracklist.tracks = [{"title": "Track 1", "artist": "Artist 1"}, {"title": "Track 2", "artist": "Artist 2"}]

        # Call the actual __repr__ method
        repr_str = Tracklist.__repr__(tracklist)
        assert "Tracklist" in repr_str
        assert str(tracklist.id) in repr_str
        assert "manual" in repr_str
        assert "2" in repr_str  # Number of tracks

    def test_model_inheritance(self):
        """Test that models inherit from Base correctly."""
        assert issubclass(Recording, Base)
        assert issubclass(Metadata, Base)
        assert issubclass(Tracklist, Base)

    def test_model_type_hints(self):
        """Test that models have proper type hints."""
        # Check Recording type annotations
        recording_annotations = Recording.__annotations__
        assert "id" in recording_annotations
        assert "file_path" in recording_annotations
        assert "file_name" in recording_annotations
        assert "sha256_hash" in recording_annotations
        assert "xxh128_hash" in recording_annotations
        assert "created_at" in recording_annotations
        assert "metadata_items" in recording_annotations
        assert "tracklists" in recording_annotations

        # Check Metadata type annotations
        metadata_annotations = Metadata.__annotations__
        assert "id" in metadata_annotations
        assert "recording_id" in metadata_annotations
        assert "key" in metadata_annotations
        assert "value" in metadata_annotations
        assert "recording" in metadata_annotations

        # Check Tracklist type annotations
        tracklist_annotations = Tracklist.__annotations__
        assert "id" in tracklist_annotations
        assert "recording_id" in tracklist_annotations
        assert "source" in tracklist_annotations
        assert "cue_file_path" in tracklist_annotations
        assert "tracks" in tracklist_annotations
        assert "recording" in tracklist_annotations
