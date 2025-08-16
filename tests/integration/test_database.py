"""Integration tests for database operations."""

import pytest
from uuid import UUID
from sqlalchemy.exc import IntegrityError

from shared.core_types.src.models import Recording, Metadata, Tracklist
from shared.core_types.src.repositories import (
    RecordingRepository, 
    MetadataRepository, 
    TracklistRepository
)
from shared.core_types.src.neo4j_repository import Neo4jRepository


class TestPostgreSQLOperations:
    """Integration tests for PostgreSQL database operations."""
    
    def test_create_recording(self, db_manager, sample_recording_data):
        """Test creating a recording in PostgreSQL."""
        repo = RecordingRepository(db_manager)
        
        recording = repo.create(**sample_recording_data)
        
        assert recording.id is not None
        assert isinstance(recording.id, UUID)
        assert recording.file_path == sample_recording_data["file_path"]
        assert recording.file_name == sample_recording_data["file_name"]
        assert recording.sha256_hash == sample_recording_data["sha256_hash"]
    
    def test_get_recording_by_id(self, db_manager, sample_recording_data):
        """Test retrieving a recording by ID."""
        repo = RecordingRepository(db_manager)
        
        # Create recording
        created = repo.create(**sample_recording_data)
        
        # Retrieve by ID
        retrieved = repo.get_by_id(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.file_path == created.file_path
    
    def test_get_recording_by_file_path(self, db_manager, sample_recording_data):
        """Test retrieving a recording by file path."""
        repo = RecordingRepository(db_manager)
        
        # Create recording
        created = repo.create(**sample_recording_data)
        
        # Retrieve by file path
        retrieved = repo.get_by_file_path(sample_recording_data["file_path"])
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.file_path == sample_recording_data["file_path"]
    
    def test_update_recording(self, db_manager, sample_recording_data):
        """Test updating a recording."""
        repo = RecordingRepository(db_manager)
        
        # Create recording
        created = repo.create(**sample_recording_data)
        
        # Update recording
        updated = repo.update(
            created.id,
            file_name="updated_name.mp3",
            sha256_hash="newhash123"
        )
        
        assert updated is not None
        assert updated.file_name == "updated_name.mp3"
        assert updated.sha256_hash == "newhash123"
        assert updated.file_path == sample_recording_data["file_path"]  # Unchanged
    
    def test_delete_recording(self, db_manager, sample_recording_data):
        """Test deleting a recording."""
        repo = RecordingRepository(db_manager)
        
        # Create recording
        created = repo.create(**sample_recording_data)
        
        # Delete recording
        result = repo.delete(created.id)
        assert result is True
        
        # Verify deletion
        retrieved = repo.get_by_id(created.id)
        assert retrieved is None
    
    def test_create_metadata(self, db_manager, sample_recording_data, sample_metadata_items):
        """Test creating metadata for a recording."""
        recording_repo = RecordingRepository(db_manager)
        metadata_repo = MetadataRepository(db_manager)
        
        # Create recording
        recording = recording_repo.create(**sample_recording_data)
        
        # Create metadata
        for item in sample_metadata_items:
            metadata = metadata_repo.create(
                recording_id=recording.id,
                key=item["key"],
                value=item["value"]
            )
            assert metadata.id is not None
            assert metadata.key == item["key"]
            assert metadata.value == item["value"]
    
    def test_get_metadata_by_recording(self, db_manager, sample_recording_data, sample_metadata_items):
        """Test retrieving all metadata for a recording."""
        recording_repo = RecordingRepository(db_manager)
        metadata_repo = MetadataRepository(db_manager)
        
        # Create recording and metadata
        recording = recording_repo.create(**sample_recording_data)
        for item in sample_metadata_items:
            metadata_repo.create(recording.id, item["key"], item["value"])
        
        # Retrieve metadata
        metadata_list = metadata_repo.get_by_recording(recording.id)
        
        assert len(metadata_list) == len(sample_metadata_items)
        
        # Verify all metadata items are present
        keys = [m.key for m in metadata_list]
        for item in sample_metadata_items:
            assert item["key"] in keys
    
    def test_create_tracklist(self, db_manager, sample_recording_data, sample_tracks):
        """Test creating a tracklist with tracks."""
        recording_repo = RecordingRepository(db_manager)
        tracklist_repo = TracklistRepository(db_manager)
        
        # Create recording
        recording = recording_repo.create(**sample_recording_data)
        
        # Create tracklist
        tracklist = tracklist_repo.create(
            recording_id=recording.id,
            source="1001tracklists.com",
            tracks=sample_tracks,
            cue_file_path="/music/test_set.cue"
        )
        
        assert tracklist.id is not None
        assert tracklist.recording_id == recording.id
        assert tracklist.source == "1001tracklists.com"
        assert len(tracklist.tracks) == len(sample_tracks)
        assert tracklist.cue_file_path == "/music/test_set.cue"
    
    def test_tracklist_uniqueness(self, db_manager, sample_recording_data, sample_tracks):
        """Test that only one tracklist can exist per recording."""
        recording_repo = RecordingRepository(db_manager)
        tracklist_repo = TracklistRepository(db_manager)
        
        # Create recording
        recording = recording_repo.create(**sample_recording_data)
        
        # Create first tracklist
        tracklist1 = tracklist_repo.create(
            recording_id=recording.id,
            source="source1",
            tracks=sample_tracks
        )
        assert tracklist1 is not None
        
        # Attempt to create second tracklist should fail
        with pytest.raises(IntegrityError):
            tracklist_repo.create(
                recording_id=recording.id,
                source="source2",
                tracks=sample_tracks
            )
    
    def test_cascade_delete(self, db_manager, sample_recording_data, 
                           sample_metadata_items, sample_tracks):
        """Test that deleting a recording cascades to metadata and tracklist."""
        recording_repo = RecordingRepository(db_manager)
        metadata_repo = MetadataRepository(db_manager)
        tracklist_repo = TracklistRepository(db_manager)
        
        # Create recording with metadata and tracklist
        recording = recording_repo.create(**sample_recording_data)
        
        for item in sample_metadata_items:
            metadata_repo.create(recording.id, item["key"], item["value"])
        
        tracklist_repo.create(recording.id, "test", sample_tracks)
        
        # Delete recording
        recording_repo.delete(recording.id)
        
        # Verify cascade deletion
        metadata_list = metadata_repo.get_by_recording(recording.id)
        assert len(metadata_list) == 0
        
        tracklist = tracklist_repo.get_by_recording(recording.id)
        assert tracklist is None
    
    def test_bulk_operations(self, db_manager):
        """Test bulk create operations."""
        repo = RecordingRepository(db_manager)
        
        # Prepare bulk data
        recordings_data = [
            {"file_path": f"/music/track{i}.mp3", "file_name": f"track{i}.mp3"}
            for i in range(5)
        ]
        
        # Bulk create
        created = repo.bulk_create(recordings_data)
        
        assert len(created) == 5
        for recording in created:
            assert recording.id is not None
            assert recording.file_path.startswith("/music/track")


class TestNeo4jOperations:
    """Integration tests for Neo4j graph database operations."""
    
    @pytest.fixture
    def neo4j_repo(self):
        """Create Neo4j repository instance."""
        import os
        repo = Neo4jRepository(
            uri=os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "changeme")
        )
        
        # Clear database and create constraints
        repo.clear_database()
        repo.create_constraints()
        
        yield repo
        
        repo.close()
    
    def test_create_recording_node(self, neo4j_repo):
        """Test creating a recording node in Neo4j."""
        from uuid import uuid4
        
        recording_id = uuid4()
        result = neo4j_repo.create_recording_node(
            recording_id=recording_id,
            file_name="test.mp3",
            file_path="/music/test.mp3"
        )
        
        assert result is not None
        assert result["uuid"] == str(recording_id)
        assert result["file_name"] == "test.mp3"
    
    def test_add_metadata_relationships(self, neo4j_repo, sample_metadata_items):
        """Test creating metadata relationships in Neo4j."""
        from uuid import uuid4
        
        recording_id = uuid4()
        
        # Create recording node
        neo4j_repo.create_recording_node(recording_id, "test.mp3", "/music/test.mp3")
        
        # Add metadata relationships
        for item in sample_metadata_items:
            neo4j_repo.add_metadata_relationship(
                recording_id, item["key"], item["value"]
            )
        
        # Retrieve metadata
        metadata = neo4j_repo.get_recording_metadata(recording_id)
        
        assert len(metadata) == len(sample_metadata_items)
        
        # Verify all metadata is present
        keys = [m["key"] for m in metadata]
        for item in sample_metadata_items:
            assert item["key"] in keys
    
    def test_add_tracklist_with_tracks(self, neo4j_repo, sample_tracks):
        """Test creating tracklist with tracks in Neo4j."""
        from uuid import uuid4
        
        recording_id = uuid4()
        
        # Create recording node
        neo4j_repo.create_recording_node(recording_id, "set.mp3", "/music/set.mp3")
        
        # Add tracklist with tracks
        neo4j_repo.add_tracklist_with_tracks(
            recording_id=recording_id,
            source="manual",
            tracks=sample_tracks
        )
        
        # Retrieve tracks
        tracks = neo4j_repo.get_tracklist_tracks(recording_id)
        
        assert len(tracks) == len(sample_tracks)
        
        # Verify track order
        for i, track in enumerate(tracks):
            assert track["title"] == sample_tracks[i]["title"]
            assert track["artist"] == sample_tracks[i]["artist"]
            assert track["start_time"] == sample_tracks[i]["start_time"]
    
    def test_delete_recording_node(self, neo4j_repo, sample_metadata_items, sample_tracks):
        """Test deleting a recording node and its relationships."""
        from uuid import uuid4
        
        recording_id = uuid4()
        
        # Create recording with metadata and tracklist
        neo4j_repo.create_recording_node(recording_id, "test.mp3", "/music/test.mp3")
        
        for item in sample_metadata_items:
            neo4j_repo.add_metadata_relationship(
                recording_id, item["key"], item["value"]
            )
        
        neo4j_repo.add_tracklist_with_tracks(recording_id, "test", sample_tracks)
        
        # Delete recording
        result = neo4j_repo.delete_recording_node(recording_id)
        assert result is True
        
        # Verify deletion
        node = neo4j_repo.get_recording_node(recording_id)
        assert node is None
        
        # Verify relationships are also deleted
        metadata = neo4j_repo.get_recording_metadata(recording_id)
        assert len(metadata) == 0
        
        tracks = neo4j_repo.get_tracklist_tracks(recording_id)
        assert len(tracks) == 0