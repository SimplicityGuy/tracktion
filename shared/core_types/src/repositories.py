"""Repository pattern implementations for database operations."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import Recording, Metadata, Tracklist
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class RecordingRepository:
    """Repository for Recording entity operations."""
    
    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def create(self, file_path: str, file_name: str, 
               sha256_hash: Optional[str] = None,
               xxh128_hash: Optional[str] = None) -> Recording:
        """Create a new recording.
        
        Args:
            file_path: Full path to the file
            file_name: Name of the file
            sha256_hash: Optional SHA256 hash of file
            xxh128_hash: Optional XXH128 hash of file
            
        Returns:
            Created Recording instance
            
        Raises:
            IntegrityError: If recording with same hash already exists
        """
        with self.db.get_db_session() as session:
            recording = Recording(
                file_path=file_path,
                file_name=file_name,
                sha256_hash=sha256_hash,
                xxh128_hash=xxh128_hash
            )
            session.add(recording)
            session.flush()
            session.refresh(recording)
            return recording
    
    def get_by_id(self, recording_id: UUID) -> Optional[Recording]:
        """Get recording by ID.
        
        Args:
            recording_id: UUID of the recording
            
        Returns:
            Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            return session.query(Recording).filter(
                Recording.id == recording_id
            ).first()
    
    def get_by_file_path(self, file_path: str) -> Optional[Recording]:
        """Get recording by file path.
        
        Args:
            file_path: Full path to the file
            
        Returns:
            Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            return session.query(Recording).filter(
                Recording.file_path == file_path
            ).first()
    
    def get_all(self, limit: Optional[int] = None, 
                offset: Optional[int] = None) -> List[Recording]:
        """Get all recordings with optional pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of Recording instances
        """
        with self.db.get_db_session() as session:
            query = session.query(Recording)
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def update(self, recording_id: UUID, **kwargs: Any) -> Optional[Recording]:
        """Update a recording.
        
        Args:
            recording_id: UUID of the recording
            **kwargs: Fields to update
            
        Returns:
            Updated Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            recording = session.query(Recording).filter(
                Recording.id == recording_id
            ).first()
            
            if not recording:
                return None
            
            for key, value in kwargs.items():
                if hasattr(recording, key):
                    setattr(recording, key, value)
            
            session.flush()
            session.refresh(recording)
            return recording
    
    def delete(self, recording_id: UUID) -> bool:
        """Delete a recording.
        
        Args:
            recording_id: UUID of the recording
            
        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            recording = session.query(Recording).filter(
                Recording.id == recording_id
            ).first()
            
            if not recording:
                return False
            
            session.delete(recording)
            return True
    
    def bulk_create(self, recordings: List[Dict[str, Any]]) -> List[Recording]:
        """Create multiple recordings in a single transaction.
        
        Args:
            recordings: List of recording data dictionaries
            
        Returns:
            List of created Recording instances
        """
        with self.db.get_db_session() as session:
            recording_objects = [
                Recording(**data) for data in recordings
            ]
            session.bulk_save_objects(recording_objects, return_defaults=True)
            return recording_objects


class MetadataRepository:
    """Repository for Metadata entity operations."""
    
    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def create(self, recording_id: UUID, key: str, value: str) -> Metadata:
        """Create metadata for a recording.
        
        Args:
            recording_id: UUID of the recording
            key: Metadata key
            value: Metadata value
            
        Returns:
            Created Metadata instance
        """
        with self.db.get_db_session() as session:
            metadata = Metadata(
                recording_id=recording_id,
                key=key,
                value=value
            )
            session.add(metadata)
            session.flush()
            session.refresh(metadata)
            return metadata
    
    def get_by_recording(self, recording_id: UUID) -> List[Metadata]:
        """Get all metadata for a recording.
        
        Args:
            recording_id: UUID of the recording
            
        Returns:
            List of Metadata instances
        """
        with self.db.get_db_session() as session:
            return session.query(Metadata).filter(
                Metadata.recording_id == recording_id
            ).all()
    
    def get_by_key(self, recording_id: UUID, key: str) -> Optional[Metadata]:
        """Get specific metadata by key for a recording.
        
        Args:
            recording_id: UUID of the recording
            key: Metadata key
            
        Returns:
            Metadata instance or None if not found
        """
        with self.db.get_db_session() as session:
            return session.query(Metadata).filter(
                Metadata.recording_id == recording_id,
                Metadata.key == key
            ).first()
    
    def update(self, metadata_id: UUID, value: str) -> Optional[Metadata]:
        """Update metadata value.
        
        Args:
            metadata_id: UUID of the metadata
            value: New value
            
        Returns:
            Updated Metadata instance or None if not found
        """
        with self.db.get_db_session() as session:
            metadata = session.query(Metadata).filter(
                Metadata.id == metadata_id
            ).first()
            
            if not metadata:
                return None
            
            metadata.value = value
            session.flush()
            session.refresh(metadata)
            return metadata
    
    def delete(self, metadata_id: UUID) -> bool:
        """Delete metadata.
        
        Args:
            metadata_id: UUID of the metadata
            
        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            metadata = session.query(Metadata).filter(
                Metadata.id == metadata_id
            ).first()
            
            if not metadata:
                return False
            
            session.delete(metadata)
            return True
    
    def bulk_create(self, recording_id: UUID, 
                    metadata_items: List[Dict[str, str]]) -> List[Metadata]:
        """Create multiple metadata entries for a recording.
        
        Args:
            recording_id: UUID of the recording
            metadata_items: List of dictionaries with 'key' and 'value'
            
        Returns:
            List of created Metadata instances
        """
        with self.db.get_db_session() as session:
            metadata_objects = [
                Metadata(recording_id=recording_id, key=item['key'], value=item['value'])
                for item in metadata_items
            ]
            session.bulk_save_objects(metadata_objects, return_defaults=True)
            return metadata_objects


class TracklistRepository:
    """Repository for Tracklist entity operations."""
    
    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def create(self, recording_id: UUID, source: str,
               tracks: Optional[List[Dict[str, Any]]] = None,
               cue_file_path: Optional[str] = None) -> Tracklist:
        """Create a tracklist for a recording.
        
        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist
            tracks: List of track dictionaries
            cue_file_path: Path to CUE file
            
        Returns:
            Created Tracklist instance
            
        Raises:
            IntegrityError: If tracklist already exists for recording
        """
        with self.db.get_db_session() as session:
            tracklist = Tracklist(
                recording_id=recording_id,
                source=source,
                tracks=tracks,
                cue_file_path=cue_file_path
            )
            
            # Validate tracks structure
            if not tracklist.validate_tracks():
                raise ValueError("Invalid tracks structure")
            
            session.add(tracklist)
            session.flush()
            session.refresh(tracklist)
            return tracklist
    
    def get_by_recording(self, recording_id: UUID) -> Optional[Tracklist]:
        """Get tracklist for a recording.
        
        Args:
            recording_id: UUID of the recording
            
        Returns:
            Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            return session.query(Tracklist).filter(
                Tracklist.recording_id == recording_id
            ).first()
    
    def get_by_id(self, tracklist_id: UUID) -> Optional[Tracklist]:
        """Get tracklist by ID.
        
        Args:
            tracklist_id: UUID of the tracklist
            
        Returns:
            Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            return session.query(Tracklist).filter(
                Tracklist.id == tracklist_id
            ).first()
    
    def update(self, tracklist_id: UUID, **kwargs: Any) -> Optional[Tracklist]:
        """Update a tracklist.
        
        Args:
            tracklist_id: UUID of the tracklist
            **kwargs: Fields to update
            
        Returns:
            Updated Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            tracklist = session.query(Tracklist).filter(
                Tracklist.id == tracklist_id
            ).first()
            
            if not tracklist:
                return None
            
            for key, value in kwargs.items():
                if hasattr(tracklist, key):
                    setattr(tracklist, key, value)
            
            # Validate if tracks were updated
            if 'tracks' in kwargs and not tracklist.validate_tracks():
                raise ValueError("Invalid tracks structure")
            
            session.flush()
            session.refresh(tracklist)
            return tracklist
    
    def delete(self, tracklist_id: UUID) -> bool:
        """Delete a tracklist.
        
        Args:
            tracklist_id: UUID of the tracklist
            
        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            tracklist = session.query(Tracklist).filter(
                Tracklist.id == tracklist_id
            ).first()
            
            if not tracklist:
                return False
            
            session.delete(tracklist)
            return True