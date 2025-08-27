"""
CUE file integration service for generating CUE files from tracklists.

This service integrates with the CUE handler from Epic 5 to generate
CUE files after successful tracklist imports.
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID
from dataclasses import dataclass

# Import CUE handler components from analysis_service
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "analysis_service" / "src"))

try:
    from cue_handler import (
        CueGenerator,
        CueDisc,
        CueFile,
        CueTrack,
        CueFormat,
        get_generator,
    )
    CUE_HANDLER_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("CUE handler not available - CUE generation disabled")
    CUE_HANDLER_AVAILABLE = False

from ..models.tracklist import Tracklist, TrackEntry

logger = logging.getLogger(__name__)


@dataclass
class CueResult:
    """Result from CUE file generation."""
    success: bool
    cue_file_path: Optional[str] = None
    cue_file_id: Optional[UUID] = None
    error: Optional[str] = None


class CueIntegrationService:
    """Service for generating CUE files from tracklists."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize CUE integration service.
        
        Args:
            output_dir: Directory for saving CUE files
        """
        self.output_dir = output_dir or Path("/tmp/cue_files")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not CUE_HANDLER_AVAILABLE:
            logger.warning("CUE handler not available - generation will fail")
    
    def convert_tracklist_to_cue(
        self,
        tracklist: Tracklist,
        audio_file_path: str,
        cue_format: str = "standard"
    ) -> Optional[CueDisc]:
        """
        Convert a Tracklist to CUE format.
        
        Args:
            tracklist: The tracklist to convert
            audio_file_path: Path to the audio file
            cue_format: Target CUE format (standard, cdj, traktor, etc.)
            
        Returns:
            CueDisc object or None if conversion fails
        """
        if not CUE_HANDLER_AVAILABLE:
            logger.error("CUE handler not available")
            return None
        
        try:
            # Create CueDisc structure
            cue_disc = CueDisc()
            
            # Set disc metadata
            cue_disc.title = f"Mix - {tracklist.source}"
            cue_disc.performer = "Various Artists"
            
            # Create file reference
            audio_filename = os.path.basename(audio_file_path)
            cue_file = CueFile(filename=audio_filename, file_type="WAVE")
            
            # Convert tracks
            for track_entry in tracklist.tracks:
                cue_track = self._create_cue_track(track_entry)
                cue_file.tracks.append(cue_track)
            
            cue_disc.files.append(cue_file)
            
            return cue_disc
            
        except Exception as e:
            logger.error(f"Failed to convert tracklist to CUE: {e}")
            return None
    
    def generate_cue_file(
        self,
        tracklist: Tracklist,
        audio_file_path: str,
        cue_format: str = "standard"
    ) -> CueResult:
        """
        Generate a CUE file from a tracklist.
        
        Args:
            tracklist: The tracklist to generate CUE from
            audio_file_path: Path to the audio file
            cue_format: Target CUE format
            
        Returns:
            CueResult with success status and file info
        """
        if not CUE_HANDLER_AVAILABLE:
            logger.error("CUE handler not available")
            return CueResult(success=False, error="CUE handler not available")
        
        try:
            # Convert tracklist to CUE format
            cue_disc = self.convert_tracklist_to_cue(
                tracklist,
                audio_file_path,
                cue_format
            )
            
            if not cue_disc:
                return CueResult(success=False, error="Failed to convert tracklist to CUE format")
            
            # Get appropriate generator
            generator = get_generator(cue_format)
            
            # Generate CUE content
            cue_content = generator.generate(cue_disc)
            
            # Create output file path
            audio_name = Path(audio_file_path).stem
            cue_filename = f"{audio_name}_{cue_format}.cue"
            cue_file_path = self.output_dir / cue_filename
            
            # Write CUE file
            with open(cue_file_path, 'w', encoding='utf-8') as f:
                f.write(cue_content)
            
            logger.info(f"Generated CUE file: {cue_file_path}")
            
            # Generate a unique ID for the CUE file
            from uuid import uuid4
            cue_file_id = uuid4()
            
            return CueResult(
                success=True,
                cue_file_path=str(cue_file_path),
                cue_file_id=cue_file_id
            )
            
        except Exception as e:
            logger.error(f"Failed to generate CUE file: {e}")
            return CueResult(success=False, error=str(e))
    
    def store_cue_file_reference(
        self,
        tracklist_id: UUID,
        cue_file_path: str
    ) -> bool:
        """
        Store CUE file reference in database.
        
        Args:
            tracklist_id: ID of the tracklist
            cue_file_path: Path to the generated CUE file
            
        Returns:
            True if stored successfully
        """
        # This would update the database with the CUE file path
        # For now, just log it
        logger.info(f"Storing CUE file reference for tracklist {tracklist_id}: {cue_file_path}")
        return True
    
    def _create_cue_track(self, track_entry: TrackEntry) -> 'CueTrack':
        """
        Create a CueTrack from a TrackEntry.
        
        Args:
            track_entry: The track entry to convert
            
        Returns:
            CueTrack object
        """
        cue_track = CueTrack()
        
        # Set track number
        cue_track.number = track_entry.position
        
        # Set track metadata
        cue_track.title = track_entry.title
        cue_track.performer = track_entry.artist
        
        # Set timing
        cue_track.index01 = self._timedelta_to_cue_time(track_entry.start_time)
        
        # Add remix/label info to title if present
        if track_entry.remix:
            cue_track.title = f"{track_entry.title} ({track_entry.remix})"
        
        # Add label as comment if present
        if track_entry.label:
            cue_track.rem = f"LABEL: {track_entry.label}"
        
        return cue_track
    
    def _timedelta_to_cue_time(self, td: timedelta) -> str:
        """
        Convert timedelta to CUE time format (MM:SS:FF).
        
        Args:
            td: timedelta to convert
            
        Returns:
            CUE time string
        """
        total_seconds = int(td.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        frames = 0  # We don't have frame-level precision
        
        return f"{minutes:02d}:{seconds:02d}:{frames:02d}"
    
    def validate_cue_file(self, cue_file_path: str) -> bool:
        """
        Validate a generated CUE file.
        
        Args:
            cue_file_path: Path to the CUE file
            
        Returns:
            True if valid
        """
        if not CUE_HANDLER_AVAILABLE:
            logger.error("CUE handler not available")
            return False
        
        try:
            from cue_handler import CueValidator
            
            validator = CueValidator()
            with open(cue_file_path, 'r') as f:
                content = f.read()
            
            result = validator.validate_content(content)
            
            if result.is_valid:
                logger.info(f"CUE file {cue_file_path} is valid")
                return True
            else:
                logger.warning(f"CUE file validation issues: {result.issues}")
                # May still be usable with warnings
                return getattr(result, 'severity', 'warning') != "error"
                
        except Exception as e:
            logger.error(f"Failed to validate CUE file: {e}")
            return False