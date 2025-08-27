"""CUE file editor module for modifying CUE sheets."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import logging

from .parser import CueParser
from .generator import CueGenerator, CueDisc, CueFile
from .models import CueSheet, Track, CueTime
from .backup import BackupManager

logger = logging.getLogger(__name__)


class CueEditor:
    """Editor for modifying CUE files while preserving format and validity."""

    def __init__(self, backup_manager: Optional[BackupManager] = None):
        """Initialize CUE editor.

        Args:
            backup_manager: Optional backup manager instance
        """
        self.parser = CueParser()
        self.generator = CueGenerator()
        self.backup_manager = backup_manager or BackupManager()
        self.cue_sheet: Optional[CueSheet] = None
        self.original_format: Optional[str] = None
        self.original_path: Optional[Path] = None
        self._dirty = False
        self._original_content: Optional[str] = None  # For preserving formatting
        self._format_style: Dict[str, Any] = {}  # Store formatting preferences

    @property
    def dirty(self) -> bool:
        """Check if CUE sheet has unsaved modifications."""
        return self._dirty

    def load_cue_file(self, filepath: Union[str, Path]) -> CueSheet:
        """Load a CUE file for editing.

        Args:
            filepath: Path to the CUE file

        Returns:
            Loaded CueSheet object
        """
        filepath = Path(filepath)
        self.original_path = filepath

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Preserve original content for format preservation
        self._original_content = content
        self._detect_formatting_style(content)

        self.cue_sheet = self.parser.parse(content)
        self.original_format = self._detect_format(self.cue_sheet)
        self._dirty = False

        logger.info(f"Loaded CUE file: {filepath} (format: {self.original_format})")
        return self.cue_sheet

    def save_cue_file(
        self, filepath: Optional[Union[str, Path]] = None, create_backup: bool = True, format_type: Optional[str] = None
    ) -> Path:
        """Save the CUE sheet to file.

        Args:
            filepath: Output path (uses original if not specified)
            create_backup: Whether to create backup before saving
            format_type: CUE format to use (preserves original if not specified)

        Returns:
            Path to saved file
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        filepath = Path(filepath) if filepath else self.original_path
        if not filepath:
            raise ValueError("No filepath specified")

        # Create backup if needed
        if create_backup and filepath.exists():
            self.backup_manager.create_backup(filepath)

        # Use original format if not specified
        if not format_type:
            format_type = self.original_format or "standard"

        # Convert CueSheet to generator format
        disc, files = self._convert_to_generator_format(self.cue_sheet)

        # Generate CUE content
        content = self.generator.generate(disc, files)

        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        self._dirty = False
        logger.info(f"Saved CUE file: {filepath} (format: {format_type})")
        return filepath

    def preserve_format(self) -> str:
        """Get the preserved original format type.

        Returns:
            Original format type
        """
        return self.original_format or "standard"

    def _detect_format(self, cue_sheet: CueSheet) -> str:
        """Detect the format variant of the CUE sheet.

        Args:
            cue_sheet: CueSheet to analyze

        Returns:
            Detected format type
        """
        # Check REM fields for generator hints
        if cue_sheet.rem_fields:
            for key, value in cue_sheet.rem_fields.items():
                combined = f"{key} {value}".lower()
                if "traktor" in combined:
                    return "traktor"
                elif "serato" in combined:
                    return "serato"
                elif "rekordbox" in combined:
                    return "rekordbox"
                elif "kodi" in combined or "xbmc" in combined:
                    return "kodi"
                elif "cdj" in combined or "pioneer" in combined:
                    return "cdj"

        # Check for CDJ-specific FLAGS
        for file_ref in cue_sheet.files:
            for track in file_ref.tracks:
                if hasattr(track, "flags") and track.flags:
                    return "cdj"

        # Default to standard
        return "standard"

    def _detect_formatting_style(self, content: str) -> None:
        """Detect formatting preferences from original content.

        Args:
            content: Original CUE file content
        """
        lines = content.splitlines()

        # Detect indentation style
        self._format_style["indent"] = ""
        for line in lines:
            if line.startswith("  "):
                # Count spaces for indentation
                indent_count = len(line) - len(line.lstrip())
                self._format_style["indent"] = " " * (indent_count // 2)
                break
            elif line.startswith("\t"):
                self._format_style["indent"] = "\t"
                break

        # Detect quote style
        self._format_style["quotes"] = '"'
        if '"' in content and "'" not in content:
            self._format_style["quotes"] = '"'
        elif "'" in content and '"' not in content:
            self._format_style["quotes"] = "'"

        # Detect line endings (preserve original)
        if "\r\n" in content:
            self._format_style["line_ending"] = "\r\n"
        else:
            self._format_style["line_ending"] = "\n"

        # Detect command order preference
        command_order = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("REM"):
                command = line.split()[0] if line.split() else ""
                if command and command not in command_order:
                    command_order.append(command)
        self._format_style["command_order"] = command_order

        # Detect spacing patterns
        self._format_style["blank_lines_between_tracks"] = False
        for i in range(1, len(lines) - 1):
            if lines[i].strip() == "" and "TRACK" in lines[i - 1] and "TRACK" in lines[i + 1]:
                self._format_style["blank_lines_between_tracks"] = True
                break

    def _mark_dirty(self) -> None:
        """Mark the CUE sheet as modified."""
        self._dirty = True

    def _convert_to_generator_format(self, cue_sheet: CueSheet) -> Tuple[CueDisc, List[CueFile]]:
        """Convert CueSheet to generator format (CueDisc, list[CueFile]).

        Args:
            cue_sheet: CueSheet to convert

        Returns:
            Tuple of (CueDisc, list[CueFile])
        """
        from .generator import CueTrack

        # Create disc metadata
        disc = CueDisc(
            title=cue_sheet.title,
            performer=cue_sheet.performer,
            catalog=cue_sheet.catalog,
            cdtextfile=cue_sheet.cdtextfile,
            rem_fields=cue_sheet.rem_fields,
        )

        # Convert files and tracks
        files = []
        for file_ref in cue_sheet.files:
            tracks = []
            for track in file_ref.tracks:
                # Convert indices to milliseconds
                indices_ms = {}
                for idx_num, idx_time in track.indices.items():
                    indices_ms[idx_num] = idx_time.to_frames() * 1000 // 75

                cue_track = CueTrack(
                    number=track.number,
                    title=track.title or "",
                    performer=track.performer or "",
                    songwriter=track.songwriter,
                    isrc=track.isrc,
                    flags=track.flags if isinstance(track.flags, list) else [],
                    start_time_ms=indices_ms.get(1, 0),
                    indices=indices_ms,
                    rem_fields=track.rem_fields,
                )
                tracks.append(cue_track)

            cue_file = CueFile(filename=file_ref.filename, file_type=file_ref.file_type, tracks=tracks)
            files.append(cue_file)

        return disc, files

    # Track Management Operations (Task 3)

    def add_track(
        self,
        title: str,
        performer: Optional[str] = None,
        start_time: Optional[Union[str, CueTime]] = None,
        track_type: str = "AUDIO",
        file_index: int = -1,
    ) -> Track:
        """Add a new track to the CUE sheet.

        Args:
            title: Track title
            performer: Track performer (optional)
            start_time: Start time as string or CueTime
            track_type: Track type (default: AUDIO)
            file_index: Which file to add to (-1 for last)

        Returns:
            Created Track object
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Ensure we have at least one file
        if not self.cue_sheet.files:
            from .models import FileReference

            file_ref = FileReference(filename="audio.wav", file_type="WAVE")
            self.cue_sheet.files.append(file_ref)

        # Get target file
        if file_index < 0:
            file_ref = self.cue_sheet.files[-1]
        else:
            file_ref = self.cue_sheet.files[file_index]

        # Calculate track number
        all_tracks = []
        for f in self.cue_sheet.files:
            all_tracks.extend(f.tracks)
        next_number = len(all_tracks) + 1

        # Create new track
        track = Track(number=next_number, track_type=track_type)
        track.title = title
        track.performer = performer

        # Set start time
        if start_time:
            if isinstance(start_time, str):
                start_time = CueTime.from_string(start_time)
            track.indices[1] = start_time
        else:
            # Auto-calculate based on last track (add 3 seconds)
            if all_tracks:
                last_track = all_tracks[-1]
                if 1 in last_track.indices:
                    last_time = last_track.indices[1]
                    new_frames = last_time.to_frames() + (3 * 75)  # Add 3 seconds
                    track.indices[1] = CueTime.from_frames(new_frames)
                else:
                    track.indices[1] = CueTime(0, 0, 0)
            else:
                track.indices[1] = CueTime(0, 0, 0)

        # Add to file
        file_ref.tracks.append(track)
        self._mark_dirty()

        logger.info(f"Added track {next_number}: {title}")
        return track

    def remove_track(self, track_number: int) -> bool:
        """Remove a track and renumber remaining tracks.

        Args:
            track_number: Track number to remove (1-based)

        Returns:
            True if track was removed
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        removed = False
        for file_ref in self.cue_sheet.files:
            for i, track in enumerate(file_ref.tracks):
                if track.number == track_number:
                    file_ref.tracks.pop(i)
                    removed = True
                    break
            if removed:
                break

        if removed:
            # Renumber remaining tracks
            track_num = 1
            for file_ref in self.cue_sheet.files:
                for track in file_ref.tracks:
                    track.number = track_num
                    track_num += 1

            self._mark_dirty()
            logger.info(f"Removed track {track_number}")

        return removed

    def reorder_tracks(self, new_order: List[int]) -> None:
        """Reorder tracks according to new order.

        Args:
            new_order: List of track numbers in desired order
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Collect all tracks
        all_tracks = []
        for file_ref in self.cue_sheet.files:
            all_tracks.extend(file_ref.tracks)

        # Validate new order
        if sorted(new_order) != list(range(1, len(all_tracks) + 1)):
            raise ValueError("Invalid track order - must contain all track numbers")

        # Create track map
        track_map = {track.number: track for track in all_tracks}

        # Clear tracks from all files
        for file_ref in self.cue_sheet.files:
            file_ref.tracks = []

        # Add tracks in new order to first file
        # (In practice, might want to preserve file associations)
        file_ref = self.cue_sheet.files[0]
        for i, old_num in enumerate(new_order, 1):
            track = track_map[old_num]
            track.number = i
            file_ref.tracks.append(track)

        self._mark_dirty()
        logger.info(f"Reordered tracks: {new_order}")

    def insert_track(
        self,
        position: int,
        title: str,
        performer: Optional[str] = None,
        start_time: Optional[Union[str, CueTime]] = None,
    ) -> Track:
        """Insert a track at specific position.

        Args:
            position: Position to insert at (1-based)
            title: Track title
            performer: Track performer
            start_time: Start time

        Returns:
            Created Track object
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Add at end first
        track = self.add_track(title, performer, start_time)

        # Get current order
        all_tracks = []
        for file_ref in self.cue_sheet.files:
            all_tracks.extend(file_ref.tracks)

        # Create new order with inserted track
        new_order = []
        for i in range(1, len(all_tracks)):  # Exclude the just-added track
            if i < position:
                new_order.append(i)
            else:
                new_order.append(i)
        # Insert the new track at position
        new_order.insert(position - 1, len(all_tracks))

        # Reorder
        self.reorder_tracks(new_order)

        return track

    def merge_tracks(self, track1_num: int, track2_num: int) -> Track:
        """Merge two adjacent tracks.

        Args:
            track1_num: First track number
            track2_num: Second track number (must be track1_num + 1)

        Returns:
            Merged track
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        if track2_num != track1_num + 1:
            raise ValueError("Can only merge adjacent tracks")

        # Find tracks
        track1 = None
        track2 = None
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track1_num:
                    track1 = track
                elif track.number == track2_num:
                    track2 = track

        if not track1 or not track2:
            raise ValueError("Track not found")

        # Merge metadata (keep track1's data, append track2's title)
        track1.title = f"{track1.title or ''} / {track2.title or ''}"

        # Remove track2
        self.remove_track(track2_num)

        logger.info(f"Merged tracks {track1_num} and {track2_num}")
        return track1

    def split_track(self, track_num: int, split_time: Union[str, CueTime], new_title: Optional[str] = None) -> Track:
        """Split a track at specified time.

        Args:
            track_num: Track number to split
            split_time: Time to split at
            new_title: Title for new track (optional)

        Returns:
            New track created from split
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Find track
        original_track = None
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_num:
                    original_track = track
                    break

        if not original_track:
            raise ValueError(f"Track {track_num} not found")

        # Parse split time
        if isinstance(split_time, str):
            split_time = CueTime.from_string(split_time)

        # Validate split time
        if 1 in original_track.indices:
            start_time = original_track.indices[1]
            if split_time.to_frames() <= start_time.to_frames():
                raise ValueError("Split time must be after track start")

        # Insert new track after original
        if new_title:
            # Update original track title
            original_track.title = f"{original_track.title} (Part 1)"
            new_track_title = f"{new_title} (Part 2)"
        else:
            new_track_title = f"{original_track.title} (Part 2)"

        new_track = self.insert_track(track_num + 1, new_track_title, original_track.performer, split_time)

        self._mark_dirty()
        logger.info(f"Split track {track_num} at {split_time}")
        return new_track

    # Timestamp Adjustment Operations (Task 4)

    def adjust_track_time(self, track_number: int, new_time: Union[str, CueTime], ripple: bool = True) -> None:
        """Adjust track timestamp with optional ripple effect.

        Args:
            track_number: Track number to adjust (1-based)
            new_time: New INDEX 01 time for the track
            ripple: If True, adjust all subsequent tracks by the same delta
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Parse time if string
        if isinstance(new_time, str):
            new_time = CueTime.from_string(new_time)

        # Find track
        target_track = None
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_number:
                    target_track = track
                    break

        if not target_track:
            raise ValueError(f"Track {track_number} not found")

        # Calculate delta
        old_time = target_track.indices.get(1)
        if not old_time:
            # No existing INDEX 01, just set it
            target_track.indices[1] = new_time
            self._mark_dirty()
            return

        delta_frames = new_time.to_frames() - old_time.to_frames()

        # Update target track
        target_track.indices[1] = new_time

        # Update other indices in the same track
        for idx_num, idx_time in list(target_track.indices.items()):
            if idx_num != 1:  # Don't re-adjust INDEX 01
                new_frames = idx_time.to_frames() + delta_frames
                if new_frames >= 0:
                    target_track.indices[idx_num] = CueTime.from_frames(new_frames)

        # Ripple to subsequent tracks if requested
        if ripple and delta_frames != 0:
            for file_ref in self.cue_sheet.files:
                for track in file_ref.tracks:
                    if track.number > track_number:
                        for idx_num, idx_time in list(track.indices.items()):
                            new_frames = idx_time.to_frames() + delta_frames
                            if new_frames >= 0:
                                track.indices[idx_num] = CueTime.from_frames(new_frames)

        self._mark_dirty()
        logger.info(f"Adjusted track {track_number} time to {new_time}, ripple={ripple}")

    def shift_all_times(self, offset_seconds: float) -> None:
        """Shift all track times by a global offset.

        Args:
            offset_seconds: Seconds to add (positive) or subtract (negative)
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        offset_frames = int(offset_seconds * 75)

        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                for idx_num, idx_time in list(track.indices.items()):
                    new_frames = idx_time.to_frames() + offset_frames
                    if new_frames >= 0:
                        track.indices[idx_num] = CueTime.from_frames(new_frames)
                    else:
                        # Can't go negative, set to 0
                        track.indices[idx_num] = CueTime(0, 0, 0)

        self._mark_dirty()
        logger.info(f"Shifted all times by {offset_seconds} seconds")

    def set_track_index(self, track_number: int, index_number: int, time: Union[str, CueTime]) -> None:
        """Set a specific INDEX for a track.

        Args:
            track_number: Track number (1-based)
            index_number: INDEX number (0-99)
            time: Time for the INDEX
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Parse time if string
        if isinstance(time, str):
            time = CueTime.from_string(time)

        # Find track
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_number:
                    track.indices[index_number] = time
                    self._mark_dirty()
                    logger.info(f"Set track {track_number} INDEX {index_number:02d} to {time}")
                    return

        raise ValueError(f"Track {track_number} not found")

    def add_pregap(self, track_number: int, pregap_duration: Union[str, CueTime]) -> None:
        """Add pregap (INDEX 00) to a track.

        Args:
            track_number: Track number
            pregap_duration: Duration of pregap (time before INDEX 01)
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Parse duration if string
        if isinstance(pregap_duration, str):
            pregap_duration = CueTime.from_string(pregap_duration)

        # Find track
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_number:
                    # Get INDEX 01
                    index_01 = track.indices.get(1)
                    if not index_01:
                        raise ValueError(f"Track {track_number} has no INDEX 01")

                    # Calculate INDEX 00 position
                    pregap_frames = index_01.to_frames() - pregap_duration.to_frames()
                    if pregap_frames < 0:
                        raise ValueError("Pregap would result in negative time")

                    track.indices[0] = CueTime.from_frames(pregap_frames)
                    self._mark_dirty()
                    logger.info(f"Added pregap to track {track_number}")
                    return

        raise ValueError(f"Track {track_number} not found")

    def remove_pregap(self, track_number: int) -> bool:
        """Remove pregap (INDEX 00) from a track.

        Args:
            track_number: Track number

        Returns:
            True if pregap was removed
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Find track
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_number:
                    if 0 in track.indices:
                        del track.indices[0]
                        self._mark_dirty()
                        logger.info(f"Removed pregap from track {track_number}")
                        return True
                    return False

        raise ValueError(f"Track {track_number} not found")

    # Metadata Update Operations (Task 5)

    def update_disc_metadata(self, **kwargs: Any) -> None:
        """Update disc-level metadata fields.

        Args:
            **kwargs: Metadata fields to update (title, performer, catalog, cdtextfile)
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        for field, value in kwargs.items():
            if field in ["title", "performer", "catalog", "cdtextfile"]:
                # Apply character limits
                if field in ["title", "performer"] and value and len(value) > 80:
                    value = value[:80]
                    logger.warning(f"Truncated {field} to 80 characters")

                # Validate catalog
                if field == "catalog" and value and len(value) != 13:
                    logger.warning(f"CATALOG should be 13 digits, got {len(value)}")

                setattr(self.cue_sheet, field, value)
                self._mark_dirty()
                logger.info(f"Updated disc {field}: {value}")
            else:
                logger.warning(f"Unknown disc metadata field: {field}")

    def update_track_metadata(self, track_number: int, **kwargs: Any) -> None:
        """Update track-level metadata fields.

        Args:
            track_number: Track number to update
            **kwargs: Metadata fields to update (title, performer, songwriter, isrc)
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Find track
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.number == track_number:
                    for field, value in kwargs.items():
                        if field in ["title", "performer", "songwriter", "isrc"]:
                            # Apply character limits
                            if field in ["title", "performer"] and value and len(value) > 80:
                                value = value[:80]
                                logger.warning(f"Truncated {field} to 80 characters")

                            # Validate ISRC
                            if field == "isrc" and value and len(value) != 12:
                                logger.warning(f"ISRC should be 12 characters, got {len(value)}")

                            setattr(track, field, value)
                            self._mark_dirty()
                            logger.info(f"Updated track {track_number} {field}: {value}")
                        else:
                            logger.warning(f"Unknown track metadata field: {field}")
                    return

        raise ValueError(f"Track {track_number} not found")

    def batch_update_metadata(self, updates: List[Dict[str, Any]]) -> None:
        """Batch update metadata for multiple tracks.

        Args:
            updates: List of update dictionaries with 'track' and metadata fields
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        for update in updates:
            track_num = update.get("track")
            if track_num is None:
                # Disc-level update
                disc_update = {k: v for k, v in update.items() if k != "track"}
                self.update_disc_metadata(**disc_update)
            else:
                # Track-level update
                track_update = {k: v for k, v in update.items() if k != "track"}
                self.update_track_metadata(track_num, **track_update)

    def update_rem_field(self, key: str, value: str, track_number: Optional[int] = None) -> None:
        """Update or add a REM field.

        Args:
            key: REM field key
            value: REM field value
            track_number: Track number for track-level REM, None for disc-level
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        if track_number is None:
            # Disc-level REM
            self.cue_sheet.rem_fields[key] = value
            self._mark_dirty()
            logger.info(f"Updated disc REM {key}: {value}")
        else:
            # Track-level REM
            for file_ref in self.cue_sheet.files:
                for track in file_ref.tracks:
                    if track.number == track_number:
                        track.rem_fields[key] = value
                        self._mark_dirty()
                        logger.info(f"Updated track {track_number} REM {key}: {value}")
                        return
            raise ValueError(f"Track {track_number} not found")

    # Advanced Editing Features (Task 8)

    def find_track_by_title(self, title: str, partial: bool = True) -> Optional[Track]:
        """Find a track by title.

        Args:
            title: Title to search for
            partial: Whether to allow partial matches

        Returns:
            First matching track or None
        """
        if not self.cue_sheet:
            return None

        title_lower = title.lower()
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                if track.title:
                    if partial and title_lower in track.title.lower():
                        return track
                    elif not partial and track.title.lower() == title_lower:
                        return track
        return None

    def find_track_by_time(self, time: str) -> Optional[Track]:
        """Find track at or containing the specified time.

        Args:
            time: Time in MM:SS:FF or MM:SS format

        Returns:
            Track containing the time or None
        """
        if not self.cue_sheet:
            return None

        target_time = CueTime.from_string(time)
        target_frames = target_time.to_frames()

        for file_ref in self.cue_sheet.files:
            for i, track in enumerate(file_ref.tracks):
                if 1 in track.indices:
                    start_frames = track.indices[1].to_frames()

                    # Check if this is the last track
                    if i == len(file_ref.tracks) - 1:
                        if start_frames <= target_frames:
                            return track
                    else:
                        # Get next track's start time
                        next_track = file_ref.tracks[i + 1]
                        if 1 in next_track.indices:
                            end_frames = next_track.indices[1].to_frames()
                            if start_frames <= target_frames < end_frames:
                                return track
        return None

    def auto_fix_gaps(self, min_gap_seconds: float = 2.0) -> int:
        """Remove silence gaps between tracks automatically.

        Args:
            min_gap_seconds: Minimum gap size to preserve

        Returns:
            Number of gaps fixed
        """
        if not self.cue_sheet:
            return 0

        fixes = 0
        min_gap_frames = int(min_gap_seconds * 75)  # Convert to frames

        for file_ref in self.cue_sheet.files:
            for i in range(len(file_ref.tracks) - 1):
                track = file_ref.tracks[i]
                next_track = file_ref.tracks[i + 1]

                if 1 in track.indices and 1 in next_track.indices:
                    # Calculate gap between tracks
                    track_end = track.indices[1].to_frames()
                    next_start = next_track.indices[1].to_frames()
                    gap = next_start - track_end

                    # If gap is larger than minimum, shift next track
                    if gap > min_gap_frames:
                        new_start_frames = track_end + min_gap_frames
                        next_track.indices[1] = CueTime.from_frames(new_start_frames)
                        fixes += 1
                        self._mark_dirty()

        return fixes

    def normalize_track_numbers(self) -> bool:
        """Ensure tracks are numbered sequentially starting from 1.

        Returns:
            True if any changes were made
        """
        if not self.cue_sheet:
            return False

        changed = False
        for file_ref in self.cue_sheet.files:
            for i, track in enumerate(file_ref.tracks, 1):
                if track.number != i:
                    track.number = i
                    changed = True
                    self._mark_dirty()

        return changed

    def validate_and_fix(self) -> Dict[str, List[str]]:
        """Validate CUE sheet and fix common issues.

        Returns:
            Dictionary of issues found and fixed
        """
        issues: Dict[str, List[str]] = {"fixed": [], "warnings": [], "errors": []}

        if not self.cue_sheet:
            issues["errors"].append("No CUE sheet loaded")
            return issues

        # Fix track numbering
        if self.normalize_track_numbers():
            issues["fixed"].append("Normalized track numbers")

        # Check for missing required fields
        if not self.cue_sheet.files:
            issues["errors"].append("No FILE entries found")

        for file_ref in self.cue_sheet.files:
            if not file_ref.filename:
                issues["errors"].append("FILE entry missing filename")

            # Check tracks
            if not file_ref.tracks:
                issues["warnings"].append(f"No tracks in file {file_ref.filename}")

            for track in file_ref.tracks:
                # Check for missing title
                if not track.title:
                    issues["warnings"].append(f"Track {track.number} missing title")

                # Check for missing INDEX 01
                if 1 not in track.indices:
                    issues["errors"].append(f"Track {track.number} missing INDEX 01")

                # Check for overlapping timestamps
                track_index = file_ref.tracks.index(track)
                if track_index < len(file_ref.tracks) - 1:
                    next_track = file_ref.tracks[track_index + 1]
                    if 1 in track.indices and 1 in next_track.indices:
                        if track.indices[1].to_frames() >= next_track.indices[1].to_frames():
                            issues["errors"].append(f"Track {track.number} overlaps with track {next_track.number}")

        return issues

    # Multi-File CUE Editing (Task 9)

    def update_file_reference(self, file_index: int, new_filename: str, new_filetype: str = "WAVE") -> None:
        """Update file reference for a FILE entry.

        Args:
            file_index: Index of the file to update (0-based)
            new_filename: New filename for the FILE entry
            new_filetype: New file type (default: WAVE)
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        if file_index >= len(self.cue_sheet.files):
            raise IndexError(f"File index {file_index} out of range")

        self.cue_sheet.files[file_index].filename = new_filename
        self.cue_sheet.files[file_index].file_type = new_filetype
        self._mark_dirty()
        logger.info(f"Updated file reference {file_index} to {new_filename}")

    def validate_file_transitions(self) -> List[str]:
        """Validate that transitions between FILE entries are valid.

        Returns:
            List of validation errors, empty if valid
        """
        if not self.cue_sheet:
            return ["No CUE sheet loaded"]

        errors = []

        # Check for overlapping tracks between files
        for i in range(len(self.cue_sheet.files) - 1):
            current_file = self.cue_sheet.files[i]
            next_file = self.cue_sheet.files[i + 1]

            if current_file.tracks and next_file.tracks:
                # Check that track numbers don't overlap
                last_track_num = current_file.tracks[-1].number
                first_next_track_num = next_file.tracks[0].number

                if first_next_track_num <= last_track_num:
                    errors.append(
                        f"Track numbering overlap between file {i} and {i + 1}: "
                        f"track {last_track_num} and {first_next_track_num}"
                    )

        return errors

    def consolidate_to_single_file(self, target_filename: Optional[str] = None) -> None:
        """Consolidate multi-file CUE to single file.

        Args:
            target_filename: Target filename for consolidated tracks
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        if len(self.cue_sheet.files) <= 1:
            logger.info("CUE sheet already has single file or no files")
            return

        # Collect all tracks from all files
        all_tracks = []
        for file_ref in self.cue_sheet.files:
            all_tracks.extend(file_ref.tracks)

        # Use first file's name if not specified
        if not target_filename:
            target_filename = self.cue_sheet.files[0].filename

        # Create new single file reference
        from .models import FileReference

        new_file = FileReference(filename=target_filename, file_type=self.cue_sheet.files[0].file_type)
        new_file.tracks = all_tracks

        # Renumber tracks sequentially
        for i, track in enumerate(new_file.tracks, 1):
            track.number = i

        # Replace files list
        self.cue_sheet.files = [new_file]
        self._mark_dirty()
        logger.info(f"Consolidated {len(all_tracks)} tracks to single file: {target_filename}")

    def split_by_file_references(self, split_points: List[int]) -> None:
        """Split tracks into multiple FILE entries.

        Args:
            split_points: List of track numbers where new FILE entries should start
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        if not self.cue_sheet.files:
            raise ValueError("No FILE entries in CUE sheet")

        # Get all tracks
        all_tracks = []
        for file_ref in self.cue_sheet.files:
            all_tracks.extend(file_ref.tracks)

        if not all_tracks:
            raise ValueError("No tracks to split")

        # Sort split points
        split_points = sorted(set(split_points))

        # Create new file references
        from .models import FileReference

        new_files: List[FileReference] = []
        current_start = 0

        for split_point in split_points:
            if split_point <= current_start or split_point > len(all_tracks):
                continue

            # Create file for tracks before split point
            file_tracks = all_tracks[current_start : split_point - 1]
            if file_tracks:
                new_file = FileReference(filename=f"file{len(new_files) + 1}.wav", file_type="WAVE")
                new_file.tracks = file_tracks
                new_files.append(new_file)

            current_start = split_point - 1

        # Add remaining tracks
        if current_start < len(all_tracks):
            new_file = FileReference(filename=f"file{len(new_files) + 1}.wav", file_type="WAVE")
            new_file.tracks = all_tracks[current_start:]
            new_files.append(new_file)

        # Update files
        self.cue_sheet.files = new_files
        self._mark_dirty()
        logger.info(f"Split into {len(new_files)} FILE entries")

    # Integration Methods (Task 10)

    def edit_from_tracklist(self, tracklist_data: Dict[str, Any]) -> None:
        """Update CUE sheet from Tracklist model data.

        Args:
            tracklist_data: Dictionary containing tracklist information
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        tracks = tracklist_data.get("tracks", [])
        if not tracks:
            logger.warning("No tracks in tracklist data")
            return

        # Clear existing tracks
        for file_ref in self.cue_sheet.files:
            file_ref.tracks = []

        # Add tracks from tracklist
        for track_data in tracks:
            title = track_data.get("title", "Unknown Title")
            artist = track_data.get("artist", "Unknown Artist")
            start_time = track_data.get("start_time", "00:00:00")

            self.add_track(title, artist, start_time)

        self._mark_dirty()
        logger.info(f"Updated CUE sheet with {len(tracks)} tracks from tracklist")

    def sync_to_tracklist(self) -> Dict[str, Any]:
        """Convert CUE sheet to Tracklist-compatible format.

        Returns:
            Dictionary compatible with Tracklist model
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        tracks = []
        for file_ref in self.cue_sheet.files:
            for track in file_ref.tracks:
                track_data = {
                    "title": track.title or "Unknown Title",
                    "artist": track.performer or track.songwriter or "Unknown Artist",
                    "start_time": str(track.indices.get(1, CueTime(0, 0, 0))),
                    "track_number": track.number,
                    "isrc": track.isrc,
                }

                # Add INDEX 00 as pregap if present
                if 0 in track.indices:
                    track_data["pregap"] = str(track.indices[0])

                tracks.append(track_data)

        return {
            "tracks": tracks,
            "source": "cue_editor",
            "cue_file_path": str(self.original_path) if self.original_path else None,
        }

    def batch_edit(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform batch editing operations.

        Args:
            operations: List of operation dictionaries

        Returns:
            Summary of operations performed
        """
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        results: Dict[str, Any] = {"successful": 0, "failed": 0, "errors": []}

        for op in operations:
            try:
                op_type = op.get("type")

                if op_type == "add_track":
                    title = op.get("title", "Unknown")
                    performer = op.get("performer")
                    start_time = op.get("start_time")
                    self.add_track(title, performer, start_time)
                elif op_type == "remove_track":
                    track_number = op.get("track_number")
                    if track_number is not None:
                        self.remove_track(track_number)
                elif op_type == "update_metadata":
                    track_number = op.get("track_number")
                    if track_number:
                        self.update_track_metadata(track_number, **op.get("metadata", {}))
                    else:
                        self.update_disc_metadata(**op.get("metadata", {}))
                elif op_type == "adjust_time":
                    track_number = op.get("track_number")
                    new_time = op.get("new_time")
                    ripple = op.get("ripple", True)
                    if track_number is not None and new_time is not None:
                        self.adjust_track_time(track_number, new_time, ripple)
                else:
                    raise ValueError(f"Unknown operation type: {op_type}")

                results["successful"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Operation {op_type} failed: {str(e)}")
                logger.error(f"Batch operation failed: {e}")

        return results

    def validate_before_save(self) -> Tuple[bool, List[str]]:
        """Validate CUE sheet before saving.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.cue_sheet:
            return False, ["No CUE sheet loaded"]

        errors = []

        # Check required disc metadata
        if not self.cue_sheet.files:
            errors.append("No FILE entries in CUE sheet")

        # Validate each file
        for i, file_ref in enumerate(self.cue_sheet.files):
            if not file_ref.filename:
                errors.append(f"FILE {i} missing filename")

            if not file_ref.tracks:
                errors.append(f"FILE {i} has no tracks")

            # Validate tracks
            for track in file_ref.tracks:
                if 1 not in track.indices:
                    errors.append(f"Track {track.number} missing INDEX 01")

                # Check for basic metadata
                if not track.title and not self.cue_sheet.title:
                    errors.append(f"Track {track.number} missing title")

        # Validate file transitions
        transition_errors = self.validate_file_transitions()
        errors.extend(transition_errors)

        # Use parser to validate generated content
        try:
            disc, files = self._convert_to_generator_format(self.cue_sheet)
            content = self.generator.generate(disc, files)
            # Try parsing the generated content
            self.parser.parse(content)
        except Exception as e:
            errors.append(f"Generated CUE validation failed: {str(e)}")

        return len(errors) == 0, errors

    def begin_transaction(self) -> None:
        """Begin a transaction for atomic editing."""
        if not self.cue_sheet:
            raise ValueError("No CUE sheet loaded")

        # Store current state for rollback
        import copy

        self._transaction_backup = copy.deepcopy(self.cue_sheet)
        self._transaction_dirty = self._dirty
        logger.debug("Transaction started")

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if hasattr(self, "_transaction_backup"):
            delattr(self, "_transaction_backup")
            delattr(self, "_transaction_dirty")
            logger.debug("Transaction committed")

    def rollback_transaction(self) -> None:
        """Rollback to state before transaction."""
        if not hasattr(self, "_transaction_backup"):
            raise ValueError("No transaction in progress")

        self.cue_sheet = self._transaction_backup
        self._dirty = self._transaction_dirty
        delattr(self, "_transaction_backup")
        delattr(self, "_transaction_dirty")
        logger.debug("Transaction rolled back")
