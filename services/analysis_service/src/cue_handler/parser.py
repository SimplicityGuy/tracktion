"""CUE sheet parser for reading and interpreting CUE files.

Example usage:
    >>> from cue_handler.parser import CueParser
    >>> parser = CueParser()

    # Parse from file
    >>> cue_sheet = parser.parse_file('mix.cue')
    >>> print(f"Album: {cue_sheet.title}")
    >>> print(f"Tracks: {cue_sheet.get_track_count()}")

    # Parse from string
    >>> cue_content = '''
    ... FILE "audio.mp3" MP3
    ...   TRACK 01 AUDIO
    ...     TITLE "First Track"
    ...     INDEX 01 00:00:00
    ... '''
    >>> cue_sheet = parser.parse(cue_content)

    # Access track information
    >>> for track in cue_sheet.get_all_tracks():
    ...     print(f"{track.number}. {track.title} @ {track.get_start_time()}")
"""

import logging
import re
from pathlib import Path

import chardet

from .exceptions import CueParsingError, CueValidationError, InvalidCommandError, InvalidTimeFormatError
from .models import CueSheet, CueTime, FileReference, Track


class CueParser:
    """Parser for CUE sheet files with support for multiple formats and encodings."""

    # Maximum CUE file size (10MB) - CUE files are text and should never be this large
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the CUE parser.

        Args:
            logger: Optional logger instance for debug output
        """
        self.logger = logger or logging.getLogger(__name__)
        self.reset()

    def reset(self) -> None:
        """Reset parser state for new file."""
        self.cue_sheet = CueSheet()
        self.current_file: FileReference | None = None
        self.current_track: Track | None = None
        self.line_number = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def parse_file(self, file_path: str, encoding: str | None = None) -> CueSheet:
        """Parse a CUE file from disk.

        Args:
            file_path: Path to the CUE file
            encoding: Optional encoding override, auto-detected if None

        Returns:
            Parsed CueSheet object

        Raises:
            CueParsingError: If file cannot be parsed or exceeds size limit
        """
        path = Path(file_path)
        if not path.exists():
            raise CueParsingError(f"CUE file not found: {file_path}")

        # Security: Check file size to prevent DoS attacks
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            self.logger.warning(f"CUE file rejected - too large: {file_size} bytes")
            raise CueParsingError(f"CUE file too large: {file_size} bytes (max {self.MAX_FILE_SIZE} bytes)")

        self.logger.debug(f"Parsing CUE file: {file_path} ({file_size} bytes)")

        try:
            # Auto-detect encoding if not specified
            if encoding is None:
                with Path(path).open("rb") as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected["encoding"] or "utf-8"

            with Path(path).open(encoding=encoding) as f:
                content = f.read()

            return self.parse(content)

        except UnicodeDecodeError as e:
            raise CueParsingError(f"Encoding error: {e}") from e
        except Exception as e:
            raise CueParsingError(f"Failed to read file: {e}") from e

    def parse(self, content: str) -> CueSheet:
        """Parse CUE sheet content.

        Args:
            content: CUE sheet content as string

        Returns:
            Parsed CueSheet object

        Raises:
            CueParsingError: If content cannot be parsed
        """
        self.reset()
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            self.line_number = line_num
            self._parse_line(line)

        # Finalize any pending track
        if self.current_track and self.current_file:
            self.current_file.tracks.append(self.current_track)

        # Finalize any pending file
        if self.current_file:
            self.cue_sheet.files.append(self.current_file)

        # Validate the parsed structure
        self._validate()

        # Add any errors/warnings to the cue_sheet
        self.cue_sheet.parsing_errors = self.errors
        self.cue_sheet.parsing_warnings = self.warnings

        return self.cue_sheet

    def _parse_line(self, line: str) -> None:
        """Parse a single line of CUE content."""
        # Strip whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            return

        # Handle alternative comment syntax
        if line.startswith((";", "//")):
            # Convert to REM format for processing
            line = "REM " + line[2:].strip() if line.startswith("//") else "REM " + line[1:].strip()

        # Parse the command
        parts = self._tokenize(line)
        if not parts:
            return

        command = parts[0].upper()
        args = parts[1:] if len(parts) > 1 else []

        # Dispatch to appropriate handler
        handler = getattr(self, f"_handle_{command.lower()}", None)
        if handler:
            try:
                handler(args)
            except Exception as e:
                self.errors.append(f"Line {self.line_number}: Error processing {command}: {e}")
        else:
            # Unknown command, store as custom field
            self.warnings.append(f"Line {self.line_number}: Unknown command: {command}")

    def _tokenize(self, line: str) -> list[str]:
        """Tokenize a CUE line into command and arguments."""
        tokens = []
        current_token = ""
        in_quotes = False
        quote_char = None
        escape_next = False

        i = 0
        while i < len(line):
            char = line[i]

            if escape_next:
                current_token += char
                escape_next = False
            elif not in_quotes:
                if char in "\"'":
                    in_quotes = True
                    quote_char = char
                elif char in " \t":
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                else:
                    current_token += char
            elif char == "\\" and i + 1 < len(line) and line[i + 1] == quote_char:
                # Escaped quote - add the quote to token
                current_token += quote_char
                i += 1  # Skip the next quote character
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            else:
                current_token += char

            i += 1

        # Add final token
        if current_token:
            tokens.append(current_token)

        return tokens

    def _handle_file(self, args: list[str]) -> None:
        """Handle FILE command."""
        if len(args) < 2:
            raise InvalidCommandError("FILE requires filename and type")

        # Save current file/track if any
        if self.current_track:
            if self.current_file:
                self.current_file.tracks.append(self.current_track)
            self.current_track = None

        if self.current_file:
            self.cue_sheet.files.append(self.current_file)

        # Create new file reference
        filename = args[0]
        file_type = args[1].upper()

        # Handle FLAC files using WAVE type
        if filename.lower().endswith(".flac") and file_type == "WAVE":
            self.warnings.append(f"Line {self.line_number}: FLAC file using WAVE type (common practice)")

        self.current_file = FileReference(filename=filename, file_type=file_type)

    def _handle_track(self, args: list[str]) -> None:
        """Handle TRACK command."""
        if len(args) < 2:
            raise InvalidCommandError("TRACK requires number and type")

        # Save current track if any
        if self.current_track and self.current_file:
            self.current_file.tracks.append(self.current_track)

        # Parse track number
        track_num = int(args[0])
        if track_num < 1 or track_num > 99:
            self.errors.append(f"Line {self.line_number}: Invalid track number {track_num} (must be 01-99)")

        track_type = args[1].upper()
        self.current_track = Track(number=track_num, track_type=track_type)

    def _handle_index(self, args: list[str]) -> None:
        """Handle INDEX command."""
        if len(args) < 2:
            raise InvalidCommandError("INDEX requires number and time")

        if not self.current_track:
            self.errors.append(f"Line {self.line_number}: INDEX without TRACK")
            return

        index_num = int(args[0])
        time_str = args[1]

        try:
            cue_time = CueTime.from_string(time_str)
            self.current_track.indices[index_num] = cue_time
        except InvalidTimeFormatError as e:
            self.errors.append(f"Line {self.line_number}: Invalid time format: {e}")

    def _handle_title(self, args: list[str]) -> None:
        """Handle TITLE command."""
        if not args:
            return

        title = " ".join(args)

        # Check character limit
        if len(title) > 80:
            self.warnings.append(f"Line {self.line_number}: TITLE exceeds 80 character limit")

        if self.current_track:
            self.current_track.title = title
        else:
            self.cue_sheet.title = title

    def _handle_performer(self, args: list[str]) -> None:
        """Handle PERFORMER command."""
        if not args:
            return

        performer = " ".join(args)

        # Check character limit
        if len(performer) > 80:
            self.warnings.append(f"Line {self.line_number}: PERFORMER exceeds 80 character limit")

        if self.current_track:
            self.current_track.performer = performer
        else:
            self.cue_sheet.performer = performer

    def _handle_songwriter(self, args: list[str]) -> None:
        """Handle SONGWRITER command."""
        if not args:
            return

        if self.current_track:
            self.current_track.songwriter = " ".join(args)

    def _handle_catalog(self, args: list[str]) -> None:
        """Handle CATALOG command."""
        if not args:
            return

        catalog = args[0]
        # Validate UPC/EAN (13 digits)
        if not re.match(r"^\d{13}$", catalog):
            self.warnings.append(f"Line {self.line_number}: CATALOG should be 13 digits")

        self.cue_sheet.catalog = catalog

    def _handle_cdtextfile(self, args: list[str]) -> None:
        """Handle CDTEXTFILE command."""
        if args:
            self.cue_sheet.cdtextfile = args[0]

    def _handle_flags(self, args: list[str]) -> None:
        """Handle FLAGS command."""
        if not self.current_track:
            self.errors.append(f"Line {self.line_number}: FLAGS without TRACK")
            return

        valid_flags = {"DCP", "4CH", "PRE", "SCMS"}
        for flag in args:
            flag_upper = flag.upper()
            if flag_upper not in valid_flags:
                self.warnings.append(f"Line {self.line_number}: Unknown flag: {flag}")
            # Always add the flag (even if unknown)
            self.current_track.flags.append(flag_upper)

    def _handle_isrc(self, args: list[str]) -> None:
        """Handle ISRC command."""
        if not self.current_track:
            self.errors.append(f"Line {self.line_number}: ISRC without TRACK")
            return

        if not args:
            return

        isrc = args[0]
        # Validate ISRC format (12 characters)
        if len(isrc) != 12:
            self.warnings.append(f"Line {self.line_number}: ISRC should be 12 characters")

        self.current_track.isrc = isrc

    def _handle_pregap(self, args: list[str]) -> None:
        """Handle PREGAP command."""
        if not self.current_track:
            self.errors.append(f"Line {self.line_number}: PREGAP without TRACK")
            return

        if args:
            try:
                self.current_track.pregap = CueTime.from_string(args[0])
            except InvalidTimeFormatError as e:
                self.errors.append(f"Line {self.line_number}: Invalid pregap time: {e}")

    def _handle_postgap(self, args: list[str]) -> None:
        """Handle POSTGAP command."""
        if not self.current_track:
            self.errors.append(f"Line {self.line_number}: POSTGAP without TRACK")
            return

        if args:
            try:
                self.current_track.postgap = CueTime.from_string(args[0])
            except InvalidTimeFormatError as e:
                self.errors.append(f"Line {self.line_number}: Invalid postgap time: {e}")

    def _handle_rem(self, args: list[str]) -> None:
        """Handle REM command."""
        if not args:
            return

        # Parse REM field
        rem_key = args[0].upper()
        rem_value = " ".join(args[1:]) if len(args) > 1 else ""

        # Handle known REM fields
        known_rem_fields = {
            "GENRE",
            "DATE",
            "DISCID",
            "COMMENT",
            "DISCNUMBER",
            "COMPOSER",
            "REPLAYGAIN_ALBUM_GAIN",
            "REPLAYGAIN_ALBUM_PEAK",
            "REPLAYGAIN_TRACK_GAIN",
            "REPLAYGAIN_TRACK_PEAK",
        }

        if rem_key in known_rem_fields:
            # Handle specific REM fields
            if rem_key == "DISCNUMBER":
                try:
                    disc_num = int(rem_value)
                    if disc_num < 1 or disc_num > 15:
                        self.warnings.append(f"Line {self.line_number}: DISCNUMBER should be 1-15")
                except ValueError:
                    self.errors.append(f"Line {self.line_number}: Invalid DISCNUMBER value")

            elif rem_key in ["REPLAYGAIN_ALBUM_GAIN", "REPLAYGAIN_TRACK_GAIN"]:
                # Validate gain format (e.g., "-7.89 dB")
                if not re.match(r"^[+-]?\d+(\.\d+)?\s*dB?$", rem_value, re.IGNORECASE):
                    self.warnings.append(f"Line {self.line_number}: Invalid {rem_key} format")

            elif rem_key in ["REPLAYGAIN_ALBUM_PEAK", "REPLAYGAIN_TRACK_PEAK"]:
                # Validate peak format (0.0-1.0)
                try:
                    peak = float(rem_value)
                    if peak < 0.0 or peak > 1.0:
                        self.warnings.append(f"Line {self.line_number}: {rem_key} should be 0.0-1.0")
                except ValueError:
                    self.errors.append(f"Line {self.line_number}: Invalid {rem_key} value")

        # Store REM field
        if self.current_track:
            self.current_track.rem_fields[rem_key] = rem_value
        else:
            self.cue_sheet.rem_fields[rem_key] = rem_value

    def _validate(self) -> None:
        """Validate the parsed CUE sheet structure."""
        # Check for required elements
        if not self.cue_sheet.files:
            self.errors.append("No FILE commands found")

        # Validate track numbering and indices
        for file_ref in self.cue_sheet.files:
            prev_track_num = 0

            for track in file_ref.tracks:
                # Check sequential track numbering
                if track.number != prev_track_num + 1:
                    self.warnings.append(f"Non-sequential track number: {track.number}")
                prev_track_num = track.number

                # Check for required INDEX 01
                if 1 not in track.indices:
                    self.errors.append(f"Track {track.number}: Missing required INDEX 01")

                # Check INDEX ordering
                if (
                    0 in track.indices
                    and 1 in track.indices
                    and track.indices[0].to_frames() >= track.indices[1].to_frames()
                ):
                    self.errors.append(f"Track {track.number}: INDEX 00 must be before INDEX 01")

        # Report validation results
        if self.errors:
            error_msg = f"Validation failed with {len(self.errors)} errors: "
            error_msg += "; ".join(self.errors)
            self.logger.error(f"CUE validation failed: {error_msg}")
            raise CueValidationError(error_msg)

        # Log successful parse summary
        self.logger.info(
            f"Successfully parsed CUE sheet: {self.cue_sheet.get_track_count()} tracks, {len(self.warnings)} warnings"
        )
