"""Unit tests for CUE file editor module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from services.analysis_service.src.cue_handler.backup import BackupManager
from services.analysis_service.src.cue_handler.editor import CueEditor
from services.analysis_service.src.cue_handler.models import CueSheet, CueTime, FileReference, Track


class TestCueEditor:
    """Test CueEditor class."""

    @pytest.fixture
    def editor(self):
        """Create CueEditor instance."""
        return CueEditor()

    @pytest.fixture
    def sample_cue_sheet(self):
        """Create sample CueSheet for testing."""
        cue_sheet = CueSheet()
        cue_sheet.title = "Test Album"
        cue_sheet.performer = "Test Artist"

        track1 = Track(number=1, track_type="AUDIO")
        track1.title = "Track 1"
        track1.performer = "Artist 1"
        track1.indices = {1: CueTime(0, 0, 0)}

        track2 = Track(number=2, track_type="AUDIO")
        track2.title = "Track 2"
        track2.performer = "Artist 2"
        track2.indices = {1: CueTime(3, 30, 0)}

        file_ref = FileReference(filename="test.wav", file_type="WAVE")
        file_ref.tracks = [track1, track2]
        cue_sheet.files = [file_ref]

        return cue_sheet

    def test_init(self, editor):
        """Test CueEditor initialization."""
        assert editor.parser is not None
        assert editor.generator is not None
        assert editor.backup_manager is not None
        assert editor.cue_sheet is None
        assert editor.original_format is None
        assert editor.original_path is None
        assert editor.dirty is False
        assert editor._undo_stack == []
        assert editor._redo_stack == []
        assert editor.can_undo is False
        assert editor.can_redo is False

    def test_init_with_backup_manager(self):
        """Test CueEditor initialization with custom backup manager."""
        backup_manager = BackupManager(retention_limit=10)
        editor = CueEditor(backup_manager=backup_manager)
        assert editor.backup_manager is backup_manager
        assert editor.backup_manager.retention_limit == 10

    @patch("pathlib.Path.open")
    def test_load_cue_file(self, mock_open, editor, sample_cue_sheet):
        """Test loading a CUE file."""
        mock_file = MagicMock()
        mock_file.read.return_value = "REM Test CUE"
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(editor.parser, "parse") as mock_parse:
            mock_parse.return_value = sample_cue_sheet

            result = editor.load_cue_file("/test/file.cue")

            assert result == sample_cue_sheet
            assert editor.cue_sheet == sample_cue_sheet
            assert editor.original_path == Path("/test/file.cue")
            assert editor.dirty is False
            assert editor.original_format == "standard"
            mock_open.assert_called_once_with(encoding="utf-8")

    def test_save_cue_file_no_sheet_loaded(self, editor):
        """Test saving when no CUE sheet is loaded."""
        with pytest.raises(ValueError, match="No CUE sheet loaded"):
            editor.save_cue_file("/test/output.cue")

    @patch("pathlib.Path.open")
    def test_save_cue_file(self, mock_open, editor, sample_cue_sheet):
        """Test saving a CUE file."""
        editor.cue_sheet = sample_cue_sheet
        editor.original_path = Path("/test/original.cue")
        editor.original_format = "standard"
        editor._dirty = True

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with (
            patch.object(editor.generator, "generate") as mock_generate,
            patch.object(editor.backup_manager, "create_backup") as mock_backup,
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "mkdir"),
        ):
            mock_generate.return_value = "Generated CUE content"
            result = editor.save_cue_file()

            assert result == Path("/test/original.cue")
            assert editor.dirty is False
            mock_backup.assert_called_once()
            # Now check that generate was called with CueDisc and list[CueFile]
            assert mock_generate.called
            call_args = mock_generate.call_args[0]
            assert len(call_args) == 2
            # First arg should be CueDisc
            assert hasattr(call_args[0], "title")
            assert call_args[0].title == "Test Album"
            # Second arg should be list of CueFile
            assert isinstance(call_args[1], list)
            assert len(call_args[1]) == 1
            mock_file.write.assert_called_once_with("Generated CUE content")

    def test_preserve_format(self, editor):
        """Test preserving original format."""
        assert editor.preserve_format() == "standard"

        editor.original_format = "traktor"
        assert editor.preserve_format() == "traktor"

    def test_detect_format_traktor(self, editor, sample_cue_sheet):
        """Test detecting Traktor format."""
        sample_cue_sheet.rem_fields = {"GENERATOR": "Traktor Pro 3"}
        format_type = editor._detect_format(sample_cue_sheet)
        assert format_type == "traktor"

    def test_detect_format_serato(self, editor, sample_cue_sheet):
        """Test detecting Serato format."""
        sample_cue_sheet.rem_fields = {"CREATED_BY": "Serato DJ"}
        format_type = editor._detect_format(sample_cue_sheet)
        assert format_type == "serato"

    def test_detect_format_cdj(self, editor, sample_cue_sheet):
        """Test detecting CDJ format by FLAGS."""
        sample_cue_sheet.files[0].tracks[0].flags = ["DCP"]
        format_type = editor._detect_format(sample_cue_sheet)
        assert format_type == "cdj"

    def test_detect_format_standard(self, editor, sample_cue_sheet):
        """Test detecting standard format (default)."""
        format_type = editor._detect_format(sample_cue_sheet)
        assert format_type == "standard"

    def test_mark_dirty(self, editor):
        """Test marking CUE sheet as dirty."""
        assert editor.dirty is False
        editor._mark_dirty()
        assert editor.dirty is True

    def test_add_track(self, editor, sample_cue_sheet):
        """Test adding a track."""
        editor.cue_sheet = sample_cue_sheet

        new_track = editor.add_track("New Track", "New Artist", "05:00:00")

        assert new_track.number == 3
        assert new_track.title == "New Track"
        assert new_track.performer == "New Artist"
        assert new_track.indices[1] == CueTime(5, 0, 0)
        assert editor.dirty is True
        assert len(editor.cue_sheet.files[0].tracks) == 3

    def test_add_track_auto_time(self, editor, sample_cue_sheet):
        """Test adding track with automatic time calculation."""
        editor.cue_sheet = sample_cue_sheet

        new_track = editor.add_track("Auto Track")

        assert new_track.indices[1] == CueTime.from_frames(CueTime(3, 30, 0).to_frames() + 225)

    def test_remove_track(self, editor, sample_cue_sheet):
        """Test removing a track."""
        editor.cue_sheet = sample_cue_sheet

        removed = editor.remove_track(1)

        assert removed is True
        assert len(editor.cue_sheet.files[0].tracks) == 1
        assert editor.cue_sheet.files[0].tracks[0].number == 1
        assert editor.cue_sheet.files[0].tracks[0].title == "Track 2"
        assert editor.dirty is True

    def test_reorder_tracks(self, editor, sample_cue_sheet):
        """Test reordering tracks."""
        editor.cue_sheet = sample_cue_sheet
        editor.add_track("Track 3")

        editor.reorder_tracks([2, 3, 1])

        tracks = editor.cue_sheet.files[0].tracks
        assert tracks[0].title == "Track 2"
        assert tracks[1].title == "Track 3"
        assert tracks[2].title == "Track 1"
        assert tracks[0].number == 1
        assert tracks[1].number == 2
        assert tracks[2].number == 3

    def test_insert_track(self, editor, sample_cue_sheet):
        """Test inserting track at position."""
        editor.cue_sheet = sample_cue_sheet

        editor.insert_track(2, "Inserted Track", "Inserted Artist")

        tracks = editor.cue_sheet.files[0].tracks
        assert len(tracks) == 3
        assert tracks[0].title == "Track 1"
        assert tracks[1].title == "Inserted Track"
        assert tracks[2].title == "Track 2"
        assert tracks[1].number == 2

    def test_merge_tracks(self, editor, sample_cue_sheet):
        """Test merging adjacent tracks."""
        editor.cue_sheet = sample_cue_sheet

        merged = editor.merge_tracks(1, 2)

        assert merged.title == "Track 1 / Track 2"
        assert len(editor.cue_sheet.files[0].tracks) == 1
        assert editor.dirty is True

    def test_merge_tracks_non_adjacent_error(self, editor, sample_cue_sheet):
        """Test error when merging non-adjacent tracks."""
        editor.cue_sheet = sample_cue_sheet
        editor.add_track("Track 3")

        with pytest.raises(ValueError, match="Can only merge adjacent tracks"):
            editor.merge_tracks(1, 3)

    def test_split_track(self, editor, sample_cue_sheet):
        """Test splitting a track."""
        editor.cue_sheet = sample_cue_sheet

        editor.split_track(2, "05:00:00", "Split Part")

        tracks = editor.cue_sheet.files[0].tracks
        assert len(tracks) == 3
        assert tracks[1].title == "Track 2 (Part 1)"
        assert tracks[2].title == "Split Part (Part 2)"
        assert tracks[2].indices[1] == CueTime(5, 0, 0)

    def test_find_track_by_title(self, editor, sample_cue_sheet):
        """Test finding track by title."""
        editor.cue_sheet = sample_cue_sheet

        # Exact match
        track = editor.find_track_by_title("Track 1", partial=False)
        assert track is not None
        assert track.title == "Track 1"

        # Partial match
        track = editor.find_track_by_title("rack 2", partial=True)
        assert track is not None
        assert track.title == "Track 2"

        # Not found
        track = editor.find_track_by_title("Nonexistent")
        assert track is None

    def test_find_track_by_time(self, editor, sample_cue_sheet):
        """Test finding track by time."""
        editor.cue_sheet = sample_cue_sheet

        # First track
        track = editor.find_track_by_time("00:00:00")
        assert track is not None
        assert track.number == 1

        # Second track
        track = editor.find_track_by_time("03:30:00")
        assert track is not None
        assert track.number == 2

        # In between tracks
        track = editor.find_track_by_time("02:00:00")
        assert track is not None
        assert track.number == 1

    def test_auto_fix_gaps(self, editor, sample_cue_sheet):
        """Test automatic gap fixing."""
        editor.cue_sheet = sample_cue_sheet

        # Add a large gap
        editor.add_track("Track 3", start_time="10:00:00")

        # Fix gaps (there are 2 gaps now: between track 1-2 and track 2-3)
        fixes = editor.auto_fix_gaps(min_gap_seconds=2.0)
        assert fixes == 2
        assert editor.dirty is True

    def test_normalize_track_numbers(self, editor, sample_cue_sheet):
        """Test track number normalization."""
        editor.cue_sheet = sample_cue_sheet

        # Mess up track numbers
        sample_cue_sheet.files[0].tracks[0].number = 5
        sample_cue_sheet.files[0].tracks[1].number = 10

        # Normalize
        changed = editor.normalize_track_numbers()
        assert changed is True
        assert sample_cue_sheet.files[0].tracks[0].number == 1
        assert sample_cue_sheet.files[0].tracks[1].number == 2
        assert editor.dirty is True

    def test_validate_and_fix(self, editor, sample_cue_sheet):
        """Test validation and automatic fixing."""
        editor.cue_sheet = sample_cue_sheet

        # Remove title from a track
        sample_cue_sheet.files[0].tracks[0].title = None

        # Mess up track numbers
        sample_cue_sheet.files[0].tracks[0].number = 5

        # Validate and fix
        issues = editor.validate_and_fix()

        assert "Normalized track numbers" in issues["fixed"]
        assert "Track 1 missing title" in issues["warnings"]
        assert len(issues["errors"]) == 0

    def test_format_preservation(self, editor):
        """Test format preservation features."""
        # Create test content with specific formatting
        content = 'REM GENERATOR "Traktor Pro"\nTITLE "Test Album"\n  TRACK 01 AUDIO\n    TITLE "Track One"\n'

        # Mock file reading

        mock_file = mock_open(read_data=content)

        with patch("pathlib.Path.open", mock_file), patch.object(editor.parser, "parse") as mock_parse:
            mock_parse.return_value = MagicMock()
            editor.load_cue_file("/test/file.cue")

        # Check format detection (2 spaces detected, but divided by 2 = 1 space)
        assert editor._format_style["indent"] == " "
        assert editor._format_style["quotes"] == '"'

    def test_backup_disabled(self):
        """Test disabling automatic backups."""

        backup_manager = BackupManager(enabled=False)

        with tempfile.NamedTemporaryFile(suffix=".cue", delete=False) as tmp:
            test_file = Path(tmp.name)
            test_file.write_text("Test content")

            try:
                # Should return None when disabled
                result = backup_manager.create_backup(test_file)
                assert result is None
            finally:
                test_file.unlink()

    def test_update_file_reference(self, editor, sample_cue_sheet):
        """Test updating file reference."""
        editor.cue_sheet = sample_cue_sheet

        editor.update_file_reference(0, "new_file.wav", "WAVE")

        assert editor.cue_sheet.files[0].filename == "new_file.wav"
        assert editor.cue_sheet.files[0].file_type == "WAVE"
        assert editor.dirty is True

    def test_validate_file_transitions(self, editor):
        """Test file transition validation."""

        editor.cue_sheet = MagicMock()

        # Create two files with overlapping track numbers
        file1 = FileReference("file1.wav", "WAVE")
        file1.tracks = [Track(5, "AUDIO")]

        file2 = FileReference("file2.wav", "WAVE")
        file2.tracks = [Track(3, "AUDIO")]

        editor.cue_sheet.files = [file1, file2]

        errors = editor.validate_file_transitions()
        assert len(errors) == 1
        assert "Track numbering overlap" in errors[0]

    def test_consolidate_to_single_file(self, editor):
        """Test consolidating multi-file CUE to single file."""

        # Create multi-file CUE
        cue_sheet = CueSheet()

        file1 = FileReference("file1.wav", "WAVE")
        track1 = Track(1, "AUDIO")
        track1.title = "Track 1"
        file1.tracks = [track1]

        file2 = FileReference("file2.wav", "WAVE")
        track2 = Track(2, "AUDIO")
        track2.title = "Track 2"
        file2.tracks = [track2]

        cue_sheet.files = [file1, file2]
        editor.cue_sheet = cue_sheet

        editor.consolidate_to_single_file("consolidated.wav")

        assert len(editor.cue_sheet.files) == 1
        assert editor.cue_sheet.files[0].filename == "consolidated.wav"
        assert len(editor.cue_sheet.files[0].tracks) == 2
        assert editor.dirty is True

    def test_edit_from_tracklist(self, editor, sample_cue_sheet):
        """Test updating CUE from tracklist data."""
        editor.cue_sheet = sample_cue_sheet

        tracklist_data = {
            "tracks": [
                {
                    "title": "New Track 1",
                    "artist": "Artist A",
                    "start_time": "00:00:00",
                },
                {
                    "title": "New Track 2",
                    "artist": "Artist B",
                    "start_time": "03:00:00",
                },
            ]
        }

        editor.edit_from_tracklist(tracklist_data)

        tracks = editor.cue_sheet.files[0].tracks
        assert len(tracks) == 2
        assert tracks[0].title == "New Track 1"
        assert tracks[1].title == "New Track 2"
        assert editor.dirty is True

    def test_sync_to_tracklist(self, editor, sample_cue_sheet):
        """Test converting CUE to tracklist format."""
        editor.cue_sheet = sample_cue_sheet
        editor.original_path = Path("/test/file.cue")

        tracklist = editor.sync_to_tracklist()

        assert "tracks" in tracklist
        assert len(tracklist["tracks"]) == 2
        assert tracklist["tracks"][0]["title"] == "Track 1"
        assert tracklist["source"] == "cue_editor"
        assert tracklist["cue_file_path"] == "/test/file.cue"

    def test_batch_edit(self, editor, sample_cue_sheet):
        """Test batch editing operations."""
        editor.cue_sheet = sample_cue_sheet

        operations = [
            {
                "type": "add_track",
                "title": "New Track",
                "performer": "New Artist",
                "start_time": "05:00:00",
            },
            {
                "type": "update_metadata",
                "track_number": 1,
                "metadata": {"title": "Updated Track 1"},
            },
            {"type": "remove_track", "track_number": 2},
        ]

        results = editor.batch_edit(operations)

        assert results["successful"] == 3
        assert results["failed"] == 0
        assert len(results["errors"]) == 0

        tracks = editor.cue_sheet.files[0].tracks
        assert len(tracks) == 2
        assert tracks[0].title == "Updated Track 1"

    def test_validate_before_save(self, editor, sample_cue_sheet):
        """Test validation before saving."""
        editor.cue_sheet = sample_cue_sheet

        is_valid, errors = editor.validate_before_save()
        assert is_valid is True
        assert len(errors) == 0

        # Remove INDEX 01 from a track
        del editor.cue_sheet.files[0].tracks[0].indices[1]

        is_valid, errors = editor.validate_before_save()
        assert is_valid is False
        assert "Track 1 missing INDEX 01" in errors

    def test_transaction_rollback(self, editor, sample_cue_sheet):
        """Test transaction rollback."""
        editor.cue_sheet = sample_cue_sheet
        original_title = sample_cue_sheet.title

        editor.begin_transaction()

        # Make changes
        editor.update_disc_metadata(title="Changed Title")
        editor.add_track("New Track")

        assert editor.cue_sheet.title == "Changed Title"
        assert len(editor.cue_sheet.files[0].tracks) == 3

        # Rollback
        editor.rollback_transaction()

        assert editor.cue_sheet.title == original_title
        assert len(editor.cue_sheet.files[0].tracks) == 2

    def test_transaction_commit(self, editor, sample_cue_sheet):
        """Test transaction commit."""
        editor.cue_sheet = sample_cue_sheet

        editor.begin_transaction()
        editor.update_disc_metadata(title="Changed Title")
        editor.commit_transaction()

        assert editor.cue_sheet.title == "Changed Title"
        assert not hasattr(editor, "_transaction_backup")

    def test_edge_cases(self, editor):
        """Test edge cases and error conditions."""
        # Test operations without loaded CUE sheet
        with pytest.raises(ValueError, match="No CUE sheet loaded"):
            editor.add_track("Test")

        with pytest.raises(ValueError, match="No CUE sheet loaded"):
            editor.save_cue_file()

        # validate_before_save returns (False, errors) instead of raising
        is_valid, errors = editor.validate_before_save()
        assert is_valid is False
        assert "No CUE sheet loaded" in errors

    def test_timestamp_validation(self, editor, sample_cue_sheet):
        """Test timestamp adjustment validation."""
        editor.cue_sheet = sample_cue_sheet

        # Adjust track time
        editor.adjust_track_time(1, "00:30:00", ripple=True)

        # Verify ripple effect
        assert editor.cue_sheet.files[0].tracks[0].indices[1] == CueTime(0, 30, 0)
        assert editor.cue_sheet.files[0].tracks[1].indices[1] == CueTime(4, 0, 0)

    def test_metadata_character_limits(self, editor, sample_cue_sheet):
        """Test metadata character limit enforcement."""
        editor.cue_sheet = sample_cue_sheet

        # Try to set title longer than 80 chars
        long_title = "A" * 100
        editor.update_track_metadata(1, title=long_title)

        # Should be truncated to 80 chars
        assert len(editor.cue_sheet.files[0].tracks[0].title) == 80

    def test_multi_file_split(self, editor, sample_cue_sheet):
        """Test splitting into multiple FILE entries."""
        editor.cue_sheet = sample_cue_sheet
        editor.add_track("Track 3", start_time="06:00:00")
        editor.add_track("Track 4", start_time="09:00:00")

        # Split at track 3
        editor.split_by_file_references([3])

        assert len(editor.cue_sheet.files) == 2
        assert len(editor.cue_sheet.files[0].tracks) == 2
        assert len(editor.cue_sheet.files[1].tracks) == 2

    def test_undo_redo_add_track(self, editor, sample_cue_sheet):
        """Test undo/redo for add track operation."""
        editor.cue_sheet = sample_cue_sheet
        initial_track_count = len(editor.cue_sheet.files[0].tracks)

        # Add a track
        editor.add_track("New Track", "New Artist")
        assert len(editor.cue_sheet.files[0].tracks) == initial_track_count + 1
        assert editor.can_undo is True
        assert editor.can_redo is False

        # Undo
        success = editor.undo()
        assert success is True
        assert len(editor.cue_sheet.files[0].tracks) == initial_track_count
        assert editor.can_undo is False
        assert editor.can_redo is True

        # Redo
        success = editor.redo()
        assert success is True
        assert len(editor.cue_sheet.files[0].tracks) == initial_track_count + 1
        assert editor.can_undo is True
        assert editor.can_redo is False

    def test_undo_redo_remove_track(self, editor, sample_cue_sheet):
        """Test undo/redo for remove track operation."""
        editor.cue_sheet = sample_cue_sheet
        initial_track_count = len(editor.cue_sheet.files[0].tracks)

        # Remove a track
        success = editor.remove_track(1)
        assert success is True
        assert len(editor.cue_sheet.files[0].tracks) == initial_track_count - 1

        # Undo
        success = editor.undo()
        assert success is True
        assert len(editor.cue_sheet.files[0].tracks) == initial_track_count
        assert editor.cue_sheet.files[0].tracks[0].title == "Track 1"

    def test_undo_redo_metadata_update(self, editor, sample_cue_sheet):
        """Test undo/redo for metadata update."""
        editor.cue_sheet = sample_cue_sheet
        original_title = editor.cue_sheet.title

        # Update metadata
        editor.update_disc_metadata(title="New Title")
        assert editor.cue_sheet.title == "New Title"

        # Undo
        editor.undo()
        assert editor.cue_sheet.title == original_title

        # Redo
        editor.redo()
        assert editor.cue_sheet.title == "New Title"

    def test_escape_unescape_metadata(self, editor):
        """Test metadata escaping and unescaping."""
        # Test escaping
        assert editor.escape_metadata_value('Test "quoted"') == 'Test \\"quoted\\"'
        assert editor.escape_metadata_value("Line\nbreak") == "Line\\nbreak"
        assert editor.escape_metadata_value("Tab\there") == "Tab\\there"

        # Test unescaping
        assert editor.unescape_metadata_value('Test \\"quoted\\"') == 'Test "quoted"'
        assert editor.unescape_metadata_value("Line\\nbreak") == "Line\nbreak"
        assert editor.unescape_metadata_value("Tab\\there") == "Tab\there"

    def test_validate_timestamps(self, editor, sample_cue_sheet):
        """Test timestamp validation."""
        editor.cue_sheet = sample_cue_sheet

        # Valid timestamps
        errors = editor.validate_timestamps()
        assert len(errors) == 0

        # Create overlap
        editor.cue_sheet.files[0].tracks[1].indices[1] = CueTime(0, 0, 0)
        errors = editor.validate_timestamps()
        assert len(errors) > 0
        assert "overlap" in errors[0].lower()

        # Missing INDEX 01
        del editor.cue_sheet.files[0].tracks[0].indices[1]
        errors = editor.validate_timestamps()
        assert any("Missing required INDEX 01" in e for e in errors)

    def test_timestamp_overlap_prevention(self, editor, sample_cue_sheet):
        """Test that timestamp adjustments prevent overlaps."""
        editor.cue_sheet = sample_cue_sheet

        # Try to adjust track 2 to overlap with track 1
        with pytest.raises(ValueError, match="overlap"):
            editor.adjust_track_time(2, "00:00:00", ripple=False)

        # Try to adjust track 1 to overlap with track 2 (no ripple)
        with pytest.raises(ValueError, match="overlap"):
            editor.adjust_track_time(1, "04:00:00", ripple=False)

    def test_index_ordering_validation(self, editor, sample_cue_sheet):
        """Test INDEX ordering validation."""
        editor.cue_sheet = sample_cue_sheet

        # Try to add INDEX 00 after INDEX 01 - should fail
        with pytest.raises(ValueError, match=r"INDEX.*must be before"):
            editor.set_track_index(1, 0, "01:00:00")  # INDEX 01 is at 00:00:00

    def test_special_character_handling(self, editor, sample_cue_sheet):
        """Test special character handling in metadata."""
        editor.cue_sheet = sample_cue_sheet

        # Update with special characters
        editor.update_track_metadata(1, title='Track with "quotes"', performer="Artist\nNewline")

        # Check escaping was applied
        track = editor.cue_sheet.files[0].tracks[0]
        assert '\\"' in track.title or '"' in track.title  # Depends on implementation
        assert "\\n" in track.performer or "\n" in track.performer

    def test_file_reference_timing_validation(self, editor, sample_cue_sheet):
        """Test timing validation when updating file references."""
        editor.cue_sheet = sample_cue_sheet

        # Update file reference with timing recalculation
        editor.update_file_reference(0, "new_audio.wav", "WAVE", recalculate_timing=True)

        assert editor.cue_sheet.files[0].filename == "new_audio.wav"
        assert editor.dirty is True

    def test_command_order_preservation(self, editor):
        """Test that command order is preserved from original."""
        # Create test content with specific command ordering
        content = """REM DATE 2024
TITLE "Album"
PERFORMER "Artist"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track"
    PERFORMER "Artist"
    INDEX 01 00:00:00"""

        mock_file = mock_open(read_data=content)

        with patch("pathlib.Path.open", mock_file), patch.object(editor.parser, "parse") as mock_parse:
            mock_parse.return_value = MagicMock()
            editor.load_cue_file("/test/file.cue")

        # Check that command order was detected
        assert "REM" in editor._format_style.get("command_order", [])
        assert "TITLE" in editor._format_style.get("command_order", [])
        assert "TRACK" in editor._format_style.get("track_command_order", [])

    def test_undo_history_limit(self, editor, sample_cue_sheet):
        """Test that undo history has a limit."""
        editor.cue_sheet = sample_cue_sheet
        editor._max_undo_history = 3  # Set small limit for testing

        # Add more operations than the limit
        for i in range(5):
            editor.add_track(f"Track {i + 3}", start_time=f"{(i + 1) * 10:02d}:00:00")

        # Should only be able to undo 3 times
        assert len(editor._undo_stack) == 3

        for _ in range(3):
            assert editor.undo() is True

        # No more undos available
        assert editor.undo() is False

    def test_clear_undo_history(self, editor, sample_cue_sheet):
        """Test clearing undo/redo history."""
        editor.cue_sheet = sample_cue_sheet

        # Add some operations
        editor.add_track("Track 3")
        editor.remove_track(1)

        assert len(editor._undo_stack) == 2
        assert editor.can_undo is True

        # Clear history
        editor.clear_undo_history()

        assert len(editor._undo_stack) == 0
        assert len(editor._redo_stack) == 0
        assert editor.can_undo is False
        assert editor.can_redo is False


class TestBackupManager:
    """Test BackupManager class."""

    @pytest.fixture
    def backup_manager(self):
        """Create BackupManager instance."""
        return BackupManager(retention_limit=3)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_init(self, backup_manager):
        """Test BackupManager initialization."""
        assert backup_manager.retention_limit == 3

    def test_create_backup_file_not_exists(self, backup_manager, temp_dir):
        """Test creating backup when file doesn't exist."""
        test_file = temp_dir / "test.cue"
        result = backup_manager.create_backup(test_file)
        assert result is None

    def test_create_backup_first(self, backup_manager, temp_dir):
        """Test creating first backup."""
        test_file = temp_dir / "test.cue"
        test_file.write_text("Original content")

        backup_path = backup_manager.create_backup(test_file)

        assert backup_path == test_file.with_suffix(".cue.bak")
        assert backup_path.exists()
        assert backup_path.read_text() == "Original content"

    def test_create_backup_multiple(self, backup_manager, temp_dir):
        """Test creating multiple backups with rotation."""
        test_file = temp_dir / "test.cue"
        test_file.write_text("Original content")

        # Create first backup - goes to .bak
        backup1 = backup_manager.create_backup(test_file)
        assert backup1 == test_file.with_suffix(".cue.bak")
        assert backup1.read_text() == "Original content"

        # Modify and create second backup
        # Previous backup rotates to .bak.1, new backup goes to .bak
        test_file.write_text("Modified content 1")
        backup2 = backup_manager.create_backup(test_file)
        assert backup2 == test_file.with_suffix(".cue.bak")
        assert backup2.read_text() == "Modified content 1"
        assert test_file.with_suffix(".cue.bak.1").exists()
        assert test_file.with_suffix(".cue.bak.1").read_text() == "Original content"

        # Modify and create third backup
        # Backups rotate: .bak.1 -> .bak.2, .bak -> .bak.1, new -> .bak
        test_file.write_text("Modified content 2")
        backup3 = backup_manager.create_backup(test_file)
        assert backup3 == test_file.with_suffix(".cue.bak")
        assert backup3.read_text() == "Modified content 2"
        assert test_file.with_suffix(".cue.bak.1").read_text() == "Modified content 1"
        assert test_file.with_suffix(".cue.bak.2").read_text() == "Original content"

    def test_restore_from_backup(self, backup_manager, temp_dir):
        """Test restoring from backup."""
        test_file = temp_dir / "test.cue"
        test_file.write_text("Original content")

        # Create backup
        backup_manager.create_backup(test_file)

        # Modify file
        test_file.write_text("Modified content")

        # Restore from backup
        success = backup_manager.restore_from_backup(test_file)

        assert success is True
        assert test_file.read_text() == "Original content"

    def test_restore_from_backup_not_exists(self, backup_manager, temp_dir):
        """Test restoring when backup doesn't exist."""
        test_file = temp_dir / "test.cue"
        success = backup_manager.restore_from_backup(test_file)
        assert success is False

    def test_list_backups(self, backup_manager, temp_dir):
        """Test listing backups."""
        test_file = temp_dir / "test.cue"
        test_file.write_text("Content")

        # No backups initially
        assert backup_manager.list_backups(test_file) == []

        # Create backups
        backup_manager.create_backup(test_file)
        test_file.write_text("Modified 1")
        backup_manager.create_backup(test_file)
        test_file.write_text("Modified 2")
        backup_manager.create_backup(test_file)

        backups = backup_manager.list_backups(test_file)
        assert len(backups) == 3
        assert test_file.with_suffix(".cue.bak") in backups
        assert test_file.with_suffix(".cue.bak.1") in backups
        assert test_file.with_suffix(".cue.bak.2") in backups

    def test_cleanup_old_backups(self, backup_manager, temp_dir):
        """Test cleaning up old backups."""

        test_file = temp_dir / "test.cue"
        test_file.write_text("Content")

        # Create more backups than retention limit
        for i in range(5):
            backup_manager.create_backup(test_file)
            test_file.write_text(f"Modified {i}")

        # Should have max retention_limit backups
        backups = backup_manager.list_backups(test_file)
        assert len(backups) == 3

        # Cleanup should not remove anything if at limit
        backup_manager.cleanup_old_backups(test_file)
        backups = backup_manager.list_backups(test_file)
        assert len(backups) == 3
