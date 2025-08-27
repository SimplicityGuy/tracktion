"""Unit tests for CUE file editor module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.analysis_service.src.cue_handler.backup import BackupManager
from services.analysis_service.src.cue_handler.editor import CueEditor
from services.analysis_service.src.cue_handler.models import CueSheet, CueTime, Track


class TestCueEditor:
    """Test CueEditor class."""

    @pytest.fixture
    def editor(self):
        """Create CueEditor instance."""
        return CueEditor()

    @pytest.fixture
    def sample_cue_sheet(self):
        """Create sample CueSheet for testing."""
        from services.analysis_service.src.cue_handler.models import FileReference

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

    def test_init_with_backup_manager(self):
        """Test CueEditor initialization with custom backup manager."""
        backup_manager = BackupManager(retention_limit=10)
        editor = CueEditor(backup_manager=backup_manager)
        assert editor.backup_manager is backup_manager
        assert editor.backup_manager.retention_limit == 10

    @patch("services.analysis_service.src.cue_handler.editor.open")
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
            mock_open.assert_called_once_with(Path("/test/file.cue"), "r", encoding="utf-8")

    def test_save_cue_file_no_sheet_loaded(self, editor):
        """Test saving when no CUE sheet is loaded."""
        with pytest.raises(ValueError, match="No CUE sheet loaded"):
            editor.save_cue_file("/test/output.cue")

    @patch("services.analysis_service.src.cue_handler.editor.open")
    def test_save_cue_file(self, mock_open, editor, sample_cue_sheet):
        """Test saving a CUE file."""
        editor.cue_sheet = sample_cue_sheet
        editor.original_path = Path("/test/original.cue")
        editor.original_format = "standard"
        editor._dirty = True

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(editor.generator, "generate") as mock_generate:
            mock_generate.return_value = "Generated CUE content"
            with patch.object(editor.backup_manager, "create_backup") as mock_backup:
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "mkdir"):
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
        from unittest.mock import MagicMock, mock_open, patch

        mock_file = mock_open(read_data=content)

        with patch("builtins.open", mock_file):
            with patch.object(editor.parser, "parse") as mock_parse:
                mock_parse.return_value = MagicMock()
                editor.load_cue_file("/test/file.cue")

        # Check format detection (2 spaces detected, but divided by 2 = 1 space)
        assert editor._format_style["indent"] == " "
        assert editor._format_style["quotes"] == '"'

    def test_backup_disabled(self):
        """Test disabling automatic backups."""
        import tempfile
        from pathlib import Path

        from services.analysis_service.src.cue_handler.backup import BackupManager

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
