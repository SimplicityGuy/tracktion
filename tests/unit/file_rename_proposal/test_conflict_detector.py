"""Unit tests for conflict detector."""

from unittest.mock import patch

import pytest

from services.analysis_service.src.file_rename_proposal.conflict_detector import ConflictDetector


class TestConflictDetector:
    """Test conflict detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create a conflict detector instance."""
        return ConflictDetector()

    def test_detect_exact_path_collision(self, detector):
        """Test detection of exact path collisions."""
        proposed = "/music/artist/song.mp3"
        existing = {"/music/artist/song.mp3", "/music/artist/other.mp3"}

        result = detector.detect_conflicts(proposed, existing)

        assert len(result["conflicts"]) == 1
        assert "already exists" in result["conflicts"][0]
        assert len(result["warnings"]) == 0

    def test_detect_case_insensitive_collision(self, detector):
        """Test detection of case-insensitive collisions."""
        proposed = "/music/Artist/Song.mp3"
        existing = {"/music/artist/song.mp3", "/music/artist/other.mp3"}

        result = detector.detect_conflicts(proposed, existing)

        assert len(result["conflicts"]) == 1
        assert "Case-insensitive collision" in result["conflicts"][0]

    def test_detect_proposal_collision(self, detector):
        """Test detection of collisions with other proposals."""
        proposed = "/music/artist/song.mp3"
        existing = set()
        other_proposals = [
            {"full_proposed_path": "/music/artist/song.mp3", "recording_id": "123"},
            {"full_proposed_path": "/music/artist/other.mp3", "recording_id": "456"},
        ]

        result = detector.detect_conflicts(proposed, existing, other_proposals)

        assert len(result["conflicts"]) == 1
        assert "Conflicts with another proposal" in result["conflicts"][0]
        assert "123" in result["conflicts"][0]

    def test_detect_directory_traversal(self, detector):
        """Test detection of directory traversal attempts."""
        proposed = "/music/../../../etc/passwd"
        existing = set()

        result = detector.detect_conflicts(proposed, existing)

        assert len(result["conflicts"]) == 1
        assert "directory traversal" in result["conflicts"][0]

    def test_detect_hidden_file_warning(self, detector):
        """Test detection of hidden file creation."""
        proposed = "/music/artist/.hidden_song.mp3"
        existing = set()

        result = detector.detect_conflicts(proposed, existing)

        assert len(result["warnings"]) == 1
        assert "hidden file" in result["warnings"][0]
        assert len(result["conflicts"]) == 0

    def test_detect_backup_pattern_warning(self, detector):
        """Test detection of backup file patterns."""
        test_cases = [
            "/music/song.mp3.bak",
            "/music/song.mp3.backup",
            "/music/backup_song.mp3",
            "/music/song.mp3.1",
            "/music/song~",
        ]

        for proposed in test_cases:
            result = detector.detect_conflicts(proposed, set())
            assert len(result["warnings"]) == 1
            assert "backup file pattern" in result["warnings"][0]

    def test_detect_temp_pattern_warning(self, detector):
        """Test detection of temporary file patterns."""
        test_cases = ["/music/song.mp3.tmp", "/music/temp_song.mp3", "/music/.~lock.song.mp3", "/music/~$song.mp3"]

        for proposed in test_cases:
            result = detector.detect_conflicts(proposed, set())
            # Some files might match both backup and temp patterns
            assert len(result["warnings"]) >= 1
            # Check that at least one warning mentions temporary file pattern
            assert any("temporary file pattern" in w for w in result["warnings"])

    def test_detect_path_length_conflict(self, detector):
        """Test detection of path length violations."""
        # Create a path longer than 255 characters
        long_name = "a" * 250
        proposed = f"/music/{long_name}.mp3"
        existing = set()

        result = detector.detect_conflicts(proposed, existing)

        assert len(result["conflicts"]) == 1
        assert "exceeds maximum length" in result["conflicts"][0]

    def test_detect_system_file_conflict(self, detector):
        """Test detection of system file conflicts."""
        test_cases = ["/music/desktop.ini", "/music/Thumbs.db", "/music/.DS_Store", "/music/.git"]

        for proposed in test_cases:
            result = detector.detect_conflicts(proposed, set())
            assert len(result["conflicts"]) == 1
            assert "system file" in result["conflicts"][0]

    def test_detect_batch_conflicts(self, detector):
        """Test batch conflict detection."""
        proposals = [
            {"recording_id": "1", "full_proposed_path": "/music/song1.mp3"},
            {"recording_id": "2", "full_proposed_path": "/music/song2.mp3"},
            {"recording_id": "3", "full_proposed_path": "/music/song1.mp3"},  # Conflicts with first
        ]

        directory_contents = {
            "/music": {"song2.mp3", "other.mp3"}  # song2 already exists
        }

        results = detector.detect_batch_conflicts(proposals, directory_contents)

        # First proposal should have conflict with third
        assert len(results["1"]["conflicts"]) > 0
        assert "3" in str(results["1"]["conflicts"])

        # Second proposal should have existence conflict
        assert len(results["2"]["conflicts"]) > 0
        assert "already exists" in str(results["2"]["conflicts"])

        # Third proposal should have conflict with first
        assert len(results["3"]["conflicts"]) > 0
        assert "1" in str(results["3"]["conflicts"])

    def test_resolve_conflicts_with_existence(self, detector):
        """Test conflict resolution for existing files."""
        proposed = "/music/song.mp3"
        conflicts = ["File already exists: /music/song.mp3"]

        with patch("os.path.exists") as mock_exists:
            # First alternative exists, second doesn't
            mock_exists.side_effect = [True, False]

            alternative = detector.resolve_conflicts(proposed, conflicts)

            assert alternative == "/music/song_2.mp3"

    def test_resolve_conflicts_with_case(self, detector):
        """Test conflict resolution for case conflicts."""
        proposed = "/music/Song.MP3"
        conflicts = ["Case-insensitive collision with: /music/song.mp3"]

        alternative = detector.resolve_conflicts(proposed, conflicts)

        # Should suggest different casing
        assert alternative is not None
        assert alternative != proposed

    def test_resolve_conflicts_unresolvable(self, detector):
        """Test when conflicts cannot be resolved."""
        proposed = "/music/song.mp3"
        conflicts = ["Path contains directory traversal patterns"]

        alternative = detector.resolve_conflicts(proposed, conflicts)

        assert alternative is None

    @patch("os.path.exists")
    @patch("os.access")
    @patch("os.path.isdir")
    def test_validate_rename_safety_success(self, mock_isdir, mock_access, mock_exists, detector):
        """Test successful rename safety validation."""
        original = "/music/old.mp3"
        proposed = "/music/new.mp3"

        mock_exists.side_effect = [True, True]  # original exists, target dir exists
        mock_access.return_value = True  # Have write permission
        mock_isdir.return_value = False  # Target is not a directory

        is_safe, issues = detector.validate_rename_safety(original, proposed)

        assert is_safe is True
        assert len(issues) == 0

    @patch("os.path.exists")
    def test_validate_rename_safety_missing_original(self, mock_exists, detector):
        """Test validation when original file doesn't exist."""
        original = "/music/old.mp3"
        proposed = "/music/new.mp3"

        mock_exists.return_value = False

        is_safe, issues = detector.validate_rename_safety(original, proposed)

        assert is_safe is False
        assert len(issues) == 1
        assert "does not exist" in issues[0]

    @patch("os.path.exists")
    @patch("os.access")
    def test_validate_rename_safety_no_permission(self, mock_access, mock_exists, detector):
        """Test validation when no write permission."""
        original = "/music/old.mp3"
        proposed = "/protected/new.mp3"

        mock_exists.side_effect = [True, True]  # Both exist
        mock_access.return_value = False  # No write permission

        is_safe, issues = detector.validate_rename_safety(original, proposed)

        assert is_safe is False
        assert len(issues) == 1
        assert "No write permission" in issues[0]

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.access")
    def test_validate_rename_safety_target_is_directory(self, mock_access, mock_isdir, mock_exists, detector):
        """Test validation when target is a directory."""
        original = "/music/old.mp3"
        proposed = "/music/folder"

        mock_exists.return_value = True
        mock_access.return_value = True  # Have write permission
        mock_isdir.return_value = True

        is_safe, issues = detector.validate_rename_safety(original, proposed)

        assert is_safe is False
        # Should have at least the directory issue
        assert any("Target is a directory" in issue for issue in issues)

    @patch("os.path.exists")
    @patch("os.path.abspath")
    @patch("os.access")
    def test_validate_rename_safety_circular(self, mock_access, mock_abspath, mock_exists, detector):
        """Test validation for circular rename."""
        original = "/music/song.mp3"
        proposed = "/music/./song.mp3"

        mock_exists.return_value = True
        mock_access.return_value = True  # Have write permission
        mock_abspath.return_value = "/music/song.mp3"  # Same for both

        is_safe, issues = detector.validate_rename_safety(original, proposed)

        assert is_safe is False
        # Should have at least the "same path" issue
        assert any("same" in issue for issue in issues)

    def test_has_directory_traversal(self, detector):
        """Test directory traversal detection."""
        assert detector._has_directory_traversal("/music/../etc/passwd")
        assert detector._has_directory_traversal("../../../etc/passwd")
        assert detector._has_directory_traversal("/music/../../system")
        assert not detector._has_directory_traversal("/music/artist/song.mp3")
        assert not detector._has_directory_traversal("/music/two..dots.mp3")

    def test_is_backup_pattern(self, detector):
        """Test backup pattern detection."""
        assert detector._is_backup_pattern("song.mp3.bak")
        assert detector._is_backup_pattern("song.mp3.backup")
        assert detector._is_backup_pattern("backup_song.mp3")
        assert detector._is_backup_pattern("song.mp3.1")
        assert detector._is_backup_pattern("song~")
        assert not detector._is_backup_pattern("song.mp3")
        assert not detector._is_backup_pattern("normal_file.txt")

    def test_is_temp_pattern(self, detector):
        """Test temporary file pattern detection."""
        assert detector._is_temp_pattern("song.mp3.tmp")
        assert detector._is_temp_pattern("temp_song.mp3")
        assert detector._is_temp_pattern(".~lock.song.mp3")
        assert detector._is_temp_pattern("~$song.mp3")
        assert not detector._is_temp_pattern("song.mp3")
        assert not detector._is_temp_pattern("temperature.mp3")

    def test_is_system_file(self, detector):
        """Test system file detection."""
        assert detector._is_system_file("desktop.ini")
        assert detector._is_system_file("Thumbs.db")
        assert detector._is_system_file(".DS_Store")
        assert detector._is_system_file(".git")
        assert not detector._is_system_file("song.mp3")
        assert not detector._is_system_file("normal_file.txt")
