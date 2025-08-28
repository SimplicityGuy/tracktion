"""Unit tests for the file scanner module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from services.file_watcher.src.file_scanner import FileScanner


class TestFileScanner:
    """Tests for FileScanner class."""

    def test_supported_extensions_includes_ogg(self):
        """Test that OGG extensions are in supported formats."""
        scanner = FileScanner()
        assert ".ogg" in scanner.SUPPORTED_EXTENSIONS
        assert ".oga" in scanner.SUPPORTED_EXTENSIONS

    def test_supported_extensions_includes_legacy_formats(self):
        """Test that legacy formats are still supported."""
        scanner = FileScanner()
        expected_formats = {".mp3", ".flac", ".wav", ".wave", ".m4a", ".mp4", ".m4b", ".m4p", ".m4v", ".m4r"}
        for fmt in expected_formats:
            assert fmt in scanner.SUPPORTED_EXTENSIONS

    def test_is_audio_file_recognizes_ogg(self):
        """Test that OGG files are recognized as audio files."""
        scanner = FileScanner()

        # Test .ogg extension
        assert scanner.is_audio_file(Path("test.ogg"))
        assert scanner.is_audio_file(Path("test.OGG"))
        assert scanner.is_audio_file(Path("/path/to/file.ogg"))

        # Test .oga extension
        assert scanner.is_audio_file(Path("test.oga"))
        assert scanner.is_audio_file(Path("test.OGA"))
        assert scanner.is_audio_file(Path("/path/to/file.oga"))

    def test_is_audio_file_rejects_non_audio(self):
        """Test that non-audio files are rejected."""
        scanner = FileScanner()

        assert not scanner.is_audio_file(Path("test.txt"))
        assert not scanner.is_audio_file(Path("test.pdf"))
        assert not scanner.is_audio_file(Path("test.jpg"))
        assert not scanner.is_audio_file(Path("test"))

    def test_scan_directory_with_ogg_files(self):
        """Test scanning a directory containing OGG files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test files
            ogg_file = tmp_path / "test.ogg"
            ogg_file.write_text("fake ogg content")

            oga_file = tmp_path / "test.oga"
            oga_file.write_text("fake oga content")

            mp3_file = tmp_path / "test.mp3"
            mp3_file.write_text("fake mp3 content")

            txt_file = tmp_path / "test.txt"
            txt_file.write_text("not audio")

            scanner = FileScanner()
            results = scanner.scan_directory(tmp_path)

            # Should find 3 audio files (ogg, oga, mp3)
            assert len(results) == 3

            # Check that OGG files are included
            paths = [r["path"] for r in results]
            assert str(ogg_file.absolute()) in paths
            assert str(oga_file.absolute()) in paths
            assert str(mp3_file.absolute()) in paths

            # Check file info structure
            for result in results:
                assert "path" in result
                assert "name" in result
                assert "extension" in result
                assert "size_bytes" in result
                assert "modified_time" in result
                assert "sha256_hash" in result
                assert "xxh128_hash" in result

    def test_scan_directory_tracks_files(self):
        """Test that scanner tracks already seen files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create a test OGG file
            ogg_file = tmp_path / "test.ogg"
            ogg_file.write_text("fake ogg content")

            scanner = FileScanner()

            # First scan should find the file
            results1 = scanner.scan_directory(tmp_path)
            assert len(results1) == 1

            # Second scan should not find it again (already tracked)
            results2 = scanner.scan_directory(tmp_path)
            assert len(results2) == 0

            # Check tracked files
            assert str(ogg_file.absolute()) in scanner.tracked_files

    def test_scan_directory_with_subdirectories(self):
        """Test recursive scanning of subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create nested directory structure
            subdir = tmp_path / "subdir"
            subdir.mkdir()

            # Create OGG files in different locations
            root_ogg = tmp_path / "root.ogg"
            root_ogg.write_text("fake content")

            sub_ogg = subdir / "sub.ogg"
            sub_ogg.write_text("fake content")

            scanner = FileScanner()
            results = scanner.scan_directory(tmp_path)

            # Should find both OGG files
            assert len(results) == 2
            paths = [r["path"] for r in results]
            assert str(root_ogg.absolute()) in paths
            assert str(sub_ogg.absolute()) in paths

    def test_scan_nonexistent_directory(self):
        """Test scanning a directory that doesn't exist."""
        scanner = FileScanner()
        results = scanner.scan_directory(Path("/nonexistent/path"))
        assert results == []

    def test_scan_file_instead_of_directory(self):
        """Test scanning a file path instead of directory."""
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmpfile:
            scanner = FileScanner()
            results = scanner.scan_directory(Path(tmpfile.name))
            assert results == []

    @patch("services.file_watcher.src.file_scanner.logger")
    def test_ogg_file_logging(self, mock_logger):
        """Test that OGG files trigger specific logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create OGG file
            ogg_file = tmp_path / "test.ogg"
            ogg_file.write_text("fake content")

            scanner = FileScanner()
            scanner.scan_directory(tmp_path)

            # Check that OGG-specific logging occurred
            mock_logger.info.assert_any_call(
                "OGG Vorbis file detected",
                path=str(ogg_file.absolute()),
                size=str(len("fake content")),
                extension=".ogg",
            )

    def test_file_info_structure(self):
        """Test the structure of file info dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test OGG file
            ogg_file = tmp_path / "test_audio.ogg"
            ogg_file.write_text("fake ogg audio data")

            scanner = FileScanner()
            results = scanner.scan_directory(tmp_path)

            assert len(results) == 1
            file_info = results[0]

            # Verify all required fields
            assert file_info["path"] == str(ogg_file.absolute())
            assert file_info["name"] == "test_audio.ogg"
            assert file_info["extension"] == ".ogg"
            assert int(file_info["size_bytes"]) > 0
            assert file_info["modified_time"]
            assert file_info["sha256_hash"]  # Should have SHA256 hash
            assert file_info["xxh128_hash"]  # Should have XXH128 hash

    def test_hash_calculation(self):
        """Test file hash calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create two identical files
            ogg1 = tmp_path / "file1.ogg"
            ogg2 = tmp_path / "file2.ogg"
            content = "same content for both files"
            ogg1.write_text(content)
            ogg2.write_text(content)

            # Create a different file
            ogg3 = tmp_path / "file3.ogg"
            ogg3.write_text("different content")

            scanner = FileScanner()

            # Get hashes
            sha256_1, xxh128_1 = scanner._calculate_dual_hashes(ogg1)
            sha256_2, xxh128_2 = scanner._calculate_dual_hashes(ogg2)
            sha256_3, xxh128_3 = scanner._calculate_dual_hashes(ogg3)

            # Same content should have same hashes
            assert sha256_1 == sha256_2
            assert xxh128_1 == xxh128_2
            # Different content should have different hashes
            assert sha256_1 != sha256_3
            assert xxh128_1 != xxh128_3
