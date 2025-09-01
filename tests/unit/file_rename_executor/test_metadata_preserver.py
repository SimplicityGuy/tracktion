"""Unit tests for Metadata Preserver."""

from unittest.mock import MagicMock, patch

from services.analysis_service.src.file_rename_executor.metadata_preserver import (
    MetadataPreserver,
)


class TestMetadataPreserver:
    """Test suite for MetadataPreserver."""

    def test_snapshot_metadata_ogg_file(self):
        """Test metadata snapshot for OGG file."""
        with (
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path,
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.MutagenFile") as mock_file,
        ):
            # Setup path mock
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            # Setup audio mock
            mock_audio = MagicMock()
            mock_audio.mime = ["audio/ogg"]
            mock_audio.tags = {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
                "organization": ["Test Org"],
                "encoder": ["libVorbis 1.3.7"],
            }
            mock_audio.info.length = 180.5
            mock_audio.info.bitrate = 192000
            mock_audio.info.sample_rate = 44100
            mock_audio.info.channels = 2
            mock_audio.info.bitrate_nominal = 192000

            mock_file.return_value = mock_audio

            # Make isinstance check work
            with patch(
                "services.analysis_service.src.file_rename_executor.metadata_preserver.isinstance"
            ) as mock_isinstance:
                mock_isinstance.return_value = True  # It's an OggVorbis file

                metadata = MetadataPreserver.snapshot_metadata("/tmp/test.ogg")

                assert metadata is not None
                assert metadata["is_ogg"] is True
                assert metadata["format"] == "audio/ogg"
                assert "tags" in metadata
                assert metadata["tags"]["title"] == ["Test Song"]
                assert metadata["tags"]["artist"] == ["Test Artist"]
                assert metadata["tags"]["organization"] == ["Test Org"]
                assert metadata["info"]["bitrate"] == 192000
                assert metadata["info"]["sample_rate"] == 44100

    def test_snapshot_metadata_file_not_exists(self):
        """Test snapshot when file doesn't exist."""
        with patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            metadata = MetadataPreserver.snapshot_metadata("/tmp/nonexistent.ogg")

            assert metadata is None

    def test_snapshot_metadata_non_ogg_file(self):
        """Test metadata snapshot for non-OGG file."""
        with (
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path,
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.MutagenFile") as mock_file,
        ):
            # Setup path mock
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            # Setup audio mock for MP3
            mock_audio = MagicMock()
            mock_audio.mime = ["audio/mp3"]
            mock_tags = MagicMock()
            mock_tags.__getitem__ = lambda self, key: {
                "title": "MP3 Song",
                "artist": "MP3 Artist",
            }[key]
            mock_tags.keys = MagicMock(return_value=["title", "artist"])
            mock_audio.tags = mock_tags
            mock_audio.info.length = 200.0
            mock_audio.info.bitrate = 320000

            mock_file.return_value = mock_audio

            # Make isinstance check work
            with patch(
                "services.analysis_service.src.file_rename_executor.metadata_preserver.isinstance"
            ) as mock_isinstance:
                mock_isinstance.return_value = False  # Not an OggVorbis file

                metadata = MetadataPreserver.snapshot_metadata("/tmp/test.mp3")

                assert metadata is not None
                assert metadata["is_ogg"] is False
                assert metadata["format"] == "audio/mp3"

    def test_restore_metadata_success(self):
        """Test successful metadata restoration."""
        metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
                "album": ["Test Album"],
            },
        }

        with (
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path,
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.OggVorbis") as mock_ogg,
        ):
            # Setup path mock
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            # Setup OggVorbis mock
            mock_audio = MagicMock()
            mock_audio.tags = MagicMock()
            mock_ogg.return_value = mock_audio

            result = MetadataPreserver.restore_metadata("/tmp/renamed.ogg", metadata)

            assert result is True
            mock_audio.tags.clear.assert_called_once()
            assert mock_audio.tags.__setitem__.call_count == 3  # 3 tags
            mock_audio.save.assert_called_once()

    def test_restore_metadata_file_not_exists(self):
        """Test restore when file doesn't exist."""
        metadata = {"is_ogg": True, "tags": {"title": ["Test"]}}

        with patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            result = MetadataPreserver.restore_metadata("/tmp/nonexistent.ogg", metadata)

            assert result is False

    def test_restore_metadata_non_ogg_file(self):
        """Test restore skips non-OGG files."""
        metadata = {"is_ogg": False, "tags": {"title": "Test"}}

        with patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            result = MetadataPreserver.restore_metadata("/tmp/test.mp3", metadata)

            assert result is True  # Should return True (not an error, just skipped)

    def test_verify_metadata_success(self):
        """Test successful metadata verification."""
        original_metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
            },
            "info": {
                "bitrate": 192000,
                "sample_rate": 44100,
            },
        }

        with patch.object(MetadataPreserver, "snapshot_metadata") as mock_snapshot:
            # Return matching metadata
            mock_snapshot.return_value = original_metadata.copy()

            result = MetadataPreserver.verify_metadata("/tmp/old.ogg", "/tmp/new.ogg", original_metadata)

            assert result is True
            mock_snapshot.assert_called_once_with("/tmp/new.ogg")

    def test_verify_metadata_tag_mismatch(self):
        """Test metadata verification with tag mismatch."""
        original_metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Original Song"],
                "artist": ["Original Artist"],
            },
            "info": {},
        }

        current_metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Different Song"],  # Changed
                "artist": ["Original Artist"],
            },
            "info": {},
        }

        with patch.object(MetadataPreserver, "snapshot_metadata") as mock_snapshot:
            mock_snapshot.return_value = current_metadata

            result = MetadataPreserver.verify_metadata("/tmp/old.ogg", "/tmp/new.ogg", original_metadata)

            assert result is False

    def test_verify_metadata_missing_tag(self):
        """Test metadata verification with missing tag."""
        original_metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
                "album": ["Test Album"],
            },
            "info": {},
        }

        current_metadata = {
            "is_ogg": True,
            "tags": {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
                # 'album' is missing
            },
            "info": {},
        }

        with patch.object(MetadataPreserver, "snapshot_metadata") as mock_snapshot:
            mock_snapshot.return_value = current_metadata

            result = MetadataPreserver.verify_metadata("/tmp/old.ogg", "/tmp/new.ogg", original_metadata)

            assert result is False

    def test_format_metadata_for_display(self):
        """Test formatting metadata for display."""
        metadata = {
            "format": "audio/ogg",
            "is_ogg": True,
            "tags": {
                "title": ["Test Song"],
                "artist": ["Test Artist"],
            },
            "info": {
                "length": 180.5,
                "bitrate": 192000,
                "sample_rate": 44100,
                "channels": 2,
                "bitrate_nominal": 192000,
            },
        }

        result = MetadataPreserver.format_metadata_for_display(metadata)

        assert "Format: audio/ogg" in result
        assert "Type: OGG Vorbis" in result
        assert "title: Test Song" in result
        assert "artist: Test Artist" in result
        assert "Duration: 180.50 seconds" in result
        assert "Bitrate: 192000 bps" in result
        assert "Sample Rate: 44100 Hz" in result
        assert "Channels: 2" in result
        assert "Nominal Bitrate: 192000 bps" in result

    def test_format_metadata_for_display_empty(self):
        """Test formatting empty metadata."""
        result = MetadataPreserver.format_metadata_for_display(None)
        assert result == "No metadata"

        result = MetadataPreserver.format_metadata_for_display({})
        assert "Format: unknown" in result

    def test_snapshot_metadata_exception_handling(self):
        """Test exception handling in snapshot_metadata."""
        with (
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path,
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.MutagenFile") as mock_file,
        ):
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            # Simulate exception
            mock_file.side_effect = Exception("Test error")

            metadata = MetadataPreserver.snapshot_metadata("/tmp/test.ogg")

            assert metadata is None

    def test_restore_metadata_exception_handling(self):
        """Test exception handling in restore_metadata."""
        metadata = {
            "is_ogg": True,
            "tags": {"title": ["Test"]},
        }

        with (
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.Path") as mock_path,
            patch("services.analysis_service.src.file_rename_executor.metadata_preserver.OggVorbis") as mock_ogg,
        ):
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path.return_value = mock_path_obj

            # Simulate exception
            mock_ogg.side_effect = Exception("Test error")

            result = MetadataPreserver.restore_metadata("/tmp/test.ogg", metadata)

            assert result is False
