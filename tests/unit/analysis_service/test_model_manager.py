"""
Unit tests for the Model Manager module.

Tests model downloading, caching, loading, and verification functionality
using mocks to avoid actual network calls and TensorFlow dependencies.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.model_manager import ModelManager


class TestModelManager:
    """Test cases for ModelManager class."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary directory for models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def model_manager(self, temp_models_dir):
        """Create a ModelManager instance with temp directory."""
        return ModelManager(models_dir=str(temp_models_dir), auto_download=False, lazy_load=True)

    def test_initialization(self, temp_models_dir):
        """Test ModelManager initialization."""
        manager = ModelManager(
            models_dir=str(temp_models_dir),
            auto_download=True,
            verify_checksum=True,
            lazy_load=False,
        )

        assert manager.models_dir == temp_models_dir
        assert manager.auto_download is True
        assert manager.verify_checksum is True
        assert manager.lazy_load is False
        assert len(manager._loaded_models) == 0
        assert temp_models_dir.exists()

    def test_initialization_creates_directory(self):
        """Test that initialization creates models directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "new_models_dir"
            ModelManager(models_dir=str(models_dir))
            assert models_dir.exists()

    def test_load_manifest_success(self):
        """Test successful manifest loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "genre_discogs_effnet": {
                    "checksum": "abc123",
                    "filename": "genre_discogs-effnet-bs64-1.pb",
                }
            }
            manifest_path = Path(tmpdir) / "manifest.json"
            with Path(manifest_path).open("w") as f:
                json.dump(manifest_data, f)

            manager = ModelManager(models_dir=str(tmpdir))
            assert manager._model_metadata["genre_discogs_effnet"]["checksum"] == "abc123"

    def test_load_manifest_missing_file(self):
        """Test manifest loading when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fresh manager - KNOWN_MODELS should be copied not referenced
            manager = ModelManager(models_dir=str(tmpdir))
            # Should not raise error, just proceed without manifest
            # Check all models have no checksum when no manifest
            for model_id in manager._model_metadata:
                assert manager._model_metadata[model_id]["checksum"] is None

    def test_save_manifest(self, model_manager, temp_models_dir):
        """Test saving manifest with checksums."""
        model_manager._model_metadata["genre_discogs_effnet"]["checksum"] = "test_checksum"
        model_manager._save_manifest()

        manifest_path = temp_models_dir / "manifest.json"
        assert manifest_path.exists()

        with manifest_path.open() as f:
            manifest = json.load(f)
        assert manifest["genre_discogs_effnet"]["checksum"] == "test_checksum"

    @patch("services.analysis_service.src.model_manager.urlopen")
    def test_download_model_success(self, mock_urlopen, model_manager, temp_models_dir):
        """Test successful model download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1000"
        mock_response.read.side_effect = [b"test_data", b""]  # Data then EOF
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Mock checksum calculation
        with patch.object(model_manager, "_calculate_checksum", return_value="checksum123"):
            result = model_manager.download_model("genre_discogs_effnet")

        assert result is True
        model_path = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        assert model_path.exists()
        assert model_manager._model_metadata["genre_discogs_effnet"]["checksum"] == "checksum123"

    @patch("services.analysis_service.src.model_manager.urlopen")
    def test_download_model_with_progress_callback(self, mock_urlopen, model_manager, temp_models_dir):
        """Test model download with progress callback."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "100"
        mock_response.read.side_effect = [b"test" * 10, b"data" * 10, b"more" * 5, b""]
        mock_urlopen.return_value.__enter__.return_value = mock_response

        progress_calls = []

        def progress_callback(model_id, progress):
            progress_calls.append((model_id, progress))

        with patch.object(model_manager, "_calculate_checksum", return_value="checksum123"):
            result = model_manager.download_model("genre_discogs_effnet", progress_callback)

        assert result is True
        assert len(progress_calls) > 0
        assert progress_calls[0][0] == "genre_discogs_effnet"

    def test_download_model_unknown_model(self, model_manager):
        """Test download with unknown model ID."""
        result = model_manager.download_model("unknown_model")
        assert result is False

    @patch("services.analysis_service.src.model_manager.urlopen")
    def test_download_model_network_error(self, mock_urlopen, model_manager):
        """Test download with network error."""
        mock_urlopen.side_effect = Exception("Network error")
        result = model_manager.download_model("genre_discogs_effnet")
        assert result is False

    def test_calculate_checksum(self, model_manager, temp_models_dir):
        """Test checksum calculation."""
        test_file = temp_models_dir / "test.txt"
        test_file.write_bytes(b"test content")

        checksum = model_manager._calculate_checksum(test_file)
        expected = hashlib.sha256(b"test content").hexdigest()
        assert checksum == expected

    def test_verify_model_no_checksum(self, temp_models_dir):
        """Test model verification when no checksum is available."""
        # Create a fresh manager with no checksum
        manager = ModelManager(models_dir=str(temp_models_dir), auto_download=False)

        test_file = temp_models_dir / "test.pb"
        test_file.write_bytes(b"model data")

        # Ensure no checksum is set
        manager._model_metadata["genre_discogs_effnet"]["checksum"] = None

        result = manager._verify_model("genre_discogs_effnet", test_file)
        assert result is True  # Should pass when no checksum available

    def test_verify_model_checksum_match(self, model_manager, temp_models_dir):
        """Test model verification with matching checksum."""
        test_file = temp_models_dir / "test.pb"
        test_file.write_bytes(b"model data")

        expected_checksum = hashlib.sha256(b"model data").hexdigest()
        model_manager._model_metadata["genre_discogs_effnet"]["checksum"] = expected_checksum

        result = model_manager._verify_model("genre_discogs_effnet", test_file)
        assert result is True

    def test_verify_model_checksum_mismatch(self, model_manager, temp_models_dir):
        """Test model verification with mismatched checksum."""
        test_file = temp_models_dir / "test.pb"
        test_file.write_bytes(b"model data")

        model_manager._model_metadata["genre_discogs_effnet"]["checksum"] = "wrong_checksum"

        result = model_manager._verify_model("genre_discogs_effnet", test_file)
        assert result is False

    def test_verify_model_disabled(self, temp_models_dir):
        """Test model verification when disabled."""
        manager = ModelManager(models_dir=str(temp_models_dir), verify_checksum=False)
        test_file = temp_models_dir / "test.pb"
        result = manager._verify_model("genre_discogs_effnet", test_file)
        assert result is True  # Should always pass when disabled

    def test_get_model_path_exists(self, model_manager, temp_models_dir):
        """Test getting path for existing model."""
        model_file = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model_file.write_bytes(b"model data")

        with patch.object(model_manager, "_verify_model", return_value=True):
            path = model_manager.get_model_path("genre_discogs_effnet")

        assert path == model_file

    def test_get_model_path_auto_download(self, temp_models_dir):
        """Test auto-download when model doesn't exist."""
        manager = ModelManager(models_dir=str(temp_models_dir), auto_download=True)

        with patch.object(manager, "download_model", return_value=True) as mock_download:
            path = manager.get_model_path("genre_discogs_effnet")

        mock_download.assert_called_once_with("genre_discogs_effnet")
        assert path == temp_models_dir / "genre_discogs-effnet-bs64-1.pb"

    def test_get_model_path_unknown_model(self, model_manager):
        """Test getting path for unknown model."""
        path = model_manager.get_model_path("unknown_model")
        assert path is None

    def test_get_model_path_verification_fails(self, model_manager, temp_models_dir):
        """Test getting path when verification fails."""
        model_file = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model_file.write_bytes(b"model data")

        with patch.object(model_manager, "_verify_model", return_value=False):
            path = model_manager.get_model_path("genre_discogs_effnet")

        assert path is None

    def test_load_model_success(self, model_manager, temp_models_dir):
        """Test successful model loading."""
        model_file = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model_file.write_bytes(b"model data")

        mock_model = Mock()

        # Mock tensorflow module
        mock_tf = MagicMock()
        mock_tf.saved_model.load.return_value = mock_model

        with (
            patch.dict("sys.modules", {"tensorflow": mock_tf}),
            patch.object(model_manager, "get_model_path", return_value=model_file),
        ):
            model = model_manager.load_model("genre_discogs_effnet")

        assert model == mock_model
        assert model_manager._loaded_models["genre_discogs_effnet"] == mock_model
        mock_tf.saved_model.load.assert_called_once_with(str(model_file))

    def test_load_model_already_loaded(self, model_manager):
        """Test loading already loaded model."""
        mock_model = Mock()
        model_manager._loaded_models["genre_discogs_effnet"] = mock_model

        # Mock tensorflow but it shouldn't be called
        mock_tf = MagicMock()

        with patch.dict("sys.modules", {"tensorflow": mock_tf}):
            model = model_manager.load_model("genre_discogs_effnet")

        assert model == mock_model
        mock_tf.saved_model.load.assert_not_called()

    def test_load_model_path_not_found(self, model_manager):
        """Test loading model when path not found."""
        with patch.object(model_manager, "get_model_path", return_value=None):
            model = model_manager.load_model("genre_discogs_effnet")

        assert model is None

    def test_load_model_tensorflow_not_installed(self, model_manager, temp_models_dir):
        """Test loading model when TensorFlow not installed."""
        model_file = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model_file.write_bytes(b"model data")

        with (
            patch.object(model_manager, "get_model_path", return_value=model_file),
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named tensorflow"),
            ),
        ):
            model = model_manager.load_model("genre_discogs_effnet")

        assert model is None

    def test_preload_models(self, model_manager, temp_models_dir):
        """Test preloading multiple models."""
        # Create model files
        for model_id in ["genre_discogs_effnet", "mood_happy"]:
            metadata = model_manager._model_metadata[model_id]
            model_file = temp_models_dir / metadata["filename"]
            model_file.write_bytes(b"model data")

        # Mock tensorflow
        mock_tf = MagicMock()
        mock_tf.saved_model.load.return_value = Mock()

        with (
            patch.dict("sys.modules", {"tensorflow": mock_tf}),
            patch.object(model_manager, "_verify_model", return_value=True),
        ):
            results = model_manager.preload_models(["genre_discogs_effnet", "mood_happy", "unknown"])

        assert results["genre_discogs_effnet"] is True
        assert results["mood_happy"] is True
        assert results["unknown"] is False

    def test_get_all_models(self, model_manager):
        """Test getting list of all models."""
        models = model_manager.get_all_models()
        assert "genre_discogs_effnet" in models
        assert "mood_happy" in models
        assert len(models) == len(ModelManager.KNOWN_MODELS)

    def test_get_model_info(self, model_manager):
        """Test getting model metadata."""
        info = model_manager.get_model_info("genre_discogs_effnet")
        assert info is not None
        assert info["filename"] == "genre_discogs-effnet-bs64-1.pb"
        assert info["size_mb"] == 85

        info = model_manager.get_model_info("unknown")
        assert info is None

    def test_clear_cache(self, model_manager, temp_models_dir):
        """Test clearing model cache."""
        # Create some model files
        model1 = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model2 = temp_models_dir / "mood_happy-audioset-vggish-1.pb"
        model1.write_bytes(b"model1")
        model2.write_bytes(b"model2")

        # Add to loaded models
        model_manager._loaded_models["test"] = Mock()

        model_manager.clear_cache()

        assert not model1.exists()
        assert not model2.exists()
        assert len(model_manager._loaded_models) == 0

    def test_get_cache_size(self, model_manager, temp_models_dir):
        """Test getting total cache size."""
        # Create model files with known sizes
        model1 = temp_models_dir / "genre_discogs-effnet-bs64-1.pb"
        model2 = temp_models_dir / "mood_happy-audioset-vggish-1.pb"
        model1.write_bytes(b"x" * 1024 * 1024)  # 1 MB
        model2.write_bytes(b"y" * 1024 * 512)  # 0.5 MB

        size = model_manager.get_cache_size()
        assert size == pytest.approx(1.5, rel=0.01)

    def test_get_cache_size_empty(self, model_manager):
        """Test cache size when no models cached."""

        size = model_manager.get_cache_size()
        assert size == 0.0
