"""
Model Manager for TensorFlow models used in audio analysis.

This module handles downloading, caching, and loading of pre-trained
TensorFlow models from Essentia's model repository.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages TensorFlow models for audio analysis.

    Handles model downloading, caching, versioning, and lazy loading
    for optimal memory usage and startup performance.
    """

    # Model repository base URL
    MODEL_REPO_BASE = "https://essentia.upf.edu/models/"

    # Known models with their metadata
    KNOWN_MODELS = {
        "genre_discogs_effnet": {
            "filename": "genre_discogs-effnet-bs64-1.pb",
            "url": "genre_discogs-effnet-bs64-1.pb",
            "size_mb": 85,
            "checksum": None,  # Will be populated from manifest
            "description": "Genre classification using Discogs EffNet",
        },
        "mood_happy": {
            "filename": "mood_happy-audioset-vggish-1.pb",
            "url": "mood_happy-audioset-vggish-1.pb",
            "size_mb": 280,
            "checksum": None,
            "description": "Mood detection - happiness dimension",
        },
        "mood_sad": {
            "filename": "mood_sad-audioset-vggish-1.pb",
            "url": "mood_sad-audioset-vggish-1.pb",
            "size_mb": 280,
            "checksum": None,
            "description": "Mood detection - sadness dimension",
        },
        "mood_aggressive": {
            "filename": "mood_aggressive-audioset-vggish-1.pb",
            "url": "mood_aggressive-audioset-vggish-1.pb",
            "size_mb": 280,
            "checksum": None,
            "description": "Mood detection - aggressive dimension",
        },
        "mood_relaxed": {
            "filename": "mood_relaxed-audioset-vggish-1.pb",
            "url": "mood_relaxed-audioset-vggish-1.pb",
            "size_mb": 280,
            "checksum": None,
            "description": "Mood detection - relaxed dimension",
        },
        "danceability": {
            "filename": "danceability-audioset-vggish-1.pb",
            "url": "danceability-audioset-vggish-1.pb",
            "size_mb": 280,
            "checksum": None,
            "description": "Danceability prediction",
        },
    }

    def __init__(
        self,
        models_dir: Optional[str] = None,
        auto_download: bool = True,
        verify_checksum: bool = True,
        lazy_load: bool = True,
    ):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory to store cached models (default: service/models/)
            auto_download: Automatically download missing models
            verify_checksum: Verify model integrity using checksums
            lazy_load: Load models only when requested (optimizes startup)
        """
        self.models_dir = Path(models_dir or "services/analysis_service/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download
        self.verify_checksum = verify_checksum
        self.lazy_load = lazy_load
        self._loaded_models: Dict[str, Any] = {}
        # Deep copy to prevent cross-instance mutation
        import copy

        self._model_metadata = copy.deepcopy(self.KNOWN_MODELS)

        # Load manifest with checksums if available
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load model manifest with checksums if available."""
        manifest_path = self.models_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    for model_id, metadata in manifest.items():
                        if model_id in self._model_metadata:
                            self._model_metadata[model_id]["checksum"] = metadata.get("checksum")
                logger.info(f"Loaded model manifest from {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {str(e)}")

    def _save_manifest(self) -> None:
        """Save model manifest with checksums."""
        manifest_path = self.models_dir / "manifest.json"
        manifest = {}
        for model_id, metadata in self._model_metadata.items():
            if metadata.get("checksum"):
                manifest[model_id] = {"checksum": metadata["checksum"], "filename": metadata["filename"]}
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved model manifest to {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to save manifest: {str(e)}")

    def download_model(self, model_id: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Download a model from the repository.

        Args:
            model_id: Model identifier from KNOWN_MODELS
            progress_callback: Optional callback for download progress

        Returns:
            True if download successful, False otherwise
        """
        if model_id not in self._model_metadata:
            logger.error(f"Unknown model: {model_id}")
            return False

        metadata = self._model_metadata[model_id]
        model_url = self.MODEL_REPO_BASE + str(metadata["url"])
        model_path = self.models_dir / str(metadata["filename"])

        # Check if already exists
        if model_path.exists() and self._verify_model(model_id, model_path):
            logger.info(f"Model {model_id} already cached at {model_path}")
            return True

        logger.info(f"Downloading model {model_id} from {model_url}")

        try:
            # Create request with headers
            req = Request(model_url, headers={"User-Agent": "Tracktion/1.0"})

            # Download with progress tracking
            with urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(model_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(model_id, progress)

            # Calculate and store checksum
            if self.verify_checksum:
                checksum = self._calculate_checksum(model_path)
                self._model_metadata[model_id]["checksum"] = checksum
                self._save_manifest()

            logger.info(f"Successfully downloaded model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {str(e)}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_model(self, model_id: str, model_path: Path) -> bool:
        """
        Verify model integrity using checksum.

        Args:
            model_id: Model identifier
            model_path: Path to model file

        Returns:
            True if model is valid or checksum not available
        """
        if not self.verify_checksum:
            return True

        if not model_path.exists():
            return False

        metadata = self._model_metadata[model_id]
        expected_checksum = metadata.get("checksum")

        if not expected_checksum:
            logger.debug(f"No checksum available for {model_id}, skipping verification")
            return True

        actual_checksum = self._calculate_checksum(model_path)
        if actual_checksum != expected_checksum:
            logger.error(f"Checksum mismatch for {model_id}: expected {expected_checksum}, got {actual_checksum}")
            return False

        logger.debug(f"Checksum verified for {model_id}")
        return True

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get the path to a cached model.

        Args:
            model_id: Model identifier

        Returns:
            Path to model file if available, None otherwise
        """
        if model_id not in self._model_metadata:
            logger.error(f"Unknown model: {model_id}")
            return None

        metadata = self._model_metadata[model_id]
        model_path = self.models_dir / str(metadata["filename"])

        if not model_path.exists():
            if self.auto_download:
                if self.download_model(model_id):
                    return model_path
            logger.error(f"Model {model_id} not found and auto-download disabled")
            return None

        if self.verify_checksum and not self._verify_model(model_id, model_path):
            logger.error(f"Model {model_id} failed verification")
            return None

        return model_path

    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load a TensorFlow model (lazy loading).

        Args:
            model_id: Model identifier

        Returns:
            Loaded model object or None if loading fails
        """
        # Check if already loaded
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        model_path = self.get_model_path(model_id)
        if not model_path:
            return None

        try:
            # Import TensorFlow only when needed (lazy import)
            import tensorflow as tf

            # Load the model
            logger.info(f"Loading model {model_id} from {model_path}")
            model = tf.saved_model.load(str(model_path))

            if self.lazy_load:
                self._loaded_models[model_id] = model

            return model

        except ImportError:
            logger.error("TensorFlow not installed. Install with: uv pip install tensorflow")
            return None
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None

    def preload_models(self, model_ids: List[str]) -> Dict[str, bool]:
        """
        Preload multiple models for better performance.

        Args:
            model_ids: List of model identifiers

        Returns:
            Dictionary of model_id -> success status
        """
        results = {}
        for model_id in model_ids:
            model = self.load_model(model_id)
            results[model_id] = model is not None
        return results

    def get_all_models(self) -> List[str]:
        """Get list of all available model IDs."""
        return list(self._model_metadata.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        return self._model_metadata.get(model_id)

    def clear_cache(self) -> None:
        """Clear all cached models from disk."""
        for model_id, metadata in self._model_metadata.items():
            model_path = self.models_dir / str(metadata["filename"])
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Removed cached model {model_id}")

        # Clear loaded models from memory
        self._loaded_models.clear()

    def get_cache_size(self) -> float:
        """
        Get total size of cached models in MB.

        Returns:
            Total size in megabytes
        """
        total_size = 0
        for model_id, metadata in self._model_metadata.items():
            model_path = self.models_dir / str(metadata["filename"])
            if model_path.exists():
                total_size += model_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
