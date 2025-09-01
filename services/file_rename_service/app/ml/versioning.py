"""Model versioning and deployment management."""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import MLModel, ModelStatus

logger = logging.getLogger(__name__)

# Constants
MIN_ACCURACY_THRESHOLD = 0.7  # Minimum accuracy required for model deployment
MIN_DEPLOYMENT_HISTORY_FOR_ROLLBACK = 2  # Minimum history entries required for rollback


class ModelVersionManager:
    """Manage model versions, deployments, and rollbacks."""

    def __init__(self, model_dir: str = "models/", deployment_dir: str = "deployed/"):
        """Initialize version manager with storage directories."""
        self.model_dir = Path(model_dir)
        self.deployment_dir = Path(deployment_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

        self.deployment_history_file = self.deployment_dir / "deployment_history.json"
        self.ab_test_config_file = self.deployment_dir / "ab_test_config.json"

    def list_models(self) -> list[MLModel]:
        """List all available model versions."""
        models = []

        for metadata_file in self.model_dir.glob("metadata_*.json"):
            with metadata_file.open() as f:
                metadata_dict = json.load(f)

            model = MLModel(
                id=metadata_dict["id"],
                version=metadata_dict["version"],
                algorithm=metadata_dict["algorithm"],
                created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                training_metrics=metadata_dict["training_metrics"],
                hyperparameters=metadata_dict["hyperparameters"],
                feature_config=metadata_dict["feature_config"],
                status=ModelStatus(metadata_dict["status"]),
                file_path=metadata_dict["file_path"],
                training_duration=metadata_dict.get("training_duration"),
                sample_count=metadata_dict.get("sample_count"),
            )
            models.append(model)

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def deploy_model(self, version: str, force: bool = False) -> bool:
        """Deploy a specific model version to production."""
        # Find model
        model_path = self.model_dir / f"model_random_forest_{version}.pkl"
        metadata_path = self.model_dir / f"metadata_{version}.json"

        if not model_path.exists() or not metadata_path.exists():
            logger.error(f"Model version {version} not found")
            return False

        # Check if model meets deployment criteria
        with metadata_path.open() as f:
            metadata = json.load(f)

        if not force:
            # Check accuracy threshold
            accuracy = metadata["training_metrics"].get("accuracy", 0)
            if accuracy < MIN_ACCURACY_THRESHOLD:
                logger.error(f"Model accuracy {accuracy:.2f} below threshold 0.70")
                return False

        # Archive current deployment
        self._archive_current_deployment()

        # Copy model to deployment directory
        deployed_model_path = self.deployment_dir / "model.pkl"
        deployed_metadata_path = self.deployment_dir / "metadata.json"

        shutil.copy2(model_path, deployed_model_path)
        shutil.copy2(metadata_path, deployed_metadata_path)

        # Update deployment history
        self._update_deployment_history(version, metadata)

        # Update model status
        metadata["status"] = ModelStatus.DEPLOYED.value
        with Path(metadata_path).open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully deployed model version {version}")
        return True

    def rollback_model(self, target_version: str | None = None) -> bool:
        """Rollback to a previous model version."""
        history = self._get_deployment_history()

        if not history:
            logger.error("No deployment history found")
            return False

        if target_version:
            # Rollback to specific version
            if not any(d["version"] == target_version for d in history):
                logger.error(f"Version {target_version} not found in deployment history")
                return False
        else:
            # Rollback to previous version
            if len(history) < MIN_DEPLOYMENT_HISTORY_FOR_ROLLBACK:
                logger.error("No previous version to rollback to")
                return False
            target_version = history[-2]["version"]

        # Deploy target version
        return self.deploy_model(target_version, force=True)

    def setup_ab_test(
        self,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        duration_hours: int = 24,
    ) -> dict[str, Any]:
        """Setup A/B testing between two model versions."""
        # Verify both models exist
        model_a_path = self.model_dir / f"model_random_forest_{version_a}.pkl"
        model_b_path = self.model_dir / f"model_random_forest_{version_b}.pkl"

        if not model_a_path.exists() or not model_b_path.exists():
            raise FileNotFoundError("One or both model versions not found")

        # Create A/B test configuration
        ab_config = {
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "start_time": datetime.now(tz=UTC).isoformat(),
            "duration_hours": duration_hours,
            "metrics": {
                "version_a": {"requests": 0, "approvals": 0},
                "version_b": {"requests": 0, "approvals": 0},
            },
        }

        # Save configuration
        with Path(self.ab_test_config_file).open("w") as f:
            json.dump(ab_config, f, indent=2)

        logger.info(
            f"A/B test configured: {version_a} ({traffic_split * 100:.0f}%) vs "
            f"{version_b} ({(1 - traffic_split) * 100:.0f}%)"
        )

        return ab_config

    def get_ab_test_status(self) -> dict[str, Any] | None:
        """Get current A/B test status and metrics."""
        if not self.ab_test_config_file.exists():
            return None

        with self.ab_test_config_file.open() as f:
            config: dict[str, Any] = json.load(f)

        # Calculate elapsed time
        start_time = datetime.fromisoformat(config["start_time"])
        elapsed_hours = (datetime.now(tz=UTC) - start_time).total_seconds() / 3600

        config["elapsed_hours"] = elapsed_hours
        config["is_active"] = elapsed_hours < config["duration_hours"]

        # Calculate approval rates
        for version in ["version_a", "version_b"]:
            metrics = config["metrics"][version]
            if metrics["requests"] > 0:
                metrics["approval_rate"] = metrics["approvals"] / metrics["requests"]
            else:
                metrics["approval_rate"] = 0

        return config

    def update_ab_metrics(self, version: str, approved: bool) -> None:
        """Update A/B test metrics for a prediction."""
        if not self.ab_test_config_file.exists():
            return

        with self.ab_test_config_file.open() as f:
            config = json.load(f)

        # Update metrics
        if version == config["version_a"]:
            config["metrics"]["version_a"]["requests"] += 1
            if approved:
                config["metrics"]["version_a"]["approvals"] += 1
        elif version == config["version_b"]:
            config["metrics"]["version_b"]["requests"] += 1
            if approved:
                config["metrics"]["version_b"]["approvals"] += 1

        # Save updated configuration
        with Path(self.ab_test_config_file).open("w") as f:
            json.dump(config, f, indent=2)

    def cleanup_old_models(self, keep_last: int = 5) -> int:
        """Remove old model versions, keeping the most recent ones."""
        models = self.list_models()

        if len(models) <= keep_last:
            return 0

        # Keep deployed model and most recent ones
        models_to_delete = models[keep_last:]
        deleted_count = 0

        for model in models_to_delete:
            if model.status != ModelStatus.DEPLOYED:
                # Delete model files
                model_path = Path(model.file_path)
                metadata_path = self.model_dir / f"metadata_{model.version}.json"

                if model_path.exists():
                    model_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()

                deleted_count += 1
                logger.info(f"Deleted model version {model.version}")

        return deleted_count

    def _archive_current_deployment(self) -> None:
        """Archive the currently deployed model."""
        deployed_model = self.deployment_dir / "model.pkl"
        deployed_metadata = self.deployment_dir / "metadata.json"

        if deployed_model.exists() and deployed_metadata.exists():
            # Get version from metadata
            with deployed_metadata.open() as f:
                metadata = json.load(f)
            version = metadata["version"]

            # Create archive directory
            archive_dir = self.deployment_dir / "archive" / version
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Move files to archive
            shutil.move(str(deployed_model), str(archive_dir / "model.pkl"))
            shutil.move(str(deployed_metadata), str(archive_dir / "metadata.json"))

            logger.info(f"Archived deployment version {version}")

    def _update_deployment_history(self, version: str, metadata: dict[str, Any]) -> None:
        """Update deployment history log."""
        history = self._get_deployment_history()

        history.append(
            {
                "version": version,
                "deployed_at": datetime.now(tz=UTC).isoformat(),
                "accuracy": metadata["training_metrics"].get("accuracy", 0),
                "f1_score": metadata["training_metrics"].get("f1_score", 0),
            }
        )

        # Keep last 50 deployments
        history = history[-50:]

        with Path(self.deployment_history_file).open("w") as f:
            json.dump(history, f, indent=2)

    def _get_deployment_history(self) -> list[dict[str, Any]]:
        """Get deployment history."""
        if not self.deployment_history_file.exists():
            return []

        with self.deployment_history_file.open() as f:
            history: list[dict[str, Any]] = json.load(f)
            return history
