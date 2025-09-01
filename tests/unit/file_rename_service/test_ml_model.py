"""Unit tests for ML model components."""

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from services.file_rename_service.app.ml.features import FeatureExtractor
from services.file_rename_service.app.ml.models import (
    FeedbackData,
    MLModel,
    ModelAlgorithm,
    ModelMetrics,
    ModelStatus,
    TrainingData,
)
from services.file_rename_service.app.ml.predictor import Predictor
from services.file_rename_service.app.ml.trainer import Trainer
from services.file_rename_service.app.ml.versioning import ModelVersionManager


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_init_default_config(self):
        """Test feature extractor initialization with default config."""
        extractor = FeatureExtractor()
        assert extractor.max_sequence_length == 50
        assert extractor.embedding_dim == 100
        assert not extractor.use_embeddings

    def test_init_custom_config(self):
        """Test feature extractor initialization with custom config."""
        config = {
            "max_sequence_length": 100,
            "embedding_dim": 200,
            "use_embeddings": True,
        }
        extractor = FeatureExtractor(config)
        assert extractor.max_sequence_length == 100
        assert extractor.embedding_dim == 200
        assert extractor.use_embeddings

    def test_fit_vocabulary(self):
        """Test vocabulary building from training data."""
        extractor = FeatureExtractor()
        training_data = [
            {
                "tokens": [
                    {"type": "word", "value": "file"},
                    {"type": "number", "value": "123"},
                ],
            },
            {
                "tokens": [
                    {"type": "word", "value": "document"},
                    {"type": "extension", "value": "pdf"},
                ],
            },
        ]

        extractor.fit(training_data)

        assert len(extractor.vocabulary) > 4  # Special tokens + actual tokens
        assert "<PAD>" in extractor.vocabulary
        assert "<UNK>" in extractor.vocabulary
        assert "word:file" in extractor.vocabulary

    def test_encode_tokens(self):
        """Test token encoding to integer sequences."""
        extractor = FeatureExtractor()
        training_data = [
            {
                "tokens": [
                    {"type": "word", "value": "test"},
                    {"type": "number", "value": "42"},
                ],
            }
        ]

        extractor.fit(training_data)

        tokens = [
            {"type": "word", "value": "test"},
            {"type": "number", "value": "42"},
        ]

        encoded = extractor._encode_tokens(tokens)

        assert len(encoded) == extractor.max_sequence_length
        assert encoded[0] == extractor.vocabulary["<START>"]
        assert encoded[-1] == extractor.vocabulary["<PAD>"] or encoded[-2] == extractor.vocabulary["<END>"]

    def test_extract_statistics(self):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor()
        tokens = [
            {"type": "word", "value": "hello"},
            {"type": "separator", "value": "_"},
            {"type": "number", "value": "123"},
            {"type": "extension", "value": "txt"},
        ]

        stats = extractor._extract_statistics(tokens, "hello_123.txt")

        assert stats["num_words"] == 1
        assert stats["num_numbers"] == 1
        assert stats["num_separators"] == 1
        assert stats["num_extensions"] == 1
        assert stats["filename_length"] == len("hello_123.txt")
        assert stats["has_snakecase"] == 1.0

    def test_extract_features_complete(self):
        """Test complete feature extraction pipeline."""
        extractor = FeatureExtractor()
        tokens = [
            {"type": "word", "value": "report"},
            {"type": "date", "value": "2024-01-15"},
            {"type": "extension", "value": "pdf"},
        ]

        features = extractor.extract_features(tokens, "report_2024-01-15.pdf")

        assert "token_encoding" in features
        assert "num_words" in features
        assert "num_dates" in features
        assert "extension" in features
        assert features["has_date"] is True


class TestTrainer:
    """Test model training functionality."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return [
            TrainingData(
                filename_original=f"file_{i}.txt",
                filename_renamed=f"renamed_{i % 5}.txt",  # 5 different classes
                tokens=[
                    {"type": "word", "value": "file"},
                    {"type": "separator", "value": "_"},
                    {"type": "number", "value": str(i)},
                    {"type": "extension", "value": "txt"},
                ],
                user_approved=True,
                confidence_score=0.8,
            )
            for i in range(50)
        ]

    def test_trainer_init(self, temp_model_dir):
        """Test trainer initialization."""
        trainer = Trainer(temp_model_dir)
        assert trainer.model_dir.exists()
        assert trainer.model is None

    def test_prepare_training_data(self, temp_model_dir, sample_training_data):
        """Test training data preparation."""
        trainer = Trainer(temp_model_dir)
        X, y = trainer._prepare_training_data(sample_training_data)  # noqa: N806

        assert X.shape[0] == len(sample_training_data)
        assert y.shape[0] == len(sample_training_data)
        assert len(np.unique(y)) == 5  # 5 different classes

    def test_create_random_forest(self, temp_model_dir):
        """Test Random Forest model creation."""
        trainer = Trainer(temp_model_dir)
        hyperparams = {"n_estimators": 50, "max_depth": 5}

        model = trainer._create_random_forest(hyperparams)

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_train_model(self, temp_model_dir, sample_training_data):
        """Test complete model training pipeline."""
        trainer = Trainer(temp_model_dir)

        model_metadata = trainer.train(
            training_data=sample_training_data,
            algorithm=ModelAlgorithm.RANDOM_FOREST,
            hyperparameters={"n_estimators": 10},  # Small for testing
            test_size=0.3,
            validation_size=0.2,
        )

        assert model_metadata.algorithm == ModelAlgorithm.RANDOM_FOREST
        assert model_metadata.status == ModelStatus.TRAINING
        assert model_metadata.training_duration > 0
        assert model_metadata.sample_count == len(sample_training_data)
        assert "accuracy" in model_metadata.training_metrics
        assert Path(model_metadata.file_path).exists()

    def test_model_evaluation_metrics(self, temp_model_dir, sample_training_data):
        """Test model evaluation metrics calculation."""
        trainer = Trainer(temp_model_dir)

        model_metadata = trainer.train(
            training_data=sample_training_data,
            algorithm=ModelAlgorithm.RANDOM_FOREST,
            hyperparameters={"n_estimators": 10},
        )

        metrics = model_metadata.training_metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert "confusion_matrix" in metrics

    def test_save_and_load_model(self, temp_model_dir, sample_training_data):
        """Test model saving and loading."""
        trainer = Trainer(temp_model_dir)

        # Train and save model
        model_metadata = trainer.train(
            training_data=sample_training_data,
            algorithm=ModelAlgorithm.RANDOM_FOREST,
        )

        # Load model
        loaded_metadata = trainer.load_model(model_metadata.file_path)

        assert loaded_metadata.id == model_metadata.id
        assert loaded_metadata.version == model_metadata.version
        assert trainer.model is not None

    @pytest.mark.benchmark
    def test_training_performance(self, temp_model_dir):
        """Test training performance requirements."""
        # Create 10,000 samples
        large_training_data = [
            TrainingData(
                filename_original=f"file_{i}.txt",
                filename_renamed=f"renamed_{i % 20}.txt",
                tokens=[
                    {"type": "word", "value": f"word_{i % 10}"},
                    {"type": "number", "value": str(i)},
                ],
                user_approved=True,
                confidence_score=0.8,
            )
            for i in range(10000)
        ]

        trainer = Trainer(temp_model_dir)

        start_time = time.time()
        model_metadata = trainer.train(
            training_data=large_training_data,
            algorithm=ModelAlgorithm.RANDOM_FOREST,
            hyperparameters={"n_estimators": 50},
        )
        training_time = time.time() - start_time

        # Should complete in less than 5 minutes (300 seconds)
        assert training_time < 300, f"Training took {training_time:.2f}s, expected <300s"

        # Should achieve >70% accuracy
        assert model_metadata.training_metrics["accuracy"] > 0.7


class TestPredictor:
    """Test model prediction functionality."""

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create a trained model for testing."""
        trainer = Trainer(str(tmp_path))

        training_data = [
            TrainingData(
                filename_original=f"file_{i}.txt",
                filename_renamed=f"renamed_{i % 3}.txt",
                tokens=[
                    {"type": "word", "value": "file"},
                    {"type": "number", "value": str(i % 10)},
                    {"type": "extension", "value": "txt"},
                ],
                user_approved=True,
                confidence_score=0.8,
            )
            for i in range(100)
        ]

        model_metadata = trainer.train(training_data)
        return tmp_path, model_metadata

    def test_predictor_init(self, tmp_path):
        """Test predictor initialization."""
        predictor = Predictor(str(tmp_path))
        assert not predictor.model_loaded
        assert len(predictor.prediction_cache) == 0

    def test_load_latest_model(self, trained_model):
        """Test loading the latest model."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))

        model_metadata = predictor.load_latest_model()

        assert predictor.model_loaded
        assert model_metadata is not None

    def test_predict_single(self, trained_model):
        """Test single filename prediction."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))
        predictor.load_latest_model()

        result = predictor.predict(
            filename="test_file.txt",
            tokens=[
                {"type": "word", "value": "test"},
                {"type": "word", "value": "file"},
                {"type": "extension", "value": "txt"},
            ],
        )

        assert result["filename_original"] == "test_file.txt"
        assert len(result["predictions"]) > 0
        assert "suggested_name" in result["predictions"][0]
        assert "confidence" in result["predictions"][0]
        assert result["inference_time_ms"] < 50  # Should be <50ms

    def test_predict_with_cache(self, trained_model):
        """Test prediction caching."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))
        predictor.load_latest_model()

        tokens = [{"type": "word", "value": "test"}]

        # First prediction
        result1 = predictor.predict("test.txt", tokens, use_cache=True)
        time1 = result1["inference_time_ms"]

        # Second prediction (should be cached)
        result2 = predictor.predict("test.txt", tokens, use_cache=True)

        assert result2["predictions"] == result1["predictions"]
        # Cache hit should be instant (very low inference time)
        assert result2["inference_time_ms"] <= time1

    def test_predict_batch(self, trained_model):
        """Test batch prediction."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))
        predictor.load_latest_model()

        samples = [
            ("file1.txt", [{"type": "word", "value": "file1"}]),
            ("file2.txt", [{"type": "word", "value": "file2"}]),
            ("file3.txt", [{"type": "word", "value": "file3"}]),
        ]

        results = predictor.predict_batch(samples)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["filename_original"] == samples[i][0]

    def test_feedback_buffer(self, trained_model):
        """Test feedback buffering."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))

        feedback = FeedbackData(
            prediction_id="test_123",
            filename_original="test.txt",
            suggested_name="renamed.txt",
            actual_name="actual.txt",
            user_approved=True,
        )

        predictor.add_feedback(feedback)

        assert len(predictor.feedback_buffer) == 1
        assert predictor.feedback_buffer[0] == feedback

    @pytest.mark.benchmark
    def test_inference_performance(self, trained_model):
        """Test inference performance requirements."""
        model_dir, _ = trained_model
        predictor = Predictor(str(model_dir))
        predictor.load_latest_model()

        tokens = [
            {"type": "word", "value": "performance"},
            {"type": "word", "value": "test"},
            {"type": "extension", "value": "txt"},
        ]

        # Run multiple predictions
        inference_times = []
        for i in range(100):
            result = predictor.predict(f"file_{i}.txt", tokens, use_cache=False)
            inference_times.append(result["inference_time_ms"])

        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)

        # Average should be well below 50ms
        assert avg_inference_time < 50, f"Avg inference time {avg_inference_time:.2f}ms, expected <50ms"
        # Max should not exceed 100ms
        assert max_inference_time < 100, f"Max inference time {max_inference_time:.2f}ms, expected <100ms"


class TestVersionManager:
    """Test model versioning functionality."""

    @pytest.fixture
    def version_manager(self, tmp_path):
        """Create version manager with temp directories."""
        model_dir = tmp_path / "models"
        deployment_dir = tmp_path / "deployed"
        return ModelVersionManager(str(model_dir), str(deployment_dir))

    @pytest.fixture
    def sample_models(self, version_manager):
        """Create sample model files."""
        models = []
        for i in range(3):
            version = f"2024010{i}_120000"
            model_path = version_manager.model_dir / f"model_random_forest_{version}.pkl"
            metadata_path = version_manager.model_dir / f"metadata_{version}.json"

            # Create dummy files
            model_path.write_text("dummy model")

            metadata = {
                "id": f"model_{i}",
                "version": version,
                "algorithm": "random_forest",
                "created_at": f"2024-01-0{i + 1}T12:00:00",
                "training_metrics": {
                    "accuracy": 0.75 + i * 0.05,
                    "f1_score": 0.73 + i * 0.05,
                },
                "hyperparameters": {},
                "feature_config": {},
                "status": "training",
                "file_path": str(model_path),
            }

            with Path(metadata_path).open("w") as f:
                json.dump(metadata, f)

            models.append(version)

        return models

    def test_list_models(self, version_manager, sample_models):
        """Test listing available models."""
        models = version_manager.list_models()

        assert len(models) == 3
        # Should be sorted by creation date (newest first)
        assert models[0].version == sample_models[2]

    def test_deploy_model(self, version_manager, sample_models):
        """Test model deployment."""
        version = sample_models[2]  # Best accuracy

        success = version_manager.deploy_model(version)

        assert success
        assert (version_manager.deployment_dir / "model.pkl").exists()
        assert (version_manager.deployment_dir / "metadata.json").exists()

    def test_deploy_model_accuracy_threshold(self, version_manager, sample_models):
        """Test deployment fails if accuracy below threshold."""
        # Create a low-accuracy model
        version = "20240104_120000"
        model_path = version_manager.model_dir / f"model_random_forest_{version}.pkl"
        metadata_path = version_manager.model_dir / f"metadata_{version}.json"

        model_path.write_text("dummy model")

        metadata = {
            "id": "low_accuracy",
            "version": version,
            "algorithm": "random_forest",
            "created_at": "2024-01-04T12:00:00",
            "training_metrics": {"accuracy": 0.65},  # Below 0.7 threshold
            "hyperparameters": {},
            "feature_config": {},
            "status": "training",
            "file_path": str(model_path),
        }

        with Path(metadata_path).open("w") as f:
            json.dump(metadata, f)

        success = version_manager.deploy_model(version, force=False)
        assert not success

        # Should succeed with force=True
        success = version_manager.deploy_model(version, force=True)
        assert success

    def test_rollback_model(self, version_manager, sample_models):
        """Test model rollback."""
        # Deploy first model
        version_manager.deploy_model(sample_models[0], force=True)
        # Deploy second model
        version_manager.deploy_model(sample_models[1], force=True)

        # Rollback to previous
        success = version_manager.rollback_model()

        assert success
        metadata_path = version_manager.deployment_dir / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)
        assert metadata["version"] == sample_models[0]

    def test_ab_test_setup(self, version_manager, sample_models):
        """Test A/B test configuration."""
        config = version_manager.setup_ab_test(
            version_a=sample_models[0],
            version_b=sample_models[1],
            traffic_split=0.6,
            duration_hours=12,
        )

        assert config["version_a"] == sample_models[0]
        assert config["version_b"] == sample_models[1]
        assert config["traffic_split"] == 0.6
        assert config["duration_hours"] == 12

    def test_ab_test_metrics_update(self, version_manager, sample_models):
        """Test A/B test metrics updating."""
        version_manager.setup_ab_test(
            version_a=sample_models[0],
            version_b=sample_models[1],
        )

        # Update metrics
        version_manager.update_ab_metrics(sample_models[0], approved=True)
        version_manager.update_ab_metrics(sample_models[0], approved=False)
        version_manager.update_ab_metrics(sample_models[1], approved=True)

        status = version_manager.get_ab_test_status()

        assert status["metrics"]["version_a"]["requests"] == 2
        assert status["metrics"]["version_a"]["approvals"] == 1
        assert status["metrics"]["version_b"]["requests"] == 1
        assert status["metrics"]["version_b"]["approvals"] == 1

    def test_cleanup_old_models(self, version_manager, sample_models):
        """Test cleanup of old model versions."""
        # Create more models
        for i in range(3, 8):
            version = f"2024010{i}_120000"
            model_path = version_manager.model_dir / f"model_random_forest_{version}.pkl"
            metadata_path = version_manager.model_dir / f"metadata_{version}.json"

            model_path.write_text("dummy model")
            metadata = {
                "id": f"model_{i}",
                "version": version,
                "algorithm": "random_forest",
                "created_at": f"2024-01-0{i + 1}T12:00:00",
                "training_metrics": {"accuracy": 0.75},
                "hyperparameters": {},
                "feature_config": {},
                "status": "training",
                "file_path": str(model_path),
            }
            with Path(metadata_path).open("w") as f:
                json.dump(metadata, f)

        # Should have 8 models total
        assert len(version_manager.list_models()) == 8

        # Cleanup keeping only 5
        deleted_count = version_manager.cleanup_old_models(keep_last=5)

        assert deleted_count == 3
        assert len(version_manager.list_models()) == 5


class TestModels:
    """Test model data structures."""

    def test_ml_model_to_dict(self):
        """Test MLModel serialization."""
        model = MLModel(
            id="test_id",
            version="v1.0",
            algorithm=ModelAlgorithm.RANDOM_FOREST,
            created_at=datetime.now(tz=UTC),
            training_metrics={"accuracy": 0.85},
            hyperparameters={"n_estimators": 100},
            feature_config={"max_length": 50},
            status=ModelStatus.DEPLOYED,
            file_path="/path/to/model.pkl",
        )

        model_dict = model.to_dict()

        assert model_dict["id"] == "test_id"
        assert model_dict["algorithm"] == "random_forest"
        assert model_dict["status"] == "deployed"

    def test_training_data_to_dict(self):
        """Test TrainingData serialization."""
        data = TrainingData(
            filename_original="test.txt",
            filename_renamed="renamed.txt",
            tokens=[{"type": "word", "value": "test"}],
            user_approved=True,
            confidence_score=0.9,
        )

        data_dict = data.to_dict()

        assert data_dict["filename_original"] == "test.txt"
        assert data_dict["user_approved"] is True
        assert data_dict["confidence_score"] == 0.9

    def test_model_metrics_to_dict(self):
        """Test ModelMetrics serialization."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            confusion_matrix=[[10, 2], [1, 15]],
            per_category_metrics={"class1": {"precision": 0.9, "recall": 0.85, "f1_score": 0.87}},
            validation_samples=100,
            test_samples=50,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["accuracy"] == 0.85
        assert metrics_dict["confusion_matrix"] == [[10, 2], [1, 15]]
        assert "class1" in metrics_dict["per_category_metrics"]
