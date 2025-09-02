"""ML model management API routers."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field

from services.file_rename_service.app.ml.models import FeedbackData, ModelAlgorithm, TrainingData
from services.file_rename_service.app.ml.predictor import Predictor
from services.file_rename_service.app.ml.trainer import Trainer
from services.file_rename_service.app.ml.versioning import ModelVersionManager

logger = logging.getLogger(__name__)

# Create router
ml_router = APIRouter(prefix="/model", tags=["ML Model Management"])

# Initialize ML components
trainer = Trainer()
predictor = Predictor()
version_manager = ModelVersionManager()


# Pydantic models for API
class TrainRequest(BaseModel):
    """Request model for training."""

    algorithm: str = Field(default="random_forest", description="ML algorithm to use")
    hyperparameters: dict[str, Any] | None = Field(default=None, description="Model hyperparameters")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    validation_size: float = Field(default=0.1, ge=0.1, le=0.3, description="Validation set size")


class TrainResponse(BaseModel):
    """Response model for training."""

    job_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""

    job_id: str
    status: str
    progress: float
    metrics: dict[str, float] | None = None
    error: str | None = None


class PredictRequest(BaseModel):
    """Request model for prediction."""

    filename: str
    tokens: list[dict[str, Any]]
    use_cache: bool = True
    return_probabilities: bool = False
    top_k: int = 3


class PredictResponse(BaseModel):
    """Response model for prediction."""

    filename_original: str
    predictions: list[dict[str, Any]]
    inference_time_ms: float
    model_version: str


class FeedbackRequest(BaseModel):
    """Request model for feedback."""

    prediction_id: str
    filename_original: str
    suggested_name: str
    actual_name: str
    user_approved: bool
    weight: float = 1.0


class FeedbackResponse(BaseModel):
    """Response model for feedback."""

    message: str
    feedback_count: int


class ModelDeployRequest(BaseModel):
    """Request model for model deployment."""

    version: str
    force: bool = False


class ModelDeployResponse(BaseModel):
    """Response model for model deployment."""

    success: bool
    version: str
    message: str


class ModelRollbackRequest(BaseModel):
    """Request model for model rollback."""

    target_version: str | None = None


class ABTestRequest(BaseModel):
    """Request model for A/B test setup."""

    version_a: str
    version_b: str
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_hours: int = Field(default=24, ge=1, le=168)


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics."""

    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_avg_ms: float
    sample_count: int


# Global training jobs tracker (in production, use Redis or database)
training_jobs: dict[str, dict[str, Any]] = {}


async def run_training_job(job_id: str, training_data: list[TrainingData], request: TrainRequest) -> None:
    """Background task to run model training."""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 0.1

        # Convert algorithm string to enum
        algorithm = ModelAlgorithm(request.algorithm)

        # Run training
        model_metadata = trainer.train(
            training_data=training_data,
            algorithm=algorithm,
            hyperparameters=request.hyperparameters,
            test_size=request.test_size,
            validation_size=request.validation_size,
        )

        # Update job status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 1.0
        training_jobs[job_id]["metrics"] = model_metadata.training_metrics
        training_jobs[job_id]["model_version"] = model_metadata.version

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


@ml_router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    """
    Start a model training job.

    Initiates an asynchronous training job with the specified algorithm
    and hyperparameters. Returns a job ID for tracking progress.
    """
    try:
        # Generate job ID
        job_id = f"train_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"

        # TODO: Load training data from database
        # For now, create mock training data
        training_data = [
            TrainingData(
                filename_original=f"file_{i}.txt",
                filename_renamed=f"renamed_file_{i}.txt",
                tokens=[
                    {"type": "word", "value": "file"},
                    {"type": "separator", "value": "_"},
                    {"type": "number", "value": str(i)},
                    {"type": "extension", "value": "txt"},
                ],
                user_approved=True,
                confidence_score=0.8,
            )
            for i in range(100)
        ]

        # Initialize job tracking
        training_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "started_at": datetime.now(tz=UTC).isoformat(),
        }

        # Start background training
        background_tasks.add_task(run_training_job, job_id, training_data, request)

        return TrainResponse(
            job_id=job_id,
            status="started",
            message="Training job started successfully",
        )

    except Exception as e:
        logger.error(f"Error starting training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """
    Get the status of a training job.

    Returns the current status, progress, and metrics (if completed)
    for the specified training job.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )

    job = training_jobs[job_id]
    return TrainingStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        metrics=job.get("metrics"),
        error=job.get("error"),
    )


@ml_router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Get rename predictions for a filename.

    Uses the currently deployed model to generate rename suggestions
    for the provided filename and tokens.
    """
    try:
        result = predictor.predict(
            filename=request.filename,
            tokens=request.tokens,
            use_cache=request.use_cache,
            return_probabilities=request.return_probabilities,
            top_k=request.top_k,
        )

        return PredictResponse(**result)

    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model available",
        ) from err
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.post("/feedback", response_model=FeedbackResponse)
async def submit_model_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit feedback on model predictions.

    Provides user feedback on model predictions to improve future
    suggestions through online learning.
    """
    try:
        feedback = FeedbackData(
            prediction_id=request.prediction_id,
            filename_original=request.filename_original,
            suggested_name=request.suggested_name,
            actual_name=request.actual_name,
            user_approved=request.user_approved,
            weight=request.weight,
        )

        predictor.add_feedback(feedback)

        return FeedbackResponse(
            message="Feedback received successfully",
            feedback_count=len(predictor.feedback_buffer),
        )

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.post("/deploy", response_model=ModelDeployResponse)
async def deploy_model(request: ModelDeployRequest) -> ModelDeployResponse:
    """
    Deploy a specific model version.

    Deploys the specified model version to production, replacing
    the currently deployed model.
    """
    try:
        success = version_manager.deploy_model(request.version, request.force)

        if success:
            # Reload model in predictor
            predictor.load_specific_model(request.version)

            return ModelDeployResponse(
                success=True,
                version=request.version,
                message=f"Model version {request.version} deployed successfully",
            )
        return ModelDeployResponse(
            success=False,
            version=request.version,
            message="Deployment failed - check model metrics",
        )

    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.post("/rollback", response_model=ModelDeployResponse)
async def rollback_model(request: ModelRollbackRequest) -> ModelDeployResponse:
    """
    Rollback to a previous model version.

    Rolls back to the specified version or the previous deployment
    if no version is specified.
    """
    try:
        success = version_manager.rollback_model(request.target_version)

        if success:
            # Reload model in predictor
            if request.target_version:
                predictor.load_specific_model(request.target_version)
            else:
                predictor.load_latest_model()

            return ModelDeployResponse(
                success=True,
                version=request.target_version or "previous",
                message="Model rolled back successfully",
            )
        return ModelDeployResponse(
            success=False,
            version=request.target_version or "previous",
            message="Rollback failed",
        )

    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.post("/ab-test", response_model=dict[str, Any])
async def setup_ab_test(request: ABTestRequest) -> dict[str, Any]:
    """
    Setup A/B testing between two model versions.

    Configures A/B testing to compare performance between two
    model versions with specified traffic split.
    """
    try:
        return version_manager.setup_ab_test(
            version_a=request.version_a,
            version_b=request.version_b,
            traffic_split=request.traffic_split,
            duration_hours=request.duration_hours,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Error setting up A/B test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.get("/ab-test/status", response_model=dict[str, Any] | None)
async def get_ab_test_status() -> dict[str, Any] | None:
    """
    Get current A/B test status.

    Returns the current A/B test configuration and metrics,
    or null if no test is active.
    """
    try:
        return version_manager.get_ab_test_status()

    except Exception as e:
        logger.error(f"Error getting A/B test status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    version: str | None = Query(default=None, description="Model version (latest if not specified)"),
) -> ModelMetricsResponse:
    """
    Get performance metrics for a model.

    Returns detailed performance metrics for the specified model version
    or the currently deployed model.
    """
    try:
        if version:
            models_list = version_manager.list_models()
            model = next((m for m in models_list if m.version == version), None)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version {version} not found",
                )
            # Convert MLModel to dict for consistent handling
            model_dict = model.to_dict()
        else:
            model_info = predictor.get_model_info()
            if not model_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No model currently loaded",
                )
            model_dict = model_info

        metrics = model_dict.get("training_metrics", {})

        return ModelMetricsResponse(
            version=model_dict.get("version", ""),
            accuracy=metrics.get("accuracy", 0),
            precision=metrics.get("precision", 0),
            recall=metrics.get("recall", 0),
            f1_score=metrics.get("f1_score", 0),
            inference_time_avg_ms=30.0,  # TODO: Track actual inference times
            sample_count=model_dict.get("sample_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@ml_router.get("/versions", response_model=list[dict[str, Any]])
async def list_model_versions() -> list[dict[str, Any]]:
    """
    List all available model versions.

    Returns a list of all trained model versions with their
    metadata and performance metrics.
    """
    try:
        models = version_manager.list_models()
        return [model.to_dict() for model in models]

    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
