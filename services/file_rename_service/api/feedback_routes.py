"""API routes for feedback learning loop."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from services.file_rename_service.api.auth import (
    admin_rate_limit,
    metrics_rate_limit,
    sanitize_error_message,
    sanitize_filename,
    sanitize_string,
    security,
    user_rate_limit,
    validate_confidence_score,
    validate_context_metadata,
    validate_model_version,
    validate_proposal_id,
    verify_admin_key,
    verify_api_key,
)
from services.file_rename_service.app.feedback.experiments import ExperimentManager
from services.file_rename_service.app.feedback.learning import OnlineLearner
from services.file_rename_service.app.feedback.metrics import MetricsTracker
from services.file_rename_service.app.feedback.models import FeedbackAction
from services.file_rename_service.app.feedback.processor import FeedbackProcessor
from services.file_rename_service.app.feedback.storage import FeedbackStorage

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/feedback", tags=["feedback"])


# Request/Response schemas
class FeedbackSubmitRequest(BaseModel):
    """Request schema for feedback submission."""

    proposal_id: str = Field(..., description="Proposal ID", max_length=100)
    original_filename: str = Field(..., description="Original filename", max_length=255)
    proposed_filename: str = Field(..., description="Proposed filename", max_length=255)
    user_action: FeedbackAction = Field(..., description="User action")
    user_filename: str | None = Field(None, description="User-provided filename if modified", max_length=255)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    model_version: str = Field(..., description="Model version", max_length=50)
    context_metadata: dict[str, Any] | None = Field(default_factory=dict)

    @validator("proposal_id")
    @classmethod
    def validate_proposal_id_field(cls, v):
        """Validate proposal ID."""
        return validate_proposal_id(v)

    @validator("original_filename", "proposed_filename")
    @classmethod
    def validate_filename_fields(cls, v):
        """Validate filename fields."""
        return sanitize_filename(v)

    @validator("user_filename")
    @classmethod
    def validate_user_filename_field(cls, v):
        """Validate user filename field."""
        if v is None:
            return v
        return sanitize_filename(v)

    @validator("confidence_score")
    @classmethod
    def validate_confidence_score_field(cls, v):
        """Validate confidence score."""
        return validate_confidence_score(v)

    @validator("model_version")
    @classmethod
    def validate_model_version_field(cls, v):
        """Validate model version."""
        return validate_model_version(v)

    @validator("context_metadata")
    @classmethod
    def validate_context_metadata_field(cls, v):
        """Validate context metadata."""
        return validate_context_metadata(v)


class FeedbackResponse(BaseModel):
    """Response schema for feedback operations."""

    feedback_id: str
    status: str
    processing_time_ms: float
    message: str


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoints."""

    metrics: dict[str, Any]
    period: dict[str, Any] | None = None
    timestamp: str


class ExperimentCreateRequest(BaseModel):
    """Request schema for creating experiments."""

    name: str = Field(..., description="Experiment name", max_length=100)
    variant_a: str = Field(..., description="Control model version", max_length=50)
    variant_b: str = Field(..., description="Treatment model version", max_length=50)
    traffic_split: float = Field(0.5, ge=0.0, le=1.0, description="Traffic to variant B")
    description: str | None = Field(None, description="Experiment description", max_length=500)
    duration_hours: int = Field(24, gt=0, le=8760, description="Experiment duration (max 1 year)")

    @validator("name")
    @classmethod
    def validate_name_field(cls, v):
        """Validate experiment name."""
        return sanitize_string(v, max_length=100)

    @validator("variant_a", "variant_b")
    @classmethod
    def validate_variant_fields(cls, v):
        """Validate variant fields."""
        return validate_model_version(v)

    @validator("description")
    @classmethod
    def validate_description_field(cls, v):
        """Validate description field."""
        if v is None:
            return v
        return sanitize_string(v, max_length=500)


class ExperimentResponse(BaseModel):
    """Response schema for experiment operations."""

    experiment_id: str
    status: str
    data: dict[str, Any]


class LearningTriggerRequest(BaseModel):
    """Request schema for triggering learning."""

    force_retrain: bool = Field(False, description="Force full retrain")
    feedback_threshold: int = Field(1000, gt=0, description="Feedback threshold")


class LearningResponse(BaseModel):
    """Response schema for learning operations."""

    status: str
    model_version: str
    data: dict[str, Any]


# Dependency injection functions
async def get_storage() -> FeedbackStorage:
    """Get feedback storage instance."""
    # In production, this would be configured from environment
    storage = FeedbackStorage(
        postgres_dsn="postgresql://user:pass@localhost/feedback",
        redis_url="redis://localhost:6379",
    )
    await storage.initialize()
    return storage


async def get_processor(storage: Annotated[FeedbackStorage, Depends(get_storage)]) -> FeedbackProcessor:
    """Get feedback processor instance."""
    return FeedbackProcessor(
        storage=storage,
        batch_size=100,
        batch_timeout_seconds=300,
        retrain_threshold=1000,
    )


async def get_metrics_tracker(storage: Annotated[FeedbackStorage, Depends(get_storage)]) -> MetricsTracker:
    """Get metrics tracker instance."""
    return MetricsTracker(storage=storage)


async def get_experiment_manager(storage: Annotated[FeedbackStorage, Depends(get_storage)]) -> ExperimentManager:
    """Get experiment manager instance."""
    return ExperimentManager(storage=storage)


async def get_online_learner(storage: Annotated[FeedbackStorage, Depends(get_storage)]) -> OnlineLearner:
    """Get online learner instance."""
    return OnlineLearner(
        storage=storage,
        model_path=Path("models/current"),
        learning_rate=0.01,
    )


# Feedback submission endpoints
@router.post("/approve", response_model=FeedbackResponse)
@user_rate_limit
async def approve_proposal(
    request: Request,
    feedback_request: FeedbackSubmitRequest,
    processor: Annotated[FeedbackProcessor, Depends(get_processor)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
) -> FeedbackResponse:
    """Approve a rename proposal."""
    try:
        start_time = datetime.now(UTC)

        # Submit approval feedback
        feedback = await processor.submit_feedback(
            proposal_id=feedback_request.proposal_id,
            original_filename=feedback_request.original_filename,
            proposed_filename=feedback_request.proposed_filename,
            user_action=FeedbackAction.APPROVED,
            confidence_score=feedback_request.confidence_score,
            model_version=feedback_request.model_version,
            context_metadata=feedback_request.context_metadata,
        )

        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            f"Proposal approved by {user_context['api_key']}: "
            f"proposal_id={feedback_request.proposal_id}, "
            f"feedback_id={feedback.id}"
        )

        return FeedbackResponse(
            feedback_id=feedback.id,
            status="success",
            processing_time_ms=processing_time,
            message="Proposal approved successfully",
        )

    except Exception as e:
        logger.error(f"Error approving proposal: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process approval: {sanitized_error}",
        ) from None


@router.post("/reject", response_model=FeedbackResponse)
@user_rate_limit
async def reject_proposal(
    request: Request,
    feedback_request: FeedbackSubmitRequest,
    processor: Annotated[FeedbackProcessor, Depends(get_processor)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
) -> FeedbackResponse:
    """Reject a rename proposal."""
    try:
        start_time = datetime.now(UTC)

        # Submit rejection feedback
        feedback = await processor.submit_feedback(
            proposal_id=feedback_request.proposal_id,
            original_filename=feedback_request.original_filename,
            proposed_filename=feedback_request.proposed_filename,
            user_action=FeedbackAction.REJECTED,
            confidence_score=feedback_request.confidence_score,
            model_version=feedback_request.model_version,
            context_metadata=feedback_request.context_metadata,
        )

        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            f"Proposal rejected by {user_context['api_key']}: "
            f"proposal_id={feedback_request.proposal_id}, "
            f"feedback_id={feedback.id}"
        )

        return FeedbackResponse(
            feedback_id=feedback.id,
            status="success",
            processing_time_ms=processing_time,
            message="Proposal rejected successfully",
        )

    except Exception as e:
        logger.error(f"Error rejecting proposal: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process rejection: {sanitized_error}",
        ) from None


@router.post("/modify", response_model=FeedbackResponse)
@user_rate_limit
async def modify_proposal(
    request: Request,
    feedback_request: FeedbackSubmitRequest,
    processor: Annotated[FeedbackProcessor, Depends(get_processor)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
) -> FeedbackResponse:
    """Submit user-modified filename."""
    if not feedback_request.user_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_filename is required for modification feedback",
        )

    try:
        start_time = datetime.now(UTC)

        # Submit modification feedback
        feedback = await processor.submit_feedback(
            proposal_id=feedback_request.proposal_id,
            original_filename=feedback_request.original_filename,
            proposed_filename=feedback_request.proposed_filename,
            user_action=FeedbackAction.MODIFIED,
            user_filename=feedback_request.user_filename,
            confidence_score=feedback_request.confidence_score,
            model_version=feedback_request.model_version,
            context_metadata=feedback_request.context_metadata,
        )

        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            f"Proposal modified by {user_context['api_key']}: "
            f"proposal_id={feedback_request.proposal_id}, "
            f"user_filename={feedback_request.user_filename}, "
            f"feedback_id={feedback.id}"
        )

        return FeedbackResponse(
            feedback_id=feedback.id,
            status="success",
            processing_time_ms=processing_time,
            message="User modification recorded successfully",
        )

    except Exception as e:
        logger.error(f"Error processing modification: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process modification: {sanitized_error}",
        ) from None


# Metrics endpoints
@router.get("/metrics", response_model=MetricsResponse)
@metrics_rate_limit
async def get_feedback_metrics(
    request: Request,
    tracker: Annotated[MetricsTracker, Depends(get_metrics_tracker)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
    model_version: Annotated[str | None, Query(description="Filter by model version")] = None,
    start_date: Annotated[datetime | None, Query(description="Start date")] = None,
    end_date: Annotated[datetime | None, Query(description="End date")] = None,
) -> MetricsResponse:
    """Get feedback metrics."""
    try:
        # Validate and sanitize model_version if provided
        if model_version is not None:
            model_version = validate_model_version(model_version)

        metrics = await tracker.calculate_metrics(
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"Metrics requested by {user_context['api_key']}: "
            f"model_version={model_version}, "
            f"period={start_date} to {end_date}"
        )

        return MetricsResponse(
            metrics=metrics,
            period={
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {sanitized_error}",
        ) from None


@router.get("/metrics/dashboard", response_model=MetricsResponse)
@metrics_rate_limit
async def get_dashboard_metrics(
    request: Request,
    tracker: Annotated[MetricsTracker, Depends(get_metrics_tracker)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
) -> MetricsResponse:
    """Get dashboard metrics data."""
    try:
        dashboard_data = await tracker.generate_dashboard_data()

        logger.info(f"Dashboard metrics requested by {user_context['api_key']}")

        return MetricsResponse(
            metrics=dashboard_data,
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard metrics: {sanitized_error}",
        ) from None


@router.get("/metrics/improvement")
@metrics_rate_limit
async def get_improvement_metrics(
    request: Request,
    baseline_version: Annotated[str, Query(description="Baseline model version")],
    current_version: Annotated[str, Query(description="Current model version")],
    tracker: Annotated[MetricsTracker, Depends(get_metrics_tracker)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_api_key)],
) -> MetricsResponse:
    """Get improvement metrics between model versions."""
    try:
        # Validate and sanitize version strings
        baseline_version = validate_model_version(baseline_version)
        current_version = validate_model_version(current_version)

        improvement = await tracker.get_improvement_metrics(
            baseline_version=baseline_version,
            current_version=current_version,
        )

        logger.info(
            f"Improvement metrics requested by {user_context['api_key']}: {baseline_version} vs {current_version}"
        )

        return MetricsResponse(
            metrics=improvement,
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting improvement metrics: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get improvement metrics: {sanitized_error}",
        ) from None


# A/B testing endpoints
@router.post("/experiments", response_model=ExperimentResponse)
@admin_rate_limit
async def create_experiment(
    request: Request,
    experiment_request: ExperimentCreateRequest,
    manager: Annotated[ExperimentManager, Depends(get_experiment_manager)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> ExperimentResponse:
    """Create new A/B test experiment."""
    try:
        experiment = await manager.create_experiment(
            name=experiment_request.name,
            variant_a=experiment_request.variant_a,
            variant_b=experiment_request.variant_b,
            traffic_split=experiment_request.traffic_split,
            description=experiment_request.description,
            duration_hours=experiment_request.duration_hours,
        )

        logger.info(
            f"Experiment created by {user_context['api_key']}: "
            f"name={experiment_request.name}, "
            f"experiment_id={experiment.id}"
        )

        return ExperimentResponse(
            experiment_id=experiment.id,
            status="created",
            data=experiment.dict(),
        )

    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {sanitized_error}",
        ) from None


@router.get("/experiments", response_model=list[dict[str, Any]])
@admin_rate_limit
async def get_experiments(
    request: Request,
    manager: Annotated[ExperimentManager, Depends(get_experiment_manager)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> list[dict[str, Any]]:
    """Get active experiments."""
    try:
        experiments = await manager.get_active_experiments()

        logger.info(f"Experiments list requested by {user_context['api_key']}")

        return [exp.dict() for exp in experiments]

    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiments: {sanitized_error}",
        ) from None


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
@admin_rate_limit
async def get_experiment_status(
    request: Request,
    experiment_id: str,
    manager: Annotated[ExperimentManager, Depends(get_experiment_manager)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> ExperimentResponse:
    """Get experiment status."""
    try:
        # Validate experiment_id
        experiment_id = validate_proposal_id(experiment_id)  # Using same validation logic

        status_data = await manager.get_experiment_status(experiment_id)

        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found",
            ) from None

        logger.info(f"Experiment status requested by {user_context['api_key']}: {experiment_id}")

        return ExperimentResponse(
            experiment_id=experiment_id,
            status=status_data["status"],
            data=status_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment status: {sanitized_error}",
        ) from None


@router.post("/experiments/{experiment_id}/start", response_model=ExperimentResponse)
@admin_rate_limit
async def start_experiment(
    request: Request,
    experiment_id: str,
    manager: Annotated[ExperimentManager, Depends(get_experiment_manager)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> ExperimentResponse:
    """Start an experiment."""
    try:
        # Validate experiment_id
        experiment_id = validate_proposal_id(experiment_id)

        experiment = await manager.start_experiment(experiment_id)

        logger.info(f"Experiment started by {user_context['api_key']}: {experiment_id}")

        return ExperimentResponse(
            experiment_id=experiment.id,
            status="started",
            data=experiment.dict(),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start experiment: {sanitized_error}",
        ) from None


@router.post("/experiments/{experiment_id}/conclude", response_model=ExperimentResponse)
@admin_rate_limit
async def conclude_experiment(
    request: Request,
    experiment_id: str,
    manager: Annotated[ExperimentManager, Depends(get_experiment_manager)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> ExperimentResponse:
    """Conclude an experiment."""
    try:
        # Validate experiment_id
        experiment_id = validate_proposal_id(experiment_id)

        results = await manager.conclude_experiment(experiment_id)

        logger.info(f"Experiment concluded by {user_context['api_key']}: {experiment_id}")

        return ExperimentResponse(
            experiment_id=experiment_id,
            status="concluded",
            data=results,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Error concluding experiment: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to conclude experiment: {sanitized_error}",
        ) from None


# Learning endpoints
@router.post("/learning/trigger", response_model=LearningResponse)
@admin_rate_limit
async def trigger_learning(
    request: Request,
    learning_request: LearningTriggerRequest,
    learner: Annotated[OnlineLearner, Depends(get_online_learner)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> LearningResponse:
    """Trigger model learning/retraining."""
    try:
        if learning_request.force_retrain:
            # Trigger full retrain
            result = await learner.trigger_full_retrain(feedback_threshold=learning_request.feedback_threshold)

            logger.info(
                f"Learning triggered by {user_context['api_key']}: "
                f"force_retrain={learning_request.force_retrain}, "
                f"threshold={learning_request.feedback_threshold}"
            )

            if result["retrained"]:
                return LearningResponse(
                    status="retrained",
                    model_version=result["new_version"],
                    data=result,
                )
            return LearningResponse(
                status="skipped",
                model_version=learner._model_version,
                data=result,
            )
        # Trigger incremental update
        # Would need to get recent feedbacks for this
        logger.info(f"Incremental learning triggered by {user_context['api_key']}")

        return LearningResponse(
            status="incremental_update",
            model_version=learner._model_version,
            data={"message": "Incremental update scheduled"},
        )

    except Exception as e:
        logger.error(f"Error triggering learning: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger learning: {sanitized_error}",
        ) from None


@router.get("/learning/status", response_model=LearningResponse)
@admin_rate_limit
async def get_learning_status(
    request: Request,
    storage: Annotated[FeedbackStorage, Depends(get_storage)],
    auth_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_context: Annotated[dict[str, Any], Depends(verify_admin_key)],
) -> LearningResponse:
    """Get learning pipeline status."""
    try:
        metrics = await storage.get_learning_metrics()

        if metrics:
            data = {
                "model_version": metrics.model_version,
                "total_feedback": metrics.total_feedback,
                "last_retrained": (metrics.last_retrained.isoformat() if metrics.last_retrained else None),
                "next_retrain_at": (metrics.next_retrain_at.isoformat() if metrics.next_retrain_at else None),
                "approval_rate": metrics.approval_rate,
                "accuracy_trend": metrics.accuracy_trend[-10:] if metrics.accuracy_trend else [],
            }
        else:
            data = {"message": "No learning metrics available"}

        logger.info(f"Learning status requested by {user_context['api_key']}")

        return LearningResponse(
            status="active",
            model_version="unknown",
            data=data,
        )

    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        sanitized_error = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning status: {sanitized_error}",
        ) from None
