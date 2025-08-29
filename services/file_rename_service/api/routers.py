"""API routers for File Rename Service."""

import logging
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from services.file_rename_service.api.schemas import (
    ErrorResponse,
    PatternResponse,
    RenameAnalyzeRequest,
    RenameAnalyzeResponse,
    RenameFeedbackRequest,
    RenameFeedbackResponse,
    RenameHistoryResponse,
    RenameProposalRequest,
    RenameProposalResponse,
)
from services.file_rename_service.models.database import (
    Pattern,
    RenameHistory,
    UserFeedback,
    get_session_factory,
)
from services.file_rename_service.utils.rabbitmq import (
    MessageTopics,
    rabbitmq_manager,
)

logger = logging.getLogger(__name__)

# Create router
rename_router = APIRouter(prefix="/rename", tags=["Rename"])


# Dependency to get database session
def get_db() -> Generator[Session]:
    """Get database session."""
    session_local = get_session_factory()
    db = session_local()
    try:
        yield db
    finally:
        db.close()


@rename_router.post(
    "/analyze",
    response_model=RenameAnalyzeResponse,
    responses={400: {"model": ErrorResponse}},
)
async def analyze_patterns(
    request: RenameAnalyzeRequest,
    db: Session = Depends(get_db),
) -> RenameAnalyzeResponse:
    """
    Analyze filename patterns.

    Analyzes a list of filenames to detect patterns, categories,
    and provide suggestions for naming consistency.
    """
    try:
        # TODO: Implement actual pattern analysis logic
        # For now, return mock response

        # Send to RabbitMQ for async processing if connected
        if rabbitmq_manager.is_connected:
            await rabbitmq_manager.publish(
                MessageTopics.PATTERN_ANALYZE,
                {
                    "filenames": request.filenames,
                    "context": request.context,
                    "include_patterns": request.include_patterns,
                    "include_categories": request.include_categories,
                },
            )

        # Return mock response
        return RenameAnalyzeResponse(
            patterns=[
                {
                    "type": "sequential",
                    "pattern": "track{number}.{extension}",
                    "matches": len(request.filenames),
                }
            ],
            categories=["music", "numbered"],
            confidence=0.85,
            suggestions=["Consider adding artist and title information"],
        )

    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@rename_router.post(
    "/propose",
    response_model=RenameProposalResponse,
    responses={400: {"model": ErrorResponse}},
)
async def propose_rename(
    request: RenameProposalRequest,
    db: Session = Depends(get_db),
) -> RenameProposalResponse:
    """
    Generate rename proposals.

    Generates one or more rename proposals for a given filename
    using pattern matching and ML models.
    """
    try:
        # TODO: Implement actual proposal generation logic
        # For now, return mock response

        # Send to RabbitMQ for async processing if connected
        if rabbitmq_manager.is_connected:
            await rabbitmq_manager.publish(
                MessageTopics.RENAME_REQUEST,
                {
                    "original_name": request.original_name,
                    "file_path": request.file_path,
                    "metadata": request.metadata,
                    "use_ml": request.use_ml,
                    "pattern_type": request.pattern_type.value if request.pattern_type else None,
                },
            )

        # Create rename history entry
        history = RenameHistory(
            original_name=request.original_name,
            proposed_name=f"renamed_{request.original_name}",
            file_path=request.file_path,
            confidence_score=0.75,
            extra_metadata=request.metadata or {},
        )
        db.add(history)
        db.commit()

        # Return mock response
        from services.file_rename_service.api.schemas import RenameProposal

        proposal = RenameProposal(
            proposed_name=f"renamed_{request.original_name}",
            confidence=0.75,
            pattern_used="template",
            reasoning="Applied standard naming template",
        )

        return RenameProposalResponse(
            original_name=request.original_name,
            proposals=[proposal],
            recommended=proposal,
        )

    except Exception as e:
        logger.error(f"Error generating rename proposal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@rename_router.post(
    "/feedback",
    response_model=RenameFeedbackResponse,
    responses={400: {"model": ErrorResponse}},
)
async def submit_feedback(
    request: RenameFeedbackRequest,
    db: Session = Depends(get_db),
) -> RenameFeedbackResponse:
    """
    Submit user feedback on rename operations.

    Allows users to provide feedback on rename proposals,
    which is used to improve future suggestions.
    """
    try:
        # Find or create rename history entry
        if request.rename_history_id:
            history = db.query(RenameHistory).filter(RenameHistory.id == request.rename_history_id).first()
            if not history:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Rename history entry not found",
                )
        else:
            # Create new history entry
            history = RenameHistory(
                original_name=request.original_name,
                proposed_name=request.proposed_name,
                final_name=request.final_name,
                was_accepted=request.was_accepted,
                feedback_rating=request.rating,
            )
            db.add(history)
            db.flush()

        # Create feedback entry
        feedback = UserFeedback(
            rename_history_id=history.id,
            feedback_type="user_feedback",
            corrected_name=request.corrected_name,
            rating=request.rating,
            comment=request.comment,
            is_helpful=request.was_accepted,
        )
        db.add(feedback)

        # Update history with feedback
        history.was_accepted = request.was_accepted  # type: ignore[assignment]
        history.feedback_rating = request.rating  # type: ignore[assignment]
        history.user_feedback = request.comment  # type: ignore[assignment]

        # Send to RabbitMQ for pattern learning
        if rabbitmq_manager.is_connected:
            await rabbitmq_manager.publish(
                MessageTopics.RENAME_FEEDBACK,
                {
                    "history_id": history.id,
                    "original_name": request.original_name,
                    "proposed_name": request.proposed_name,
                    "final_name": request.final_name,
                    "was_accepted": request.was_accepted,
                    "rating": request.rating,
                    "corrected_name": request.corrected_name,
                },
            )

        db.commit()

        return RenameFeedbackResponse(
            feedback_id=feedback.id,  # type: ignore[arg-type]
            message="Feedback submitted successfully",
            patterns_updated=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@rename_router.get(
    "/patterns",
    response_model=list[PatternResponse],
    responses={400: {"model": ErrorResponse}},
)
async def get_patterns(
    category: str | None = Query(default=None, description="Filter by category"),
    pattern_type: str | None = Query(default=None, description="Filter by pattern type"),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of records to return"),
    db: Session = Depends(get_db),
) -> list[PatternResponse]:
    """
    Retrieve learned patterns.

    Returns a list of learned filename patterns with optional filtering.
    """
    try:
        query = db.query(Pattern).filter(Pattern.is_active)

        if category:
            query = query.filter(Pattern.category == category)
        if pattern_type:
            query = query.filter(Pattern.pattern_type == pattern_type)

        patterns = query.order_by(Pattern.confidence_score.desc()).offset(skip).limit(limit).all()

        return [
            PatternResponse(
                id=p.id,  # type: ignore[arg-type]
                pattern_type=p.pattern_type,  # type: ignore[arg-type]
                pattern_value=p.pattern_value,  # type: ignore[arg-type]
                description=p.description,  # type: ignore[arg-type]
                category=p.category,  # type: ignore[arg-type]
                frequency=p.frequency,  # type: ignore[arg-type]
                confidence_score=p.confidence_score,  # type: ignore[arg-type]
                created_at=p.created_at,  # type: ignore[arg-type]
                updated_at=p.updated_at,  # type: ignore[arg-type]
            )
            for p in patterns
        ]

    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@rename_router.get(
    "/history",
    response_model=list[RenameHistoryResponse],
    responses={400: {"model": ErrorResponse}},
)
async def get_history(
    was_accepted: bool | None = Query(default=None, description="Filter by acceptance status"),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of records to return"),
    db: Session = Depends(get_db),
) -> list[RenameHistoryResponse]:
    """
    Get rename history.

    Returns the history of rename operations with optional filtering.
    """
    try:
        query = db.query(RenameHistory)

        if was_accepted is not None:
            query = query.filter(RenameHistory.was_accepted == was_accepted)

        history = query.order_by(RenameHistory.created_at.desc()).offset(skip).limit(limit).all()

        return [
            RenameHistoryResponse(
                id=h.id,  # type: ignore[arg-type]
                original_name=h.original_name,  # type: ignore[arg-type]
                proposed_name=h.proposed_name,  # type: ignore[arg-type]
                final_name=h.final_name,  # type: ignore[arg-type]
                confidence_score=h.confidence_score,  # type: ignore[arg-type]
                was_accepted=h.was_accepted,  # type: ignore[arg-type]
                feedback_rating=h.feedback_rating,  # type: ignore[arg-type]
                created_at=h.created_at,  # type: ignore[arg-type]
            )
            for h in history
        ]

    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
