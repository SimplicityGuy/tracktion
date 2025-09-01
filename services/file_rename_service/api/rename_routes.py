"""API routes for rename proposal operations."""

import logging
from collections.abc import Generator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from services.file_rename_service.api.schemas import (
    BatchProposalRequest,
    BatchProposalResponse,
    ErrorResponse,
    RenameProposalRequest,
    RenameProposalResponse,
    TemplateRequest,
    TemplateResponse,
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
)
from services.file_rename_service.app.cache import proposal_cache
from services.file_rename_service.app.proposal.batch_processor import BatchProcessor
from services.file_rename_service.app.proposal.conflicts import FilenameConflictResolver
from services.file_rename_service.app.proposal.generator import ProposalGenerator
from services.file_rename_service.app.proposal.models import NamingTemplate, RenameProposal
from services.file_rename_service.app.proposal.templates import template_manager
from services.file_rename_service.models.database import get_session_factory

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
proposal_router = APIRouter(prefix="/rename", tags=["Rename Proposals"])


# Dependency to get database session
def get_db() -> Generator[Session]:
    """Get database session."""
    session_local = get_session_factory()
    db = session_local()
    try:
        yield db
    finally:
        db.close()


@proposal_router.post(
    "/propose",
    response_model=RenameProposalResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate single file rename proposal",
    description="Generate a rename proposal for a single file using ML models and templates",
)
async def propose_single_rename(
    request: RenameProposalRequest,
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI Depends pattern requires function call in default
) -> RenameProposalResponse:
    """
    Generate rename proposal for a single file.

    This endpoint analyzes a single filename and generates rename proposals using:
    - Machine learning models for pattern recognition
    - User-defined or system templates
    - Historical naming patterns

    Args:
        request: Request containing filename and metadata
        db: Database session

    Returns:
        Response with proposed filename(s) and confidence scores

    Raises:
        HTTPException: If filename is invalid or processing fails
    """
    try:
        logger.info(f"Generating proposal for: {request.original_name}")

        # Check cache first
        cache_key = f"proposal:{request.original_name}:{hash(str(request.metadata))}"
        cached_result = proposal_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached proposal for: {request.original_name}")
            return cached_result  # type: ignore[no-any-return]

        # Initialize proposal generator
        generator = ProposalGenerator()

        # Get user templates if available
        templates = None
        user_id = getattr(request, "user_id", None)
        if user_id:
            templates_list = await template_manager.get_user_templates(user_id)
            templates = [template for template in templates_list if template.is_active]

        # Generate the proposal
        proposal = await generator.generate_proposal(
            filename=request.original_name,
            templates=templates,
        )

        # Convert to response format
        response_proposal = RenameProposal(
            proposed_name=proposal.proposed_filename,
            confidence=proposal.confidence_score,
            pattern_used=", ".join(proposal.patterns_used) if proposal.patterns_used else None,
            reasoning=proposal.explanation,
        )

        # Create alternatives list
        alternatives = [
            RenameProposal(
                proposed_name=alt,
                confidence=max(
                    0.1, proposal.confidence_score - 0.1 - (i * 0.05)
                ),  # Decrease confidence for alternatives
                pattern_used="alternative",
                reasoning=f"Alternative suggestion based on {proposal.explanation}",
            )
            for i, alt in enumerate(proposal.alternatives[:5])  # Limit to 5 alternatives
        ]

        response = RenameProposalResponse(
            original_name=request.original_name,
            proposals=[response_proposal, *alternatives],
            recommended=response_proposal,
        )

        # Cache the response for 30 minutes
        proposal_cache.set(cache_key, response, ttl=1800)

        return response

    except ValueError as e:
        logger.error(f"Validation error for '{request.original_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filename or parameters: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"Error generating proposal for '{request.original_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate proposal: {e!s}",
        ) from e


@proposal_router.post(
    "/propose/batch",
    response_model=BatchProposalResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate batch rename proposals",
    description="Generate rename proposals for multiple files with parallel processing",
)
async def propose_batch_rename(
    request: BatchProposalRequest,
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI Depends pattern requires function call in default
) -> BatchProposalResponse:
    """
    Generate rename proposals for multiple files in batch.

    This endpoint processes multiple files in parallel to generate rename proposals.
    It supports:
    - Parallel processing for improved performance
    - Individual metadata per file
    - Progress tracking and error handling
    - Partial success handling

    Args:
        request: Request containing list of filenames and options
        db: Database session

    Returns:
        Response with successful proposals and failed files

    Raises:
        HTTPException: If request is invalid or processing fails
    """
    try:
        logger.info(f"Starting batch proposal generation for {len(request.filenames)} files")

        # Validate input lengths match
        if request.file_paths and len(request.file_paths) != len(request.filenames):
            raise ValueError("file_paths length must match filenames length")

        if request.metadata_list and len(request.metadata_list) != len(request.filenames):
            raise ValueError("metadata_list length must match filenames length")

        # Initialize batch processor
        processor = BatchProcessor(
            max_concurrent_tasks=request.max_concurrent,
        )

        # Get user templates (assuming user_id is available from auth context)
        # TODO: Get user_id from authentication context
        user_id = "system"  # Placeholder for authentication
        templates_list = await template_manager.get_user_templates(user_id)
        templates = [template for template in templates_list if template.is_active]

        # Process batch
        result = await processor.process_batch(
            filenames=request.filenames,
            templates=templates,
        )

        # Convert proposals to response format
        successful_responses = []
        for proposal in result.successful_proposals:
            response_proposal = RenameProposal(
                proposed_name=proposal.proposed_filename,
                confidence=proposal.confidence_score,
                pattern_used=", ".join(proposal.patterns_used) if proposal.patterns_used else None,
                reasoning=proposal.explanation,
            )

            # Create alternatives list
            alternatives = [
                RenameProposal(
                    proposed_name=alt,
                    confidence=max(0.1, proposal.confidence_score - 0.1 - (i * 0.05)),
                    pattern_used="alternative",
                    reasoning=f"Alternative for {proposal.original_filename}",
                )
                for i, alt in enumerate(proposal.alternatives[:3])  # Limit alternatives for batch
            ]

            successful_responses.append(
                RenameProposalResponse(
                    original_name=proposal.original_filename,
                    proposals=[response_proposal, *alternatives],
                    recommended=response_proposal,
                )
            )

        return BatchProposalResponse(
            successful_proposals=successful_responses,
            failed_files=result.failed_files,
            total_files=result.total_files,
            success_rate=result.success_rate,
            processing_time=result.processing_time,
            errors=result.errors,
        )

    except ValueError as e:
        logger.error(f"Validation error in batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"Error in batch proposal generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch: {e!s}",
        ) from e


@proposal_router.get(
    "/templates",
    response_model=list[TemplateResponse],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get naming templates",
    description="Retrieve available naming templates for the user",
)
async def get_templates(
    user_id: str = Query(..., description="User ID to get templates for"),
    search: str | None = Query(default=None, description="Search query for template names"),
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI Depends pattern requires function call in default
) -> list[TemplateResponse]:
    """
    Get naming templates for a user.

    Retrieves all active naming templates available to the user, with optional search filtering.

    Args:
        user_id: ID of the user to get templates for
        search: Optional search query to filter templates
        db: Database session

    Returns:
        List of naming templates

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Retrieving templates for user: {user_id}")

        if search:
            templates = await template_manager.search_templates(user_id, search)
        else:
            templates = await template_manager.get_user_templates(user_id)

        # Convert to response format
        return [
            TemplateResponse(
                id=template.id,
                name=template.name,
                pattern=template.pattern,
                user_id=template.user_id,
                description=template.description,
                usage_count=template.usage_count,
                is_active=template.is_active,
                created_at=template.created_at,
            )
            for template in templates
        ]

    except Exception as e:
        logger.error(f"Error retrieving templates for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve templates: {e!s}",
        ) from e


@proposal_router.post(
    "/templates",
    response_model=TemplateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid template"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Save custom template",
    description="Create a new custom naming template",
)
async def save_template(
    request: TemplateRequest,
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI Depends pattern requires function call in default
) -> TemplateResponse:
    """
    Save a custom naming template.

    Creates a new naming template that can be used for generating rename proposals.

    Args:
        request: Request containing template details
        db: Database session

    Returns:
        The created template

    Raises:
        HTTPException: If template is invalid or creation fails
    """
    try:
        logger.info(f"Creating template: {request.name}")

        # Create naming template
        template = NamingTemplate(
            id="",  # Will be generated by save_template
            name=request.name,
            pattern=request.pattern,
            user_id=request.user_id,
            description=request.description,
            usage_count=0,
            is_active=True,
        )

        # Save template (validation happens in template_manager)
        template_id = await template_manager.save_template(template)

        # Retrieve saved template
        saved_template = await template_manager.get_template(template_id)
        if not saved_template:
            raise ValueError("Failed to retrieve saved template")

        return TemplateResponse(
            id=saved_template.id,
            name=saved_template.name,
            pattern=saved_template.pattern,
            user_id=saved_template.user_id,
            description=saved_template.description,
            usage_count=saved_template.usage_count,
            is_active=saved_template.is_active,
            created_at=saved_template.created_at,
        )

    except ValueError as e:
        logger.error(f"Validation error creating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid template: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {e!s}",
        ) from e


@proposal_router.post(
    "/validate",
    response_model=ValidateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Validate proposed filenames",
    description="Check proposed filenames for conflicts and validation issues",
)
async def validate_filenames(
    request: ValidateRequest,
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI Depends pattern requires function call in default
) -> ValidateResponse:
    """
    Validate proposed filenames for conflicts and issues.

    Checks proposed filenames against:
    - Filename validity (invalid characters, reserved names, etc.)
    - Conflicts with existing files
    - Optional filesystem checks

    Args:
        request: Request containing filenames to validate
        db: Database session

    Returns:
        Validation results with conflicts and suggestions

    Raises:
        HTTPException: If validation fails
    """
    try:
        logger.info(f"Validating {len(request.proposed_names)} filenames")

        # Initialize conflict resolver
        resolver = FilenameConflictResolver()

        # Validate each proposed filename
        results = []
        conflicts_detected = 0
        invalid_count = 0

        for proposed_name in request.proposed_names:
            issues = []
            conflicts = []
            suggested_resolution = None

            # Check filename validity
            is_valid = resolver.validate_filename(proposed_name)
            if not is_valid:
                invalid_count += 1
                issues.append("Invalid filename format or characters")

            # Check for conflicts with existing files
            conflict = resolver.detect_conflicts(proposed_name, request.existing_files)
            if conflict:
                conflicts_detected += 1
                conflicts.append(conflict.existing_file)
                suggested_resolution = conflict.proposed_action

            # Check filesystem if requested
            if request.check_filesystem and Path(proposed_name).exists() and proposed_name not in conflicts:
                conflicts_detected += 1
                conflicts.append(proposed_name)
                suggested_resolution = "File already exists on filesystem"

            results.append(
                ValidationResult(
                    filename=proposed_name,
                    is_valid=is_valid and len(conflicts) == 0,
                    conflicts=conflicts,
                    issues=issues,
                    suggested_resolution=suggested_resolution,
                )
            )

        # Generate general suggestions
        suggestions = []
        if invalid_count > 0:
            suggestions.append(f"{invalid_count} filenames have invalid format - check for invalid characters")
        if conflicts_detected > 0:
            suggestions.append(f"{conflicts_detected} conflicts detected - consider using unique naming patterns")
        if conflicts_detected > len(request.proposed_names) * 0.5:
            suggestions.append("High conflict rate - review existing files and naming conventions")

        return ValidateResponse(
            results=results,
            overall_valid=conflicts_detected == 0 and invalid_count == 0,
            conflicts_detected=conflicts_detected,
            invalid_count=invalid_count,
            suggestions=suggestions,
        )

    except Exception as e:
        logger.error(f"Error validating filenames: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate filenames: {e!s}",
        ) from e
