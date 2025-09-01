"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PatternType(str, Enum):
    """Types of filename patterns."""

    REGEX = "regex"
    TOKEN = "token"
    TEMPLATE = "template"
    LEARNED = "learned"


class RenameAnalyzeRequest(BaseModel):
    """Request schema for analyzing filename patterns."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filenames": ["track01.mp3", "track02.mp3", "track03.mp3"],
                "context": {"album": "Greatest Hits", "artist": "The Band"},
            }
        }
    )

    filenames: list[str] = Field(..., min_length=1, description="List of filenames to analyze")
    context: dict[str, Any] | None = Field(default=None, description="Additional context for analysis")
    include_patterns: bool = Field(default=True, description="Include pattern detection")
    include_categories: bool = Field(default=True, description="Include categorization")


class RenameAnalyzeResponse(BaseModel):
    """Response schema for filename pattern analysis."""

    patterns: list[dict[str, Any]] = Field(description="Detected patterns")
    categories: list[str] = Field(description="Detected categories")
    confidence: float = Field(description="Overall confidence score")
    suggestions: list[str] = Field(description="Suggested improvements")


class RenameProposalRequest(BaseModel):
    """Request schema for generating rename proposals."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_name": "track01.mp3",
                "file_path": "/music/album/",
                "metadata": {
                    "title": "Song Title",
                    "artist": "Artist Name",
                    "track": 1,
                },
            }
        }
    )

    original_name: str = Field(..., description="Original filename")
    file_path: str | None = Field(default=None, description="File path")
    metadata: dict[str, Any] | None = Field(default=None, description="File metadata")
    use_ml: bool = Field(default=True, description="Use ML models for proposal")
    pattern_type: PatternType | None = Field(default=None, description="Preferred pattern type")


class RenameProposal(BaseModel):
    """Schema for a single rename proposal."""

    proposed_name: str = Field(description="Proposed new filename")
    confidence: float = Field(description="Confidence score (0-1)")
    pattern_used: str | None = Field(default=None, description="Pattern used for proposal")
    reasoning: str | None = Field(default=None, description="Explanation for proposal")


class RenameProposalResponse(BaseModel):
    """Response schema for rename proposals."""

    original_name: str = Field(description="Original filename")
    proposals: list[RenameProposal] = Field(description="List of rename proposals")
    recommended: RenameProposal | None = Field(default=None, description="Recommended proposal")


class RenameFeedbackRequest(BaseModel):
    """Request schema for submitting user feedback."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rename_history_id": 123,
                "was_accepted": True,
                "rating": 5,
                "corrected_name": "01 - Artist - Song Title.mp3",
            }
        }
    )

    rename_history_id: int | None = Field(default=None, description="ID of rename history entry")
    original_name: str = Field(description="Original filename")
    proposed_name: str = Field(description="Proposed filename")
    final_name: str | None = Field(default=None, description="Final chosen filename")
    was_accepted: bool = Field(description="Whether proposal was accepted")
    rating: int | None = Field(default=None, ge=1, le=5, description="Rating (1-5)")
    corrected_name: str | None = Field(default=None, description="User-corrected filename")
    comment: str | None = Field(default=None, description="User comment")


class RenameFeedbackResponse(BaseModel):
    """Response schema for feedback submission."""

    feedback_id: int = Field(description="Feedback entry ID")
    message: str = Field(description="Confirmation message")
    patterns_updated: bool = Field(description="Whether patterns were updated")


class PatternResponse(BaseModel):
    """Response schema for pattern information."""

    id: int = Field(description="Pattern ID")
    pattern_type: str = Field(description="Type of pattern")
    pattern_value: str = Field(description="Pattern value/expression")
    description: str | None = Field(default=None, description="Pattern description")
    category: str | None = Field(default=None, description="Pattern category")
    frequency: int = Field(description="Usage frequency")
    confidence_score: float = Field(description="Confidence score")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class RenameHistoryResponse(BaseModel):
    """Response schema for rename history."""

    id: int = Field(description="History entry ID")
    original_name: str = Field(description="Original filename")
    proposed_name: str = Field(description="Proposed filename")
    final_name: str | None = Field(default=None, description="Final filename")
    confidence_score: float | None = Field(default=None, description="Confidence score")
    was_accepted: bool | None = Field(default=None, description="Whether accepted")
    feedback_rating: int | None = Field(default=None, description="User rating")
    created_at: datetime = Field(description="Timestamp")


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of records to return")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    status_code: int = Field(description="HTTP status code")


# New schemas for proposal endpoints


class BatchProposalRequest(BaseModel):
    """Request schema for batch rename proposals."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filenames": ["track01.mp3", "track02.mp3"],
                "file_paths": ["/music/album/", "/music/album/"],
                "metadata_list": [
                    {"title": "Song 1", "artist": "Artist", "track": 1},
                    {"title": "Song 2", "artist": "Artist", "track": 2},
                ],
                "use_ml": True,
            }
        }
    )

    filenames: list[str] = Field(..., min_length=1, description="List of filenames to process")
    file_paths: list[str] | None = Field(default=None, description="List of file paths (optional)")
    metadata_list: list[dict[str, Any]] | None = Field(default=None, description="List of metadata for each file")
    use_ml: bool = Field(default=True, description="Use ML models for proposals")
    pattern_type: PatternType | None = Field(default=None, description="Preferred pattern type")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Maximum concurrent processing tasks")


class BatchProposalResponse(BaseModel):
    """Response schema for batch rename proposals."""

    successful_proposals: list[RenameProposalResponse] = Field(description="Successfully generated proposals")
    failed_files: list[str] = Field(description="Files that failed processing")
    total_files: int = Field(description="Total number of files processed")
    success_rate: float = Field(description="Success rate percentage")
    processing_time: float = Field(description="Total processing time in seconds")
    errors: dict[str, str] = Field(default_factory=dict, description="Error messages for failed files")


class TemplateRequest(BaseModel):
    """Request schema for creating/updating a template."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Concert Recording",
                "pattern": "{artist} - {date} - {venue} - {quality}",
                "description": "Template for concert recordings",
                "user_id": "user123",
            }
        }
    )

    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    pattern: str = Field(..., description="Template pattern string")
    description: str | None = Field(default=None, description="Template description")
    user_id: str = Field(..., description="User ID who owns the template")


class TemplateResponse(BaseModel):
    """Response schema for template information."""

    id: str = Field(description="Template ID")
    name: str = Field(description="Template name")
    pattern: str = Field(description="Template pattern")
    user_id: str = Field(description="User ID who owns the template")
    description: str | None = Field(default=None, description="Template description")
    usage_count: int = Field(description="Number of times used")
    is_active: bool = Field(description="Whether template is active")
    created_at: datetime = Field(description="Creation timestamp")


class ValidateRequest(BaseModel):
    """Request schema for validation endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "proposed_names": ["Artist - 2023-01-01 - Venue.mp3", "Artist - Song Title.mp3"],
                "existing_files": ["existing_file.mp3", "another_file.mp3"],
                "check_filesystem": True,
            }
        }
    )

    proposed_names: list[str] = Field(..., min_length=1, description="List of proposed filenames to validate")
    existing_files: list[str] = Field(default_factory=list, description="List of existing filenames to check against")
    check_filesystem: bool = Field(default=False, description="Whether to check actual filesystem for conflicts")


class ValidationResult(BaseModel):
    """Schema for individual validation result."""

    filename: str = Field(description="The filename that was validated")
    is_valid: bool = Field(description="Whether the filename is valid")
    conflicts: list[str] = Field(default_factory=list, description="List of conflicting files")
    issues: list[str] = Field(default_factory=list, description="List of validation issues")
    suggested_resolution: str | None = Field(default=None, description="Suggested resolution if conflicts exist")


class ValidateResponse(BaseModel):
    """Response schema for validation endpoint."""

    results: list[ValidationResult] = Field(description="Validation results for each filename")
    overall_valid: bool = Field(description="Whether all filenames are valid")
    conflicts_detected: int = Field(description="Number of conflicts detected")
    invalid_count: int = Field(description="Number of invalid filenames")
    suggestions: list[str] = Field(default_factory=list, description="General suggestions for resolution")
