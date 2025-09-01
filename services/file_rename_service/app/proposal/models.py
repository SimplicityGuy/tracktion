"""Data models for the rename proposal system."""

from datetime import datetime

from pydantic import BaseModel, Field


class ConflictResolution(BaseModel):
    """Model for handling filename conflicts."""

    strategy: str = Field(
        ..., description="Resolution strategy for conflicts", pattern="^(append_number|skip|replace)$"
    )
    existing_file: str = Field(..., description="Path/name of the existing conflicting file")
    proposed_action: str = Field(..., description="Description of the proposed action to resolve the conflict")


class RenameProposal(BaseModel):
    """Model for a single rename proposal."""

    original_filename: str = Field(..., description="The original filename to be renamed")
    proposed_filename: str = Field(..., description="The proposed new filename")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the rename proposal (0.0 to 1.0)"
    )
    explanation: str = Field(..., description="Human-readable explanation of why this rename is proposed")
    patterns_used: list[str] = Field(
        default_factory=list, description="List of pattern names/IDs that were used to generate this proposal"
    )
    alternatives: list[str] = Field(default_factory=list, description="List of alternative filename suggestions")
    conflict_status: str = Field(
        default="none", description="Status of potential filename conflicts", pattern="^(none|duplicate|similar)$"
    )
    conflict_resolution: ConflictResolution | None = Field(
        default=None, description="Conflict resolution details if conflicts exist"
    )


class NamingTemplate(BaseModel):
    """Model for user-defined naming templates."""

    id: str = Field(..., description="Unique identifier for the template")
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name for the template")
    pattern: str = Field(..., description="Template pattern string (e.g., '{artist} - {date} - {venue} - {quality}')")
    user_id: str = Field(..., description="Identifier of the user who created this template")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the template was created")
    usage_count: int = Field(default=0, ge=0, description="Number of times this template has been used")
    description: str | None = Field(default=None, description="Optional description of the template's purpose")
    is_active: bool = Field(default=True, description="Whether the template is currently active/available for use")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
