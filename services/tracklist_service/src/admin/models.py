"""Admin models for manual override and recovery operations."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SelectorType(str, Enum):
    """Types of selectors that can be updated."""

    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    REGEX = "regex"


class ParserTestResult(BaseModel):
    """Result of parser testing operation."""

    success: bool
    extracted_data: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time_ms: int
    strategy_used: str
    quality_score: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectorUpdate(BaseModel):
    """Update request for parser selectors."""

    page_type: str = Field(..., description="Type of page (e.g., 'tracklist', 'search')")
    field_name: str = Field(..., description="Field being extracted (e.g., 'title', 'artist')")
    selector_type: SelectorType
    selector_value: str = Field(..., description="The selector string")
    priority: int = Field(default=1, description="Priority in fallback chain (lower = higher priority)")
    test_url: str | None = Field(None, description="URL to test the selector against")
    notes: str | None = Field(None, description="Admin notes about the change")


class ManualDataCorrection(BaseModel):
    """Manual data correction request."""

    tracklist_id: str
    field_corrections: dict[str, Any] = Field(..., description="Field name to corrected value mapping")
    reason: str = Field(..., description="Reason for manual correction")
    admin_user: str = Field(..., description="Admin user making the correction")
    preserve_original: bool = Field(default=True, description="Keep original data for audit")


class RollbackRequest(BaseModel):
    """Request to rollback parser configuration."""

    target_version: str = Field(..., description="Version to rollback to")
    reason: str = Field(..., description="Reason for rollback")
    admin_user: str = Field(..., description="Admin user performing rollback")
    force: bool = Field(default=False, description="Force rollback even if version is old")


class AdminOperation(BaseModel):
    """Admin operation record for audit trail."""

    operation_id: str
    operation_type: str
    timestamp: datetime
    admin_user: str
    details: dict[str, Any]
    success: bool
    error_message: str | None = None


class ParserHealthStatus(BaseModel):
    """Current parser health and configuration status."""

    current_version: str
    success_rate_24h: float
    total_extractions_24h: int
    failed_extractions_24h: int
    active_strategies: dict[str, int]
    last_config_update: datetime
    alerts_active: list[str]
    system_status: str
    performance_metrics: dict[str, float]


class SelectorTestRequest(BaseModel):
    """Request to test a specific selector."""

    url: str = Field(..., description="URL to test against")
    page_type: str = Field(..., description="Type of page being tested")
    field_name: str = Field(..., description="Field to extract")
    selector_type: SelectorType
    selector_value: str = Field(..., description="Selector to test")
    expected_result: str | None = Field(None, description="Expected extraction result")


class ConfigurationSnapshot(BaseModel):
    """Snapshot of current parser configuration."""

    version: str
    timestamp: datetime
    strategies: dict[str, list[dict[str, Any]]]
    success_rates: dict[str, float]
    metadata: dict[str, Any]
    admin_notes: str | None = None
