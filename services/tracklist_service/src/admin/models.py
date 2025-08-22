"""Admin models for manual override and recovery operations."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
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
    extracted_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: int
    strategy_used: str
    quality_score: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SelectorUpdate(BaseModel):
    """Update request for parser selectors."""

    page_type: str = Field(..., description="Type of page (e.g., 'tracklist', 'search')")
    field_name: str = Field(..., description="Field being extracted (e.g., 'title', 'artist')")
    selector_type: SelectorType
    selector_value: str = Field(..., description="The selector string")
    priority: int = Field(default=1, description="Priority in fallback chain (lower = higher priority)")
    test_url: Optional[str] = Field(None, description="URL to test the selector against")
    notes: Optional[str] = Field(None, description="Admin notes about the change")


class ManualDataCorrection(BaseModel):
    """Manual data correction request."""

    tracklist_id: str
    field_corrections: Dict[str, Any] = Field(..., description="Field name to corrected value mapping")
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
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class ParserHealthStatus(BaseModel):
    """Current parser health and configuration status."""

    current_version: str
    success_rate_24h: float
    total_extractions_24h: int
    failed_extractions_24h: int
    active_strategies: Dict[str, int]
    last_config_update: datetime
    alerts_active: List[str]
    system_status: str
    performance_metrics: Dict[str, float]


class SelectorTestRequest(BaseModel):
    """Request to test a specific selector."""

    url: str = Field(..., description="URL to test against")
    page_type: str = Field(..., description="Type of page being tested")
    field_name: str = Field(..., description="Field to extract")
    selector_type: SelectorType
    selector_value: str = Field(..., description="Selector to test")
    expected_result: Optional[str] = Field(None, description="Expected extraction result")


class ConfigurationSnapshot(BaseModel):
    """Snapshot of current parser configuration."""

    version: str
    timestamp: datetime
    strategies: Dict[str, List[Dict[str, Any]]]
    success_rates: Dict[str, float]
    metadata: Dict[str, Any]
    admin_notes: Optional[str] = None
