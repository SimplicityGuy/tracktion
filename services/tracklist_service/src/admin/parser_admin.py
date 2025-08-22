"""Parser administration API endpoints for manual override and recovery."""

import logging
import time
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

import requests  # type: ignore
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import (
    SelectorUpdate,
    ManualDataCorrection,
    RollbackRequest,
    ParserTestResult,
    AdminOperation,
    ParserHealthStatus,
    SelectorTestRequest,
    ConfigurationSnapshot,
)
from ..scrapers.adaptive_parser import AdaptiveParser
from ..scrapers.resilient_extractor import (
    ResilientExtractor,
    CSSStrategy,
    XPathStrategy,
    TextStrategy,
    RegexStrategy,
)
from ..monitoring.alert_manager import AlertManager
from ..monitoring.structure_monitor import StructureMonitor
from ..cache.fallback_cache import FallbackCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()


async def verify_admin_access(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify admin access token and return admin user ID."""
    # TODO: Implement proper authentication
    # For now, return a placeholder admin user
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing admin token")

    # Placeholder validation - implement actual token validation
    if credentials.credentials == "admin-token":
        return "admin-user"

    raise HTTPException(status_code=403, detail="Invalid admin token")


# Initialize components
_adaptive_parser: Optional[AdaptiveParser] = None
_resilient_extractor: Optional[ResilientExtractor] = None
_alert_manager: Optional[AlertManager] = None
_structure_monitor: Optional[StructureMonitor] = None
_fallback_cache: Optional[FallbackCache] = None


def get_adaptive_parser() -> AdaptiveParser:
    """Get or create adaptive parser instance."""
    global _adaptive_parser
    if _adaptive_parser is None:
        _adaptive_parser = AdaptiveParser()
    return _adaptive_parser


def get_resilient_extractor() -> ResilientExtractor:
    """Get or create resilient extractor instance."""
    global _resilient_extractor
    if _resilient_extractor is None:
        _resilient_extractor = ResilientExtractor()
    return _resilient_extractor


def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def get_structure_monitor() -> StructureMonitor:
    """Get or create structure monitor instance."""
    global _structure_monitor
    if _structure_monitor is None:
        _structure_monitor = StructureMonitor()
    return _structure_monitor


def get_fallback_cache() -> FallbackCache:
    """Get or create fallback cache instance."""
    global _fallback_cache
    if _fallback_cache is None:
        _fallback_cache = FallbackCache()
    return _fallback_cache


@router.post("/selectors/update", response_model=Dict[str, Any])
async def update_selectors(
    updates: SelectorUpdate,
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(verify_admin_access),
) -> Dict[str, Any]:
    """Update parser selectors with new configuration.

    Args:
        updates: Selector update configuration
        background_tasks: FastAPI background tasks
        admin_user: Authenticated admin user

    Returns:
        Update result with operation details
    """
    try:
        adaptive_parser = get_adaptive_parser()

        # Create strategy instance based on type
        strategy_map = {
            "css": CSSStrategy,
            "xpath": XPathStrategy,
            "text": TextStrategy,
            "regex": RegexStrategy,
        }

        strategy_class = strategy_map.get(updates.selector_type.value)
        if not strategy_class:
            raise HTTPException(status_code=400, detail=f"Unsupported selector type: {updates.selector_type}")

        # Test selector if test URL provided
        test_result = None
        if updates.test_url:
            test_result = await test_selector_internal(
                updates.test_url,
                updates.page_type,
                updates.field_name,
                updates.selector_type.value,
                updates.selector_value,
            )

            if not test_result["success"]:
                logger.warning(f"Selector test failed for {updates.field_name}: {test_result['error_message']}")

        # Update configuration
        operation_id = str(uuid.uuid4())

        # Record admin operation
        admin_operation = AdminOperation(
            operation_id=operation_id,
            operation_type="selector_update",
            timestamp=datetime.now(UTC),
            admin_user=admin_user,
            details={
                "page_type": updates.page_type,
                "field_name": updates.field_name,
                "selector_type": updates.selector_type.value,
                "selector_value": updates.selector_value,
                "priority": updates.priority,
                "test_result": test_result,
                "notes": updates.notes,
            },
            success=True,
        )

        # Background task to update configuration
        background_tasks.add_task(
            apply_selector_update,
            adaptive_parser,
            updates,
            admin_operation,
        )

        return {
            "success": True,
            "operation_id": operation_id,
            "message": "Selector update queued for processing",
            "test_result": test_result,
        }

    except Exception as e:
        logger.error(f"Error updating selectors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Selector update failed: {str(e)}")


async def apply_selector_update(
    adaptive_parser: AdaptiveParser,
    updates: SelectorUpdate,
    admin_operation: AdminOperation,
) -> None:
    """Apply selector update in background."""
    try:
        # Update parser configuration
        await adaptive_parser.update_selector_config(
            page_type=updates.page_type,
            field_name=updates.field_name,
            selector_type=updates.selector_type.value,
            selector_value=updates.selector_value,
            priority=updates.priority,
        )

        # Trigger hot reload
        adaptive_parser.hot_reload_config()

        logger.info(f"Successfully applied selector update: {admin_operation.operation_id}")

    except Exception as e:
        logger.error(f"Failed to apply selector update {admin_operation.operation_id}: {e}")
        admin_operation.success = False
        admin_operation.error_message = str(e)


@router.post("/parser/test", response_model=ParserTestResult)
async def test_parser(
    test_request: SelectorTestRequest,
    admin_user: str = Depends(verify_admin_access),
) -> ParserTestResult:
    """Test parser with specific selector configuration.

    Args:
        test_request: Parser test configuration
        admin_user: Authenticated admin user

    Returns:
        Test result with extracted data and metrics
    """
    try:
        result = await test_selector_internal(
            test_request.url,
            test_request.page_type,
            test_request.field_name,
            test_request.selector_type.value,
            test_request.selector_value,
            test_request.expected_result,
        )

        return ParserTestResult(**result)

    except Exception as e:
        logger.error(f"Error testing parser: {e}", exc_info=True)
        return ParserTestResult(
            success=False,
            error_message=str(e),
            execution_time_ms=0,
            strategy_used="none",
            quality_score=0.0,
        )


async def test_selector_internal(
    url: str,
    page_type: str,
    field_name: str,
    selector_type: str,
    selector_value: str,
    expected_result: Optional[str] = None,
) -> Dict[str, Any]:
    """Internal selector testing logic."""
    start_time = time.time()

    try:
        # Fetch page content
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Create strategy instance
        strategy_map = {
            "css": CSSStrategy,
            "xpath": XPathStrategy,
            "text": TextStrategy,
            "regex": RegexStrategy,
        }

        strategy_class = strategy_map.get(selector_type)
        if not strategy_class:
            raise ValueError(f"Unsupported selector type: {selector_type}")

        strategy = strategy_class(selector_value)

        # Extract data
        extracted_data = strategy.extract(soup, field_name)

        execution_time = int((time.time() - start_time) * 1000)

        # Calculate quality score
        quality_score = 1.0 if extracted_data.value else 0.0
        if expected_result and extracted_data.value:
            # Simple similarity check
            similarity = 1.0 if expected_result.lower() in extracted_data.value.lower() else 0.5
            quality_score = similarity

        warnings = []
        if not extracted_data.value:
            warnings.append("No data extracted")
        elif expected_result and expected_result.lower() not in extracted_data.value.lower():
            warnings.append("Extracted data differs from expected result")

        return {
            "success": bool(extracted_data.value),
            "extracted_data": extracted_data.to_dict() if extracted_data.value else None,
            "execution_time_ms": execution_time,
            "strategy_used": selector_type,
            "quality_score": quality_score,
            "warnings": warnings,
            "metadata": {
                "page_type": page_type,
                "field_name": field_name,
                "selector": selector_value,
                "url": url,
            },
        }

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "error_message": str(e),
            "execution_time_ms": execution_time,
            "strategy_used": selector_type,
            "quality_score": 0.0,
            "warnings": [f"Test failed: {str(e)}"],
            "metadata": {
                "page_type": page_type,
                "field_name": field_name,
                "selector": selector_value,
                "url": url,
            },
        }


@router.post("/data/correct", response_model=Dict[str, Any])
async def correct_data(
    correction: ManualDataCorrection,
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(verify_admin_access),
) -> Dict[str, Any]:
    """Manually correct extracted data.

    Args:
        correction: Data correction request
        background_tasks: FastAPI background tasks
        admin_user: Authenticated admin user

    Returns:
        Correction operation result
    """
    try:
        operation_id = str(uuid.uuid4())

        # Record admin operation
        admin_operation = AdminOperation(
            operation_id=operation_id,
            operation_type="manual_correction",
            timestamp=datetime.now(UTC),
            admin_user=admin_user,
            details={
                "tracklist_id": correction.tracklist_id,
                "field_corrections": correction.field_corrections,
                "reason": correction.reason,
                "preserve_original": correction.preserve_original,
            },
            success=True,
        )

        # Background task to apply correction
        background_tasks.add_task(
            apply_data_correction,
            correction,
            admin_operation,
        )

        return {
            "success": True,
            "operation_id": operation_id,
            "message": "Data correction queued for processing",
            "corrected_fields": list(correction.field_corrections.keys()),
        }

    except Exception as e:
        logger.error(f"Error correcting data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data correction failed: {str(e)}")


async def apply_data_correction(
    correction: ManualDataCorrection,
    admin_operation: AdminOperation,
) -> None:
    """Apply data correction in background."""
    try:
        fallback_cache = get_fallback_cache()

        # Get current data
        cache_key = f"tracklist:{correction.tracklist_id}"
        current_data = await fallback_cache.get_with_fallback(cache_key)

        if not current_data:
            raise ValueError(f"Tracklist {correction.tracklist_id} not found in cache")

        # Preserve original if requested
        if correction.preserve_original:
            original_key = f"tracklist:original:{correction.tracklist_id}:{datetime.now(UTC).isoformat()}"
            await fallback_cache.set_with_quality(original_key, current_data, quality_score=1.0)

        # Apply corrections
        corrected_data = current_data.copy()
        for field, value in correction.field_corrections.items():
            corrected_data[field] = value

        # Add correction metadata
        corrected_data["manual_corrections"] = corrected_data.get("manual_corrections", [])
        corrected_data["manual_corrections"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "admin_user": correction.admin_user,
                "reason": correction.reason,
                "fields": list(correction.field_corrections.keys()),
            }
        )

        # Update cache
        await fallback_cache.set_with_quality(cache_key, corrected_data, quality_score=1.0)

        logger.info(f"Successfully applied data correction: {admin_operation.operation_id}")

    except Exception as e:
        logger.error(f"Failed to apply data correction {admin_operation.operation_id}: {e}")
        admin_operation.success = False
        admin_operation.error_message = str(e)


@router.post("/parser/rollback", response_model=Dict[str, Any])
async def rollback_parser(
    rollback_request: RollbackRequest,
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(verify_admin_access),
) -> Dict[str, Any]:
    """Rollback parser to previous version.

    Args:
        rollback_request: Rollback configuration
        background_tasks: FastAPI background tasks
        admin_user: Authenticated admin user

    Returns:
        Rollback operation result
    """
    try:
        adaptive_parser = get_adaptive_parser()

        # Validate target version exists
        if not await adaptive_parser.version_exists(rollback_request.target_version):
            raise HTTPException(status_code=404, detail=f"Version {rollback_request.target_version} not found")

        operation_id = str(uuid.uuid4())

        # Record admin operation
        admin_operation = AdminOperation(
            operation_id=operation_id,
            operation_type="parser_rollback",
            timestamp=datetime.now(UTC),
            admin_user=admin_user,
            details={
                "target_version": rollback_request.target_version,
                "reason": rollback_request.reason,
                "force": rollback_request.force,
            },
            success=True,
        )

        # Background task to perform rollback
        background_tasks.add_task(
            perform_rollback,
            adaptive_parser,
            rollback_request,
            admin_operation,
        )

        return {
            "success": True,
            "operation_id": operation_id,
            "message": f"Rollback to version {rollback_request.target_version} queued",
            "target_version": rollback_request.target_version,
        }

    except Exception as e:
        logger.error(f"Error initiating rollback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")


async def perform_rollback(
    adaptive_parser: AdaptiveParser,
    rollback_request: RollbackRequest,
    admin_operation: AdminOperation,
) -> None:
    """Perform parser rollback in background."""
    try:
        # Perform rollback
        await adaptive_parser.rollback_to_version(rollback_request.target_version, force=rollback_request.force)

        # Trigger hot reload with rolled back configuration
        adaptive_parser.hot_reload_config()

        logger.info(f"Successfully rolled back parser to version {rollback_request.target_version}")

    except Exception as e:
        logger.error(f"Failed to perform rollback {admin_operation.operation_id}: {e}")
        admin_operation.success = False
        admin_operation.error_message = str(e)


@router.get("/parser/health", response_model=ParserHealthStatus)
async def get_parser_health(
    admin_user: str = Depends(verify_admin_access),
) -> ParserHealthStatus:
    """Get current parser health and status.

    Args:
        admin_user: Authenticated admin user

    Returns:
        Parser health status
    """
    try:
        adaptive_parser = get_adaptive_parser()
        alert_manager = get_alert_manager()

        # Get current version
        current_version = adaptive_parser.get_current_version()

        # Get health metrics
        health_status = await alert_manager.check_parser_health()

        # Get active alerts
        active_alerts = await alert_manager.get_active_alerts()

        return ParserHealthStatus(
            current_version=current_version.version,
            success_rate_24h=health_status.success_rate,
            total_extractions_24h=len(health_status.failed_extractions) + 100,  # Placeholder calculation
            failed_extractions_24h=len(health_status.failed_extractions),
            active_strategies={"css": 5, "xpath": 3, "text": 2, "regex": 1},  # Placeholder data
            last_config_update=current_version.created_at,
            alerts_active=[alert.message for alert in active_alerts],
            system_status="healthy" if health_status.healthy else "degraded",
            performance_metrics=health_status.metrics,
        )

    except Exception as e:
        logger.error(f"Error getting parser health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get parser health: {str(e)}")


@router.get("/configuration/snapshot", response_model=ConfigurationSnapshot)
async def get_configuration_snapshot(
    admin_user: str = Depends(verify_admin_access),
) -> ConfigurationSnapshot:
    """Get current parser configuration snapshot.

    Args:
        admin_user: Authenticated admin user

    Returns:
        Configuration snapshot
    """
    try:
        adaptive_parser = get_adaptive_parser()

        current_version = adaptive_parser.get_current_version()

        return ConfigurationSnapshot(
            version=current_version.version,
            timestamp=current_version.created_at,
            strategies=current_version.strategies,
            success_rates={
                strategy: rate for strategy, rate in current_version.metadata.get("success_rates", {}).items()
            },
            metadata=current_version.metadata,
            admin_notes=current_version.metadata.get("admin_notes"),
        )

    except Exception as e:
        logger.error(f"Error getting configuration snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.get("/operations/history")
async def get_operation_history(
    limit: int = 50,
    operation_type: Optional[str] = None,
    admin_user: str = Depends(verify_admin_access),
) -> List[Dict[str, Any]]:
    """Get history of admin operations.

    Args:
        limit: Maximum number of operations to return
        operation_type: Filter by operation type
        admin_user: Authenticated admin user

    Returns:
        List of admin operations
    """
    try:
        # TODO: Implement operation history storage and retrieval
        # For now, return placeholder
        return [
            {
                "operation_id": "placeholder",
                "operation_type": "system_status",
                "timestamp": datetime.now(UTC).isoformat(),
                "admin_user": admin_user,
                "details": {"message": "Operation history not yet implemented"},
                "success": True,
            }
        ]

    except Exception as e:
        logger.error(f"Error getting operation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get operation history: {str(e)}")
