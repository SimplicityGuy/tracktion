"""
Wave-based orchestration for progressive audio analysis pipeline.

This module implements a multi-stage wave processing system that progressively
refines analysis results through coordinated execution stages.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WaveStage(Enum):
    """Stages of wave-based processing."""

    PREPARATION = "preparation"
    INITIAL_ANALYSIS = "initial_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    DEEP_ANALYSIS = "deep_analysis"
    AGGREGATION = "aggregation"
    POST_PROCESSING = "post_processing"
    FINALIZATION = "finalization"


class WaveStatus(Enum):
    """Status of a wave execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class WaveConfig:
    """Configuration for wave orchestration."""

    # Wave stages configuration
    enabled_stages: List[WaveStage] = field(
        default_factory=lambda: [
            WaveStage.PREPARATION,
            WaveStage.INITIAL_ANALYSIS,
            WaveStage.FEATURE_EXTRACTION,
            WaveStage.DEEP_ANALYSIS,
            WaveStage.AGGREGATION,
            WaveStage.POST_PROCESSING,
            WaveStage.FINALIZATION,
        ]
    )

    # Execution settings
    max_parallel_items_per_stage: int = 10
    stage_timeout_seconds: int = 300  # 5 minutes per stage
    max_retries_per_stage: int = 2
    inter_stage_delay_ms: int = 100

    # Resource management
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    adaptive_scaling: bool = True

    # Quality gates
    min_success_rate_per_stage: float = 0.8
    require_all_stages: bool = False
    allow_partial_results: bool = True

    # Progressive refinement
    progressive_mode: bool = True
    early_termination_on_quality: bool = True
    quality_threshold: float = 0.95


@dataclass
class WaveContext:
    """Context for a wave execution."""

    wave_id: str
    start_time: datetime
    current_stage: Optional[WaveStage] = None
    completed_stages: Set[WaveStage] = field(default_factory=set)
    failed_stages: Set[WaveStage] = field(default_factory=set)
    stage_results: Dict[WaveStage, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: WaveStatus = WaveStatus.PENDING


@dataclass
class StageResult:
    """Result of a stage execution."""

    stage: WaveStage
    status: WaveStatus
    data: Any
    metrics: Dict[str, Any]
    duration_ms: float
    error: Optional[str] = None


class WaveOrchestrator:
    """
    Orchestrates multi-stage wave processing for audio analysis.

    Implements progressive refinement through coordinated execution stages
    with resource management and quality gates.
    """

    def __init__(
        self,
        config: Optional[WaveConfig] = None,
        async_processor: Optional[Any] = None,
        batch_processor: Optional[Any] = None,
    ):
        """
        Initialize the wave orchestrator.

        Args:
            config: Wave configuration
            async_processor: AsyncAudioProcessor for parallel execution
            batch_processor: BatchProcessor for batch operations
        """
        self.config = config or WaveConfig()
        self.async_processor = async_processor
        self.batch_processor = batch_processor

        # Stage handlers
        self.stage_handlers: Dict[WaveStage, Callable] = {}
        self.stage_dependencies: Dict[WaveStage, List[WaveStage]] = defaultdict(list)

        # Wave tracking
        self.active_waves: Dict[str, WaveContext] = {}
        self.wave_lock = asyncio.Lock()

        # Metrics
        self.total_waves_processed = 0
        self.successful_waves = 0
        self.failed_waves = 0

        self._setup_default_dependencies()
        logger.info(f"WaveOrchestrator initialized with {len(self.config.enabled_stages)} stages")

    def _setup_default_dependencies(self) -> None:
        """Set up default stage dependencies."""
        # Define the default pipeline flow
        self.stage_dependencies[WaveStage.INITIAL_ANALYSIS] = [WaveStage.PREPARATION]
        self.stage_dependencies[WaveStage.FEATURE_EXTRACTION] = [WaveStage.INITIAL_ANALYSIS]
        self.stage_dependencies[WaveStage.DEEP_ANALYSIS] = [WaveStage.FEATURE_EXTRACTION]
        self.stage_dependencies[WaveStage.AGGREGATION] = [
            WaveStage.INITIAL_ANALYSIS,
            WaveStage.FEATURE_EXTRACTION,
            WaveStage.DEEP_ANALYSIS,
        ]
        self.stage_dependencies[WaveStage.POST_PROCESSING] = [WaveStage.AGGREGATION]
        self.stage_dependencies[WaveStage.FINALIZATION] = [WaveStage.POST_PROCESSING]

    def register_stage_handler(
        self,
        stage: WaveStage,
        handler: Callable,
        dependencies: Optional[List[WaveStage]] = None,
    ) -> None:
        """
        Register a handler for a specific stage.

        Args:
            stage: The stage to register the handler for
            handler: The handler function (can be async or sync)
            dependencies: Optional list of stages that must complete before this stage
        """
        self.stage_handlers[stage] = handler
        if dependencies:
            self.stage_dependencies[stage] = dependencies
        logger.debug(f"Registered handler for stage {stage.value}")

    async def execute_wave(
        self,
        wave_id: str,
        input_data: Any,
        stages: Optional[List[WaveStage]] = None,
    ) -> WaveContext:
        """
        Execute a wave of processing through multiple stages.

        Args:
            wave_id: Unique identifier for this wave
            input_data: Input data for the wave
            stages: Optional list of stages to execute (defaults to all enabled)

        Returns:
            WaveContext with execution results
        """
        stages = stages or self.config.enabled_stages
        context = WaveContext(
            wave_id=wave_id,
            start_time=datetime.now(),
        )

        async with self.wave_lock:
            self.active_waves[wave_id] = context

        try:
            context.status = WaveStatus.IN_PROGRESS
            logger.info(f"Starting wave {wave_id} with {len(stages)} stages")

            # Execute stages in order
            for stage in stages:
                if not await self._can_execute_stage(stage, context):
                    logger.warning(f"Skipping stage {stage.value} due to unmet dependencies")
                    continue

                result = await self._execute_stage(stage, input_data, context)

                if result.status == WaveStatus.FAILED:
                    if not self.config.allow_partial_results:
                        context.status = WaveStatus.FAILED
                        break
                    context.failed_stages.add(stage)
                else:
                    context.completed_stages.add(stage)
                    context.stage_results[stage] = result.data

                    # Check for early termination based on quality
                    if self.config.early_termination_on_quality:
                        # Check if the latest result has a quality metric
                        if isinstance(result.data, dict) and "quality" in result.data:
                            quality = result.data["quality"]
                            if quality >= self.config.quality_threshold:
                                logger.info(f"Early termination: quality threshold {quality:.2f} reached")
                                break

                # Inter-stage delay
                if self.config.inter_stage_delay_ms > 0:
                    await asyncio.sleep(self.config.inter_stage_delay_ms / 1000)

            # Determine final status
            if context.failed_stages and not self.config.allow_partial_results:
                context.status = WaveStatus.FAILED
                self.failed_waves += 1
            elif context.completed_stages:
                if len(context.completed_stages) == len(stages):
                    context.status = WaveStatus.COMPLETED
                    self.successful_waves += 1
                else:
                    context.status = WaveStatus.PARTIAL
            else:
                context.status = WaveStatus.FAILED
                self.failed_waves += 1

            self.total_waves_processed += 1

            # Calculate final metrics
            context.metrics = self._calculate_wave_metrics(context)

            logger.info(
                f"Wave {wave_id} completed with status {context.status.value}, "
                f"completed stages: {len(context.completed_stages)}/{len(stages)}"
            )

            return context

        except Exception as e:
            logger.error(f"Wave {wave_id} failed with error: {str(e)}")
            context.status = WaveStatus.FAILED
            context.errors.append({"error": str(e), "timestamp": datetime.now()})
            self.failed_waves += 1
            return context

        finally:
            async with self.wave_lock:
                if wave_id in self.active_waves:
                    del self.active_waves[wave_id]

    async def _can_execute_stage(self, stage: WaveStage, context: WaveContext) -> bool:
        """
        Check if a stage can be executed based on dependencies.

        Args:
            stage: Stage to check
            context: Current wave context

        Returns:
            True if stage can be executed
        """
        dependencies = self.stage_dependencies.get(stage, [])
        for dep in dependencies:
            if dep not in context.completed_stages:
                return False
        return True

    async def _execute_stage(
        self,
        stage: WaveStage,
        input_data: Any,
        context: WaveContext,
    ) -> StageResult:
        """
        Execute a single stage of the wave.

        Args:
            stage: Stage to execute
            input_data: Input data for the stage
            context: Wave context

        Returns:
            StageResult with execution outcome
        """
        start_time = datetime.now()
        context.current_stage = stage

        try:
            handler = self.stage_handlers.get(stage)
            if not handler:
                # Use default handler if no specific handler registered
                handler = self._default_stage_handler

            logger.debug(f"Executing stage {stage.value} for wave {context.wave_id}")

            # Prepare stage input based on previous results
            stage_input = self._prepare_stage_input(stage, input_data, context)

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_handler(handler, stage_input, context),
                timeout=self.config.stage_timeout_seconds,
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return StageResult(
                stage=stage,
                status=WaveStatus.COMPLETED,
                data=result,
                metrics={"duration_ms": duration_ms},
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            logger.error(f"Stage {stage.value} timed out after {self.config.stage_timeout_seconds}s")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=stage,
                status=WaveStatus.FAILED,
                data=None,
                metrics={"duration_ms": duration_ms},
                duration_ms=duration_ms,
                error="Stage timeout",
            )

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {str(e)}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=stage,
                status=WaveStatus.FAILED,
                data=None,
                metrics={"duration_ms": duration_ms},
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _execute_handler(
        self,
        handler: Callable,
        input_data: Any,
        context: WaveContext,
    ) -> Any:
        """
        Execute a handler function (async or sync).

        Args:
            handler: Handler function
            input_data: Input data
            context: Wave context

        Returns:
            Handler result
        """
        if asyncio.iscoroutinefunction(handler):
            return await handler(input_data, context)
        else:
            # Run sync handler in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, handler, input_data, context)

    def _prepare_stage_input(
        self,
        stage: WaveStage,
        original_input: Any,
        context: WaveContext,
    ) -> Any:
        """
        Prepare input for a stage based on previous results.

        Args:
            stage: Current stage
            original_input: Original input data
            context: Wave context

        Returns:
            Prepared input for the stage
        """
        # For initial stages, use original input
        if stage in [WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS]:
            return original_input

        # For later stages, combine original input with previous results
        stage_input = {
            "original": original_input,
            "previous_results": dict(context.stage_results),
            "stage": stage.value,
            "wave_id": context.wave_id,
        }

        return stage_input

    async def _default_stage_handler(self, input_data: Any, context: WaveContext) -> Any:
        """
        Default handler for stages without specific handlers.

        Args:
            input_data: Stage input
            context: Wave context

        Returns:
            Processed data
        """
        logger.debug(f"Default handler for stage {context.current_stage.value if context.current_stage else 'unknown'}")
        # Simulate some processing
        await asyncio.sleep(0.1)
        return {"processed": True, "stage": context.current_stage.value if context.current_stage else None}

    def _calculate_quality_score(self, context: WaveContext) -> float:
        """
        Calculate quality score for current wave state.

        Args:
            context: Wave context

        Returns:
            Quality score between 0 and 1
        """
        if not context.completed_stages:
            return 0.0

        # Base score on completed stages
        stage_score = len(context.completed_stages) / len(self.config.enabled_stages)

        # Adjust for failed stages
        if context.failed_stages:
            failure_penalty = len(context.failed_stages) / len(self.config.enabled_stages)
            stage_score -= failure_penalty * 0.5

        # Consider stage-specific quality metrics if available
        quality_sum = 0.0
        quality_count = 0
        for stage, result in context.stage_results.items():
            if isinstance(result, dict) and "quality" in result:
                quality_sum += result["quality"]
                quality_count += 1

        if quality_count > 0:
            avg_quality = quality_sum / quality_count
            return (stage_score + avg_quality) / 2

        return max(0.0, min(1.0, stage_score))

    def _calculate_wave_metrics(self, context: WaveContext) -> Dict[str, Any]:
        """
        Calculate metrics for the completed wave.

        Args:
            context: Wave context

        Returns:
            Dictionary of metrics
        """
        total_duration = (datetime.now() - context.start_time).total_seconds()

        # Calculate stage durations
        stage_durations = {}
        for stage, result in context.stage_results.items():
            if isinstance(result, dict) and "duration_ms" in result:
                stage_durations[stage.value] = result["duration_ms"]

        return {
            "total_duration_seconds": total_duration,
            "completed_stages": len(context.completed_stages),
            "failed_stages": len(context.failed_stages),
            "success_rate": len(context.completed_stages) / len(self.config.enabled_stages)
            if self.config.enabled_stages
            else 0,
            "quality_score": self._calculate_quality_score(context),
            "stage_durations": stage_durations,
            "errors": len(context.errors),
        }

    async def execute_parallel_waves(
        self,
        waves: List[Tuple[str, Any]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, WaveContext]:
        """
        Execute multiple waves in parallel.

        Args:
            waves: List of (wave_id, input_data) tuples
            max_concurrent: Maximum concurrent waves

        Returns:
            Dictionary mapping wave_id to WaveContext
        """
        max_concurrent = max_concurrent or self.config.max_parallel_items_per_stage
        results = {}

        # Process waves in batches
        for i in range(0, len(waves), max_concurrent):
            batch = waves[i : i + max_concurrent]
            tasks = []

            for wave_id, input_data in batch:
                task = asyncio.create_task(self.execute_wave(wave_id, input_data))
                tasks.append((wave_id, task))

            # Wait for batch to complete
            for wave_id, task in tasks:
                try:
                    context = await task
                    results[wave_id] = context
                except Exception as e:
                    logger.error(f"Failed to execute wave {wave_id}: {str(e)}")
                    results[wave_id] = WaveContext(
                        wave_id=wave_id,
                        start_time=datetime.now(),
                        status=WaveStatus.FAILED,
                        errors=[{"error": str(e)}],
                    )

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with statistics
        """
        success_rate = (
            (self.successful_waves / self.total_waves_processed * 100) if self.total_waves_processed > 0 else 0
        )

        return {
            "total_waves": self.total_waves_processed,
            "successful_waves": self.successful_waves,
            "failed_waves": self.failed_waves,
            "success_rate": success_rate,
            "active_waves": len(self.active_waves),
            "enabled_stages": [stage.value for stage in self.config.enabled_stages],
        }

    async def cancel_wave(self, wave_id: str) -> bool:
        """
        Cancel an active wave.

        Args:
            wave_id: Wave ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self.wave_lock:
            if wave_id in self.active_waves:
                context = self.active_waves[wave_id]
                context.status = WaveStatus.CANCELLED
                logger.info(f"Cancelled wave {wave_id}")
                return True
        return False

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and clean up resources."""
        logger.info("Shutting down WaveOrchestrator")

        # Cancel all active waves
        async with self.wave_lock:
            for wave_id in list(self.active_waves.keys()):
                await self.cancel_wave(wave_id)

        logger.info(
            f"WaveOrchestrator shutdown complete. "
            f"Processed {self.total_waves_processed} waves "
            f"(success: {self.successful_waves}, failed: {self.failed_waves})"
        )
