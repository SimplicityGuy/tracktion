"""
Unit tests for the WaveOrchestrator implementation.

Tests multi-stage wave processing, progressive refinement, and resource management.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from services.analysis_service.src.wave_orchestrator import (
    WaveConfig,
    WaveContext,
    WaveOrchestrator,
    WaveStage,
    WaveStatus,
)


@pytest.fixture
def wave_config():
    """Create a test wave configuration."""
    return WaveConfig(
        enabled_stages=[
            WaveStage.PREPARATION,
            WaveStage.INITIAL_ANALYSIS,
            WaveStage.FEATURE_EXTRACTION,
            WaveStage.AGGREGATION,
            WaveStage.FINALIZATION,
        ],
        max_parallel_items_per_stage=5,
        stage_timeout_seconds=10,
        max_retries_per_stage=1,
        inter_stage_delay_ms=10,
        min_success_rate_per_stage=0.7,
        allow_partial_results=True,
        progressive_mode=True,
        early_termination_on_quality=True,
        quality_threshold=0.9,
    )


@pytest.fixture
def wave_orchestrator(wave_config):
    """Create a WaveOrchestrator instance for testing."""
    return WaveOrchestrator(config=wave_config)


class TestWaveOrchestratorInit:
    """Test WaveOrchestrator initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        orchestrator = WaveOrchestrator()
        assert orchestrator.config is not None
        assert len(orchestrator.config.enabled_stages) == 7
        assert orchestrator.total_waves_processed == 0
        assert orchestrator.successful_waves == 0
        assert orchestrator.failed_waves == 0

    def test_init_with_custom_config(self, wave_config):
        """Test initialization with custom configuration."""
        orchestrator = WaveOrchestrator(config=wave_config)
        assert orchestrator.config == wave_config
        assert len(orchestrator.config.enabled_stages) == 5

    def test_init_with_processors(self, wave_config):
        """Test initialization with async and batch processors."""
        async_processor = MagicMock()
        batch_processor = MagicMock()
        orchestrator = WaveOrchestrator(
            config=wave_config,
            async_processor=async_processor,
            batch_processor=batch_processor,
        )
        assert orchestrator.async_processor == async_processor
        assert orchestrator.batch_processor == batch_processor

    def test_default_dependencies_setup(self, wave_orchestrator):
        """Test that default stage dependencies are set up correctly."""
        deps = wave_orchestrator.stage_dependencies
        assert WaveStage.PREPARATION in deps[WaveStage.INITIAL_ANALYSIS]
        assert WaveStage.INITIAL_ANALYSIS in deps[WaveStage.FEATURE_EXTRACTION]
        assert WaveStage.POST_PROCESSING in deps[WaveStage.FINALIZATION]


class TestStageHandlerRegistration:
    """Test stage handler registration."""

    def test_register_stage_handler(self, wave_orchestrator):
        """Test registering a stage handler."""
        handler = MagicMock()
        wave_orchestrator.register_stage_handler(WaveStage.INITIAL_ANALYSIS, handler)
        assert wave_orchestrator.stage_handlers[WaveStage.INITIAL_ANALYSIS] == handler

    def test_register_handler_with_dependencies(self, wave_orchestrator):
        """Test registering a handler with custom dependencies."""
        handler = MagicMock()
        dependencies = [WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS]
        wave_orchestrator.register_stage_handler(
            WaveStage.FEATURE_EXTRACTION,
            handler,
            dependencies=dependencies,
        )
        assert wave_orchestrator.stage_dependencies[WaveStage.FEATURE_EXTRACTION] == dependencies


class TestWaveExecution:
    """Test wave execution."""

    @pytest.mark.asyncio
    async def test_execute_wave_basic(self, wave_orchestrator):
        """Test basic wave execution."""

        # Register simple handlers
        async def prep_handler(data, context):
            return {"prepared": True}

        async def analysis_handler(data, context):
            return {"analyzed": True}

        wave_orchestrator.register_stage_handler(WaveStage.PREPARATION, prep_handler)
        wave_orchestrator.register_stage_handler(WaveStage.INITIAL_ANALYSIS, analysis_handler)

        # Execute wave with only two stages
        context = await wave_orchestrator.execute_wave(
            wave_id="test_wave_1",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS],
        )

        assert context.wave_id == "test_wave_1"
        assert context.status == WaveStatus.COMPLETED
        assert WaveStage.PREPARATION in context.completed_stages
        assert WaveStage.INITIAL_ANALYSIS in context.completed_stages
        assert len(context.failed_stages) == 0

    @pytest.mark.asyncio
    async def test_execute_wave_with_failure(self, wave_orchestrator):
        """Test wave execution with stage failure."""

        async def failing_handler(data, context):
            raise Exception("Stage failed")

        wave_orchestrator.register_stage_handler(WaveStage.PREPARATION, failing_handler)

        context = await wave_orchestrator.execute_wave(
            wave_id="test_wave_2",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION],  # Use PREPARATION which has no dependencies
        )

        assert context.status == WaveStatus.FAILED
        assert WaveStage.PREPARATION in context.failed_stages
        assert len(context.completed_stages) == 0

    @pytest.mark.asyncio
    async def test_execute_wave_with_partial_results(self, wave_orchestrator):
        """Test wave execution with partial results allowed."""

        async def good_handler(data, context):
            return {"success": True}

        async def bad_handler(data, context):
            raise Exception("Failed")

        wave_orchestrator.register_stage_handler(WaveStage.PREPARATION, good_handler)
        wave_orchestrator.register_stage_handler(WaveStage.INITIAL_ANALYSIS, bad_handler)
        wave_orchestrator.config.allow_partial_results = True

        context = await wave_orchestrator.execute_wave(
            wave_id="test_wave_3",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS],
        )

        assert context.status == WaveStatus.PARTIAL
        assert WaveStage.PREPARATION in context.completed_stages
        assert WaveStage.INITIAL_ANALYSIS in context.failed_stages

    @pytest.mark.asyncio
    async def test_execute_wave_with_timeout(self, wave_orchestrator):
        """Test wave execution with stage timeout."""

        async def slow_handler(data, context):
            await asyncio.sleep(20)  # Longer than timeout
            return {"done": True}

        wave_orchestrator.register_stage_handler(WaveStage.PREPARATION, slow_handler)
        wave_orchestrator.config.stage_timeout_seconds = 1

        context = await wave_orchestrator.execute_wave(
            wave_id="test_wave_4",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION],  # Use PREPARATION which has no dependencies
        )

        assert context.status == WaveStatus.FAILED
        assert WaveStage.PREPARATION in context.failed_stages

    @pytest.mark.asyncio
    async def test_execute_wave_with_dependencies(self, wave_orchestrator):
        """Test wave execution respects stage dependencies."""
        executed_stages = []

        async def tracking_handler(stage_name):
            async def handler(data, context):
                executed_stages.append(stage_name)
                return {stage_name: True}

            return handler

        # Register handlers that track execution order
        wave_orchestrator.register_stage_handler(
            WaveStage.PREPARATION,
            await tracking_handler("prep"),
        )
        wave_orchestrator.register_stage_handler(
            WaveStage.INITIAL_ANALYSIS,
            await tracking_handler("analysis"),
        )
        wave_orchestrator.register_stage_handler(
            WaveStage.FEATURE_EXTRACTION,
            await tracking_handler("features"),
        )

        context = await wave_orchestrator.execute_wave(
            wave_id="test_wave_5",
            input_data={"test": "data"},
            stages=[
                WaveStage.PREPARATION,
                WaveStage.INITIAL_ANALYSIS,
                WaveStage.FEATURE_EXTRACTION,
            ],
        )

        assert context.status == WaveStatus.COMPLETED
        # Verify execution order respects dependencies
        assert executed_stages.index("prep") < executed_stages.index("analysis")
        assert executed_stages.index("analysis") < executed_stages.index("features")


class TestEarlyTermination:
    """Test early termination based on quality."""

    @pytest.mark.asyncio
    async def test_early_termination_on_quality(self, wave_orchestrator):
        """Test that wave terminates early when quality threshold is met."""
        executed_stages = []

        def quality_handler(stage_name, quality):
            async def handler(data, context):
                executed_stages.append(stage_name)
                return {"quality": quality}

            return handler

        # Register handlers with increasing quality scores
        wave_orchestrator.register_stage_handler(
            WaveStage.PREPARATION,
            quality_handler("prep", 0.5),
        )
        wave_orchestrator.register_stage_handler(
            WaveStage.INITIAL_ANALYSIS,
            quality_handler("analysis", 0.95),  # Meets threshold
        )
        wave_orchestrator.register_stage_handler(
            WaveStage.FEATURE_EXTRACTION,
            quality_handler("features", 1.0),  # Should not execute
        )

        wave_orchestrator.config.early_termination_on_quality = True
        wave_orchestrator.config.quality_threshold = 0.9

        await wave_orchestrator.execute_wave(
            wave_id="test_wave_6",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS, WaveStage.FEATURE_EXTRACTION],
        )

        assert len(executed_stages) == 2  # Only first two stages
        assert "features" not in executed_stages


class TestParallelWaveExecution:
    """Test parallel wave execution."""

    @pytest.mark.asyncio
    async def test_execute_parallel_waves(self, wave_orchestrator):
        """Test executing multiple waves in parallel."""

        async def simple_handler(data, context):
            await asyncio.sleep(0.1)  # Simulate work
            return {"wave_id": context.wave_id}

        wave_orchestrator.register_stage_handler(WaveStage.INITIAL_ANALYSIS, simple_handler)

        waves = [(f"wave_{i}", {"data": i}) for i in range(10)]

        results = await wave_orchestrator.execute_parallel_waves(
            waves,
            max_concurrent=3,
        )

        assert len(results) == 10
        for wave_id, context in results.items():
            assert context.wave_id == wave_id
            assert context.status in [WaveStatus.COMPLETED, WaveStatus.PARTIAL]

    @pytest.mark.asyncio
    async def test_parallel_waves_with_failures(self, wave_orchestrator):
        """Test parallel wave execution with some failures."""

        async def mixed_handler(data, context):
            # Fail every third wave
            if int(context.wave_id.split("_")[1]) % 3 == 0:
                raise Exception("Intentional failure")
            return {"success": True}

        wave_orchestrator.register_stage_handler(WaveStage.PREPARATION, mixed_handler)

        waves = [(f"wave_{i}", {"data": i}) for i in range(9)]

        results = await wave_orchestrator.execute_parallel_waves(waves, max_concurrent=3)

        failed_count = sum(1 for ctx in results.values() if ctx.status == WaveStatus.FAILED)
        assert failed_count == 3  # waves 0, 3, 6


class TestWaveMetrics:
    """Test wave metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_wave_metrics_calculation(self, wave_orchestrator):
        """Test that wave metrics are calculated correctly."""

        async def timed_handler(delay):
            async def handler(data, context):
                await asyncio.sleep(delay)
                return {"duration_ms": delay * 1000}

            return handler

        wave_orchestrator.register_stage_handler(
            WaveStage.PREPARATION,
            await timed_handler(0.1),
        )
        wave_orchestrator.register_stage_handler(
            WaveStage.INITIAL_ANALYSIS,
            await timed_handler(0.2),
        )

        context = await wave_orchestrator.execute_wave(
            wave_id="metrics_wave",
            input_data={"test": "data"},
            stages=[WaveStage.PREPARATION, WaveStage.INITIAL_ANALYSIS],
        )

        assert "total_duration_seconds" in context.metrics
        assert context.metrics["completed_stages"] == 2
        assert context.metrics["failed_stages"] == 0
        assert context.metrics["success_rate"] == 0.4  # 2 out of 5 enabled stages

    def test_get_statistics(self, wave_orchestrator):
        """Test getting orchestrator statistics."""
        wave_orchestrator.total_waves_processed = 10
        wave_orchestrator.successful_waves = 7
        wave_orchestrator.failed_waves = 3

        stats = wave_orchestrator.get_statistics()

        assert stats["total_waves"] == 10
        assert stats["successful_waves"] == 7
        assert stats["failed_waves"] == 3
        assert stats["success_rate"] == 70.0
        assert len(stats["enabled_stages"]) == 5


class TestWaveCancellation:
    """Test wave cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_active_wave(self, wave_orchestrator):
        """Test cancelling an active wave."""

        async def long_handler(data, context):
            for _ in range(10):
                await asyncio.sleep(0.5)
                if context.status == WaveStatus.CANCELLED:
                    return {"cancelled": True}
            return {"completed": True}

        wave_orchestrator.register_stage_handler(WaveStage.INITIAL_ANALYSIS, long_handler)

        # Start wave execution
        wave_task = asyncio.create_task(
            wave_orchestrator.execute_wave(
                wave_id="cancellable_wave",
                input_data={"test": "data"},
                stages=[WaveStage.INITIAL_ANALYSIS],
            )
        )

        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        cancelled = await wave_orchestrator.cancel_wave("cancellable_wave")

        assert cancelled is True

        # Clean up
        wave_task.cancel()
        try:
            await wave_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_wave(self, wave_orchestrator):
        """Test cancelling a non-existent wave."""
        cancelled = await wave_orchestrator.cancel_wave("nonexistent")
        assert cancelled is False


class TestStageInputPreparation:
    """Test stage input preparation."""

    def test_prepare_stage_input_initial_stages(self, wave_orchestrator):
        """Test input preparation for initial stages."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )

        original_input = {"original": "data"}

        # Initial stages should get original input
        prep_input = wave_orchestrator._prepare_stage_input(
            WaveStage.PREPARATION,
            original_input,
            context,
        )
        assert prep_input == original_input

        analysis_input = wave_orchestrator._prepare_stage_input(
            WaveStage.INITIAL_ANALYSIS,
            original_input,
            context,
        )
        assert analysis_input == original_input

    def test_prepare_stage_input_later_stages(self, wave_orchestrator):
        """Test input preparation for later stages."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )
        context.stage_results[WaveStage.PREPARATION] = {"prep": "result"}
        context.stage_results[WaveStage.INITIAL_ANALYSIS] = {"analysis": "result"}

        original_input = {"original": "data"}

        # Later stages should get combined input
        feature_input = wave_orchestrator._prepare_stage_input(
            WaveStage.FEATURE_EXTRACTION,
            original_input,
            context,
        )

        assert feature_input["original"] == original_input
        assert feature_input["previous_results"][WaveStage.PREPARATION] == {"prep": "result"}
        assert feature_input["previous_results"][WaveStage.INITIAL_ANALYSIS] == {"analysis": "result"}
        assert feature_input["stage"] == "feature_extraction"
        assert feature_input["wave_id"] == "test"


class TestQualityScore:
    """Test quality score calculation."""

    def test_calculate_quality_score_no_stages(self, wave_orchestrator):
        """Test quality score with no completed stages."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )

        score = wave_orchestrator._calculate_quality_score(context)
        assert score == 0.0

    def test_calculate_quality_score_all_completed(self, wave_orchestrator):
        """Test quality score with all stages completed."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )
        # Mark all enabled stages as completed
        for stage in wave_orchestrator.config.enabled_stages:
            context.completed_stages.add(stage)

        score = wave_orchestrator._calculate_quality_score(context)
        assert score == 1.0

    def test_calculate_quality_score_with_failures(self, wave_orchestrator):
        """Test quality score with some failed stages."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )
        # Complete 3 out of 5 stages
        context.completed_stages.add(WaveStage.PREPARATION)
        context.completed_stages.add(WaveStage.INITIAL_ANALYSIS)
        context.completed_stages.add(WaveStage.FEATURE_EXTRACTION)
        # Fail 1 stage
        context.failed_stages.add(WaveStage.AGGREGATION)

        score = wave_orchestrator._calculate_quality_score(context)
        # 3/5 completed = 0.6, minus penalty for 1/5 failed = 0.6 - 0.1 = 0.5
        assert 0.4 <= score <= 0.6

    def test_calculate_quality_score_with_stage_metrics(self, wave_orchestrator):
        """Test quality score with stage-specific quality metrics."""
        context = WaveContext(
            wave_id="test",
            start_time=datetime.now(),
        )
        context.completed_stages.add(WaveStage.PREPARATION)
        context.completed_stages.add(WaveStage.INITIAL_ANALYSIS)

        # Add quality metrics to stage results
        context.stage_results[WaveStage.PREPARATION] = {"quality": 0.8}
        context.stage_results[WaveStage.INITIAL_ANALYSIS] = {"quality": 0.9}

        score = wave_orchestrator._calculate_quality_score(context)
        # (stage_score + avg_quality) / 2 = (0.4 + 0.85) / 2 = 0.625
        assert 0.6 <= score <= 0.7


class TestShutdown:
    """Test orchestrator shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(self, wave_orchestrator):
        """Test orchestrator shutdown."""

        # Add some fake active waves
        wave_orchestrator.active_waves["wave1"] = WaveContext(
            wave_id="wave1",
            start_time=datetime.now(),
        )
        wave_orchestrator.active_waves["wave2"] = WaveContext(
            wave_id="wave2",
            start_time=datetime.now(),
        )

        await wave_orchestrator.shutdown()

        # All active waves should be cancelled
        assert len(wave_orchestrator.active_waves) == 0
