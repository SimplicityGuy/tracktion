"""Integration tests for the feedback system components.

Tests the complete feedback flow from submission to storage to learning,
including API endpoints, database transactions, cache consistency,
A/B testing, and resource management under load.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from services.file_rename_service.api.feedback_routes import router as feedback_router
from services.file_rename_service.app.feedback.experiments import ExperimentManager
from services.file_rename_service.app.feedback.learning import OnlineLearner
from services.file_rename_service.app.feedback.metrics import MetricsTracker
from services.file_rename_service.app.feedback.models import ExperimentStatus, Feedback, FeedbackAction
from services.file_rename_service.app.feedback.processor import (
    BackpressureError,
    BackpressureStrategy,
    FeedbackProcessor,
)
from services.file_rename_service.app.feedback.storage import FeedbackStorage

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test Configuration
TEST_POSTGRES_DSN = os.getenv("TEST_POSTGRES_DSN", "postgresql://tracktion_user:changeme@localhost:5432/test_feedback")
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost:6379/1",  # Use database 1 for testing
)
TEST_API_KEY = "test-api-key-123"
TEST_ADMIN_KEY = "test-admin-key-789"


# Test Database Setup
@asynccontextmanager
async def setup_test_database() -> AsyncGenerator[tuple[str, str]]:
    """Setup isolated test database and Redis instance."""
    # Create unique database name
    db_suffix = str(uuid4())[:8]
    test_db_name = f"test_feedback_{db_suffix}"

    # Connect to postgres to create test database
    base_dsn = TEST_POSTGRES_DSN.rsplit("/", 1)[0]
    admin_conn = await asyncpg.connect(f"{base_dsn}/postgres")

    try:
        # Create test database
        await admin_conn.execute(f'CREATE DATABASE "{test_db_name}"')
        test_dsn = f"{base_dsn}/{test_db_name}"

        # Use Redis database with unique suffix
        redis_db = hash(db_suffix) % 16  # Use hash to get consistent db number
        test_redis_url = f"{TEST_REDIS_URL.rsplit('/', 1)[0]}/{redis_db}"

        # Clear Redis database
        redis_client = Redis.from_url(test_redis_url, decode_responses=True)
        await redis_client.flushdb()
        await redis_client.aclose()

        yield test_dsn, test_redis_url

    finally:
        # Cleanup: Drop test database
        try:
            # Terminate connections to test database
            await admin_conn.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{test_db_name}'"
            )
            await admin_conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
        except Exception as e:
            logger.warning(f"Failed to cleanup test database: {e}")
        finally:
            await admin_conn.close()


@pytest_asyncio.fixture
async def test_storage() -> AsyncGenerator[FeedbackStorage]:
    """Create isolated test storage instance."""
    async with setup_test_database() as (test_dsn, test_redis_url):
        storage = FeedbackStorage(
            postgres_dsn=test_dsn,
            redis_url=test_redis_url,
            cache_ttl_seconds=60,
        )
        await storage.initialize()
        try:
            yield storage
        finally:
            await storage.close()


@pytest_asyncio.fixture
async def test_processor(test_storage: FeedbackStorage) -> FeedbackProcessor:
    """Create test feedback processor."""
    processor = FeedbackProcessor(
        storage=test_storage,
        batch_size=5,  # Small batch size for testing
        batch_timeout_seconds=1,  # Short timeout for testing
        retrain_threshold=10,  # Low threshold for testing
        max_pending_size=20,
        backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
    )
    await processor.start_monitoring()
    return processor


@pytest_asyncio.fixture
async def test_experiment_manager(test_storage: FeedbackStorage) -> ExperimentManager:
    """Create test experiment manager."""
    return ExperimentManager(
        storage=test_storage,
        min_sample_size=5,  # Low threshold for testing
        significance_level=0.05,
        power_threshold=0.8,
    )


@pytest_asyncio.fixture
async def test_metrics_tracker(test_storage: FeedbackStorage) -> MetricsTracker:
    """Create test metrics tracker."""
    return MetricsTracker(storage=test_storage, window_size=10)


@pytest_asyncio.fixture
async def mock_online_learner(test_storage: FeedbackStorage) -> OnlineLearner:
    """Create mock online learner for testing."""
    learner = OnlineLearner(
        storage=test_storage,
        model_path=Path("/tmp/test_models"),
        learning_rate=0.01,
        min_feedback_for_update=3,
    )
    # Mock the actual ML operations
    learner._apply_gradient_update = AsyncMock(return_value={"loss": 0.1})
    learner._save_model = AsyncMock()
    learner._load_model = AsyncMock()
    return learner


@pytest_asyncio.fixture
async def test_app(
    test_storage: FeedbackStorage,
    test_processor: FeedbackProcessor,
    test_experiment_manager: ExperimentManager,
    test_metrics_tracker: MetricsTracker,
    mock_online_learner: OnlineLearner,
) -> FastAPI:
    """Create test FastAPI application with mocked dependencies."""
    app = FastAPI()

    # Mock the dependency injection
    async def get_test_storage():
        return test_storage

    async def get_test_processor():
        return test_processor

    async def get_test_experiment_manager():
        return test_experiment_manager

    async def get_test_metrics_tracker():
        return test_metrics_tracker

    async def get_test_online_learner():
        return mock_online_learner

    # Override dependencies
    feedback_router.dependency_overrides = {
        "get_storage": get_test_storage,
        "get_processor": get_test_processor,
        "get_experiment_manager": get_test_experiment_manager,
        "get_metrics_tracker": get_test_metrics_tracker,
        "get_online_learner": get_test_online_learner,
    }

    # Include router
    app.include_router(feedback_router)

    # Mock authentication for testing
    with (
        patch("services.file_rename_service.api.feedback_routes.verify_api_key") as mock_api_key,
        patch("services.file_rename_service.api.feedback_routes.verify_admin_key") as mock_admin_key,
    ):
        mock_api_key.return_value = {"api_key": "test-key", "is_admin": False, "permissions": ["user"]}

        mock_admin_key.return_value = {"api_key": "admin-key", "is_admin": True, "permissions": ["admin"]}

        yield app

    # Cleanup
    feedback_router.dependency_overrides = {}


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(test_app)


# Helper Functions
def create_sample_feedback(
    action: FeedbackAction = FeedbackAction.APPROVED, confidence: float = 0.8, model_version: str = "v1.0.0"
) -> dict[str, Any]:
    """Create sample feedback data for testing."""
    feedback_data = {
        "proposal_id": str(uuid4()),
        "original_filename": f"old_{uuid4().hex[:8]}.txt",
        "proposed_filename": f"new_{uuid4().hex[:8]}.txt",
        "user_action": action.value,
        "confidence_score": confidence,
        "model_version": model_version,
        "context_metadata": {"file_size": 1024, "file_type": "text", "user_agent": "test-client"},
    }

    if action == FeedbackAction.MODIFIED:
        feedback_data["user_filename"] = f"custom_{uuid4().hex[:8]}.txt"

    return feedback_data


# Integration Tests
@pytest.mark.asyncio
class TestEndToEndFeedbackFlow:
    """Test complete end-to-end feedback flow."""

    async def test_complete_feedback_submission_and_processing(
        self, test_storage: FeedbackStorage, test_processor: FeedbackProcessor
    ):
        """Test complete feedback submission, storage, and batch processing."""
        # Submit multiple feedbacks
        feedbacks = []
        for i in range(7):  # More than batch size to trigger processing
            feedback = await test_processor.submit_feedback(
                proposal_id=f"prop-{i}",
                original_filename=f"old{i}.txt",
                proposed_filename=f"new{i}.txt",
                user_action=FeedbackAction.APPROVED if i < 5 else FeedbackAction.REJECTED,
                confidence_score=0.7 + i * 0.02,
                model_version="v1.0.0",
                context_metadata={"test_id": i},
            )
            feedbacks.append(feedback)

        # Wait for batch processing
        await asyncio.sleep(0.1)

        # Verify feedbacks are stored
        stored_feedbacks = await test_storage.get_feedbacks(limit=10)
        assert len(stored_feedbacks) == 7

        # Verify batch was processed
        batches = await test_storage.get_batch_for_learning(processed=True)
        assert len(batches) > 0

        # Verify learning metrics were updated
        metrics = await test_storage.get_learning_metrics()
        assert metrics is not None
        assert metrics.total_feedback > 0
        assert 0.0 <= metrics.approval_rate <= 1.0

    async def test_feedback_flow_with_learning_integration(
        self, test_processor: FeedbackProcessor, mock_online_learner: OnlineLearner
    ):
        """Test feedback flow triggers learning updates."""
        # Submit enough feedbacks to trigger learning
        for i in range(12):  # Above retrain threshold
            await test_processor.submit_feedback(
                proposal_id=f"prop-{i}",
                original_filename=f"old{i}.txt",
                proposed_filename=f"new{i}.txt",
                user_action=FeedbackAction.APPROVED,
                confidence_score=0.8,
                model_version="v1.0.0",
            )

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify learning was triggered (via storage mark_retrain_triggered)
        metrics = await test_processor.storage.get_learning_metrics()
        assert metrics is not None

    async def test_feedback_processing_with_errors_and_recovery(
        self, test_storage: FeedbackStorage, test_processor: FeedbackProcessor
    ):
        """Test error handling and recovery in feedback processing."""
        # Mock storage error for testing
        original_store_batch = test_storage.store_batch
        call_count = 0

        async def failing_store_batch(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated database error")
            return await original_store_batch(batch)

        test_storage.store_batch = failing_store_batch

        try:
            # Submit feedbacks that should trigger batch processing
            for i in range(6):
                await test_processor.submit_feedback(
                    proposal_id=f"prop-{i}",
                    original_filename=f"old{i}.txt",
                    proposed_filename=f"new{i}.txt",
                    user_action=FeedbackAction.APPROVED,
                    confidence_score=0.8,
                    model_version="v1.0.0",
                )

            # Wait for processing
            await asyncio.sleep(0.1)

            # Force batch processing again (should succeed on retry)
            await test_processor.force_batch_processing()

            # Verify eventual consistency
            stored_feedbacks = await test_storage.get_feedbacks()
            assert len(stored_feedbacks) >= 6

        finally:
            test_storage.store_batch = original_store_batch


@pytest.mark.asyncio
class TestAPIEndpointsWithAuthentication:
    """Test API endpoints with authentication and rate limiting."""

    def test_feedback_approval_endpoint(self, test_client: TestClient):
        """Test feedback approval endpoint."""
        feedback_data = create_sample_feedback(FeedbackAction.APPROVED)

        response = test_client.post(
            "/feedback/approve", json=feedback_data, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "feedback_id" in result
        assert "processing_time_ms" in result

    def test_feedback_rejection_endpoint(self, test_client: TestClient):
        """Test feedback rejection endpoint."""
        feedback_data = create_sample_feedback(FeedbackAction.REJECTED)

        response = test_client.post(
            "/feedback/reject", json=feedback_data, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

    def test_feedback_modification_endpoint(self, test_client: TestClient):
        """Test feedback modification endpoint."""
        feedback_data = create_sample_feedback(FeedbackAction.MODIFIED)

        response = test_client.post(
            "/feedback/modify", json=feedback_data, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

    def test_feedback_modification_without_user_filename(self, test_client: TestClient):
        """Test feedback modification endpoint validation."""
        feedback_data = create_sample_feedback(FeedbackAction.MODIFIED)
        feedback_data.pop("user_filename")  # Remove required field

        response = test_client.post(
            "/feedback/modify", json=feedback_data, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 400
        assert "user_filename is required" in response.json()["detail"]

    def test_metrics_endpoint_with_authentication(self, test_client: TestClient):
        """Test metrics endpoint requires authentication."""
        # Without authentication
        response = test_client.get("/feedback/metrics")
        assert response.status_code == 401

        # With authentication
        response = test_client.get("/feedback/metrics", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert response.status_code == 200
        result = response.json()
        assert "metrics" in result
        assert "timestamp" in result

    def test_admin_endpoints_require_admin_key(self, test_client: TestClient):
        """Test admin endpoints require admin privileges."""
        # Test with regular API key
        experiment_data = {
            "name": "Test Experiment",
            "variant_a": "v1.0.0",
            "variant_b": "v2.0.0",
            "traffic_split": 0.5,
            "duration_hours": 24,
        }

        response = test_client.post(
            "/feedback/experiments", json=experiment_data, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )
        assert response.status_code == 403

        # Test with admin key (would work with proper admin mock)
        response = test_client.post(
            "/feedback/experiments", json=experiment_data, headers={"Authorization": f"Bearer {TEST_ADMIN_KEY}"}
        )
        # Should succeed due to mock admin verification
        assert response.status_code == 200

    def test_input_validation_and_sanitization(self, test_client: TestClient):
        """Test input validation and sanitization."""
        # Test with invalid data
        invalid_feedback = {
            "proposal_id": "",  # Empty
            "original_filename": "../../etc/passwd",  # Path traversal attempt
            "proposed_filename": "file<script>alert('xss')</script>.txt",  # XSS attempt
            "user_action": "invalid_action",  # Invalid enum
            "confidence_score": 1.5,  # Out of range
            "model_version": "",  # Empty
        }

        response = test_client.post(
            "/feedback/approve", json=invalid_feedback, headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
class TestDatabaseTransactionsAndRollback:
    """Test database transaction behavior and rollback scenarios."""

    async def test_feedback_storage_transaction_rollback(self, test_storage: FeedbackStorage):
        """Test transaction rollback on storage errors."""
        # Create feedback
        feedback = Feedback(
            id=str(uuid4()),
            proposal_id="test-prop",
            original_filename="old.txt",
            proposed_filename="new.txt",
            user_action=FeedbackAction.APPROVED,
            confidence_score=0.8,
            model_version="v1.0.0",
        )

        # Mock Redis error to test transaction rollback
        original_incr = test_storage._redis_client.incr
        test_storage._redis_client.incr = AsyncMock(side_effect=Exception("Redis error"))

        try:
            # Should fail and rollback
            with pytest.raises(Exception):  # noqa: B017
                await test_storage.store_feedback(feedback)

            # Verify feedback was not stored due to rollback
            stored_feedbacks = await test_storage.get_feedbacks()
            feedback_ids = [f.id for f in stored_feedbacks]
            assert feedback.id not in feedback_ids

        finally:
            test_storage._redis_client.incr = original_incr

    async def test_batch_processing_transaction_integrity(
        self, test_storage: FeedbackStorage, test_processor: FeedbackProcessor
    ):
        """Test batch processing maintains transaction integrity."""
        # Submit feedbacks
        feedback_ids = []
        for i in range(6):
            feedback = await test_processor.submit_feedback(
                proposal_id=f"prop-{i}",
                original_filename=f"old{i}.txt",
                proposed_filename=f"new{i}.txt",
                user_action=FeedbackAction.APPROVED,
                confidence_score=0.8,
                model_version="v1.0.0",
            )
            feedback_ids.append(feedback.id)

        # Wait for batch processing
        await asyncio.sleep(0.1)

        # Verify all feedbacks are stored
        stored_feedbacks = await test_storage.get_feedbacks()
        stored_ids = {f.id for f in stored_feedbacks}

        for feedback_id in feedback_ids:
            assert feedback_id in stored_ids

    async def test_concurrent_feedback_submission(self, test_processor: FeedbackProcessor):
        """Test concurrent feedback submissions maintain data consistency."""
        # Submit feedbacks concurrently
        tasks = []
        for i in range(15):
            task = test_processor.submit_feedback(
                proposal_id=f"concurrent-prop-{i}",
                original_filename=f"old{i}.txt",
                proposed_filename=f"new{i}.txt",
                user_action=FeedbackAction.APPROVED if i % 2 == 0 else FeedbackAction.REJECTED,
                confidence_score=0.7 + (i % 3) * 0.1,
                model_version="v1.0.0",
            )
            tasks.append(task)

        # Wait for all submissions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions
        for result in results:
            assert not isinstance(result, Exception)

        # Wait for batch processing
        await asyncio.sleep(0.2)

        # Verify all feedbacks were processed correctly
        stored_feedbacks = await test_processor.storage.get_feedbacks()
        assert len(stored_feedbacks) >= 15

        # Verify resource statistics
        stats = await test_processor.get_resource_stats()
        assert stats["total_processed"] >= 15
        assert stats["dropped_items"] == 0  # Should be no drops with sufficient queue size


@pytest.mark.asyncio
class TestCacheConsistency:
    """Test cache consistency between Redis and PostgreSQL."""

    async def test_cache_invalidation_on_updates(self, test_storage: FeedbackStorage):
        """Test cache is properly invalidated on updates."""
        # Store initial metrics to populate cache
        await test_storage.update_learning_metrics(
            {
                "model_versions": ["v1.0.0"],
                "total": 100,
                "approval_rate": 0.8,
                "rejection_rate": 0.15,
                "modification_rate": 0.05,
            }
        )

        # Get metrics (should cache them)
        metrics1 = await test_storage.get_learning_metrics()
        assert metrics1 is not None
        assert metrics1.total_feedback == 100

        # Update metrics
        await test_storage.update_learning_metrics(
            {
                "model_versions": ["v1.0.0"],
                "total": 150,
                "approval_rate": 0.85,
                "rejection_rate": 0.1,
                "modification_rate": 0.05,
            }
        )

        # Get metrics again (should be updated, not cached)
        metrics2 = await test_storage.get_learning_metrics()
        assert metrics2 is not None
        assert metrics2.total_feedback == 250  # Should be cumulative

    async def test_feedback_count_cache_consistency(
        self, test_storage: FeedbackStorage, test_processor: FeedbackProcessor
    ):
        """Test feedback count cache consistency."""
        # Submit several feedbacks
        for i in range(8):
            await test_processor.submit_feedback(
                proposal_id=f"count-prop-{i}",
                original_filename=f"old{i}.txt",
                proposed_filename=f"new{i}.txt",
                user_action=FeedbackAction.APPROVED,
                confidence_score=0.8,
                model_version="v1.0.0",
            )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Get count from cache
        cached_count = await test_storage.get_feedback_count_since_retrain()

        # Get count from database directly
        conn = await asyncpg.connect(test_storage.postgres_dsn)
        try:
            db_count = await conn.fetchval("SELECT COUNT(*) FROM feedback")
        finally:
            await conn.close()

        # Counts should be consistent
        assert cached_count == db_count
        assert cached_count >= 8

    async def test_cache_recovery_after_redis_failure(self, test_storage: FeedbackStorage):
        """Test system recovers gracefully from Redis failures."""
        # Store some feedbacks first
        feedback = Feedback(
            id=str(uuid4()),
            proposal_id="cache-test-prop",
            original_filename="old.txt",
            proposed_filename="new.txt",
            user_action=FeedbackAction.APPROVED,
            confidence_score=0.8,
            model_version="v1.0.0",
        )
        await test_storage.store_feedback(feedback)

        # Simulate Redis failure
        original_client = test_storage._redis_client
        test_storage._redis_client = None

        try:
            # Should still work with database fallback
            count = await test_storage.get_feedback_count_since_retrain()
            assert count >= 0  # Should get count from database

        finally:
            test_storage._redis_client = original_client


@pytest.mark.asyncio
class TestABTestingIntegration:
    """Test A/B testing with real model variants."""

    async def test_complete_experiment_lifecycle(
        self, test_experiment_manager: ExperimentManager, test_processor: FeedbackProcessor
    ):
        """Test complete A/B experiment lifecycle."""
        # Create experiment
        experiment = await test_experiment_manager.create_experiment(
            name="Integration Test Experiment",
            variant_a="v1.0.0",
            variant_b="v2.0.0",
            traffic_split=0.5,
            duration_hours=1,
        )

        assert experiment.status == ExperimentStatus.PENDING

        # Start experiment
        started_exp = await test_experiment_manager.start_experiment(experiment.id)
        assert started_exp.status == ExperimentStatus.RUNNING

        # Simulate traffic allocation and feedback
        variant_feedbacks = {"v1.0.0": [], "v2.0.0": []}

        for i in range(20):  # Generate sufficient sample size
            # Allocate variant
            variant = await test_experiment_manager.allocate_variant(experiment.id)
            assert variant in ["v1.0.0", "v2.0.0"]

            # Create feedback for allocated variant
            # Simulate v2.0.0 performing better
            success_rate = 0.6 if variant == "v1.0.0" else 0.8
            action = FeedbackAction.APPROVED if i / 20 < success_rate else FeedbackAction.REJECTED

            feedback = await test_processor.submit_feedback(
                proposal_id=f"exp-prop-{i}",
                original_filename=f"exp_old{i}.txt",
                proposed_filename=f"exp_new{i}.txt",
                user_action=action,
                confidence_score=0.8,
                model_version=variant,
            )

            # Record feedback in experiment
            await test_experiment_manager.record_feedback(experiment.id, variant, feedback)
            variant_feedbacks[variant].append(feedback)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Get experiment status
        status = await test_experiment_manager.get_experiment_status(experiment.id)
        assert status is not None
        assert status["variant_a"]["sample_size"] > 0
        assert status["variant_b"]["sample_size"] > 0

        # Force conclusion for testing
        results = await test_experiment_manager.conclude_experiment(experiment.id)

        assert "winner" in results
        assert "statistical_significance" in results
        assert results["experiment_id"] == experiment.id

    async def test_multiple_concurrent_experiments(self, test_experiment_manager: ExperimentManager):
        """Test handling multiple concurrent experiments."""
        # Create multiple experiments
        experiments = []
        for i in range(3):
            experiment = await test_experiment_manager.create_experiment(
                name=f"Concurrent Test {i}", variant_a=f"v1.{i}.0", variant_b=f"v2.{i}.0", traffic_split=0.5
            )
            await test_experiment_manager.start_experiment(experiment.id)
            experiments.append(experiment)

        # Get active experiments
        active = await test_experiment_manager.get_active_experiments()
        assert len(active) == 3

        # Verify each experiment can allocate variants
        for experiment in experiments:
            variant = await test_experiment_manager.allocate_variant(experiment.id)
            assert variant in [experiment.variant_a, experiment.variant_b]

    async def test_experiment_statistical_significance(self, test_experiment_manager: ExperimentManager):
        """Test statistical significance calculation."""
        experiment = await test_experiment_manager.create_experiment(
            name="Significance Test", variant_a="control", variant_b="treatment", traffic_split=0.5
        )
        await test_experiment_manager.start_experiment(experiment.id)

        # Manually set experiment data with clear winner
        exp = test_experiment_manager._active_experiments[experiment.id]
        exp.sample_size_a = 100
        exp.sample_size_b = 100
        exp.metrics_a = {
            "total": 100,
            "approved": 50,
            "rejected": 50,
            "modified": 0,
            "approval_rate": 0.5,
            "rejection_rate": 0.5,
            "modification_rate": 0.0,
        }
        exp.metrics_b = {
            "total": 100,
            "approved": 80,
            "rejected": 20,
            "modified": 0,
            "approval_rate": 0.8,
            "rejection_rate": 0.2,
            "modification_rate": 0.0,
        }

        # Calculate significance
        p_value = test_experiment_manager._calculate_significance(exp)
        assert p_value is not None
        assert 0.0 <= p_value <= 1.0

        # With such a large difference, should be significant
        assert p_value < 0.05


@pytest.mark.asyncio
class TestResourceManagementUnderLoad:
    """Test resource management and behavior under load."""

    async def test_backpressure_handling_with_high_load(self, test_storage: FeedbackStorage):
        """Test backpressure handling under high load."""
        processor = FeedbackProcessor(
            storage=test_storage,
            batch_size=10,
            batch_timeout_seconds=1,
            retrain_threshold=100,
            max_pending_size=15,  # Small queue for testing
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
            memory_warning_threshold_mb=50.0,
        )

        # Start monitoring
        await processor.start_monitoring()

        try:
            # Flood with feedbacks to trigger backpressure
            tasks = []
            for i in range(25):  # More than max_pending_size
                task = processor.submit_feedback(
                    proposal_id=f"load-prop-{i}",
                    original_filename=f"load_old{i}.txt",
                    proposed_filename=f"load_new{i}.txt",
                    user_action=FeedbackAction.APPROVED,
                    confidence_score=0.8,
                    model_version="v1.0.0",
                )
                tasks.append(task)

            # Some should succeed, oldest should be dropped
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete without exceptions (due to DROP_OLDEST strategy)
            for result in results:
                assert not isinstance(result, Exception)

            # Check resource stats
            stats = await processor.get_resource_stats()
            assert stats["dropped_items"] > 0  # Some items should be dropped
            assert stats["queue_utilization"] <= 1.0

        finally:
            await processor.cleanup()

    async def test_backpressure_reject_new_strategy(self, test_storage: FeedbackStorage):
        """Test REJECT_NEW backpressure strategy."""
        processor = FeedbackProcessor(
            storage=test_storage,
            batch_size=20,  # Large batch to prevent processing
            max_pending_size=5,  # Very small queue
            backpressure_strategy=BackpressureStrategy.REJECT_NEW,
        )

        try:
            # Fill up the queue
            for i in range(5):
                await processor.submit_feedback(
                    proposal_id=f"queue-prop-{i}",
                    original_filename=f"queue_old{i}.txt",
                    proposed_filename=f"queue_new{i}.txt",
                    user_action=FeedbackAction.APPROVED,
                    confidence_score=0.8,
                    model_version="v1.0.0",
                )

            # Next submission should raise BackpressureError
            with pytest.raises(BackpressureError):
                await processor.submit_feedback(
                    proposal_id="overflow-prop",
                    original_filename="overflow.txt",
                    proposed_filename="overflow_new.txt",
                    user_action=FeedbackAction.APPROVED,
                    confidence_score=0.8,
                    model_version="v1.0.0",
                )

            # Check stats
            stats = await processor.get_resource_stats()
            assert stats["rejected_items"] > 0

        finally:
            await processor.cleanup()

    async def test_memory_monitoring_and_warnings(self, test_storage: FeedbackStorage):
        """Test memory monitoring and warning system."""
        processor = FeedbackProcessor(
            storage=test_storage,
            batch_size=50,  # Large batch to accumulate items
            memory_warning_threshold_mb=1.0,  # Very low threshold
        )

        await processor.start_monitoring()

        try:
            # Add many feedbacks to increase memory usage
            for i in range(30):
                await processor.submit_feedback(
                    proposal_id=f"memory-prop-{i}",
                    original_filename=f"memory_old{i}.txt",
                    proposed_filename=f"memory_new{i}.txt",
                    user_action=FeedbackAction.APPROVED,
                    confidence_score=0.8,
                    model_version="v1.0.0",
                    context_metadata={"large_data": "x" * 1000},  # Add bulk
                )

            # Wait for monitoring to detect high memory usage
            await asyncio.sleep(0.5)

            # Check resource stats
            stats = await processor.get_resource_stats()
            assert stats["memory_usage_mb"] >= 0
            assert stats["monitoring_active"] is True

        finally:
            await processor.cleanup()

    async def test_concurrent_batch_processing_safety(self, test_storage: FeedbackStorage):
        """Test thread safety of concurrent batch processing."""
        processor = FeedbackProcessor(storage=test_storage, batch_size=3, batch_timeout_seconds=0.1)

        try:
            # Submit feedbacks from multiple "threads" (tasks)
            async def submit_batch(batch_id: int):
                for i in range(5):
                    await processor.submit_feedback(
                        proposal_id=f"batch{batch_id}-prop-{i}",
                        original_filename=f"batch{batch_id}_old{i}.txt",
                        proposed_filename=f"batch{batch_id}_new{i}.txt",
                        user_action=FeedbackAction.APPROVED,
                        confidence_score=0.8,
                        model_version="v1.0.0",
                    )

            # Run concurrent submissions
            tasks = [submit_batch(i) for i in range(4)]
            await asyncio.gather(*tasks)

            # Wait for processing
            await asyncio.sleep(0.3)

            # Force final batch processing
            await processor.force_batch_processing()

            # Verify all feedbacks were processed
            stored_feedbacks = await test_storage.get_feedbacks()
            assert len(stored_feedbacks) == 20  # 4 batches * 5 feedbacks each

            # Verify no data corruption
            stats = await processor.get_resource_stats()
            assert stats["total_processed"] == 20
            assert stats["dropped_items"] == 0

        finally:
            await processor.cleanup()

    async def test_storage_connection_recovery(self, test_storage: FeedbackStorage):
        """Test storage connection recovery after failures."""
        # Test PostgreSQL connection recovery
        original_pool = test_storage._pg_pool

        # Simulate connection failure
        test_storage._pg_pool = None

        # Should raise error initially
        with pytest.raises(RuntimeError):
            await test_storage.get_feedbacks()

        # Restore connection
        test_storage._pg_pool = original_pool

        # Should work again
        feedbacks = await test_storage.get_feedbacks()
        assert isinstance(feedbacks, list)

    async def test_system_cleanup_and_resource_release(self, test_storage: FeedbackStorage):
        """Test proper cleanup and resource release."""
        processor = FeedbackProcessor(storage=test_storage, batch_size=5)

        # Start monitoring
        await processor.start_monitoring()

        # Submit some feedbacks
        for i in range(8):
            await processor.submit_feedback(
                proposal_id=f"cleanup-prop-{i}",
                original_filename=f"cleanup_old{i}.txt",
                proposed_filename=f"cleanup_new{i}.txt",
                user_action=FeedbackAction.APPROVED,
                confidence_score=0.8,
                model_version="v1.0.0",
            )

        # Get initial stats
        await processor.get_resource_stats()  # initial_stats

        # Cleanup
        await processor.cleanup()

        # Verify cleanup worked
        final_stats = await processor.get_resource_stats()
        assert final_stats["monitoring_active"] is False
        assert final_stats["pending_feedback_count"] == 0


# Performance and Stress Tests
@pytest.mark.asyncio
@pytest.mark.slow
class TestPerformanceUnderLoad:
    """Performance tests for high-load scenarios."""

    async def test_high_throughput_feedback_processing(self, test_storage: FeedbackStorage):
        """Test system performance under high throughput."""
        processor = FeedbackProcessor(
            storage=test_storage, batch_size=50, batch_timeout_seconds=0.5, max_pending_size=1000
        )

        try:
            start_time = datetime.now(UTC)

            # Submit many feedbacks rapidly
            tasks = []
            for i in range(200):
                task = processor.submit_feedback(
                    proposal_id=f"perf-prop-{i}",
                    original_filename=f"perf_old{i}.txt",
                    proposed_filename=f"perf_new{i}.txt",
                    user_action=(FeedbackAction.APPROVED if i % 2 == 0 else FeedbackAction.REJECTED),
                    confidence_score=0.7 + (i % 3) * 0.1,
                    model_version="v1.0.0",
                )
                tasks.append(task)

            # Wait for all submissions
            await asyncio.gather(*tasks)

            # Wait for batch processing
            await asyncio.sleep(1.0)

            end_time = datetime.now(UTC)
            total_time = (end_time - start_time).total_seconds()

            # Verify performance metrics
            assert total_time < 5.0  # Should complete within 5 seconds

            # Verify all feedbacks were processed
            stats = await processor.get_resource_stats()
            assert stats["total_processed"] == 200

            # Calculate throughput
            throughput = 200 / total_time
            assert throughput > 50  # At least 50 feedbacks/second

            logger.info(f"Processed 200 feedbacks in {total_time:.2f}s (throughput: {throughput:.1f}/s)")

        finally:
            await processor.cleanup()


if __name__ == "__main__":
    # Run tests with: uv run pytest tests/integration/file_rename_service/test_feedback_integration.py -v
    pytest.main([__file__, "-v", "--tb=short"])
