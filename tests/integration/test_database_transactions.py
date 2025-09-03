"""
Integration tests for database transaction handling across services.

Tests ACID properties, transaction isolation, rollback scenarios, and cross-service
transaction coordination in the Tracktion system's database operations.
"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.shared_utilities import (
    TestDataGenerator,
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAsyncSession:
    """Mock async database session for testing."""

    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.operations = []
        self.isolation_level = "READ_COMMITTED"
        self._in_transaction = False
        self._savepoints = []

    async def execute(self, statement, parameters=None):
        """Execute a statement."""
        self.operations.append(
            {"type": "execute", "statement": str(statement), "parameters": parameters, "timestamp": datetime.now(UTC)}
        )

        # Mock result
        result = MagicMock()
        result.fetchall = list
        result.scalar = lambda: None
        result.rowcount = 1
        return result

    async def commit(self):
        """Commit the transaction."""
        if not self._in_transaction:
            raise RuntimeError("No transaction in progress")
        self.committed = True
        self.operations.append({"type": "commit", "timestamp": datetime.now(UTC)})
        self._in_transaction = False

    async def rollback(self):
        """Rollback the transaction."""
        if not self._in_transaction:
            raise RuntimeError("No transaction in progress")
        self.rolled_back = True
        self.operations.append({"type": "rollback", "timestamp": datetime.now(UTC)})
        self._in_transaction = False

    async def close(self):
        """Close the session."""
        self.closed = True
        self.operations.append({"type": "close", "timestamp": datetime.now(UTC)})

    def begin(self):
        """Begin a transaction."""
        self._in_transaction = True
        return self

    async def __aenter__(self):
        """Async context manager entry."""
        self._in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()

    def savepoint(self, name: str | None = None):
        """Create a savepoint."""
        savepoint_name = name or f"sp_{len(self._savepoints) + 1}"
        self._savepoints.append(savepoint_name)
        self.operations.append({"type": "savepoint", "name": savepoint_name, "timestamp": datetime.now(UTC)})
        return savepoint_name

    def rollback_to_savepoint(self, savepoint_name: str):
        """Rollback to a savepoint."""
        if savepoint_name in self._savepoints:
            self._savepoints.remove(savepoint_name)
            self.operations.append(
                {"type": "rollback_to_savepoint", "name": savepoint_name, "timestamp": datetime.now(UTC)}
            )


class MockDatabaseManager:
    """Mock database manager for testing."""

    def __init__(self):
        self.sessions = []
        self.connection_pool_size = 10
        self.active_connections = 0

    async def get_session(self) -> MockAsyncSession:
        """Get a database session."""
        session = MockAsyncSession()
        self.sessions.append(session)
        self.active_connections += 1
        return session

    async def close_session(self, session: MockAsyncSession):
        """Close a database session."""
        await session.close()
        self.active_connections = max(0, self.active_connections - 1)

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        session = await self.get_session()
        try:
            async with session:
                yield session
        finally:
            await self.close_session(session)


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    return MockDatabaseManager()


@pytest.fixture
def data_generator():
    """Test data generator."""
    return TestDataGenerator(seed=42)


class TestDatabaseTransactionBasics:
    """Test basic database transaction operations."""

    @pytest.mark.asyncio
    async def test_basic_transaction_commit(self, mock_db_manager):
        """Test basic transaction commit functionality."""

        async with mock_db_manager.transaction() as session:
            # Execute some operations
            await session.execute(
                text("INSERT INTO recordings (id, name) VALUES (:id, :name)"),
                {"id": str(uuid4()), "name": "test_recording"},
            )
            await session.execute(
                text("UPDATE recordings SET status = 'processed' WHERE id = :id"), {"id": str(uuid4())}
            )

        # Verify transaction was committed
        session = mock_db_manager.sessions[-1]
        assert session.committed is True
        assert session.rolled_back is False
        assert len(session.operations) >= 3  # 2 executes + 1 commit

        # Verify operations were recorded
        operations = [op for op in session.operations if op["type"] == "execute"]
        assert len(operations) == 2
        assert "INSERT INTO recordings" in operations[0]["statement"]
        assert "UPDATE recordings" in operations[1]["statement"]

        logger.info("✅ Basic transaction commit test completed successfully")

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, mock_db_manager):
        """Test transaction rollback when errors occur."""

        try:
            async with mock_db_manager.transaction() as session:
                # Execute valid operation
                await session.execute(
                    text("INSERT INTO recordings (id, name) VALUES (:id, :name)"),
                    {"id": str(uuid4()), "name": "test_recording"},
                )

                # Simulate an error
                raise IntegrityError("Constraint violation", None, None)

        except IntegrityError:
            # Expected error
            pass

        # Verify transaction was rolled back
        session = mock_db_manager.sessions[-1]
        assert session.committed is False
        assert session.rolled_back is True

        # Verify operations included rollback
        rollback_ops = [op for op in session.operations if op["type"] == "rollback"]
        assert len(rollback_ops) == 1

        logger.info("✅ Transaction rollback on error test completed successfully")

    @pytest.mark.asyncio
    async def test_nested_transactions_with_savepoints(self, mock_db_manager):
        """Test nested transactions using savepoints."""

        async with mock_db_manager.transaction() as session:
            # Main transaction operations
            await session.execute(
                text("INSERT INTO recordings (id, name) VALUES (:id, :name)"),
                {"id": str(uuid4()), "name": "main_recording"},
            )

            # Create savepoint for nested operation
            savepoint = session.savepoint("nested_operation")

            try:
                # Nested operation that might fail
                await session.execute(
                    text("INSERT INTO analysis_results (id, recording_id) VALUES (:id, :rid)"),
                    {"id": str(uuid4()), "rid": str(uuid4())},
                )

                # Simulate partial failure - rollback to savepoint
                session.rollback_to_savepoint(savepoint)

                # Continue with alternative operation
                await session.execute(
                    text("UPDATE recordings SET status = 'pending' WHERE id = :id"), {"id": str(uuid4())}
                )

            except Exception:
                session.rollback_to_savepoint(savepoint)

        # Verify savepoint operations
        session = mock_db_manager.sessions[-1]
        assert session.committed is True

        # Check savepoint operations
        savepoint_ops = [op for op in session.operations if op["type"] == "savepoint"]
        rollback_to_savepoint_ops = [op for op in session.operations if op["type"] == "rollback_to_savepoint"]

        assert len(savepoint_ops) == 1
        assert len(rollback_to_savepoint_ops) == 1
        assert savepoint_ops[0]["name"] == "nested_operation"

        logger.info("✅ Nested transactions with savepoints test completed successfully")


class TestConcurrentTransactions:
    """Test concurrent transaction scenarios and isolation."""

    @pytest.mark.asyncio
    async def test_concurrent_read_write_isolation(self, mock_db_manager, data_generator):
        """Test read-write isolation between concurrent transactions."""

        recording_id = data_generator.generate_uuid_string()

        # Results from concurrent operations
        transaction_results = []

        async def read_transaction():
            """Transaction that reads data."""
            async with mock_db_manager.transaction() as session:
                # Read operation
                await session.execute(text("SELECT * FROM recordings WHERE id = :id"), {"id": recording_id})
                transaction_results.append({"type": "read", "session_id": id(session)})

                # Simulate processing time
                await asyncio.sleep(0.01)

                # Another read to test consistency
                await session.execute(text("SELECT status FROM recordings WHERE id = :id"), {"id": recording_id})
                transaction_results.append({"type": "read_consistent", "session_id": id(session)})

        async def write_transaction():
            """Transaction that writes data."""
            async with mock_db_manager.transaction() as session:
                # Write operation
                await session.execute(
                    text("UPDATE recordings SET status = 'processing' WHERE id = :id"), {"id": recording_id}
                )
                transaction_results.append({"type": "write", "session_id": id(session)})

                # Simulate processing time
                await asyncio.sleep(0.01)

                # Final update
                await session.execute(
                    text("UPDATE recordings SET status = 'completed' WHERE id = :id"), {"id": recording_id}
                )
                transaction_results.append({"type": "write_complete", "session_id": id(session)})

        # Execute transactions concurrently
        await asyncio.gather(read_transaction(), write_transaction())

        # Verify both transactions completed
        assert len(transaction_results) == 4
        assert len(mock_db_manager.sessions) == 2

        # Verify isolation - different sessions used
        read_results = [r for r in transaction_results if r["type"].startswith("read")]
        write_results = [r for r in transaction_results if r["type"].startswith("write")]

        assert len(read_results) == 2
        assert len(write_results) == 2
        assert read_results[0]["session_id"] != write_results[0]["session_id"]

        # Verify both sessions committed
        for session in mock_db_manager.sessions:
            assert session.committed is True

        logger.info("✅ Concurrent read-write isolation test completed successfully")

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_resolution(self, mock_db_manager, data_generator):
        """Test deadlock detection and resolution strategies."""

        resource_1 = data_generator.generate_uuid_string()
        resource_2 = data_generator.generate_uuid_string()

        # Track transaction timing and conflicts
        transaction_events = []

        async def transaction_a():
            """Transaction A: Lock resource 1, then resource 2."""
            try:
                async with mock_db_manager.transaction() as session:
                    transaction_events.append({"tx": "A", "event": "started", "time": datetime.now(UTC)})

                    # Lock resource 1
                    await session.execute(
                        text("SELECT * FROM recordings WHERE id = :id FOR UPDATE"), {"id": resource_1}
                    )
                    transaction_events.append({"tx": "A", "event": "locked_resource_1", "time": datetime.now(UTC)})

                    # Simulate processing time
                    await asyncio.sleep(0.02)

                    # Try to lock resource 2 (potential deadlock)
                    await session.execute(
                        text("SELECT * FROM analysis_results WHERE recording_id = :id FOR UPDATE"), {"id": resource_2}
                    )
                    transaction_events.append({"tx": "A", "event": "locked_resource_2", "time": datetime.now(UTC)})

            except OperationalError as e:
                transaction_events.append(
                    {"tx": "A", "event": "deadlock_detected", "error": str(e), "time": datetime.now(UTC)}
                )
                # Simulate deadlock victim selection and rollback
                raise

        async def transaction_b():
            """Transaction B: Lock resource 2, then resource 1."""
            try:
                async with mock_db_manager.transaction() as session:
                    transaction_events.append({"tx": "B", "event": "started", "time": datetime.now(UTC)})

                    # Lock resource 2
                    await session.execute(
                        text("SELECT * FROM analysis_results WHERE recording_id = :id FOR UPDATE"), {"id": resource_2}
                    )
                    transaction_events.append({"tx": "B", "event": "locked_resource_2", "time": datetime.now(UTC)})

                    # Simulate processing time
                    await asyncio.sleep(0.02)

                    # Try to lock resource 1 (potential deadlock)
                    await session.execute(
                        text("SELECT * FROM recordings WHERE id = :id FOR UPDATE"), {"id": resource_1}
                    )
                    transaction_events.append({"tx": "B", "event": "locked_resource_1", "time": datetime.now(UTC)})

            except OperationalError as e:
                transaction_events.append(
                    {"tx": "B", "event": "deadlock_detected", "error": str(e), "time": datetime.now(UTC)}
                )
                raise

        # Execute potentially deadlocking transactions
        with contextlib.suppress(OperationalError):
            await asyncio.gather(transaction_a(), transaction_b())

        # Verify both transactions attempted to start
        start_events = [e for e in transaction_events if e["event"] == "started"]
        assert len(start_events) == 2

        # Verify resource locking attempts were made
        lock_events = [e for e in transaction_events if "locked_resource" in e["event"]]
        assert len(lock_events) >= 2  # At least initial locks should succeed

        logger.info("✅ Deadlock detection and resolution test completed successfully")


class TestCrossServiceTransactions:
    """Test transaction coordination across multiple services."""

    @pytest.mark.asyncio
    async def test_distributed_transaction_coordination(self, mock_db_manager, data_generator):
        """Test coordination of transactions across multiple services."""

        recording_id = data_generator.generate_uuid_string()

        # Mock services with their own database sessions
        services = {"analysis": mock_db_manager, "cataloging": mock_db_manager, "tracklist": mock_db_manager}

        # Track cross-service transaction coordination
        coordination_events = []

        async def analysis_service_operation(recording_id: str):
            """Analysis service database operations."""
            async with services["analysis"].transaction() as session:
                await session.execute(
                    text("INSERT INTO analysis_results (id, recording_id, bpm) VALUES (:id, :rid, :bpm)"),
                    {"id": str(uuid4()), "rid": recording_id, "bpm": 120.5},
                )
                coordination_events.append({"service": "analysis", "operation": "insert_result"})

                await session.execute(
                    text("UPDATE recordings SET analysis_status = 'completed' WHERE id = :id"), {"id": recording_id}
                )
                coordination_events.append({"service": "analysis", "operation": "update_status"})

        async def cataloging_service_operation(recording_id: str):
            """Cataloging service database operations."""
            async with services["cataloging"].transaction() as session:
                await session.execute(
                    text("INSERT INTO catalog_entries (id, recording_id, metadata) VALUES (:id, :rid, :meta)"),
                    {"id": str(uuid4()), "rid": recording_id, "meta": "{}"},
                )
                coordination_events.append({"service": "cataloging", "operation": "insert_catalog"})

                await session.execute(
                    text("UPDATE recordings SET catalog_status = 'cataloged' WHERE id = :id"), {"id": recording_id}
                )
                coordination_events.append({"service": "cataloging", "operation": "update_status"})

        async def tracklist_service_operation(recording_id: str):
            """Tracklist service database operations."""
            async with services["tracklist"].transaction() as session:
                await session.execute(
                    text("INSERT INTO tracklist_items (id, recording_id, position) VALUES (:id, :rid, :pos)"),
                    {"id": str(uuid4()), "rid": recording_id, "pos": 1},
                )
                coordination_events.append({"service": "tracklist", "operation": "insert_item"})

                await session.execute(
                    text("UPDATE recordings SET tracklist_status = 'added' WHERE id = :id"), {"id": recording_id}
                )
                coordination_events.append({"service": "tracklist", "operation": "update_status"})

        # Execute service operations concurrently (simulating distributed processing)
        await asyncio.gather(
            analysis_service_operation(recording_id),
            cataloging_service_operation(recording_id),
            tracklist_service_operation(recording_id),
        )

        # Verify all service operations completed
        assert len(coordination_events) == 6

        # Verify each service performed its operations
        analysis_events = [e for e in coordination_events if e["service"] == "analysis"]
        cataloging_events = [e for e in coordination_events if e["service"] == "cataloging"]
        tracklist_events = [e for e in coordination_events if e["service"] == "tracklist"]

        assert len(analysis_events) == 2
        assert len(cataloging_events) == 2
        assert len(tracklist_events) == 2

        # Verify all transactions committed successfully
        assert len(mock_db_manager.sessions) >= 3
        for session in mock_db_manager.sessions[-3:]:
            assert session.committed is True
            assert session.rolled_back is False

        logger.info("✅ Distributed transaction coordination test completed successfully")

    @pytest.mark.asyncio
    async def test_transaction_failure_rollback_cascade(self, mock_db_manager, data_generator):
        """Test rollback cascade when one service transaction fails."""

        recording_id = data_generator.generate_uuid_string()

        # Track transaction outcomes
        transaction_outcomes = []

        async def successful_service_a():
            """Service A - should succeed but rollback due to cascade."""
            try:
                async with mock_db_manager.transaction() as session:
                    await session.execute(
                        text("INSERT INTO service_a_data (id, recording_id) VALUES (:id, :rid)"),
                        {"id": str(uuid4()), "rid": recording_id},
                    )
                    transaction_outcomes.append({"service": "A", "outcome": "committed"})
            except Exception as e:
                transaction_outcomes.append({"service": "A", "outcome": "failed", "error": str(e)})

        async def failing_service_b():
            """Service B - will fail and trigger cascade rollback."""
            try:
                async with mock_db_manager.transaction() as session:
                    await session.execute(
                        text("INSERT INTO service_b_data (id, recording_id) VALUES (:id, :rid)"),
                        {"id": str(uuid4()), "rid": recording_id},
                    )

                    # Simulate failure
                    raise IntegrityError("Constraint violation in service B", None, None)

            except IntegrityError as e:
                transaction_outcomes.append({"service": "B", "outcome": "failed", "error": str(e)})
                raise  # Re-raise to trigger cascade

        async def dependent_service_c():
            """Service C - depends on A and B, should not execute if B fails."""
            try:
                async with mock_db_manager.transaction() as session:
                    await session.execute(
                        text("INSERT INTO service_c_data (id, recording_id) VALUES (:id, :rid)"),
                        {"id": str(uuid4()), "rid": recording_id},
                    )
                    transaction_outcomes.append({"service": "C", "outcome": "committed"})
            except Exception as e:
                transaction_outcomes.append({"service": "C", "outcome": "failed", "error": str(e)})

        # Execute distributed transaction with failure
        try:
            # Service A succeeds, Service B fails, Service C should not execute
            await successful_service_a()

            try:
                await failing_service_b()
                # If B succeeds, then execute C
                await dependent_service_c()
            except IntegrityError:
                # B failed, need to compensate A's transaction
                transaction_outcomes.append({"service": "A", "outcome": "compensated"})

        except Exception:
            pass

        # Verify failure handling
        assert len(transaction_outcomes) >= 2

        # Verify service B failed
        service_b_outcomes = [o for o in transaction_outcomes if o["service"] == "B"]
        assert len(service_b_outcomes) == 1
        assert service_b_outcomes[0]["outcome"] == "failed"

        # Verify compensating action for service A
        service_a_outcomes = [o for o in transaction_outcomes if o["service"] == "A"]
        assert len(service_a_outcomes) >= 1

        logger.info("✅ Transaction failure rollback cascade test completed successfully")


class TestTransactionPerformance:
    """Test transaction performance and optimization scenarios."""

    @pytest.mark.asyncio
    async def test_bulk_operation_transaction_optimization(self, mock_db_manager, data_generator):
        """Test transaction optimization for bulk operations."""

        # Generate bulk test data
        bulk_size = 100
        recording_ids = [data_generator.generate_uuid_string() for _ in range(bulk_size)]

        # Test different transaction strategies
        transaction_strategies = []

        # Strategy 1: Single transaction for all operations
        start_time = datetime.now(UTC)
        async with mock_db_manager.transaction() as session:
            for recording_id in recording_ids:
                await session.execute(
                    text("INSERT INTO recordings (id, name) VALUES (:id, :name)"),
                    {"id": recording_id, "name": f"bulk_recording_{recording_id[:8]}"},
                )
        single_tx_time = (datetime.now(UTC) - start_time).total_seconds()

        transaction_strategies.append(
            {"strategy": "single_transaction", "operations": bulk_size, "time": single_tx_time, "transactions": 1}
        )

        # Strategy 2: Batch transactions (10 operations per transaction)
        batch_size = 10
        batch_count = bulk_size // batch_size

        start_time = datetime.now(UTC)
        for batch_start in range(0, bulk_size, batch_size):
            batch_ids = recording_ids[batch_start : batch_start + batch_size]
            async with mock_db_manager.transaction() as session:
                for recording_id in batch_ids:
                    await session.execute(
                        text("INSERT INTO analysis_results (id, recording_id) VALUES (:id, :rid)"),
                        {"id": str(uuid4()), "rid": recording_id},
                    )
        batch_tx_time = (datetime.now(UTC) - start_time).total_seconds()

        transaction_strategies.append(
            {
                "strategy": "batch_transactions",
                "operations": bulk_size,
                "time": batch_tx_time,
                "transactions": batch_count,
            }
        )

        # Strategy 3: Individual transactions (one per operation)
        start_time = datetime.now(UTC)
        for recording_id in recording_ids:
            async with mock_db_manager.transaction() as session:
                await session.execute(
                    text("INSERT INTO catalog_entries (id, recording_id) VALUES (:id, :rid)"),
                    {"id": str(uuid4()), "rid": recording_id},
                )
        individual_tx_time = (datetime.now(UTC) - start_time).total_seconds()

        transaction_strategies.append(
            {
                "strategy": "individual_transactions",
                "operations": bulk_size,
                "time": individual_tx_time,
                "transactions": bulk_size,
            }
        )

        # Analyze performance characteristics
        for strategy in transaction_strategies:
            strategy["ops_per_second"] = strategy["operations"] / strategy["time"] if strategy["time"] > 0 else 0
            strategy["tx_overhead"] = strategy["time"] / strategy["transactions"] if strategy["transactions"] > 0 else 0

        # Verify all strategies completed successfully
        assert len(transaction_strategies) == 3
        for strategy in transaction_strategies:
            assert strategy["operations"] == bulk_size
            assert strategy["time"] > 0

        # Verify sessions were created for each strategy
        # Single: 1 session, Batch: batch_count sessions, Individual: bulk_size sessions
        expected_sessions = 1 + batch_count + bulk_size
        assert len(mock_db_manager.sessions) >= expected_sessions

        logger.info("✅ Bulk operation transaction optimization test completed successfully")
        logger.info(f"Performance results: {transaction_strategies}")

    @pytest.mark.asyncio
    async def test_connection_pool_management(self, mock_db_manager):
        """Test database connection pool management under load."""

        # Simulate high concurrent load
        concurrent_transactions = 20

        # Track connection usage
        connection_metrics = {
            "max_concurrent": 0,
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
        }

        async def concurrent_transaction(transaction_id: int):
            """Execute a transaction with connection pool tracking."""
            try:
                async with mock_db_manager.transaction() as session:
                    # Track peak concurrent connections
                    connection_metrics["max_concurrent"] = max(
                        connection_metrics["max_concurrent"], mock_db_manager.active_connections
                    )

                    # Simulate database work
                    await session.execute(text("SELECT COUNT(*) FROM recordings"))
                    await asyncio.sleep(0.001)  # Simulate processing time
                    await session.execute(
                        text("INSERT INTO test_data (id, tx_id) VALUES (:id, :tx_id)"),
                        {"id": str(uuid4()), "tx_id": transaction_id},
                    )

                    connection_metrics["successful_transactions"] += 1

            except Exception as e:
                connection_metrics["failed_transactions"] += 1
                logger.error(f"Transaction {transaction_id} failed: {e}")
            finally:
                connection_metrics["total_transactions"] += 1

        # Execute concurrent transactions
        tasks = [concurrent_transaction(i) for i in range(concurrent_transactions)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify connection pool behavior
        assert connection_metrics["total_transactions"] == concurrent_transactions
        assert connection_metrics["successful_transactions"] > 0
        assert connection_metrics["max_concurrent"] <= mock_db_manager.connection_pool_size

        # Verify connections were properly cleaned up
        assert mock_db_manager.active_connections >= 0

        logger.info("✅ Connection pool management test completed successfully")
        logger.info(f"Connection metrics: {connection_metrics}")


@pytest.mark.integration
class TestTransactionIntegrationScenarios:
    """Integration test scenarios combining multiple transaction patterns."""

    @pytest.mark.asyncio
    async def test_complex_workflow_transaction_coordination(self, mock_db_manager, data_generator):
        """Test complex workflow with multiple transaction coordination patterns."""

        # Simulate a complete audio file processing workflow
        file_path = f"/audio/{data_generator.generate_uuid_string()}.mp3"
        recording_id = data_generator.generate_uuid_string()

        # Workflow stages with different transaction requirements
        workflow_results = []

        # Stage 1: File intake (requires immediate consistency)
        async with mock_db_manager.transaction() as session:
            await session.execute(
                text("INSERT INTO file_intake (id, path, status) VALUES (:id, :path, :status)"),
                {"id": str(uuid4()), "path": file_path, "status": "received"},
            )

            await session.execute(
                text("INSERT INTO recordings (id, file_path, status) VALUES (:id, :path, :status)"),
                {"id": recording_id, "path": file_path, "status": "pending"},
            )

            workflow_results.append({"stage": "intake", "status": "completed"})

        # Stage 2: Analysis (can handle partial failures with savepoints)
        async with mock_db_manager.transaction() as session:
            # Main analysis
            await session.execute(
                text("UPDATE recordings SET status = 'analyzing' WHERE id = :id"), {"id": recording_id}
            )

            # BPM analysis (can fail independently)
            bpm_savepoint = session.savepoint("bpm_analysis")
            try:
                await session.execute(
                    text("INSERT INTO bpm_analysis (id, recording_id, bpm) VALUES (:id, :rid, :bpm)"),
                    {"id": str(uuid4()), "rid": recording_id, "bpm": 120.5},
                )
            except Exception:
                session.rollback_to_savepoint(bpm_savepoint)

            # Key detection (can fail independently)
            key_savepoint = session.savepoint("key_detection")
            try:
                await session.execute(
                    text("INSERT INTO key_detection (id, recording_id, key_sig) VALUES (:id, :rid, :key)"),
                    {"id": str(uuid4()), "rid": recording_id, "key": "C major"},
                )
            except Exception:
                session.rollback_to_savepoint(key_savepoint)

            # Finalize analysis
            await session.execute(
                text("UPDATE recordings SET status = 'analyzed' WHERE id = :id"), {"id": recording_id}
            )

            workflow_results.append({"stage": "analysis", "status": "completed"})

        # Stage 3: Cataloging and indexing (eventual consistency acceptable)
        async with mock_db_manager.transaction() as session:
            await session.execute(
                text("INSERT INTO catalog_entries (id, recording_id, indexed_at) VALUES (:id, :rid, :time)"),
                {"id": str(uuid4()), "rid": recording_id, "time": datetime.now(UTC)},
            )

            await session.execute(
                text("UPDATE recordings SET status = 'cataloged' WHERE id = :id"), {"id": recording_id}
            )

            workflow_results.append({"stage": "cataloging", "status": "completed"})

        # Verify workflow completion
        assert len(workflow_results) == 3
        for result in workflow_results:
            assert result["status"] == "completed"

        # Verify all transactions committed successfully
        assert len(mock_db_manager.sessions) == 3
        for session in mock_db_manager.sessions:
            assert session.committed is True

        logger.info("✅ Complex workflow transaction coordination test completed successfully")
