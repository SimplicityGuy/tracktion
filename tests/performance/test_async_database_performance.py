"""Performance tests for async database operations."""

import asyncio
import statistics
import time
from typing import Any
from uuid import uuid4

from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.async_repositories import (
    AsyncBatchOperations,
    AsyncMetadataRepository,
    AsyncRecordingRepository,
)


class PerformanceBenchmark:
    """Benchmark for async database operations."""

    def __init__(self, db_manager: AsyncDatabaseManager):
        """Initialize benchmark.

        Args:
            db_manager: Async database manager
        """
        self.db = db_manager
        self.recording_repo = AsyncRecordingRepository(db_manager)
        self.metadata_repo = AsyncMetadataRepository(db_manager)
        self.batch_ops = AsyncBatchOperations(db_manager)
        self.results: dict[str, list[float]] = {}

    async def benchmark_single_insert(self, count: int = 100) -> dict[str, float]:
        """Benchmark single insert operations.

        Args:
            count: Number of inserts to perform

        Returns:
            Benchmark results
        """
        times = []

        for i in range(count):
            start = time.perf_counter()
            await self.recording_repo.create(
                file_path=f"/test/file_{i}.mp3",
                file_name=f"file_{i}.mp3",
                sha256_hash=f"hash_{i}",
                xxh128_hash=f"xxhash_{i}",
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "operation": "single_insert",
            "count": count,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,  # 95th percentile
            "p99_ms": statistics.quantiles(times, n=100)[98] * 1000,  # 99th percentile
            "total_time_s": sum(times),
        }

    async def benchmark_batch_insert(self, batch_size: int = 100, batches: int = 10) -> dict[str, float]:
        """Benchmark batch insert operations.

        Args:
            batch_size: Size of each batch
            batches: Number of batches

        Returns:
            Benchmark results
        """
        times = []

        for batch_num in range(batches):
            recordings_data = [
                {
                    "file_path": f"/test/batch_{batch_num}_file_{i}.mp3",
                    "file_name": f"batch_{batch_num}_file_{i}.mp3",
                    "sha256_hash": f"hash_{batch_num}_{i}",
                    "xxh128_hash": f"xxhash_{batch_num}_{i}",
                }
                for i in range(batch_size)
            ]

            start = time.perf_counter()
            await self.batch_ops.bulk_insert_recordings(recordings_data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "operation": "batch_insert",
            "batch_size": batch_size,
            "batches": batches,
            "total_records": batch_size * batches,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
            "p99_ms": statistics.quantiles(times, n=100)[98] * 1000 if len(times) >= 100 else max(times) * 1000,
            "records_per_second": (batch_size * batches) / sum(times),
            "total_time_s": sum(times),
        }

    async def benchmark_concurrent_queries(self, concurrency: int = 10, queries: int = 100) -> dict[str, float]:
        """Benchmark concurrent query execution.

        Args:
            concurrency: Number of concurrent queries
            queries: Total number of queries

        Returns:
            Benchmark results
        """

        async def run_query(query_id: int) -> float:
            """Run a single query and return execution time."""
            start = time.perf_counter()
            # Mix of different query types
            if query_id % 3 == 0:
                await self.recording_repo.get_all(limit=10)
            elif query_id % 3 == 1:
                await self.recording_repo.get_by_file_path(f"/test/file_{query_id}.mp3")
            else:
                recording_id = uuid4()
                await self.recording_repo.get_by_id(recording_id)
            return time.perf_counter() - start

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def run_with_semaphore(query_id: int) -> float:
            async with semaphore:
                return await run_query(query_id)

        start_time = time.perf_counter()
        times = await asyncio.gather(*[run_with_semaphore(i) for i in range(queries)])
        total_time = time.perf_counter() - start_time

        return {
            "operation": "concurrent_queries",
            "concurrency": concurrency,
            "total_queries": queries,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
            "p99_ms": statistics.quantiles(times, n=100)[98] * 1000 if len(times) >= 100 else max(times) * 1000,
            "queries_per_second": queries / total_time,
            "total_time_s": total_time,
        }

    async def benchmark_connection_pool(self, operations: int = 1000) -> dict[str, float]:
        """Benchmark connection pool efficiency.

        Args:
            operations: Number of operations to perform

        Returns:
            Benchmark results
        """
        connection_times = []
        query_times = []

        for _i in range(operations):
            # Measure connection acquisition time
            start_conn = time.perf_counter()
            async with self.db.get_db_session() as session:
                conn_time = time.perf_counter() - start_conn
                connection_times.append(conn_time)

                # Measure query execution time
                start_query = time.perf_counter()
                result = await session.execute("SELECT 1")
                await result.fetchone()
                query_time = time.perf_counter() - start_query
                query_times.append(query_time)

        return {
            "operation": "connection_pool",
            "operations": operations,
            "conn_acquisition_mean_ms": statistics.mean(connection_times) * 1000,
            "conn_acquisition_p95_ms": statistics.quantiles(connection_times, n=20)[18] * 1000,
            "query_mean_ms": statistics.mean(query_times) * 1000,
            "query_p95_ms": statistics.quantiles(query_times, n=20)[18] * 1000,
            "total_time_s": sum(connection_times) + sum(query_times),
        }

    async def benchmark_streaming(self, total_records: int = 10000, chunk_size: int = 1000) -> dict[str, float]:
        """Benchmark streaming large datasets.

        Args:
            total_records: Total number of records to stream
            chunk_size: Size of each chunk

        Returns:
            Benchmark results
        """
        # First, insert test data
        await self._prepare_test_data(total_records)

        # Benchmark streaming
        start = time.perf_counter()
        chunks_processed = 0
        records_processed = 0

        async for chunk in self.batch_ops.stream_large_dataset(query_limit=chunk_size):
            chunks_processed += 1
            records_processed += len(chunk)
            if records_processed >= total_records:
                break

        elapsed = time.perf_counter() - start

        return {
            "operation": "streaming",
            "total_records": records_processed,
            "chunk_size": chunk_size,
            "chunks_processed": chunks_processed,
            "records_per_second": records_processed / elapsed,
            "total_time_s": elapsed,
        }

    async def _prepare_test_data(self, count: int) -> None:
        """Prepare test data for benchmarks.

        Args:
            count: Number of records to create
        """
        batch_size = 1000
        for i in range(0, count, batch_size):
            batch = [
                {
                    "file_path": f"/benchmark/file_{j}.mp3",
                    "file_name": f"file_{j}.mp3",
                    "sha256_hash": f"hash_{j}",
                    "xxh128_hash": f"xxhash_{j}",
                }
                for j in range(i, min(i + batch_size, count))
            ]
            await self.batch_ops.bulk_insert_recordings(batch)

    async def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all benchmarks and return results.

        Returns:
            All benchmark results
        """
        results = {}

        print("Running single insert benchmark...")
        results["single_insert"] = await self.benchmark_single_insert(100)

        print("Running batch insert benchmark...")
        results["batch_insert"] = await self.benchmark_batch_insert(100, 10)

        print("Running concurrent queries benchmark...")
        results["concurrent_queries"] = await self.benchmark_concurrent_queries(20, 200)

        print("Running connection pool benchmark...")
        results["connection_pool"] = await self.benchmark_connection_pool(500)

        print("Running streaming benchmark...")
        results["streaming"] = await self.benchmark_streaming(5000, 500)

        return results

    def print_results(self, results: dict[str, Any]) -> None:
        """Print benchmark results in a formatted way.

        Args:
            results: Benchmark results
        """
        print("\n" + "=" * 60)
        print("ASYNC DATABASE PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        for name, result in results.items():
            print(f"\n{name.upper().replace('_', ' ')}:")
            print("-" * 40)

            for key, value in result.items():
                if key == "operation":
                    continue
                if isinstance(value, float):
                    if key.endswith("_ms"):
                        print(f"  {key}: {value:.2f} ms")
                    elif key.endswith("_s"):
                        print(f"  {key}: {value:.2f} s")
                    elif "per_second" in key:
                        print(f"  {key}: {value:.0f}/s")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        # Performance assessment
        print("\n" + "=" * 60)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 60)

        # Check against targets
        targets_met = []
        targets_failed = []

        # Target: Query response time <100ms for 95th percentile
        if "concurrent_queries" in results:
            p95 = results["concurrent_queries"]["p95_ms"]
            if p95 < 100:
                targets_met.append(f"✅ Query P95 latency: {p95:.2f}ms < 100ms target")
            else:
                targets_failed.append(f"❌ Query P95 latency: {p95:.2f}ms > 100ms target")

        # Target: Support 1000+ concurrent operations
        if "concurrent_queries" in results:
            qps = results["concurrent_queries"]["queries_per_second"]
            if qps > 1000:
                targets_met.append(f"✅ Queries per second: {qps:.0f} > 1000 target")
            else:
                targets_failed.append(f"⚠️  Queries per second: {qps:.0f} < 1000 target")

        # Target: Connection pool efficiency
        if "connection_pool" in results:
            conn_p95 = results["connection_pool"]["conn_acquisition_p95_ms"]
            if conn_p95 < 10:
                targets_met.append(f"✅ Connection acquisition P95: {conn_p95:.2f}ms < 10ms")
            else:
                targets_failed.append(f"⚠️  Connection acquisition P95: {conn_p95:.2f}ms > 10ms")

        print("\nTargets Met:")
        for target in targets_met:
            print(f"  {target}")

        if targets_failed:
            print("\nTargets Not Met:")
            for target in targets_failed:
                print(f"  {target}")

        print("\n" + "=" * 60)


async def main():
    """Main entry point for performance benchmarks."""
    import os

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/tracktion")

    # Create database manager
    db_manager = AsyncDatabaseManager(database_url=db_url)

    # Create benchmark
    benchmark = PerformanceBenchmark(db_manager)

    try:
        # Run benchmarks
        results = await benchmark.run_all_benchmarks()

        # Print results
        benchmark.print_results(results)

    finally:
        # Cleanup
        pass


if __name__ == "__main__":
    asyncio.run(main())
