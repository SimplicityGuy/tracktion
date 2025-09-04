# Performance Optimization Examples

This document provides comprehensive performance optimization examples for Tracktion services, covering database optimization, caching strategies, async processing, and system-level improvements.

## Table of Contents

1. [Database Performance Optimization](#database-performance-optimization)
2. [Caching Strategies](#caching-strategies)
3. [Async Processing Optimization](#async-processing-optimization)
4. [Memory Management](#memory-management)
5. [I/O Optimization](#io-optimization)
6. [API Performance](#api-performance)
7. [Audio Processing Optimization](#audio-processing-optimization)
8. [Container and Deployment Optimization](#container-and-deployment-optimization)
9. [Monitoring and Profiling](#monitoring-and-profiling)
10. [Load Testing and Benchmarking](#load-testing-and-benchmarking)

## Database Performance Optimization

### 1. Query Optimization with Indexing

```python
from sqlalchemy import Index, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import time
from typing import List, Dict, Any

class DatabaseOptimizer:
    """Database performance optimization utilities."""

    def __init__(self, engine):
        self.engine = engine
        self.session_factory = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def create_performance_indexes(self):
        """Create optimized indexes for common queries."""
        indexes = [
            # Audio file metadata searches
            Index('idx_tracks_bpm_key', 'bpm', 'key_signature'),
            Index('idx_tracks_genre_year', 'genre', 'year'),
            Index('idx_tracks_duration', 'duration'),
            Index('idx_tracks_energy_mood', 'energy_level', 'mood'),

            # Analysis results
            Index('idx_analysis_file_type', 'file_path', 'analysis_type'),
            Index('idx_analysis_created', 'created_at'),
            Index('idx_analysis_status', 'status'),

            # Playlists and user data
            Index('idx_playlists_user_created', 'user_id', 'created_at'),
            Index('idx_playlist_tracks_position', 'playlist_id', 'position'),

            # Full-text search indexes
            Index('idx_tracks_title_search', text("to_tsvector('english', title)")),
            Index('idx_tracks_artist_search', text("to_tsvector('english', artist)")),
        ]

        async with self.engine.begin() as conn:
            for index in indexes:
                try:
                    await conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index.name} {index}"))
                    print(f"Created index: {index.name}")
                except Exception as e:
                    print(f"Failed to create index {index.name}: {e}")

    async def optimize_track_search(self, criteria: Dict[str, Any], limit: int = 50) -> List[Dict]:
        """Optimized track search with efficient querying."""
        async with self.session_factory() as session:
            # Build optimized query with proper indexing
            base_query = """
            SELECT
                t.id, t.title, t.artist, t.bpm, t.key_signature,
                t.energy_level, t.mood, t.duration, t.file_path
            FROM tracks t
            WHERE 1=1
            """

            params = {}
            conditions = []

            # Use indexed columns for filtering
            if 'bpm_range' in criteria:
                conditions.append("t.bpm BETWEEN :bpm_min AND :bpm_max")
                params['bpm_min'] = criteria['bpm_range'][0]
                params['bpm_max'] = criteria['bpm_range'][1]

            if 'key' in criteria:
                conditions.append("t.key_signature = :key")
                params['key'] = criteria['key']

            if 'genre' in criteria:
                conditions.append("t.genre = :genre")
                params['genre'] = criteria['genre']

            if 'energy_range' in criteria:
                conditions.append("t.energy_level BETWEEN :energy_min AND :energy_max")
                params['energy_min'] = criteria['energy_range'][0]
                params['energy_max'] = criteria['energy_range'][1]

            # Full-text search with proper indexing
            if 'search_term' in criteria:
                conditions.append("""
                    (to_tsvector('english', t.title) @@ plainto_tsquery('english', :search_term)
                     OR to_tsvector('english', t.artist) @@ plainto_tsquery('english', :search_term))
                """)
                params['search_term'] = criteria['search_term']

            # Combine conditions
            if conditions:
                base_query += " AND " + " AND ".join(conditions)

            # Add ordering and limit
            base_query += """
            ORDER BY
                CASE WHEN :search_term IS NOT NULL THEN
                    ts_rank(to_tsvector('english', t.title || ' ' || t.artist),
                            plainto_tsquery('english', :search_term))
                ELSE t.created_at
                END DESC
            LIMIT :limit
            """
            params['limit'] = limit

            # Execute optimized query
            start_time = time.time()
            result = await session.execute(text(base_query), params)
            tracks = [dict(row._mapping) for row in result]
            duration = time.time() - start_time

            print(f"Optimized search returned {len(tracks)} tracks in {duration:.3f}s")
            return tracks

    async def bulk_insert_optimization(self, tracks: List[Dict[str, Any]]):
        """Optimized bulk insert with batching and transactions."""
        batch_size = 1000
        total_inserted = 0

        async with self.session_factory() as session:
            try:
                # Disable autocommit for better performance
                await session.execute(text("BEGIN"))

                # Process in batches
                for i in range(0, len(tracks), batch_size):
                    batch = tracks[i:i + batch_size]

                    # Use bulk insert with ON CONFLICT handling
                    insert_query = """
                    INSERT INTO tracks (title, artist, bpm, key_signature, duration, file_path,
                                      energy_level, mood, genre, year, created_at)
                    VALUES (:title, :artist, :bpm, :key_signature, :duration, :file_path,
                            :energy_level, :mood, :genre, :year, NOW())
                    ON CONFLICT (file_path) DO UPDATE SET
                        title = EXCLUDED.title,
                        artist = EXCLUDED.artist,
                        bpm = EXCLUDED.bpm,
                        key_signature = EXCLUDED.key_signature,
                        updated_at = NOW()
                    """

                    await session.execute(text(insert_query), batch)
                    total_inserted += len(batch)

                    # Commit every 5 batches to manage memory
                    if (i // batch_size) % 5 == 4:
                        await session.commit()
                        await session.execute(text("BEGIN"))

                await session.commit()
                print(f"Bulk inserted {total_inserted} tracks successfully")

            except Exception as e:
                await session.rollback()
                print(f"Bulk insert failed: {e}")
                raise

# Usage
async def optimize_database():
    optimizer = DatabaseOptimizer(engine)

    # Create performance indexes
    await optimizer.create_performance_indexes()

    # Test optimized search
    search_criteria = {
        'bpm_range': [120, 130],
        'key': 'C',
        'search_term': 'dance music'
    }

    results = await optimizer.optimize_track_search(search_criteria, limit=100)
    print(f"Found {len(results)} matching tracks")
```

### 2. Connection Pool Optimization

```python
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import QueuePool
import asyncio

def create_optimized_engine(database_url: str):
    """Create database engine with optimized connection pool."""
    return create_async_engine(
        database_url,
        # Connection pool settings
        poolclass=QueuePool,
        pool_size=20,          # Base number of connections
        max_overflow=30,       # Additional connections allowed
        pool_pre_ping=True,    # Validate connections before use
        pool_recycle=3600,     # Recycle connections every hour

        # Query optimization
        echo=False,            # Disable SQL logging in production
        future=True,           # Use SQLAlchemy 2.0 API

        # Connection timeout settings
        connect_args={
            "command_timeout": 30,
            "server_settings": {
                "jit": "off",  # Disable JIT for consistent performance
                "application_name": "tracktion_service",
            }
        }
    )

class ConnectionPoolMonitor:
    """Monitor and optimize connection pool performance."""

    def __init__(self, engine):
        self.engine = engine

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics."""
        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }

    async def optimize_pool_settings(self):
        """Dynamically optimize pool settings based on usage."""
        stats = self.get_pool_stats()

        # If we're frequently hitting overflow, increase pool size
        if stats['overflow'] > stats['pool_size'] * 0.8:
            print("High overflow detected, consider increasing pool_size")

        # If we have many idle connections, consider reducing pool size
        idle_ratio = stats['checked_in'] / (stats['checked_in'] + stats['checked_out'])
        if idle_ratio > 0.8:
            print("High idle connection ratio, consider reducing pool_size")

        return {
            'recommendation': 'optimize' if stats['overflow'] > 10 else 'current_ok',
            'stats': stats
        }
```

## Caching Strategies

### 1. Multi-Level Caching System

```python
import redis
import json
import hashlib
from typing import Optional, Any, Dict, List
from functools import wraps
import asyncio
import pickle

class CacheManager:
    """Multi-level caching system with Redis and in-memory caches."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.memory_cache = {}  # Simple in-memory cache
        self.cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}

    def _generate_cache_key(self, key_parts: List[str]) -> str:
        """Generate consistent cache key from parts."""
        combined = ":".join(str(part) for part in key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    async def get(self, key: str, cache_level: str = 'both') -> Optional[Any]:
        """Get value from cache with level preference."""
        try:
            # Try memory cache first (fastest)
            if cache_level in ['both', 'memory'] and key in self.memory_cache:
                self.cache_stats['hits'] += 1
                return self.memory_cache[key]['data']

            # Try Redis cache
            if cache_level in ['both', 'redis']:
                redis_value = await self.redis_client.get(key)
                if redis_value:
                    self.cache_stats['hits'] += 1
                    data = pickle.loads(redis_value)

                    # Store in memory cache for future access
                    if cache_level == 'both':
                        self.memory_cache[key] = {'data': data}

                    return data

            self.cache_stats['misses'] += 1
            return None

        except Exception as e:
            print(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600, cache_level: str = 'both'):
        """Set value in cache with TTL."""
        try:
            if cache_level in ['both', 'memory']:
                self.memory_cache[key] = {'data': value}

            if cache_level in ['both', 'redis']:
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)

            self.cache_stats['sets'] += 1

        except Exception as e:
            print(f"Cache set error: {e}")

    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        try:
            # Invalidate Redis keys
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)

            # Invalidate memory cache entries
            to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in to_remove:
                del self.memory_cache[key]

        except Exception as e:
            print(f"Cache invalidation error: {e}")

# Caching decorators
cache_manager = CacheManager("redis://localhost:6379")

def cached(ttl: int = 3600, cache_level: str = 'both', key_prefix: str = ''):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__] + [str(arg) for arg in args]
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = cache_manager._generate_cache_key(key_parts)

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, cache_level)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, cache_level)

            return result
        return wrapper
    return decorator

# Usage examples
@cached(ttl=1800, key_prefix='track_analysis')
async def get_track_analysis(file_path: str) -> Dict[str, Any]:
    """Get cached track analysis results."""
    # Expensive analysis operation
    return await perform_audio_analysis(file_path)

@cached(ttl=3600, cache_level='redis', key_prefix='playlist_gen')
async def generate_playlist_cached(criteria: Dict[str, Any]) -> List[Dict]:
    """Generate playlist with caching."""
    return await generate_playlist_from_db(criteria)
```

### 2. Cache Warming and Preloading

```python
class CacheWarmer:
    """Proactive cache warming for better performance."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.warming_tasks = []

    async def warm_popular_tracks(self, limit: int = 1000):
        """Warm cache with popular track analyses."""
        print("Starting cache warming for popular tracks...")

        # Get most accessed tracks from database
        popular_tracks = await self.get_popular_tracks(limit)

        # Warm cache in batches
        batch_size = 50
        for i in range(0, len(popular_tracks), batch_size):
            batch = popular_tracks[i:i + batch_size]
            tasks = [
                self.warm_track_analysis(track['file_path'])
                for track in batch
            ]

            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"Warmed cache for {min(i + batch_size, len(popular_tracks))} tracks")

    async def warm_track_analysis(self, file_path: str):
        """Warm cache for specific track analysis."""
        try:
            # This will cache the result
            await get_track_analysis(file_path)
        except Exception as e:
            print(f"Failed to warm cache for {file_path}: {e}")

    async def warm_playlist_templates(self):
        """Warm cache with common playlist templates."""
        common_criteria = [
            {'genre': 'electronic', 'bpm_range': [128, 132]},
            {'genre': 'rock', 'energy_range': [0.7, 1.0]},
            {'mood': 'happy', 'bpm_range': [120, 140]},
            {'key': 'C', 'energy_range': [0.5, 0.8]},
        ]

        tasks = [
            generate_playlist_cached(criteria)
            for criteria in common_criteria
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Warmed cache for {len(common_criteria)} playlist templates")

    async def schedule_cache_warming(self):
        """Schedule regular cache warming."""
        while True:
            try:
                await self.warm_popular_tracks()
                await self.warm_playlist_templates()

                # Wait 4 hours before next warming cycle
                await asyncio.sleep(4 * 3600)

            except Exception as e:
                print(f"Cache warming error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error

# Background cache warming
cache_warmer = CacheWarmer(cache_manager)

async def start_cache_warming():
    """Start background cache warming task."""
    asyncio.create_task(cache_warmer.schedule_cache_warming())
```

## Async Processing Optimization

### 1. Concurrent Audio Processing

```python
import asyncio
import aiofiles
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import multiprocessing as mp

class AsyncAudioProcessor:
    """Optimized async audio processing with parallel execution."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.semaphore = asyncio.Semaphore(self.max_workers * 2)  # Limit concurrent tasks

    async def process_batch_optimized(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process batch of audio files with optimized concurrency."""
        # Group files by processing requirements
        fast_files = []  # Small files that can be processed quickly
        slow_files = []  # Large files requiring more processing power

        for file_path in file_paths:
            file_size = await self.get_file_size(file_path)
            if file_size < 50 * 1024 * 1024:  # 50MB threshold
                fast_files.append(file_path)
            else:
                slow_files.append(file_path)

        print(f"Processing {len(fast_files)} fast files and {len(slow_files)} slow files")

        # Process different groups with different strategies
        fast_tasks = [
            self.process_file_fast(file_path)
            for file_path in fast_files
        ]

        slow_tasks = [
            self.process_file_slow(file_path)
            for file_path in slow_files
        ]

        # Execute with proper concurrency control
        fast_results = await asyncio.gather(*fast_tasks, return_exceptions=True)
        slow_results = await asyncio.gather(*slow_tasks, return_exceptions=True)

        # Combine and filter results
        all_results = list(fast_results) + list(slow_results)
        successful_results = [
            result for result in all_results
            if not isinstance(result, Exception)
        ]

        print(f"Successfully processed {len(successful_results)} out of {len(file_paths)} files")
        return successful_results

    async def process_file_fast(self, file_path: str) -> Dict[str, Any]:
        """Process small files with thread pool."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_executor,
                self._cpu_bound_analysis,
                file_path
            )

    async def process_file_slow(self, file_path: str) -> Dict[str, Any]:
        """Process large files with process pool."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_executor,
                self._intensive_analysis,
                file_path
            )

    def _cpu_bound_analysis(self, file_path: str) -> Dict[str, Any]:
        """CPU-bound analysis for thread pool execution."""
        # Lightweight analysis operations
        try:
            # Your fast analysis code here
            return {
                'file_path': file_path,
                'analysis_type': 'fast',
                'status': 'completed'
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'status': 'failed'
            }

    def _intensive_analysis(self, file_path: str) -> Dict[str, Any]:
        """Intensive analysis for process pool execution."""
        # Heavy computational analysis
        try:
            # Your intensive analysis code here
            return {
                'file_path': file_path,
                'analysis_type': 'intensive',
                'status': 'completed'
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'status': 'failed'
            }

    async def get_file_size(self, file_path: str) -> int:
        """Get file size asynchronously."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                await f.seek(0, 2)  # Seek to end
                return await f.tell()
        except Exception:
            return 0

# Usage
processor = AsyncAudioProcessor(max_workers=4)

async def process_audio_library():
    """Process entire audio library with optimization."""
    # Get all audio files
    audio_files = await get_all_audio_files()

    # Process in optimized batches
    batch_size = 100
    all_results = []

    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}")

        batch_results = await processor.process_batch_optimized(batch)
        all_results.extend(batch_results)

        # Optional: Save intermediate results
        await save_batch_results(batch_results)

    print(f"Completed processing {len(all_results)} audio files")
    return all_results
```

### 2. Queue-Based Processing with Backpressure

```python
import asyncio
from asyncio import Queue
from typing import Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import time

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ProcessingTask:
    id: str
    data: Any
    priority: TaskPriority
    created_at: float
    max_retries: int = 3
    retry_count: int = 0

class OptimizedTaskQueue:
    """High-performance async task queue with backpressure control."""

    def __init__(self,
                 max_workers: int = 4,
                 queue_size: int = 1000,
                 backpressure_threshold: float = 0.8):
        self.max_workers = max_workers
        self.queue = Queue(maxsize=queue_size)
        self.backpressure_threshold = backpressure_threshold
        self.workers = []
        self.running = False

        # Performance metrics
        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'average_processing_time': 0.0,
            'queue_size': 0
        }

    async def add_task(self, task: ProcessingTask) -> bool:
        """Add task to queue with backpressure control."""
        current_size = self.queue.qsize()
        queue_utilization = current_size / self.queue.maxsize

        # Apply backpressure if queue is too full
        if queue_utilization > self.backpressure_threshold:
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                # Allow high priority tasks to bypass backpressure
                await self.queue.put(task)
                return True
            else:
                # Reject low priority tasks when under backpressure
                print(f"Backpressure: rejecting task {task.id} (queue {queue_utilization:.1%} full)")
                return False
        else:
            await self.queue.put(task)
            return True

    async def start_workers(self, processor_func: Callable):
        """Start worker tasks."""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}", processor_func))
            for i in range(self.max_workers)
        ]

        # Start metrics collector
        asyncio.create_task(self._metrics_collector())

    async def _worker(self, worker_name: str, processor_func: Callable):
        """Worker coroutine to process tasks."""
        while self.running:
            try:
                # Get task with timeout to avoid blocking
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                start_time = time.time()
                success = False

                try:
                    # Process the task
                    result = await processor_func(task.data)
                    success = True
                    self.metrics['tasks_processed'] += 1

                except Exception as e:
                    print(f"Task {task.id} failed: {e}")

                    # Handle retries
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await self.queue.put(task)  # Retry the task
                        self.metrics['tasks_retried'] += 1
                    else:
                        self.metrics['tasks_failed'] += 1

                # Update metrics
                processing_time = time.time() - start_time
                self.metrics['average_processing_time'] = (
                    self.metrics['average_processing_time'] * 0.9 +
                    processing_time * 0.1
                )

                self.queue.task_done()

            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                print(f"Worker {worker_name} error: {e}")

    async def _metrics_collector(self):
        """Collect and log performance metrics."""
        while self.running:
            self.metrics['queue_size'] = self.queue.qsize()

            # Log metrics every minute
            await asyncio.sleep(60)
            print(f"Queue metrics: {self.metrics}")

    async def shutdown(self):
        """Graceful shutdown of the queue."""
        self.running = False

        # Wait for all workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        # Process remaining tasks
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break

# Usage example
task_queue = OptimizedTaskQueue(max_workers=8, queue_size=5000)

async def audio_processor(audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio data asynchronously."""
    # Simulate processing time
    await asyncio.sleep(0.1)

    # Your actual audio processing logic here
    return {
        'file_path': audio_data['file_path'],
        'processed_at': time.time(),
        'status': 'completed'
    }

async def queue_based_processing():
    """Example of queue-based audio processing."""
    # Start the workers
    await task_queue.start_workers(audio_processor)

    # Add tasks to the queue
    for i in range(1000):
        task = ProcessingTask(
            id=f"audio_task_{i}",
            data={'file_path': f'/audio/file_{i}.mp3'},
            priority=TaskPriority.NORMAL if i % 10 != 0 else TaskPriority.HIGH,
            created_at=time.time()
        )

        success = await task_queue.add_task(task)
        if not success:
            print(f"Task {task.id} rejected due to backpressure")

    # Wait for all tasks to complete
    await task_queue.queue.join()

    # Shutdown gracefully
    await task_queue.shutdown()
```

## Memory Management

### 1. Memory-Efficient Data Processing

```python
import gc
import tracemalloc
from typing import Iterator, List, Dict, Any, Generator
import psutil
import asyncio

class MemoryOptimizedProcessor:
    """Memory-efficient data processing with monitoring."""

    def __init__(self, memory_limit_mb: int = 512):
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        tracemalloc.start()

    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    async def process_large_dataset_streaming(self,
                                            data_source: str,
                                            batch_size: int = 1000) -> Generator[List[Dict], None, None]:
        """Process large dataset with streaming and memory management."""

        async def data_generator() -> Iterator[Dict[str, Any]]:
            """Generate data items one at a time to minimize memory usage."""
            # Your data source implementation here
            # This could be database cursor, file reader, etc.
            for i in range(100000):  # Simulated large dataset
                yield {'id': i, 'data': f'item_{i}' * 100}  # Simulated data

        batch = []
        processed_count = 0

        async for item in data_generator():
            batch.append(item)

            # Process batch when full or memory limit reached
            if (len(batch) >= batch_size or
                self.memory_usage_mb() > self.memory_limit_mb * 0.8):

                # Process batch
                processed_batch = await self._process_batch(batch)
                yield processed_batch

                processed_count += len(batch)
                print(f"Processed {processed_count} items, "
                     f"memory: {self.memory_usage_mb():.1f}MB")

                # Clear batch and force garbage collection
                batch.clear()
                gc.collect()

                # Optional: wait for memory to stabilize
                if self.memory_usage_mb() > self.memory_limit_mb:
                    await asyncio.sleep(0.1)

        # Process remaining items
        if batch:
            processed_batch = await self._process_batch(batch)
            yield processed_batch

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items."""
        # Simulate processing
        await asyncio.sleep(0.01)

        # Transform data (in-place to save memory)
        for item in batch:
            item['processed'] = True
            item['timestamp'] = time.time()

        return batch

    async def memory_aware_audio_processing(self, file_paths: List[str]) -> List[Dict]:
        """Process audio files with memory awareness."""
        results = []

        for file_path in file_paths:
            # Check memory before processing each file
            current_memory = self.memory_usage_mb()

            if current_memory > self.memory_limit_mb * 0.9:
                print(f"Memory usage high ({current_memory:.1f}MB), forcing GC")
                gc.collect()
                await asyncio.sleep(0.1)  # Let GC complete

            # Process file
            try:
                result = await self._process_audio_file_memory_efficient(file_path)
                results.append(result)

            except MemoryError:
                print(f"Memory error processing {file_path}, skipping")
                gc.collect()
                continue

        return results

    async def _process_audio_file_memory_efficient(self, file_path: str) -> Dict[str, Any]:
        """Process single audio file with memory efficiency."""
        # Use context managers and process in chunks
        try:
            # Your memory-efficient audio processing logic here
            # Process file in chunks rather than loading entirely

            return {
                'file_path': file_path,
                'status': 'completed',
                'memory_peak': self.memory_usage_mb()
            }
        finally:
            # Ensure cleanup
            gc.collect()

# Usage
memory_processor = MemoryOptimizedProcessor(memory_limit_mb=256)

async def memory_efficient_processing():
    """Example of memory-efficient processing."""

    # Process large dataset with streaming
    async for batch in memory_processor.process_large_dataset_streaming(
        'large_dataset.db', batch_size=500
    ):
        # Process each batch as it becomes available
        print(f"Received batch with {len(batch)} items")

        # Optional: save batch results immediately
        await save_batch_to_storage(batch)

    # Process audio files with memory awareness
    audio_files = await get_audio_file_list()
    results = await memory_processor.memory_aware_audio_processing(audio_files)

    print(f"Completed processing with peak memory: "
          f"{memory_processor.memory_usage_mb():.1f}MB")
```

### 2. Object Pool for Resource Management

```python
import asyncio
from typing import Any, Dict, List, Optional, Protocol
from contextlib import asynccontextmanager
import weakref

class PoolableResource(Protocol):
    """Protocol for resources that can be pooled."""

    def reset(self) -> None:
        """Reset resource to clean state."""
        ...

    def is_valid(self) -> bool:
        """Check if resource is still valid."""
        ...

class ResourcePool:
    """Generic object pool for expensive resources."""

    def __init__(self,
                 resource_factory: callable,
                 max_size: int = 10,
                 min_size: int = 2):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size

        self.available = asyncio.Queue()
        self.in_use = weakref.WeakSet()
        self.created_count = 0

        # Initialize minimum number of resources
        asyncio.create_task(self._initialize_pool())

    async def _initialize_pool(self):
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            resource = await self._create_resource()
            await self.available.put(resource)

    async def _create_resource(self) -> Any:
        """Create new resource instance."""
        self.created_count += 1
        return await self.resource_factory()

    @asynccontextmanager
    async def acquire(self):
        """Acquire resource from pool."""
        resource = None

        try:
            # Try to get available resource
            try:
                resource = self.available.get_nowait()
            except asyncio.QueueEmpty:
                # Create new resource if under limit
                if self.created_count < self.max_size:
                    resource = await self._create_resource()
                else:
                    # Wait for available resource
                    resource = await self.available.get()

            # Validate resource
            if hasattr(resource, 'is_valid') and not resource.is_valid():
                resource = await self._create_resource()

            # Reset resource state
            if hasattr(resource, 'reset'):
                resource.reset()

            self.in_use.add(resource)
            yield resource

        finally:
            if resource:
                self.in_use.discard(resource)
                await self.available.put(resource)

# Example: Audio analysis engine pool
class AudioAnalysisEngine:
    """Example resource for pooling."""

    def __init__(self):
        self.model_loaded = False
        self.analysis_count = 0

    async def initialize(self):
        """Load heavy ML models."""
        # Simulate expensive initialization
        await asyncio.sleep(2.0)
        self.model_loaded = True
        print("Audio analysis engine initialized")

    def reset(self):
        """Reset engine state."""
        self.analysis_count = 0

    def is_valid(self) -> bool:
        """Check if engine is still valid."""
        return self.model_loaded

    async def analyze(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyze audio data."""
        if not self.model_loaded:
            raise RuntimeError("Engine not initialized")

        self.analysis_count += 1
        # Simulate analysis
        await asyncio.sleep(0.1)

        return {
            'bpm': 128,
            'key': 'C',
            'energy': 0.75,
            'analysis_id': self.analysis_count
        }

async def create_audio_engine():
    """Factory function for audio analysis engine."""
    engine = AudioAnalysisEngine()
    await engine.initialize()
    return engine

# Usage
engine_pool = ResourcePool(
    resource_factory=create_audio_engine,
    max_size=5,
    min_size=2
)

async def analyze_with_pooled_engine(audio_file_path: str) -> Dict[str, Any]:
    """Analyze audio file using pooled engine."""

    # Load audio data
    audio_data = await load_audio_data(audio_file_path)

    # Use pooled resource
    async with engine_pool.acquire() as engine:
        result = await engine.analyze(audio_data)
        result['file_path'] = audio_file_path
        return result

async def batch_analysis_with_pool(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple files using resource pool."""

    # Process files concurrently with pooled resources
    tasks = [
        analyze_with_pooled_engine(file_path)
        for file_path in file_paths
    ]

    results = await asyncio.gather(*tasks)
    return results
```

## I/O Optimization

### 1. Async File Operations

```python
import aiofiles
import aiofiles.os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
import hashlib

class AsyncFileManager:
    """Optimized async file operations."""

    def __init__(self, max_concurrent_operations: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_operations)
        self.file_cache = {}

    async def read_file_chunked(self, file_path: str, chunk_size: int = 64 * 1024) -> AsyncGenerator[bytes, None]:
        """Read file in chunks for memory efficiency."""
        async with self.semaphore:
            async with aiofiles.open(file_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    async def write_file_atomic(self, file_path: str, data: bytes):
        """Write file atomically to prevent corruption."""
        temp_path = f"{file_path}.tmp"

        async with self.semaphore:
            try:
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(data)
                    await f.fsync()  # Ensure data is written to disk

                # Atomic rename
                await aiofiles.os.rename(temp_path, file_path)

            except Exception:
                # Clean up temp file on error
                try:
                    await aiofiles.os.remove(temp_path)
                except FileNotFoundError:
                    pass
                raise

    async def parallel_file_processing(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files in parallel with I/O optimization."""

        async def process_single_file(file_path: str) -> Dict[str, Any]:
            """Process single file with error handling."""
            try:
                # Get file stats
                stat = await aiofiles.os.stat(file_path)

                # Calculate file hash efficiently
                file_hash = await self.calculate_file_hash(file_path)

                # Read file metadata
                metadata = await self.extract_file_metadata(file_path)

                return {
                    'file_path': file_path,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'hash': file_hash,
                    'metadata': metadata,
                    'status': 'success'
                }

            except Exception as e:
                return {
                    'file_path': file_path,
                    'error': str(e),
                    'status': 'error'
                }

        # Process files concurrently
        tasks = [process_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        return results

    async def calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash asynchronously."""
        hash_md5 = hashlib.md5()

        async for chunk in self.read_file_chunked(file_path):
            hash_md5.update(chunk)

        return hash_md5.hexdigest()

    async def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata efficiently."""
        path = Path(file_path)

        # Basic metadata
        metadata = {
            'name': path.name,
            'extension': path.suffix.lower(),
            'directory': str(path.parent)
        }

        # Audio-specific metadata (if audio file)
        if path.suffix.lower() in ['.mp3', '.wav', '.flac', '.aac']:
            # Use lightweight metadata extraction
            try:
                audio_metadata = await self.extract_audio_metadata_light(file_path)
                metadata.update(audio_metadata)
            except Exception as e:
                metadata['metadata_error'] = str(e)

        return metadata

    async def extract_audio_metadata_light(self, file_path: str) -> Dict[str, Any]:
        """Lightweight audio metadata extraction."""
        # This would use a lightweight library like mutagen
        # Simulated for example
        await asyncio.sleep(0.01)  # Simulate I/O

        return {
            'duration': 180.5,
            'bitrate': 320000,
            'sample_rate': 44100,
            'channels': 2
        }

# Usage example
file_manager = AsyncFileManager(max_concurrent_operations=20)

async def process_audio_library_io_optimized(library_path: str) -> List[Dict[str, Any]]:
    """Process entire audio library with I/O optimization."""

    # Find all audio files efficiently
    audio_files = []
    path = Path(library_path)

    # Use async directory traversal
    for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
        async for file_path in async_glob(path, f"**/*{ext}"):
            audio_files.append(str(file_path))

    print(f"Found {len(audio_files)} audio files")

    # Process files in batches for memory efficiency
    batch_size = 100
    all_results = []

    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        batch_results = await file_manager.parallel_file_processing(batch)
        all_results.extend(batch_results)

        print(f"Processed batch {i // batch_size + 1}/{(len(audio_files) - 1) // batch_size + 1}")

    return all_results

async def async_glob(path: Path, pattern: str):
    """Async generator for file globbing."""
    # This would be implemented using aiofiles or similar
    # Simulated for example
    for file_path in path.rglob(pattern):
        yield file_path
        await asyncio.sleep(0)  # Yield control
```

### 2. Database I/O Optimization

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any, Optional
import asyncio

class DatabaseIOOptimizer:
    """Optimize database I/O operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert_optimized(self, table_name: str, data: List[Dict[str, Any]],
                                  conflict_columns: List[str]) -> int:
        """Optimized bulk upsert with minimal I/O."""

        if not data:
            return 0

        # Prepare bulk upsert query
        columns = list(data[0].keys())
        values_placeholder = ', '.join([f":{col}" for col in columns])

        # Build conflict resolution
        conflict_cols = ', '.join(conflict_columns)
        update_cols = ', '.join([
            f"{col} = EXCLUDED.{col}"
            for col in columns if col not in conflict_columns
        ])

        query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({values_placeholder})
        ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_cols}
        """

        # Execute in batches to manage memory and locks
        batch_size = 1000
        total_processed = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            try:
                result = await self.session.execute(text(query), batch)
                total_processed += result.rowcount

                # Commit every few batches to release locks
                if (i // batch_size) % 5 == 4:
                    await self.session.commit()

            except Exception as e:
                await self.session.rollback()
                print(f"Batch upsert failed at index {i}: {e}")
                raise

        await self.session.commit()
        return total_processed

    async def streaming_query_results(self, query: str, params: Dict = None,
                                    batch_size: int = 1000) -> AsyncGenerator[List[Dict], None]:
        """Stream large query results to minimize memory usage."""

        offset = 0

        while True:
            # Add limit and offset to query
            paginated_query = f"{query} LIMIT :limit OFFSET :offset"
            query_params = (params or {}).copy()
            query_params.update({'limit': batch_size, 'offset': offset})

            # Execute paginated query
            result = await self.session.execute(text(paginated_query), query_params)
            rows = [dict(row._mapping) for row in result]

            if not rows:
                break

            yield rows
            offset += len(rows)

            # If we got fewer rows than batch size, we're done
            if len(rows) < batch_size:
                break

    async def parallel_query_execution(self, queries: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple queries in parallel."""

        async def execute_single_query(query_info: Dict[str, Any]) -> Any:
            """Execute single query with error handling."""
            try:
                result = await self.session.execute(
                    text(query_info['query']),
                    query_info.get('params', {})
                )

                if query_info.get('fetch_all', True):
                    return [dict(row._mapping) for row in result]
                else:
                    return result.rowcount

            except Exception as e:
                return {'error': str(e), 'query': query_info['query']}

        # Execute queries concurrently
        tasks = [execute_single_query(query_info) for query_info in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def batch_insert_with_conflict_handling(self, table_name: str,
                                                data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Batch insert with detailed conflict handling."""

        stats = {'inserted': 0, 'updated': 0, 'errors': 0}

        # Group data by presence of primary key for different handling
        new_records = []
        update_records = []

        for record in data:
            if 'id' in record and record['id']:
                update_records.append(record)
            else:
                new_records.append(record)

        # Handle new records with batch insert
        if new_records:
            try:
                columns = list(new_records[0].keys())
                values_placeholder = ', '.join([f":{col}" for col in columns])

                insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({values_placeholder})
                ON CONFLICT DO NOTHING
                """

                result = await self.session.execute(text(insert_query), new_records)
                stats['inserted'] = result.rowcount

            except Exception as e:
                stats['errors'] += len(new_records)
                print(f"Batch insert failed: {e}")

        # Handle updates individually for better error handling
        if update_records:
            for record in update_records:
                try:
                    # Build dynamic update query
                    set_clauses = ', '.join([
                        f"{col} = :{col}" for col in record.keys() if col != 'id'
                    ])

                    update_query = f"""
                    UPDATE {table_name} SET {set_clauses} WHERE id = :id
                    """

                    result = await self.session.execute(text(update_query), record)
                    if result.rowcount > 0:
                        stats['updated'] += 1

                except Exception as e:
                    stats['errors'] += 1
                    print(f"Update failed for record {record.get('id')}: {e}")

        await self.session.commit()
        return stats

# Usage example
async def optimize_database_operations():
    """Example of optimized database operations."""

    db_optimizer = DatabaseIOOptimizer(session)

    # Bulk upsert optimization
    track_data = [
        {'file_path': '/music/track1.mp3', 'title': 'Track 1', 'bpm': 128},
        {'file_path': '/music/track2.mp3', 'title': 'Track 2', 'bpm': 132},
        # ... more records
    ]

    processed = await db_optimizer.bulk_upsert_optimized(
        'tracks',
        track_data,
        conflict_columns=['file_path']
    )
    print(f"Processed {processed} records")

    # Stream large query results
    query = "SELECT * FROM tracks WHERE bpm BETWEEN 120 AND 140 ORDER BY title"

    async for batch in db_optimizer.streaming_query_results(query, batch_size=500):
        # Process each batch
        print(f"Processing batch of {len(batch)} records")
        await process_track_batch(batch)

    # Parallel query execution
    analysis_queries = [
        {'query': 'SELECT COUNT(*) as total FROM tracks', 'fetch_all': True},
        {'query': 'SELECT AVG(bpm) as avg_bpm FROM tracks', 'fetch_all': True},
        {'query': 'SELECT COUNT(*) as analyzed FROM tracks WHERE analysis_completed = true', 'fetch_all': True}
    ]

    results = await db_optimizer.parallel_query_execution(analysis_queries)
    print(f"Analysis results: {results}")
```

This performance optimization guide provides comprehensive examples for improving Tracktion service performance across multiple dimensions. Each technique includes practical code examples and can be adapted to your specific use cases and requirements.
