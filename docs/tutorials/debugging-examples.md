# Debugging Examples

This document provides comprehensive debugging examples for Tracktion services, covering common issues, debugging techniques, and troubleshooting strategies.

## Table of Contents

1. [Common Debugging Scenarios](#common-debugging-scenarios)
2. [Service-Specific Debugging](#service-specific-debugging)
3. [Performance Debugging](#performance-debugging)
4. [Database Debugging](#database-debugging)
5. [Message Queue Debugging](#message-queue-debugging)
6. [File Processing Debugging](#file-processing-debugging)
7. [Integration Debugging](#integration-debugging)
8. [Docker Container Debugging](#docker-container-debugging)
9. [Logging Best Practices](#logging-best-practices)
10. [Debugging Tools and Utilities](#debugging-tools-and-utilities)

## Common Debugging Scenarios

### 1. Service Startup Issues

**Problem**: Service fails to start or crashes immediately

```python
import logging
import sys
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

def debug_service_startup():
    """Debug service startup issues with comprehensive error tracking."""
    try:
        # Check environment variables
        required_env_vars = [
            'DATABASE_URL', 'REDIS_URL', 'RABBITMQ_URL',
            'JWT_SECRET_KEY', 'ANALYSIS_SERVICE_URL'
        ]

        missing_vars = []
        for var in required_env_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False

        # Test database connection
        from services.shared.database import get_database_connection
        try:
            conn = get_database_connection()
            conn.execute("SELECT 1")
            logger.info("Database connection: OK")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

        # Test Redis connection
        from services.shared.cache import get_redis_client
        try:
            redis_client = get_redis_client()
            redis_client.ping()
            logger.info("Redis connection: OK")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False

        # Test RabbitMQ connection
        from services.shared.messaging import get_rabbitmq_connection
        try:
            rabbit_conn = get_rabbitmq_connection()
            rabbit_conn.channel()
            logger.info("RabbitMQ connection: OK")
        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            return False

        logger.info("All startup checks passed")
        return True

    except Exception as e:
        logger.error(f"Startup debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Usage in your service startup
if __name__ == "__main__":
    if not debug_service_startup():
        sys.exit(1)

    # Continue with normal startup
    app.run()
```

### 2. API Response Issues

**Problem**: API returns unexpected responses or errors

```python
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def debug_api_endpoint(func):
    """Decorator to debug API endpoint issues."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Log request details
        logger.info(f"[{request_id}] Starting {func.__name__}")
        logger.debug(f"[{request_id}] Args: {args}")
        logger.debug(f"[{request_id}] Kwargs: {kwargs}")

        try:
            # Execute the endpoint
            result = await func(*args, **kwargs)

            # Log successful response
            logger.info(f"[{request_id}] Success: {func.__name__}")
            logger.debug(f"[{request_id}] Response: {json.dumps(result, default=str)[:500]}")

            return result

        except Exception as e:
            # Log error details
            logger.error(f"[{request_id}] Error in {func.__name__}: {str(e)}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")

            # Return structured error response
            error_response = {
                "error": str(e),
                "endpoint": func.__name__,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }

            raise HTTPException(
                status_code=500,
                detail=error_response
            )

    return wrapper

# Usage example
@app.post("/api/analyze")
@debug_api_endpoint
async def analyze_audio_file(file_data: dict):
    """Analyze audio file with debugging."""
    # Your analysis logic here
    pass
```

### 3. Memory Leak Detection

**Problem**: Service memory usage grows over time

```python
import gc
import psutil
import tracemalloc
from typing import Dict, List
from contextlib import contextmanager

class MemoryTracker:
    """Track memory usage and detect potential leaks."""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
        tracemalloc.start()

    def take_snapshot(self, label: str) -> Dict:
        """Take a memory snapshot with label."""
        snapshot = tracemalloc.take_snapshot()
        memory_info = self.process.memory_info()

        snapshot_data = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident memory
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
            'tracemalloc_snapshot': snapshot
        }

        self.snapshots.append(snapshot_data)
        logger.info(f"Memory snapshot '{label}': RSS={snapshot_data['rss_mb']:.1f}MB")

        return snapshot_data

    def compare_snapshots(self, snapshot1_idx: int = -2, snapshot2_idx: int = -1):
        """Compare two memory snapshots."""
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots to compare")
            return

        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]

        # Memory size comparison
        rss_diff = snap2['rss_mb'] - snap1['rss_mb']
        logger.info(f"Memory change from '{snap1['label']}' to '{snap2['label']}': {rss_diff:+.1f}MB")

        # Top memory allocations
        top_stats = snap2['tracemalloc_snapshot'].compare_to(
            snap1['tracemalloc_snapshot'], 'lineno'
        )

        logger.info("Top 10 memory allocation differences:")
        for index, stat in enumerate(top_stats[:10], 1):
            logger.info(f"{index:2d}. {stat}")

    def force_garbage_collection(self):
        """Force garbage collection and log results."""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())

        logger.info(f"Garbage collection: {collected} objects collected")
        logger.info(f"Objects before/after: {before_objects}/{after_objects}")

# Usage example
memory_tracker = MemoryTracker()

@contextmanager
def track_memory(operation_name: str):
    """Context manager to track memory usage of an operation."""
    memory_tracker.take_snapshot(f"{operation_name}_start")
    try:
        yield
    finally:
        memory_tracker.take_snapshot(f"{operation_name}_end")
        memory_tracker.compare_snapshots()

# Use in your code
async def process_large_batch():
    with track_memory("large_batch_processing"):
        # Your batch processing logic
        for i in range(1000):
            await process_audio_file(f"file_{i}.mp3")

    # Force GC after large operations
    memory_tracker.force_garbage_collection()
```

## Service-Specific Debugging

### Analysis Service Debugging

```python
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

class AnalysisServiceDebugger:
    """Debug analysis service issues."""

    def __init__(self):
        self.logger = logging.getLogger("analysis_service_debug")

    async def debug_file_analysis(self, file_path: str) -> Dict[str, Any]:
        """Debug audio file analysis issues."""
        debug_info = {
            'file_path': file_path,
            'file_exists': False,
            'file_size': 0,
            'file_format': None,
            'analysis_steps': {},
            'errors': [],
            'timing': {}
        }

        try:
            # Check file existence and properties
            path = Path(file_path)
            if path.exists():
                debug_info['file_exists'] = True
                debug_info['file_size'] = path.stat().st_size
                debug_info['file_format'] = path.suffix.lower()
                self.logger.info(f"File check: {file_path} exists, size={debug_info['file_size']} bytes")
            else:
                debug_info['errors'].append(f"File does not exist: {file_path}")
                return debug_info

            # Test BPM detection
            start_time = time.time()
            try:
                from services.analysis_service.src.bpm_detector import BPMDetector
                bpm_detector = BPMDetector()
                bpm_result = await bpm_detector.analyze(file_path)
                debug_info['analysis_steps']['bpm'] = {
                    'success': True,
                    'result': bpm_result,
                    'duration': time.time() - start_time
                }
                self.logger.info(f"BPM analysis: SUCCESS, BPM={bpm_result.get('bpm')}")
            except Exception as e:
                debug_info['analysis_steps']['bpm'] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                debug_info['errors'].append(f"BPM analysis failed: {e}")
                self.logger.error(f"BPM analysis: FAILED, error={e}")

            # Test key detection
            start_time = time.time()
            try:
                from services.analysis_service.src.key_detector import KeyDetector
                key_detector = KeyDetector()
                key_result = await key_detector.analyze(file_path)
                debug_info['analysis_steps']['key'] = {
                    'success': True,
                    'result': key_result,
                    'duration': time.time() - start_time
                }
                self.logger.info(f"Key analysis: SUCCESS, key={key_result.get('key')}")
            except Exception as e:
                debug_info['analysis_steps']['key'] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                debug_info['errors'].append(f"Key analysis failed: {e}")
                self.logger.error(f"Key analysis: FAILED, error={e}")

            # Test mood analysis
            start_time = time.time()
            try:
                from services.analysis_service.src.mood_analyzer import MoodAnalyzer
                mood_analyzer = MoodAnalyzer()
                mood_result = await mood_analyzer.analyze(file_path)
                debug_info['analysis_steps']['mood'] = {
                    'success': True,
                    'result': mood_result,
                    'duration': time.time() - start_time
                }
                self.logger.info(f"Mood analysis: SUCCESS, mood={mood_result.get('mood')}")
            except Exception as e:
                debug_info['analysis_steps']['mood'] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                debug_info['errors'].append(f"Mood analysis failed: {e}")
                self.logger.error(f"Mood analysis: FAILED, error={e}")

        except Exception as e:
            debug_info['errors'].append(f"General analysis error: {e}")
            self.logger.error(f"Analysis debug failed: {e}")

        return debug_info

# Usage
async def debug_analysis():
    debugger = AnalysisServiceDebugger()
    result = await debugger.debug_file_analysis("/path/to/audio/file.mp3")
    print(json.dumps(result, indent=2, default=str))
```

### Tracklist Service Debugging

```python
class TracklistServiceDebugger:
    """Debug tracklist service issues."""

    def __init__(self):
        self.logger = logging.getLogger("tracklist_service_debug")

    async def debug_playlist_generation(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Debug playlist generation issues."""
        debug_info = {
            'criteria': criteria,
            'database_query': {},
            'matching_tracks': 0,
            'algorithm_steps': {},
            'errors': [],
            'warnings': []
        }

        try:
            # Debug database query
            from services.tracklist_service.src.database import get_session
            from services.tracklist_service.src.models import Track

            async with get_session() as session:
                # Test basic track retrieval
                try:
                    track_count = await session.execute(
                        "SELECT COUNT(*) FROM tracks"
                    )
                    total_tracks = track_count.scalar()
                    debug_info['database_query']['total_tracks'] = total_tracks
                    self.logger.info(f"Database: {total_tracks} total tracks found")

                    if total_tracks == 0:
                        debug_info['warnings'].append("No tracks in database")
                        return debug_info

                except Exception as e:
                    debug_info['errors'].append(f"Database query failed: {e}")
                    return debug_info

                # Test criteria filtering
                filters = []
                params = {}

                if 'genre' in criteria:
                    filters.append("genre = :genre")
                    params['genre'] = criteria['genre']

                if 'bpm_range' in criteria:
                    filters.append("bpm BETWEEN :bpm_min AND :bpm_max")
                    params['bpm_min'] = criteria['bpm_range'][0]
                    params['bpm_max'] = criteria['bpm_range'][1]

                if 'key' in criteria:
                    filters.append("key_signature = :key")
                    params['key'] = criteria['key']

                # Execute filtered query
                where_clause = " AND ".join(filters) if filters else "1=1"
                query = f"SELECT COUNT(*) FROM tracks WHERE {where_clause}"

                result = await session.execute(query, params)
                matching_count = result.scalar()
                debug_info['matching_tracks'] = matching_count

                self.logger.info(f"Filtered query: {matching_count} matching tracks")

                if matching_count == 0:
                    debug_info['warnings'].append("No tracks match the criteria")
                    return debug_info

                # Test algorithm steps
                from services.tracklist_service.src.playlist_generator import PlaylistGenerator

                generator = PlaylistGenerator()

                # Debug track selection algorithm
                try:
                    selected_tracks = await generator.select_tracks(criteria)
                    debug_info['algorithm_steps']['track_selection'] = {
                        'success': True,
                        'selected_count': len(selected_tracks),
                        'first_track': selected_tracks[0] if selected_tracks else None
                    }
                    self.logger.info(f"Track selection: {len(selected_tracks)} tracks selected")
                except Exception as e:
                    debug_info['algorithm_steps']['track_selection'] = {
                        'success': False,
                        'error': str(e)
                    }
                    debug_info['errors'].append(f"Track selection failed: {e}")

                # Debug ordering algorithm
                if 'algorithm_steps' in debug_info and debug_info['algorithm_steps'].get('track_selection', {}).get('success'):
                    try:
                        ordered_playlist = await generator.order_tracks(selected_tracks, criteria)
                        debug_info['algorithm_steps']['track_ordering'] = {
                            'success': True,
                            'final_count': len(ordered_playlist),
                            'ordering_method': criteria.get('ordering', 'default')
                        }
                        self.logger.info(f"Track ordering: SUCCESS, final playlist has {len(ordered_playlist)} tracks")
                    except Exception as e:
                        debug_info['algorithm_steps']['track_ordering'] = {
                            'success': False,
                            'error': str(e)
                        }
                        debug_info['errors'].append(f"Track ordering failed: {e}")

        except Exception as e:
            debug_info['errors'].append(f"General playlist debug error: {e}")
            self.logger.error(f"Playlist debug failed: {e}")

        return debug_info
```

## Performance Debugging

### Slow Query Detection

```python
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

class QueryPerformanceTracker:
    """Track and debug slow database queries."""

    def __init__(self, slow_threshold: float = 1.0):
        self.slow_threshold = slow_threshold
        self.query_stats = []
        self.logger = logging.getLogger("query_performance")

    @asynccontextmanager
    async def track_query(self, query_name: str, query_sql: Optional[str] = None):
        """Context manager to track query performance."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time

            query_info = {
                'name': query_name,
                'sql': query_sql,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'is_slow': duration > self.slow_threshold
            }

            self.query_stats.append(query_info)

            if query_info['is_slow']:
                self.logger.warning(
                    f"SLOW QUERY: {query_name} took {duration:.2f}s "
                    f"(threshold: {self.slow_threshold}s)"
                )
                if query_sql:
                    self.logger.warning(f"SQL: {query_sql}")
            else:
                self.logger.debug(f"Query {query_name}: {duration:.2f}s")

    def get_slow_queries(self) -> List[Dict]:
        """Get all queries that exceeded the slow threshold."""
        return [q for q in self.query_stats if q['is_slow']]

    def get_performance_summary(self) -> Dict:
        """Get overall query performance summary."""
        if not self.query_stats:
            return {'total_queries': 0}

        durations = [q['duration'] for q in self.query_stats]
        slow_queries = self.get_slow_queries()

        return {
            'total_queries': len(self.query_stats),
            'slow_queries': len(slow_queries),
            'slow_query_percentage': (len(slow_queries) / len(self.query_stats)) * 100,
            'average_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'slowest_query': max(self.query_stats, key=lambda x: x['duration'])
        }

# Usage
query_tracker = QueryPerformanceTracker(slow_threshold=0.5)

async def search_tracks_with_debug(criteria: dict):
    """Search tracks with performance debugging."""

    # Debug individual queries
    async with query_tracker.track_query("count_all_tracks", "SELECT COUNT(*) FROM tracks"):
        total_count = await get_total_track_count()

    async with query_tracker.track_query("search_by_criteria", f"SELECT * FROM tracks WHERE ..."):
        results = await search_tracks_by_criteria(criteria)

    async with query_tracker.track_query("load_track_details", "SELECT * FROM track_details WHERE ..."):
        detailed_results = await load_track_details(results)

    # Print performance summary
    summary = query_tracker.get_performance_summary()
    logger.info(f"Query performance summary: {summary}")

    # Log slow queries
    slow_queries = query_tracker.get_slow_queries()
    if slow_queries:
        logger.warning(f"Found {len(slow_queries)} slow queries:")
        for query in slow_queries:
            logger.warning(f"  - {query['name']}: {query['duration']:.2f}s")

    return detailed_results
```

### Async Task Debugging

```python
import asyncio
import weakref
from typing import Set, Dict, Any

class AsyncTaskTracker:
    """Track and debug async tasks for deadlocks and resource leaks."""

    def __init__(self):
        self.active_tasks: Set[asyncio.Task] = weakref.WeakSet()
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("async_task_tracker")

    def register_task(self, task: asyncio.Task, name: str, context: Dict[str, Any] = None):
        """Register a task for tracking."""
        task_id = id(task)
        self.active_tasks.add(task)

        self.task_registry[task_id] = {
            'name': name,
            'context': context or {},
            'created_at': datetime.now(),
            'task': task
        }

        self.logger.debug(f"Registered task: {name} (ID: {task_id})")

    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active tasks."""
        active_info = {}

        for task_id, info in self.task_registry.items():
            task = info['task']
            if not task.done():
                active_info[task_id] = {
                    'name': info['name'],
                    'context': info['context'],
                    'created_at': info['created_at'],
                    'duration': datetime.now() - info['created_at'],
                    'state': 'running' if not task.cancelled() else 'cancelled'
                }

        return active_info

    def detect_long_running_tasks(self, threshold_seconds: int = 300) -> List[Dict]:
        """Detect tasks that have been running longer than threshold."""
        threshold = timedelta(seconds=threshold_seconds)
        long_running = []

        for task_id, info in self.get_active_tasks().items():
            if info['duration'] > threshold:
                long_running.append({
                    'task_id': task_id,
                    'name': info['name'],
                    'duration_seconds': info['duration'].total_seconds(),
                    'context': info['context']
                })

        return long_running

    async def debug_deadlock_detection(self):
        """Detect potential deadlocks in async tasks."""
        long_running = self.detect_long_running_tasks(threshold_seconds=60)

        if long_running:
            self.logger.warning(f"Detected {len(long_running)} potentially stuck tasks:")
            for task_info in long_running:
                self.logger.warning(
                    f"  - {task_info['name']}: {task_info['duration_seconds']:.1f}s"
                )
                self.logger.warning(f"    Context: {task_info['context']}")

        return long_running

# Usage example
task_tracker = AsyncTaskTracker()

async def track_async_operation(name: str, coro, context: Dict = None):
    """Wrapper to track async operations."""
    task = asyncio.create_task(coro)
    task_tracker.register_task(task, name, context)

    try:
        return await task
    except Exception as e:
        task_tracker.logger.error(f"Task {name} failed: {e}")
        raise
    finally:
        task_tracker.logger.debug(f"Task {name} completed")

# Use in your code
async def process_audio_files():
    files = await get_audio_files()

    tasks = []
    for i, file_path in enumerate(files):
        context = {'file_path': file_path, 'batch_index': i}
        task_coro = analyze_audio_file(file_path)

        task = track_async_operation(
            f"analyze_file_{i}",
            task_coro,
            context
        )
        tasks.append(task)

    # Monitor for deadlocks every 30 seconds
    async def monitor_deadlocks():
        while True:
            await asyncio.sleep(30)
            await task_tracker.debug_deadlock_detection()

    monitor_task = asyncio.create_task(monitor_deadlocks())

    try:
        results = await asyncio.gather(*tasks)
        return results
    finally:
        monitor_task.cancel()
```

## Database Debugging

### Connection Pool Debugging

```python
import sqlalchemy
from sqlalchemy import event, pool
from typing import Dict, List

class DatabaseConnectionDebugger:
    """Debug database connection issues and pool problems."""

    def __init__(self, engine):
        self.engine = engine
        self.connection_stats = {
            'created': 0,
            'closed': 0,
            'checked_out': 0,
            'checked_in': 0,
            'invalidated': 0,
            'active_connections': []
        }
        self.logger = logging.getLogger("db_connection_debug")
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for connection tracking."""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            self.connection_stats['created'] += 1
            connection_id = id(dbapi_connection)
            self.connection_stats['active_connections'].append({
                'id': connection_id,
                'created_at': datetime.now(),
                'thread_id': threading.current_thread().ident
            })
            self.logger.debug(f"Connection created: {connection_id}")

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_connection, connection_record):
            self.connection_stats['closed'] += 1
            connection_id = id(dbapi_connection)
            self.connection_stats['active_connections'] = [
                conn for conn in self.connection_stats['active_connections']
                if conn['id'] != connection_id
            ]
            self.logger.debug(f"Connection closed: {connection_id}")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            self.connection_stats['checked_out'] += 1
            self.logger.debug(f"Connection checked out: {id(dbapi_connection)}")

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            self.connection_stats['checked_in'] += 1
            self.logger.debug(f"Connection checked in: {id(dbapi_connection)}")

    def get_pool_status(self) -> Dict:
        """Get current connection pool status."""
        pool = self.engine.pool

        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'stats': self.connection_stats.copy(),
            'active_connection_count': len(self.connection_stats['active_connections'])
        }

    def detect_connection_leaks(self) -> List[Dict]:
        """Detect potential connection leaks."""
        now = datetime.now()
        long_lived_connections = []

        for conn in self.connection_stats['active_connections']:
            age = now - conn['created_at']
            if age.total_seconds() > 300:  # 5 minutes
                long_lived_connections.append({
                    'connection_id': conn['id'],
                    'age_seconds': age.total_seconds(),
                    'thread_id': conn['thread_id']
                })

        if long_lived_connections:
            self.logger.warning(f"Detected {len(long_lived_connections)} potentially leaked connections")
            for conn in long_lived_connections:
                self.logger.warning(
                    f"  - Connection {conn['connection_id']}: "
                    f"{conn['age_seconds']:.1f}s old (thread {conn['thread_id']})"
                )

        return long_lived_connections

# Usage
engine = create_async_engine("postgresql://...")
db_debugger = DatabaseConnectionDebugger(engine)

async def debug_database_operations():
    """Debug database operations with connection monitoring."""

    # Perform some database operations
    async with AsyncSession(engine) as session:
        # Your database operations here
        result = await session.execute("SELECT COUNT(*) FROM tracks")
        track_count = result.scalar()

    # Check pool status
    pool_status = db_debugger.get_pool_status()
    logger.info(f"Pool status: {pool_status}")

    # Check for leaks
    leaks = db_debugger.detect_connection_leaks()
    if leaks:
        logger.warning("Connection leaks detected!")
```

## Message Queue Debugging

### RabbitMQ Connection Issues

```python
import pika
import json
from typing import Dict, Any, Optional

class RabbitMQDebugger:
    """Debug RabbitMQ connection and message processing issues."""

    def __init__(self, connection_params: pika.ConnectionParameters):
        self.connection_params = connection_params
        self.logger = logging.getLogger("rabbitmq_debug")
        self.message_stats = {
            'published': 0,
            'consumed': 0,
            'failed': 0,
            'redelivered': 0
        }

    async def test_connection(self) -> Dict[str, Any]:
        """Test RabbitMQ connection and return diagnostic info."""
        debug_info = {
            'connection': {'success': False, 'error': None},
            'channel': {'success': False, 'error': None},
            'queue_declare': {'success': False, 'error': None},
            'server_info': {}
        }

        try:
            # Test connection
            connection = pika.BlockingConnection(self.connection_params)
            debug_info['connection']['success'] = True
            self.logger.info("RabbitMQ connection: SUCCESS")

            try:
                # Test channel creation
                channel = connection.channel()
                debug_info['channel']['success'] = True
                self.logger.info("RabbitMQ channel creation: SUCCESS")

                # Get server properties
                debug_info['server_info'] = {
                    'server_properties': connection._impl.server_properties,
                    'is_open': connection.is_open
                }

                try:
                    # Test queue declaration
                    test_queue = 'debug_test_queue'
                    channel.queue_declare(queue=test_queue, durable=True)
                    debug_info['queue_declare']['success'] = True
                    self.logger.info("Queue declaration: SUCCESS")

                    # Clean up test queue
                    channel.queue_delete(queue=test_queue)

                except Exception as e:
                    debug_info['queue_declare']['error'] = str(e)
                    self.logger.error(f"Queue declaration: FAILED - {e}")

                channel.close()

            except Exception as e:
                debug_info['channel']['error'] = str(e)
                self.logger.error(f"Channel creation: FAILED - {e}")

            connection.close()

        except Exception as e:
            debug_info['connection']['error'] = str(e)
            self.logger.error(f"RabbitMQ connection: FAILED - {e}")

        return debug_info

    def debug_message_publishing(self, exchange: str, routing_key: str,
                                message: Dict[str, Any]) -> Dict[str, Any]:
        """Debug message publishing with detailed logging."""
        debug_info = {
            'exchange': exchange,
            'routing_key': routing_key,
            'message_size': len(json.dumps(message).encode('utf-8')),
            'published': False,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

        try:
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()

            # Publish message with confirmation
            channel.confirm_delivery()

            result = channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    timestamp=int(time.time()),
                    message_id=str(uuid.uuid4())
                )
            )

            if result:
                debug_info['published'] = True
                self.message_stats['published'] += 1
                self.logger.info(
                    f"Message published: exchange={exchange}, "
                    f"routing_key={routing_key}, size={debug_info['message_size']}"
                )
            else:
                debug_info['error'] = "Message not confirmed by broker"
                self.message_stats['failed'] += 1
                self.logger.error("Message publishing not confirmed")

            connection.close()

        except Exception as e:
            debug_info['error'] = str(e)
            self.message_stats['failed'] += 1
            self.logger.error(f"Message publishing failed: {e}")

        return debug_info

    def debug_message_consumption(self, queue: str, timeout: int = 10) -> Dict[str, Any]:
        """Debug message consumption with timeout."""
        debug_info = {
            'queue': queue,
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'timeout': timeout,
            'errors': []
        }

        try:
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()

            # Get queue information
            method_frame = channel.queue_declare(queue=queue, passive=True)
            message_count = method_frame.method.message_count
            self.logger.info(f"Queue {queue} has {message_count} messages")

            def callback(ch, method, properties, body):
                debug_info['messages_received'] += 1
                try:
                    # Try to parse message
                    message = json.loads(body.decode('utf-8'))
                    self.logger.debug(f"Received message: {message}")

                    # Simulate processing
                    # Your message processing logic would go here

                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    debug_info['messages_processed'] += 1
                    self.message_stats['consumed'] += 1

                except Exception as e:
                    debug_info['messages_failed'] += 1
                    debug_info['errors'].append(str(e))
                    self.logger.error(f"Message processing failed: {e}")

                    # Reject and requeue message
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                    self.message_stats['redelivered'] += 1

            # Set up consumer with timeout
            channel.basic_consume(queue=queue, on_message_callback=callback)

            # Start consuming with timeout
            connection.add_timeout(timeout, lambda: connection.close())
            channel.start_consuming()

        except Exception as e:
            debug_info['errors'].append(str(e))
            self.logger.error(f"Message consumption failed: {e}")

        return debug_info

# Usage
rabbitmq_params = pika.ConnectionParameters(
    host='localhost',
    port=5672,
    virtual_host='/',
    credentials=pika.PlainCredentials('guest', 'guest')
)

rabbitmq_debugger = RabbitMQDebugger(rabbitmq_params)

async def debug_messaging_system():
    """Debug the entire messaging system."""

    # Test connection
    connection_test = await rabbitmq_debugger.test_connection()
    logger.info(f"Connection test results: {connection_test}")

    # Test message publishing
    test_message = {
        'type': 'analysis_request',
        'file_path': '/test/audio.mp3',
        'timestamp': datetime.now().isoformat()
    }

    publish_result = rabbitmq_debugger.debug_message_publishing(
        exchange='analysis_exchange',
        routing_key='audio.analysis',
        message=test_message
    )
    logger.info(f"Publishing test: {publish_result}")

    # Test message consumption
    consume_result = rabbitmq_debugger.debug_message_consumption(
        queue='analysis_queue',
        timeout=5
    )
    logger.info(f"Consumption test: {consume_result}")
```

## Docker Container Debugging

### Container Health Debugging

```python
import docker
import subprocess
from typing import Dict, List, Any

class DockerDebugger:
    """Debug Docker container issues."""

    def __init__(self):
        self.client = docker.from_env()
        self.logger = logging.getLogger("docker_debug")

    def debug_container_health(self, container_name: str) -> Dict[str, Any]:
        """Debug container health and configuration."""
        debug_info = {
            'container_name': container_name,
            'exists': False,
            'running': False,
            'health_status': None,
            'resource_usage': {},
            'network_info': {},
            'logs': [],
            'errors': []
        }

        try:
            # Get container
            container = self.client.containers.get(container_name)
            debug_info['exists'] = True
            debug_info['running'] = container.status == 'running'

            # Get container stats
            if debug_info['running']:
                stats = container.stats(stream=False)
                debug_info['resource_usage'] = {
                    'cpu_percent': self._calculate_cpu_percent(stats),
                    'memory_usage_mb': stats['memory']['usage'] / 1024 / 1024,
                    'memory_limit_mb': stats['memory']['limit'] / 1024 / 1024,
                    'network_rx_bytes': stats['networks']['eth0']['rx_bytes'],
                    'network_tx_bytes': stats['networks']['eth0']['tx_bytes']
                }

            # Get network information
            debug_info['network_info'] = {
                'ports': container.ports,
                'network_settings': container.attrs['NetworkSettings']['Networks']
            }

            # Get recent logs
            logs = container.logs(tail=50).decode('utf-8')
            debug_info['logs'] = logs.split('\n')[-10:]  # Last 10 lines

            # Check health status
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                debug_info['health_status'] = health.get('Status')
                debug_info['health_checks'] = health.get('Log', [])[-3:]  # Last 3 checks

            self.logger.info(f"Container {container_name}: running={debug_info['running']}")

        except docker.errors.NotFound:
            debug_info['errors'].append(f"Container {container_name} not found")
            self.logger.error(f"Container {container_name} not found")
        except Exception as e:
            debug_info['errors'].append(str(e))
            self.logger.error(f"Container debug failed: {e}")

        return debug_info

    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] -
                        stats['precpu_stats']['cpu_usage']['total_usage'])
            system_delta = (stats['cpu_stats']['system_cpu_usage'] -
                           stats['precpu_stats']['system_cpu_usage'])

            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass

        return 0.0

    def debug_service_connectivity(self, services: List[str]) -> Dict[str, Any]:
        """Debug connectivity between services."""
        debug_info = {
            'services': services,
            'connectivity': {},
            'dns_resolution': {},
            'port_accessibility': {}
        }

        for service in services:
            debug_info['connectivity'][service] = {}
            debug_info['dns_resolution'][service] = {}
            debug_info['port_accessibility'][service] = {}

            try:
                container = self.client.containers.get(service)

                # Test DNS resolution from other services
                for target_service in services:
                    if target_service != service:
                        try:
                            # Execute nslookup inside container
                            exec_result = container.exec_run(f"nslookup {target_service}")
                            debug_info['dns_resolution'][service][target_service] = {
                                'success': exec_result.exit_code == 0,
                                'output': exec_result.output.decode('utf-8')
                            }
                        except Exception as e:
                            debug_info['dns_resolution'][service][target_service] = {
                                'success': False,
                                'error': str(e)
                            }

                # Test port connectivity
                for target_service in services:
                    if target_service != service:
                        common_ports = [80, 443, 5432, 6379, 5672]  # HTTP, HTTPS, Postgres, Redis, RabbitMQ

                        for port in common_ports:
                            try:
                                exec_result = container.exec_run(
                                    f"timeout 3 bash -c 'cat < /dev/null > /dev/tcp/{target_service}/{port}'"
                                )

                                if target_service not in debug_info['port_accessibility'][service]:
                                    debug_info['port_accessibility'][service][target_service] = {}

                                debug_info['port_accessibility'][service][target_service][port] = {
                                    'accessible': exec_result.exit_code == 0
                                }
                            except Exception as e:
                                self.logger.debug(f"Port test failed: {e}")

            except Exception as e:
                debug_info['connectivity'][service] = {'error': str(e)}

        return debug_info

# Usage
docker_debugger = DockerDebugger()

def debug_docker_services():
    """Debug Docker services in the stack."""
    services = ['tracktion_analysis', 'tracktion_tracklist', 'tracktion_postgres', 'tracktion_redis']

    # Check individual container health
    for service in services:
        health_info = docker_debugger.debug_container_health(service)
        logger.info(f"Service {service} health: {health_info}")

        if health_info['errors']:
            logger.error(f"Service {service} errors: {health_info['errors']}")

    # Check inter-service connectivity
    connectivity_info = docker_debugger.debug_service_connectivity(services)
    logger.info(f"Service connectivity: {connectivity_info}")
```

## Logging Best Practices

### Structured Logging Setup

```python
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """Structured logging for better debugging and monitoring."""

    def __init__(self, service_name: str, log_level: str = "INFO"):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        # Add structured handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.StructuredFormatter())
        self.logger.addHandler(handler)

    class StructuredFormatter(logging.Formatter):
        """Custom formatter for structured JSON logs."""

        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)

            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_entry)

    def log_with_context(self, level: str, message: str, **context):
        """Log message with additional context."""
        extra_fields = context
        extra_fields['service'] = self.service_name

        # Add request context if available
        if hasattr(self, '_request_context'):
            extra_fields['request_id'] = self._request_context.get('request_id')
            extra_fields['user_id'] = self._request_context.get('user_id')

        getattr(self.logger, level.lower())(
            message,
            extra={'extra_fields': extra_fields}
        )

    def log_performance(self, operation: str, duration: float, **metadata):
        """Log performance metrics."""
        self.log_with_context(
            'info',
            f"Performance metric: {operation}",
            operation=operation,
            duration_seconds=duration,
            **metadata
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            **(context or {})
        }

        self.log_with_context(
            'error',
            f"Error occurred: {str(error)}",
            **error_context
        )

# Usage
structured_logger = StructuredLogger('tracktion_analysis')

# Example logging in your code
async def analyze_audio_file_with_logging(file_path: str):
    """Analyze audio file with comprehensive logging."""
    start_time = time.time()

    structured_logger.log_with_context(
        'info',
        'Starting audio analysis',
        file_path=file_path,
        operation='audio_analysis'
    )

    try:
        # Your analysis logic here
        result = await perform_analysis(file_path)

        duration = time.time() - start_time
        structured_logger.log_performance(
            'audio_analysis',
            duration,
            file_path=file_path,
            result_keys=list(result.keys())
        )

        return result

    except Exception as e:
        structured_logger.log_error(
            e,
            context={
                'file_path': file_path,
                'operation': 'audio_analysis',
                'duration': time.time() - start_time
            }
        )
        raise
```

## Debugging Tools and Utilities

### Debug Dashboard

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import json

class DebugDashboard:
    """Simple web dashboard for debugging information."""

    def __init__(self, app: FastAPI):
        self.app = app
        self.debug_data = {
            'requests': [],
            'errors': [],
            'performance': [],
            'system_info': {}
        }
        self.setup_routes()

    def setup_routes(self):
        """Setup debug dashboard routes."""

        @self.app.get("/debug/dashboard", response_class=HTMLResponse)
        async def debug_dashboard():
            """Render debug dashboard."""
            return self.render_dashboard()

        @self.app.get("/debug/data")
        async def debug_data():
            """Get debug data as JSON."""
            return self.debug_data

        @self.app.post("/debug/clear")
        async def clear_debug_data():
            """Clear debug data."""
            self.debug_data = {
                'requests': [],
                'errors': [],
                'performance': [],
                'system_info': {}
            }
            return {"message": "Debug data cleared"}

    def render_dashboard(self) -> str:
        """Render HTML dashboard."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tracktion Debug Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .error {{ background-color: #ffe6e6; }}
                .warning {{ background-color: #fff3cd; }}
                .info {{ background-color: #e6f3ff; }}
                pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>Tracktion Debug Dashboard</h1>

            <div class="section info">
                <h2>System Information</h2>
                <pre>{json.dumps(self.debug_data['system_info'], indent=2)}</pre>
            </div>

            <div class="section">
                <h2>Recent Requests ({len(self.debug_data['requests'])})</h2>
                <pre>{json.dumps(self.debug_data['requests'][-10:], indent=2, default=str)}</pre>
            </div>

            <div class="section error">
                <h2>Recent Errors ({len(self.debug_data['errors'])})</h2>
                <pre>{json.dumps(self.debug_data['errors'][-10:], indent=2, default=str)}</pre>
            </div>

            <div class="section warning">
                <h2>Performance Metrics</h2>
                <pre>{json.dumps(self.debug_data['performance'][-10:], indent=2, default=str)}</pre>
            </div>

            <div class="section">
                <button onclick="location.reload()">Refresh</button>
                <button onclick="clearData()">Clear Data</button>
            </div>

            <script>
                function clearData() {{
                    fetch('/debug/clear', {{method: 'POST'}})
                        .then(() => location.reload());
                }}

                // Auto-refresh every 30 seconds
                setTimeout(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        """

    def log_request(self, request: Request, duration: float):
        """Log request information."""
        self.debug_data['requests'].append({
            'timestamp': datetime.now().isoformat(),
            'method': request.method,
            'url': str(request.url),
            'duration': duration,
            'client_ip': request.client.host if request.client else None
        })

        # Keep only last 100 requests
        self.debug_data['requests'] = self.debug_data['requests'][-100:]

    def log_error(self, error: Exception, context: Dict = None):
        """Log error information."""
        self.debug_data['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        })

        # Keep only last 50 errors
        self.debug_data['errors'] = self.debug_data['errors'][-50:]

# Usage in FastAPI app
app = FastAPI()
debug_dashboard = DebugDashboard(app)

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """Middleware to track requests and performance."""
    start_time = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        debug_dashboard.log_request(request, duration)
        return response
    except Exception as e:
        duration = time.time() - start_time
        debug_dashboard.log_error(e, {'request_url': str(request.url), 'duration': duration})
        raise
```

This comprehensive debugging guide provides practical tools and techniques for identifying and resolving issues in the Tracktion system. Use these examples as starting points and adapt them to your specific debugging needs.
