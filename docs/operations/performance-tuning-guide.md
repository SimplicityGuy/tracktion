# Performance Tuning Guide

## Overview

This guide provides comprehensive performance optimization strategies for the Tracktion system. It covers service-specific tuning, infrastructure optimization, and monitoring techniques to ensure optimal system performance under various load conditions.

## Performance Baseline Metrics

### Target Performance Standards

#### Audio Processing Performance
- **BPM Detection**: < 3 seconds for 4-minute track
- **Key Detection**: < 2 seconds for 4-minute track
- **Mood Analysis**: < 5 seconds for 4-minute track
- **Complete Analysis**: < 8 seconds for 4-minute track
- **Batch Processing**: > 100 tracks/hour on single instance

#### API Response Times
- **Health Check**: < 100ms
- **Track Search**: < 200ms
- **Playlist Operations**: < 300ms
- **User Authentication**: < 150ms
- **File Upload**: < 5 seconds for 50MB file

#### System Resource Utilization
- **CPU Usage**: < 70% average, < 90% peak
- **Memory Usage**: < 80% of allocated memory
- **Disk I/O**: < 80% utilization
- **Network Latency**: < 10ms between services

## Service-Specific Performance Tuning

### Analysis Service Optimization

#### CPU and Memory Optimization

**Audio Processing Configuration:**
```python
# services/analysis_service/src/config.py
@dataclass
class AnalysisConfig:
    # Optimize audio processing
    audio_sample_rate: int = 44100  # Standard rate for good quality/performance balance
    chunk_size: int = 4096          # Optimize for CPU cache
    max_file_size: int = 100 * 1024 * 1024  # 100MB limit

    # Parallel processing
    max_workers: int = min(4, cpu_count())  # Limit to prevent oversubscription
    enable_gpu: bool = False  # Enable only if GPU available

    # Memory management
    audio_cache_size: int = 50  # Cache last 50 processed files
    temp_file_cleanup: bool = True
    memory_limit_mb: int = 2048
```

**Optimize BPM Detection:**
```python
# services/analysis_service/src/bpm_detector.py
class BPMDetector:
    def __init__(self, config: AnalysisConfig):
        # Pre-initialize algorithms to avoid repeated loading
        self.rhythm_extractor = es.RhythmExtractor2013(
            method="multifeature",
            frameSize=1024,      # Smaller frame for faster processing
            hopSize=512,         # Good balance of accuracy/speed
            sampleRate=config.audio_sample_rate
        )

        self.percival_estimator = es.PercivalBpmEstimator(
            frameSize=1024,
            hopSize=512,
            sampleRate=config.audio_sample_rate
        )

        # Cache for repeated detections
        self.result_cache = {}

    def detect_bpm_with_cache(self, file_path: str) -> Dict[str, Any]:
        # Use file hash as cache key
        file_hash = self._get_file_hash(file_path)

        if file_hash in self.result_cache:
            logger.debug("BPM cache hit", file_path=file_path)
            return self.result_cache[file_hash]

        result = self.detect_bmp(file_path)

        # Cache successful results only
        if result.get("confidence", 0) > 0.7:
            self.result_cache[file_hash] = result

        return result
```

**Memory Usage Monitoring:**
```python
# services/analysis_service/src/memory_monitor.py
import psutil
import gc
from typing import Optional

class MemoryMonitor:
    def __init__(self, memory_limit_mb: int = 2048):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.process = psutil.Process()

    def check_memory_usage(self) -> Optional[str]:
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss

        if current_memory > self.memory_limit:
            # Force garbage collection
            gc.collect()

            # Check again after GC
            memory_info = self.process.memory_info()
            if memory_info.rss > self.memory_limit:
                return f"Memory usage critical: {memory_info.rss / 1024 / 1024:.1f}MB"

        return None

    def cleanup_cache_if_needed(self, cache_dict: dict):
        if self.check_memory_usage():
            # Clear half of cache, keeping most recent
            cache_size = len(cache_dict)
            if cache_size > 10:
                # Remove oldest entries
                keys_to_remove = list(cache_dict.keys())[:cache_size // 2]
                for key in keys_to_remove:
                    cache_dict.pop(key, None)

                logger.info("Cache cleared due to memory pressure",
                          removed_entries=len(keys_to_remove))
```

#### Parallel Processing Setup

**Async Audio Processing:**
```python
# services/analysis_service/src/async_processor.py
import asyncio
import concurrent.futures
from typing import List, Dict, Any

class AsyncAudioProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple audio files concurrently."""

        async def process_single(file_path: str) -> Dict[str, Any]:
            async with self.semaphore:  # Limit concurrent processing
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self._process_audio_sync,
                    file_path
                )

        # Process files concurrently with limited parallelism
        tasks = [process_single(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Audio processing failed",
                           file_path=file_paths[i],
                           error=str(result))
            else:
                successful_results.append(result)

        return successful_results

    def _process_audio_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronous audio processing for thread executor."""
        # Your existing synchronous processing logic
        pass
```

### Tracklist Service Optimization

#### Database Query Optimization

**Connection Pool Configuration:**
```python
# shared/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def get_optimized_engine(database_url: str):
    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=20,           # Base connections
        max_overflow=30,        # Additional connections under load
        pool_timeout=30,        # Max wait time for connection
        pool_recycle=3600,      # Recycle connections hourly
        pool_pre_ping=True,     # Validate connections
        echo=False,             # Disable SQL logging in production

        # Connection-level optimizations
        connect_args={
            "connect_timeout": 10,
            "command_timeout": 30,
            "server_settings": {
                "application_name": "tracktion_service",
                "jit": "off",  # Disable JIT for consistent performance
            }
        }
    )
```

**Optimized Track Matching Queries:**
```python
# services/tracklist_service/src/services/optimized_matching.py
from sqlalchemy import text, func
from sqlalchemy.orm import Session

class OptimizedMatchingService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def fuzzy_search_tracks(self, title: str, artist: str, limit: int = 10) -> List[Dict]:
        """Optimized fuzzy search using PostgreSQL trigram indexes."""

        query = text("""
        SELECT
            t.id,
            t.title,
            t.artist,
            t.album,
            -- Calculate combined similarity score
            GREATEST(
                similarity(t.title, :title),
                similarity(t.artist, :artist)
            ) as similarity_score
        FROM tracks t
        WHERE
            -- Use trigram indexes for fast filtering
            (t.title % :title OR t.artist % :artist)
            -- Additional filters to reduce result set
            AND similarity(t.title, :title) > 0.3
        ORDER BY similarity_score DESC
        LIMIT :limit
        """)

        results = self.db.execute(query, {
            "title": title,
            "artist": artist,
            "limit": limit
        }).fetchall()

        return [dict(row) for row in results]

    def batch_match_tracks(self, track_requests: List[Dict]) -> List[Dict]:
        """Batch multiple track matching requests for efficiency."""

        # Use VALUES clause for bulk matching
        values_clause = ",".join([
            f"('{req['title']}', '{req['artist']}', {req['id']})"
            for req in track_requests
        ])

        query = text(f"""
        WITH search_terms AS (
            VALUES {values_clause}
        ) AS st(search_title, search_artist, request_id)

        SELECT DISTINCT ON (st.request_id)
            st.request_id,
            t.id as track_id,
            t.title,
            t.artist,
            similarity(t.title, st.search_title) +
            similarity(t.artist, st.search_artist) as total_similarity
        FROM search_terms st
        CROSS JOIN tracks t
        WHERE
            t.title % st.search_title
            OR t.artist % st.search_artist
        ORDER BY st.request_id, total_similarity DESC
        """)

        results = self.db.execute(query).fetchall()
        return [dict(row) for row in results]
```

**Database Index Optimization:**
```sql
-- Essential indexes for track matching performance
CREATE INDEX CONCURRENTLY idx_tracks_title_trgm ON tracks USING gin(title gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_tracks_artist_trgm ON tracks USING gin(artist gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_tracks_album_trgm ON tracks USING gin(album gin_trgm_ops);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_tracks_artist_title ON tracks(artist_id, title);
CREATE INDEX CONCURRENTLY idx_tracks_created_at ON tracks(created_at DESC);

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY idx_tracks_active ON tracks(id) WHERE active = true;

-- Index for playlist operations
CREATE INDEX CONCURRENTLY idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id, position);
```

#### Caching Strategy

**Redis Caching Implementation:**
```python
# services/tracklist_service/src/cache/track_cache.py
import json
import redis
from typing import Optional, Dict, List, Any
from datetime import timedelta

class TrackMatchCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = timedelta(hours=6)  # Cache matches for 6 hours

    def get_cached_matches(self, search_key: str) -> Optional[List[Dict]]:
        """Get cached track matches."""
        cache_key = f"track_matches:{search_key}"
        cached_data = self.redis.get(cache_key)

        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                self.redis.delete(cache_key)

        return None

    def cache_matches(self, search_key: str, matches: List[Dict], ttl: Optional[timedelta] = None):
        """Cache track matches with TTL."""
        cache_key = f"track_matches:{search_key}"
        ttl = ttl or self.default_ttl

        self.redis.setex(
            cache_key,
            int(ttl.total_seconds()),
            json.dumps(matches, default=str)
        )

    def generate_search_key(self, title: str, artist: str) -> str:
        """Generate consistent cache key for search terms."""
        # Normalize search terms for consistent caching
        normalized_title = title.lower().strip()
        normalized_artist = artist.lower().strip()
        return f"{normalized_title}:{normalized_artist}"

    def warm_cache_batch(self, popular_searches: List[tuple]):
        """Pre-warm cache with popular search terms."""
        from .matching_service import MatchingService

        matcher = MatchingService()

        for title, artist in popular_searches:
            search_key = self.generate_search_key(title, artist)
            if not self.get_cached_matches(search_key):
                matches = matcher.find_matches(title, artist)
                self.cache_matches(search_key, matches, ttl=timedelta(days=1))
```

### File Watcher Service Optimization

#### Efficient File Monitoring

**Optimized File Watcher Configuration:**
```python
# services/file_watcher/src/optimized_watcher.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from collections import defaultdict
from threading import Lock, Timer

class OptimizedFileHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds: float = 2.0):
        self.debounce_seconds = debounce_seconds
        self.pending_events = defaultdict(dict)
        self.lock = Lock()
        self.timers = {}

    def on_any_event(self, event):
        if event.is_directory:
            return

        # Filter file types early
        if not self._is_audio_file(event.src_path):
            return

        with self.lock:
            # Cancel existing timer for this file
            if event.src_path in self.timers:
                self.timers[event.src_path].cancel()

            # Store latest event
            self.pending_events[event.src_path] = {
                'event_type': event.event_type,
                'src_path': event.src_path,
                'timestamp': time.time()
            }

            # Set new timer
            timer = Timer(self.debounce_seconds, self._process_debounced_event, [event.src_path])
            self.timers[event.src_path] = timer
            timer.start()

    def _process_debounced_event(self, file_path: str):
        """Process event after debounce period."""
        with self.lock:
            if file_path in self.pending_events:
                event_data = self.pending_events.pop(file_path)
                self.timers.pop(file_path, None)

                # Send to processing queue
                self._queue_file_processing(event_data)

    def _is_audio_file(self, file_path: str) -> bool:
        """Quick file extension check."""
        audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg'}
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)
```

**Batch Processing Queue:**
```python
# services/file_watcher/src/batch_processor.py
import asyncio
from typing import List, Dict
import aio_pika
from collections import deque
import time

class BatchProcessor:
    def __init__(self, rabbitmq_connection, batch_size: int = 10, batch_timeout: float = 30.0):
        self.connection = rabbitmq_connection
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_files = deque()
        self.last_batch_time = time.time()

    async def add_file_for_processing(self, file_info: Dict):
        """Add file to batch processing queue."""
        self.pending_files.append(file_info)

        # Check if batch is ready
        if (len(self.pending_files) >= self.batch_size or
            time.time() - self.last_batch_time > self.batch_timeout):
            await self._process_batch()

    async def _process_batch(self):
        """Process accumulated files as a batch."""
        if not self.pending_files:
            return

        # Extract current batch
        current_batch = []
        while self.pending_files and len(current_batch) < self.batch_size:
            current_batch.append(self.pending_files.popleft())

        if current_batch:
            await self._send_batch_message(current_batch)
            self.last_batch_time = time.time()

    async def _send_batch_message(self, batch: List[Dict]):
        """Send batch message to analysis service."""
        channel = await self.connection.channel()

        message_body = {
            'batch_id': f"batch_{int(time.time())}",
            'files': batch,
            'batch_size': len(batch),
            'priority': 'normal'
        }

        await channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(message_body).encode(),
                priority=5 if len(batch) > 5 else 1  # Higher priority for larger batches
            ),
            routing_key='audio.analysis.batch'
        )
```

## Infrastructure Performance Tuning

### Docker and Container Optimization

#### Container Resource Limits

**Optimized Docker Compose Configuration:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  analysis_service:
    image: tracktion/analysis_service
    deploy:
      resources:
        limits:
          cpus: '2.0'      # Limit to 2 CPU cores
          memory: 4G       # 4GB memory limit
        reservations:
          cpus: '1.0'      # Reserve 1 CPU core
          memory: 2G       # Reserve 2GB memory
    environment:
      - MALLOC_ARENA_MAX=4  # Reduce memory fragmentation
    ulimits:
      nofile: 65536        # Increase file descriptor limit
    sysctls:
      - net.core.somaxconn=1024  # Increase connection queue

  tracklist_service:
    image: tracktion/tracklist_service
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  postgres:
    image: postgres:15
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      # PostgreSQL performance tuning
      - POSTGRES_INITDB_ARGS="--data-checksums"
    command: |
      postgres
      -c max_connections=200
      -c shared_buffers=512MB
      -c effective_cache_size=1536MB
      -c maintenance_work_mem=128MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M
    command: >
      redis-server
      --maxmemory 800mb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000

  rabbitmq:
    image: rabbitmq:3-management-alpine
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M
    environment:
      - RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.6
      - RABBITMQ_DISK_FREE_LIMIT=1GB
```

#### Docker Performance Optimization

**Container Image Optimization:**
```dockerfile
# Dockerfile.analysis_service - Optimized for performance
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r tracktion && useradd -r -g tracktion tracktion

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --compile -r requirements.txt

# Copy application code
COPY . .
RUN chown -R tracktion:tracktion /app

# Switch to non-root user
USER tracktion

# Optimize Python performance
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set memory allocator (better for multithreaded apps)
ENV MALLOC_ARENA_MAX=4

EXPOSE 8001

CMD ["python", "src/main.py"]
```

### Database Performance Tuning

#### PostgreSQL Optimization

**postgresql.conf Optimization:**
```ini
# postgresql.conf - Production performance settings

# Memory Configuration
shared_buffers = 512MB          # 25% of system RAM
effective_cache_size = 1536MB   # 75% of system RAM
work_mem = 4MB                  # Per operation memory
maintenance_work_mem = 128MB    # Maintenance operations
temp_buffers = 8MB              # Temporary tables

# Checkpoint and WAL Settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms
commit_delay = 0
commit_siblings = 5

# Query Planner Settings
random_page_cost = 1.1          # SSD storage
seq_page_cost = 1.0
cpu_tuple_cost = 0.01
cpu_index_tuple_cost = 0.005
cpu_operator_cost = 0.0025

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3

# Logging for Performance Analysis
log_min_duration_statement = 1000  # Log queries > 1 second
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Extensions
shared_preload_libraries = 'pg_stat_statements'
```

**Performance Monitoring Queries:**
```sql
-- Monitor slow queries
SELECT
    query,
    calls,
    total_time / calls as avg_time_ms,
    mean_time,
    stddev_time,
    rows / calls as avg_rows
FROM pg_stat_statements
WHERE calls > 50
ORDER BY avg_time_ms DESC
LIMIT 20;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_tup_read / NULLIF(idx_tup_fetch, 0) as read_fetch_ratio
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;

-- Monitor database size and bloat
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    CASE
        WHEN n_live_tup > 0
        THEN (n_dead_tup::float / n_live_tup::float) * 100
        ELSE 0
    END as dead_row_percent
FROM pg_stat_user_tables
ORDER BY dead_row_percent DESC;
```

#### Redis Optimization

**Redis Performance Configuration:**
```conf
# redis.conf - Performance optimized settings

# Memory Management
maxmemory 1gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence (adjust for performance vs durability trade-off)
save 900 1      # Save after 900 sec if at least 1 key changed
save 300 10     # Save after 300 sec if at least 10 keys changed
save 60 10000   # Save after 60 sec if at least 10000 keys changed

# Network and Connection
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Performance Optimizations
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Disable expensive commands in production
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG ""
```

## Application-Level Performance Optimization

### Async Programming Best Practices

**Efficient Async Service Implementation:**
```python
# services/shared/async_base_service.py
import asyncio
import aiohttp
from typing import List, Dict, Any
import time

class AsyncBaseService:
    def __init__(self, max_concurrent_requests: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=100,                    # Total connection pool size
            limit_per_host=30,           # Per host connection limit
            ttl_dns_cache=300,           # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=300,       # Keep alive timeout
            enable_cleanup_closed=True   # Cleanup closed connections
        )

        timeout = aiohttp.ClientTimeout(
            total=30,      # Total request timeout
            connect=5,     # Connection timeout
            sock_read=10   # Socket read timeout
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def batch_request(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Make multiple HTTP requests concurrently with rate limiting."""

        async def make_request(url: str) -> Dict[str, Any]:
            async with self.semaphore:  # Rate limiting
                try:
                    async with self.session.get(url) as response:
                        return {
                            'url': url,
                            'status': response.status,
                            'data': await response.json() if response.content_type == 'application/json' else await response.text(),
                            'response_time': response.headers.get('X-Response-Time')
                        }
                except asyncio.TimeoutError:
                    return {'url': url, 'error': 'timeout'}
                except Exception as e:
                    return {'url': url, 'error': str(e)}

        # Execute requests concurrently
        tasks = [make_request(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if not isinstance(r, Exception)]
```

### Memory Management and Caching

**Intelligent Caching Strategy:**
```python
# services/shared/intelligent_cache.py
import time
import threading
from typing import Any, Optional, Callable
from functools import wraps
import weakref

class IntelligentCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.creation_times[key] > self.ttl_seconds:
                    self._remove_key(key)
                    return None

                # Update access time for LRU
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            # Remove old entry if exists
            if key in self.cache:
                self._remove_key(key)

            # Check size limit
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add new entry
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times, key=self.access_times.get)
        self._remove_key(lru_key)

    def _remove_key(self, key: str):
        """Remove key from all tracking dictionaries."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)

    def cleanup_expired(self):
        """Clean up expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, creation_time in self.creation_times.items()
            if current_time - creation_time > self.ttl_seconds
        ]

        for key in expired_keys:
            self._remove_key(key)

# Cache decorator with automatic cleanup
def cached_result(cache_instance: IntelligentCache, key_func: Optional[Callable] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Try cache first
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result and cache
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)

            return result
        return wrapper
    return decorator
```

## Monitoring and Benchmarking

### Performance Metrics Collection

**Application Performance Monitoring:**
```python
# services/shared/performance_monitor.py
import time
import psutil
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

@dataclass
class PerformanceMetrics:
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()

    @asynccontextmanager
    async def measure_async(self, operation: str, **metadata):
        """Async context manager for performance measurement."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()

        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu = self.process.cpu_percent()

            metric = PerformanceMetrics(
                operation=operation,
                duration_ms=(end_time - start_time) * 1000,
                memory_usage_mb=end_memory - start_memory,
                cpu_percent=end_cpu - start_cpu,
                timestamp=start_time,
                success=success,
                metadata=metadata
            )

            self.metrics.append(metric)

            # Log slow operations
            if metric.duration_ms > 5000:  # 5 seconds
                logger.warning("Slow operation detected",
                             operation=operation,
                             duration_ms=metric.duration_ms,
                             **metadata)

    def get_performance_summary(self, operation: str = None) -> Dict:
        """Get performance summary for operations."""
        relevant_metrics = [
            m for m in self.metrics
            if operation is None or m.operation == operation
        ]

        if not relevant_metrics:
            return {}

        durations = [m.duration_ms for m in relevant_metrics]
        successful = [m for m in relevant_metrics if m.success]

        return {
            'total_operations': len(relevant_metrics),
            'successful_operations': len(successful),
            'success_rate': len(successful) / len(relevant_metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'total_memory_mb': sum(m.memory_usage_mb for m in relevant_metrics)
        }

# Usage example
monitor = PerformanceMonitor()

async def process_audio_with_monitoring(file_path: str):
    async with monitor.measure_async("audio_processing", file_path=file_path):
        # Your audio processing code
        result = await process_audio(file_path)
        return result
```

### Benchmarking Tools

**Performance Benchmark Suite:**
```python
# scripts/benchmark_suite.py
import asyncio
import time
import statistics
from typing import List, Dict, Callable, Any
import matplotlib.pyplot as plt

class BenchmarkSuite:
    def __init__(self):
        self.results = {}

    async def benchmark_function(
        self,
        func: Callable,
        test_cases: List[Any],
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict:
        """Benchmark a function with multiple test cases."""

        # Warmup
        for _ in range(warmup_iterations):
            for test_case in test_cases:
                if asyncio.iscoroutinefunction(func):
                    await func(test_case)
                else:
                    func(test_case)

        results = []
        for test_case in test_cases:
            test_results = []

            for _ in range(iterations):
                start_time = time.perf_counter()

                if asyncio.iscoroutinefunction(func):
                    await func(test_case)
                else:
                    func(test_case)

                end_time = time.perf_counter()
                test_results.append((end_time - start_time) * 1000)  # ms

            results.append({
                'test_case': str(test_case),
                'avg_time_ms': statistics.mean(test_results),
                'median_time_ms': statistics.median(test_results),
                'min_time_ms': min(test_results),
                'max_time_ms': max(test_results),
                'std_dev_ms': statistics.stdev(test_results) if len(test_results) > 1 else 0,
                'raw_times': test_results
            })

        return {
            'function': func.__name__,
            'iterations': iterations,
            'test_results': results
        }

    def generate_performance_report(self, benchmark_results: Dict):
        """Generate performance analysis report."""

        print(f"\n=== Performance Benchmark: {benchmark_results['function']} ===")
        print(f"Iterations per test: {benchmark_results['iterations']}")
        print("-" * 70)

        for result in benchmark_results['test_results']:
            print(f"Test Case: {result['test_case']}")
            print(f"  Average: {result['avg_time_ms']:.2f}ms")
            print(f"  Median:  {result['median_time_ms']:.2f}ms")
            print(f"  Range:   {result['min_time_ms']:.2f}ms - {result['max_time_ms']:.2f}ms")
            print(f"  Std Dev: {result['std_dev_ms']:.2f}ms")
            print()

    def plot_benchmark_results(self, benchmark_results: Dict, output_file: str = None):
        """Create visualization of benchmark results."""

        test_cases = [r['test_case'] for r in benchmark_results['test_results']]
        avg_times = [r['avg_time_ms'] for r in benchmark_results['test_results']]
        std_devs = [r['std_dev_ms'] for r in benchmark_results['test_results']]

        plt.figure(figsize=(12, 6))
        plt.bar(test_cases, avg_times, yerr=std_devs, capsize=5)
        plt.title(f'Performance Benchmark: {benchmark_results["function"]}')
        plt.xlabel('Test Cases')
        plt.ylabel('Average Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
        plt.show()

# Usage example
async def benchmark_audio_processing():
    from services.analysis_service.src.bpm_detector import BPMDetector

    detector = BPMDetector()
    suite = BenchmarkSuite()

    # Test with different file sizes
    test_files = [
        "test_short_30s.mp3",
        "test_medium_3min.mp3",
        "test_long_10min.mp3"
    ]

    results = await suite.benchmark_function(
        detector.detect_bpm,
        test_files,
        iterations=5
    )

    suite.generate_performance_report(results)
    suite.plot_benchmark_results(results, "bpm_detection_benchmark.png")

if __name__ == "__main__":
    asyncio.run(benchmark_audio_processing())
```

### Load Testing

**Automated Load Testing Script:**
```bash
#!/bin/bash
# scripts/load_test.sh

set -e

echo "=== Tracktion Load Testing Suite ==="

# Configuration
API_BASE_URL="http://localhost:8001"
CONCURRENT_USERS=10
TEST_DURATION=300  # 5 minutes
RAMP_UP_TIME=60   # 1 minute

# Install dependencies
if ! command -v wrk &> /dev/null; then
    echo "Installing wrk load testing tool..."
    # Ubuntu/Debian
    sudo apt-get update && sudo apt-get install -y wrk
    # macOS
    # brew install wrk
fi

# Health check endpoints test
echo "Testing health endpoints..."
wrk -t4 -c10 -d30s --latency "${API_BASE_URL}/health"

# Audio analysis endpoint test (with sample data)
echo "Testing audio analysis endpoints..."
wrk -t8 -c${CONCURRENT_USERS} -d${TEST_DURATION}s \
    -s scripts/load_test_analysis.lua \
    --latency \
    "${API_BASE_URL}/api/v1/analyze"

# Track search endpoint test
echo "Testing track search endpoints..."
wrk -t4 -c${CONCURRENT_USERS} -d${TEST_DURATION}s \
    -s scripts/load_test_search.lua \
    --latency \
    "http://localhost:8002/api/v1/tracks/search"

# Generate load test report
echo "Generating load test report..."
python scripts/analyze_load_test_results.py
```

This performance tuning guide provides a comprehensive approach to optimizing the Tracktion system. Regular monitoring and benchmarking should be performed to maintain optimal performance as the system scales.
