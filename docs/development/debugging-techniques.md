# Debugging Techniques Guide

## Overview

This guide provides comprehensive debugging strategies, tools, and techniques for the Tracktion project. Effective debugging is crucial for maintaining the complex microservices architecture and ensuring reliable audio processing functionality.

## Debugging Philosophy

### Core Principles
1. **Systematic Approach**: Use structured debugging methodologies
2. **Evidence-Based**: Collect data before making assumptions
3. **Minimal Reproduction**: Create the smallest test case that reproduces the issue
4. **Documentation**: Record findings and solutions for future reference
5. **Prevention**: Learn from bugs to prevent similar issues

### Debugging Mindset
- **Question Assumptions**: Don't assume you know where the problem is
- **Follow the Data**: Let logs and metrics guide your investigation
- **Think Systematically**: Isolate variables and test hypotheses
- **Stay Curious**: Understand not just how to fix, but why it broke

## Logging and Observability

### Structured Logging with Structlog

#### Basic Logging Setup
```python
import structlog

# Service-wide logger configuration
logger = structlog.get_logger(__name__)

# Example usage with context
logger.info(
    "Audio analysis started",
    file_path=audio_file_path,
    file_size=file_size,
    service="analysis_service",
    trace_id=trace_id
)

logger.error(
    "BPM detection failed",
    error=str(exception),
    file_path=audio_file_path,
    algorithm="rhythm_extractor",
    confidence_threshold=0.7,
    exc_info=True  # Include stack trace
)
```

#### Advanced Logging Patterns
```python
# Context managers for automatic logging
from contextlib import contextmanager

@contextmanager
def logged_operation(operation_name: str, **context):
    logger.info(f"{operation_name} started", **context)
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"{operation_name} completed",
            duration_ms=duration * 1000,
            **context
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation_name} failed",
            error=str(e),
            duration_ms=duration * 1000,
            **context,
            exc_info=True
        )
        raise

# Usage
with logged_operation("audio_processing", file_id=track_id):
    result = process_audio_file(file_path)
```

#### Debug Logging Levels
```python
# Configure different log levels for debugging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Debugging specific components
COMPONENT_LOG_LEVELS = {
    "audio_processing": "DEBUG",
    "database": "INFO",
    "messaging": "WARNING",
    "external_api": "DEBUG"
}

# Conditional debug logging
if LOG_LEVEL == "DEBUG":
    logger.debug(
        "Audio features extracted",
        features=audio_features[:5],  # Only log first 5 for brevity
        feature_count=len(audio_features),
        extraction_time=extraction_time
    )
```

### Application Metrics

#### Custom Metrics for Debugging
```python
import time
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    duration: float
    memory_usage: int
    success: bool
    metadata: Dict[str, Any]

class MetricsCollector:
    def __init__(self):
        self.metrics = []

    @contextmanager
    def measure(self, operation: str, **metadata):
        import psutil
        process = psutil.Process()

        start_time = time.time()
        memory_before = process.memory_info().rss

        try:
            yield
            success = True
        except Exception as e:
            success = False
            metadata['error'] = str(e)
            raise
        finally:
            duration = time.time() - start_time
            memory_after = process.memory_info().rss

            metric = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                duration=duration,
                memory_usage=memory_after - memory_before,
                success=success,
                metadata=metadata
            )

            self.metrics.append(metric)

            logger.debug(
                "Operation metrics collected",
                **metric.__dict__
            )

# Usage
metrics = MetricsCollector()

with metrics.measure("bpm_detection", file_size=file_size):
    bpm_result = detect_bpm(audio_file)
```

## Debugging Techniques by Component

### Audio Processing Debugging

#### BPM Detection Issues
```python
def debug_bpm_detection(audio_file_path: str):
    """Debug BPM detection with detailed logging."""

    logger.info("Starting BPM debug analysis", file=audio_file_path)

    try:
        # Load audio with detailed logging
        logger.debug("Loading audio file")
        audio = es.MonoLoader(filename=audio_file_path, sampleRate=44100)()
        logger.debug("Audio loaded",
                    duration=len(audio) / 44100,
                    sample_count=len(audio),
                    audio_preview=audio[:100].tolist())

        # Primary algorithm debugging
        logger.debug("Running RhythmExtractor2013")
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, beats, confidence, estimates, intervals = rhythm_extractor(audio)

        logger.debug("Primary algorithm results",
                    bpm=bpm,
                    confidence=confidence,
                    beats_count=len(beats),
                    intervals_mean=float(np.mean(intervals)) if len(intervals) > 0 else None,
                    intervals_std=float(np.std(intervals)) if len(intervals) > 0 else None)

        # Fallback algorithm if needed
        if confidence < 0.7:
            logger.debug("Running Percival fallback")
            percival = es.PercivalBpmEstimator()
            fallback_bpm = percival(audio)
            logger.debug("Fallback results", fallback_bpm=fallback_bpm)

            # Agreement analysis
            agreement = abs(bpm - fallback_bpm) <= 5.0
            logger.debug("Algorithm agreement",
                        primary_bpm=bpm,
                        fallback_bpm=fallback_bpm,
                        difference=abs(bpm - fallback_bpm),
                        agreement=agreement)

        return {"bpm": bpm, "confidence": confidence, "algorithm": "debug"}

    except Exception as e:
        logger.error("BPM debug failed",
                    error=str(e),
                    file=audio_file_path,
                    exc_info=True)
        raise

# Usage
debug_result = debug_bmp_detection("/path/to/problematic/file.mp3")
```

#### Audio Data Validation
```python
def validate_audio_data(audio_data: np.ndarray, file_path: str):
    """Comprehensive audio data validation for debugging."""

    issues = []

    # Basic validation
    if len(audio_data) == 0:
        issues.append("Empty audio data")

    if np.all(audio_data == 0):
        issues.append("All samples are zero (silence)")

    if np.any(np.isnan(audio_data)):
        issues.append("NaN values detected")

    if np.any(np.isinf(audio_data)):
        issues.append("Infinite values detected")

    # Dynamic range analysis
    max_val = np.max(np.abs(audio_data))
    if max_val < 0.001:  # Very quiet
        issues.append(f"Very low amplitude (max: {max_val})")
    elif max_val > 0.99:  # Near clipping
        issues.append(f"Near clipping (max: {max_val})")

    # Frequency analysis
    if len(audio_data) > 1024:
        fft = np.fft.fft(audio_data[:1024])
        freq_content = np.abs(fft)
        if np.max(freq_content[1:]) < 0.001:  # No AC content
            issues.append("No significant frequency content")

    logger.info(
        "Audio validation results",
        file=file_path,
        duration=len(audio_data) / 44100,
        sample_rate=44100,
        max_amplitude=float(max_val),
        issues=issues
    )

    return issues
```

### Database Debugging

#### SQL Query Analysis
```python
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time

# Enable SQL query logging for debugging
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()
    logger.debug("SQL Query Start",
                query=statement[:200] + "..." if len(statement) > 200 else statement,
                parameters=parameters if len(str(parameters)) < 500 else "Large parameters")

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    logger.debug("SQL Query Complete",
                duration_ms=total * 1000,
                rowcount=cursor.rowcount if hasattr(cursor, 'rowcount') else None)

# Database connection debugging
async def debug_database_operation(operation_name: str, query_func):
    """Debug database operations with detailed logging."""

    logger.info(f"Database operation starting: {operation_name}")

    try:
        async with get_async_engine().begin() as conn:
            # Check connection
            await conn.execute(text("SELECT 1"))
            logger.debug("Database connection verified")

            # Execute operation
            result = await query_func(conn)

            logger.info(f"Database operation completed: {operation_name}",
                       result_type=type(result).__name__)

            return result

    except Exception as e:
        logger.error(f"Database operation failed: {operation_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True)
        raise
```

#### Connection Pool Monitoring
```python
def monitor_connection_pool(engine):
    """Monitor database connection pool health."""

    pool = engine.pool

    logger.info("Connection pool status",
               size=pool.size(),
               checked_in=pool.checkedin(),
               checked_out=pool.checkedout(),
               overflow=pool.overflow(),
               invalid=pool.invalid())

    # Check for pool exhaustion
    if pool.checkedout() >= pool.size() + pool.overflow():
        logger.warning("Connection pool exhaustion detected",
                      checkedout=pool.checkedout(),
                      pool_size=pool.size(),
                      overflow=pool.overflow())
```

### Message Queue Debugging

#### RabbitMQ Connection Issues
```python
import pika
from pika.adapters.asyncio_connection import AsyncioConnection

class RabbitMQDebugger:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.logger = structlog.get_logger(__name__)

    async def debug_connection(self):
        """Comprehensive RabbitMQ connection debugging."""

        try:
            # Parse connection URL
            params = pika.URLParameters(self.connection_url)
            self.logger.debug("Connection parameters",
                            host=params.host,
                            port=params.port,
                            virtual_host=params.virtual_host,
                            username=params.credentials.username)

            # Test connection
            connection = pika.BlockingConnection(params)
            self.logger.info("RabbitMQ connection successful")

            # Test channel
            channel = connection.channel()
            self.logger.debug("Channel created successfully")

            # List queues
            try:
                method = channel.queue_declare(queue='', passive=True)
                self.logger.debug("Queue info", queue_info=method)
            except pika.exceptions.ChannelClosedByBroker as e:
                self.logger.debug("Queue declaration test failed", error=str(e))

            connection.close()

        except pika.exceptions.AMQPConnectionError as e:
            self.logger.error("RabbitMQ connection failed",
                            error=str(e),
                            connection_url=self.connection_url.replace(
                                params.credentials.password, "***"))
            raise

    def debug_message_flow(self, queue_name: str, routing_key: str):
        """Debug message flow through queues."""

        connection = pika.BlockingConnection(pika.URLParameters(self.connection_url))
        channel = connection.channel()

        try:
            # Check queue existence and stats
            method = channel.queue_declare(queue=queue_name, passive=True)
            self.logger.info("Queue stats",
                           queue=queue_name,
                           message_count=method.method.message_count,
                           consumer_count=method.method.consumer_count)

            # Send test message
            test_message = {"test": True, "timestamp": time.time()}
            channel.basic_publish(
                exchange='',
                routing_key=routing_key,
                body=json.dumps(test_message)
            )
            self.logger.debug("Test message sent",
                            queue=queue_name,
                            message=test_message)

        except Exception as e:
            self.logger.error("Message flow debug failed",
                            queue=queue_name,
                            error=str(e))
        finally:
            connection.close()
```

## Interactive Debugging

### Python Debugger (pdb)

#### Basic pdb Usage
```python
import pdb

def complex_audio_processing(audio_data):
    # Set breakpoint
    pdb.set_trace()

    # Process data step by step
    normalized = normalize_audio(audio_data)
    features = extract_features(normalized)
    result = analyze_features(features)

    return result

# Remote debugging for services
import pdbp  # Enhanced pdb
pdbp.set_trace()  # More features than standard pdb
```

#### Conditional Breakpoints
```python
def process_batch(audio_files):
    for i, file_path in enumerate(audio_files):
        # Only break on specific conditions
        if i == 10 or "problematic" in file_path:
            pdb.set_trace()

        result = process_file(file_path)
```

### IDE Debugging Setup

#### VS Code Debugging Configuration
```json
{
    "name": "Debug Analysis Service",
    "type": "python",
    "request": "launch",
    "program": "services/analysis_service/src/main.py",
    "console": "integratedTerminal",
    "envFile": "${workspaceFolder}/.env",
    "args": ["--debug"],
    "justMyCode": false,  // Allow debugging into libraries
    "stopOnEntry": false,
    "breakpoints": {
        "raised": ["Exception"]  // Break on any exception
    }
}
```

#### Remote Debugging
```python
# For debugging running services
import debugpy

# Start debug server
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached!")

# Your service code here
```

## Advanced Debugging Techniques

### Memory Debugging

#### Memory Usage Analysis
```python
import tracemalloc
import psutil
from memory_profiler import profile

@profile  # Line-by-line memory usage
def memory_intensive_function():
    # Your code here
    pass

def debug_memory_usage():
    """Comprehensive memory debugging."""

    # Start memory tracing
    tracemalloc.start()

    process = psutil.Process()
    memory_before = process.memory_info().rss

    try:
        # Your operation here
        result = some_operation()

        # Memory analysis
        memory_after = process.memory_info().rss
        memory_diff = memory_after - memory_before

        # Get top memory allocations
        current, peak = tracemalloc.get_traced_memory()

        logger.info("Memory usage analysis",
                   memory_before_mb=memory_before / 1024 / 1024,
                   memory_after_mb=memory_after / 1024 / 1024,
                   memory_diff_mb=memory_diff / 1024 / 1024,
                   traced_current_mb=current / 1024 / 1024,
                   traced_peak_mb=peak / 1024 / 1024)

        # Top allocations
        top_stats = tracemalloc.take_snapshot().statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug("Top memory allocation",
                        file=stat.traceback.format()[-1],
                        size_mb=stat.size / 1024 / 1024,
                        count=stat.count)

        return result

    finally:
        tracemalloc.stop()
```

### Performance Profiling

#### CPU Profiling
```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    """Profile a function's CPU usage."""

    pr = cProfile.Profile()
    pr.enable()

    try:
        result = func(*args, **kwargs)
    finally:
        pr.disable()

    # Analyze results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions

    logger.debug("CPU profiling results",
                profile_output=s.getvalue())

    return result

# Usage
result = profile_function(complex_audio_operation, audio_data)
```

#### Async Code Debugging
```python
import asyncio
import logging

# Enable asyncio debug mode
logging.getLogger('asyncio').setLevel(logging.DEBUG)
asyncio.get_event_loop().set_debug(True)

async def debug_async_operation():
    """Debug async operations with detailed logging."""

    logger.debug("Async operation started")

    # Track concurrent tasks
    tasks = []

    try:
        # Create tasks with debugging
        for i in range(10):
            task = asyncio.create_task(
                async_worker(i),
                name=f"worker-{i}"
            )
            tasks.append(task)

        # Wait with timeout and error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        logger.info("Async operation completed",
                   total_tasks=len(tasks),
                   successful=len(successful),
                   failed=len(failed))

        if failed:
            for i, error in enumerate(failed):
                logger.error(f"Task failed: {error}",
                           task_index=i,
                           error_type=type(error).__name__)

        return successful

    except Exception as e:
        logger.error("Async operation failed",
                    error=str(e),
                    pending_tasks=len([t for t in tasks if not t.done()]))

        # Cancel pending tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        raise
```

## Error Analysis and Root Cause Investigation

### Exception Analysis Framework

```python
import traceback
import sys
from typing import Any, Dict

class ErrorAnalyzer:
    def __init__(self):
        self.logger = structlog.get_logger(__name__)

    def analyze_exception(self, exc: Exception, context: Dict[str, Any] = None):
        """Comprehensive exception analysis."""

        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Basic exception info
        error_info = {
            "exception_type": exc_type.__name__ if exc_type else type(exc).__name__,
            "exception_message": str(exc),
            "exception_args": getattr(exc, 'args', None),
        }

        # Stack trace analysis
        if exc_traceback:
            stack_trace = traceback.extract_tb(exc_traceback)
            error_info["stack_trace"] = [
                {
                    "file": frame.filename,
                    "line": frame.lineno,
                    "function": frame.name,
                    "code": frame.line
                }
                for frame in stack_trace
            ]

            # Find the deepest frame in our code (not libraries)
            our_frames = [
                frame for frame in stack_trace
                if 'services/' in frame.filename
            ]

            if our_frames:
                error_info["error_location"] = {
                    "file": our_frames[-1].filename,
                    "line": our_frames[-1].lineno,
                    "function": our_frames[-1].name
                }

        # Context information
        if context:
            error_info["context"] = context

        # Common error patterns
        error_info["category"] = self._categorize_error(exc)
        error_info["suggestions"] = self._get_error_suggestions(exc)

        self.logger.error("Exception analysis", **error_info)

        return error_info

    def _categorize_error(self, exc: Exception) -> str:
        """Categorize common error types."""

        if isinstance(exc, (ConnectionError, TimeoutError)):
            return "network"
        elif isinstance(exc, (FileNotFoundError, PermissionError)):
            return "filesystem"
        elif isinstance(exc, (ValueError, TypeError)):
            return "data_validation"
        elif "essentia" in str(exc).lower():
            return "audio_processing"
        elif "database" in str(exc).lower() or "sql" in str(exc).lower():
            return "database"
        else:
            return "unknown"

    def _get_error_suggestions(self, exc: Exception) -> list:
        """Get suggestions for common errors."""

        suggestions = []
        error_msg = str(exc).lower()

        if "connection" in error_msg:
            suggestions.append("Check network connectivity and service availability")
            suggestions.append("Verify connection parameters and credentials")

        if "file not found" in error_msg:
            suggestions.append("Verify file path and permissions")
            suggestions.append("Check if file was moved or deleted")

        if "timeout" in error_msg:
            suggestions.append("Increase timeout values")
            suggestions.append("Check for network latency issues")

        if "memory" in error_msg:
            suggestions.append("Check available system memory")
            suggestions.append("Consider processing data in smaller chunks")

        return suggestions

# Usage
analyzer = ErrorAnalyzer()

try:
    risky_operation()
except Exception as e:
    analysis = analyzer.analyze_exception(e, {
        "operation": "audio_processing",
        "file_path": audio_file_path,
        "user_id": user_id
    })
```

## Testing and Debugging Integration

### Debug-Friendly Test Setup

```python
import pytest
import logging

@pytest.fixture
def debug_mode(request):
    """Enable debug mode for specific tests."""

    # Enable detailed logging for tests
    logging.getLogger().setLevel(logging.DEBUG)

    # Add debug information to test context
    yield True

    # Cleanup
    logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("debug", [True], indirect=True)
def test_audio_processing_with_debug(debug_mode):
    """Test with debug information enabled."""

    with logged_operation("test_audio_processing"):
        result = process_audio_file("test.mp3")
        assert result is not None

# Debugging test failures
def pytest_runtest_logreport(report):
    """Custom hook to capture test failure information."""

    if report.failed:
        logger.error("Test failed",
                    test_name=report.nodeid,
                    test_outcome=report.outcome,
                    test_duration=report.duration,
                    failure_message=str(report.longrepr))
```

## Production Debugging

### Safe Production Debugging

```python
import os
from typing import Optional

class ProductionDebugger:
    def __init__(self):
        self.debug_enabled = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
        self.debug_token = os.getenv("DEBUG_TOKEN")

    def safe_debug(self, token: Optional[str] = None):
        """Enable debugging only with proper authorization."""

        if not self.debug_enabled:
            return False

        if self.debug_token and token != self.debug_token:
            logger.warning("Unauthorized debug attempt",
                         provided_token=token[:4] + "..." if token else None)
            return False

        return True

    def debug_request(self, request_id: str, token: Optional[str] = None):
        """Debug a specific request in production."""

        if not self.safe_debug(token):
            return

        # Enable detailed logging for this request
        request_logger = logger.bind(request_id=request_id, debug_mode=True)

        # Add request to debug tracking
        self._track_debug_request(request_id)

# Usage in FastAPI
from fastapi import FastAPI, Header

app = FastAPI()
debugger = ProductionDebugger()

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    debug_token = request.headers.get("X-Debug-Token")
    request_id = request.headers.get("X-Request-ID")

    if debugger.safe_debug(debug_token):
        # Enable debugging for this request
        with logged_operation("request_debug", request_id=request_id):
            response = await call_next(request)
    else:
        response = await call_next(request)

    return response
```

## Debugging Tools and Utilities

### Custom Debug Tools

```python
class AudioDebugger:
    """Specialized debugging tools for audio processing."""

    @staticmethod
    def save_audio_debug_info(audio_data: np.ndarray, file_path: str, debug_dir: str = "/tmp/audio_debug"):
        """Save audio debug information."""

        os.makedirs(debug_dir, exist_ok=True)

        # Save raw audio data
        np.save(os.path.join(debug_dir, "audio_data.npy"), audio_data)

        # Save audio statistics
        stats = {
            "length": len(audio_data),
            "duration": len(audio_data) / 44100,
            "min": float(np.min(audio_data)),
            "max": float(np.max(audio_data)),
            "mean": float(np.mean(audio_data)),
            "std": float(np.std(audio_data)),
            "rms": float(np.sqrt(np.mean(audio_data**2)))
        }

        with open(os.path.join(debug_dir, "audio_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        logger.debug("Audio debug info saved",
                    debug_dir=debug_dir,
                    stats=stats)

# Usage
if os.getenv("DEBUG_AUDIO", "false").lower() == "true":
    AudioDebugger.save_audio_debug_info(audio_data, file_path)
```

## Best Practices

### Debugging Workflow
1. **Reproduce**: Create minimal reproduction case
2. **Isolate**: Identify the specific component or operation
3. **Instrument**: Add logging and monitoring
4. **Hypothesize**: Form theories about the root cause
5. **Test**: Validate hypotheses systematically
6. **Fix**: Implement and verify the solution
7. **Document**: Record findings for future reference

### Performance Debugging
- **Profile first**: Measure before optimizing
- **Focus on bottlenecks**: Fix the slowest parts first
- **Test realistic data**: Use production-like datasets
- **Monitor continuously**: Track performance over time

### Security Debugging
- **Never log secrets**: Sanitize sensitive data in logs
- **Limit debug access**: Require authentication for debug features
- **Audit debug usage**: Log all debug activities
- **Disable in production**: Remove debug code from production builds

## Next Steps

After mastering debugging techniques:
1. **Practice systematic debugging**: Apply the framework to real issues
2. **Set up monitoring**: Implement comprehensive logging and metrics
3. **Create debug tools**: Build service-specific debugging utilities
4. **Document solutions**: Maintain a knowledge base of common issues
5. **Improve prevention**: Use debugging insights to improve code quality
