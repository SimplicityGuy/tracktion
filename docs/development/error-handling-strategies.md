# Error Handling Strategies Guide

## Overview

This document outlines the comprehensive error handling strategies implemented across all Tracktion services. Each service follows consistent patterns while adapting to domain-specific requirements.

## Error Handling Philosophy

### Core Principles
1. **Fail Fast, Fail Explicitly**: Detect and report errors immediately with meaningful context
2. **Graceful Degradation**: Continue operation with reduced functionality when possible
3. **Error Context Preservation**: Maintain full error context for debugging and analysis
4. **Recovery Strategies**: Design systems with automatic recovery mechanisms
5. **Circuit Breaking**: Prevent cascading failures across service boundaries
6. **Resource Cleanup**: Ensure proper cleanup in all error scenarios

### Error Classification Framework

#### By Severity
- **Critical**: System cannot continue, immediate shutdown required
- **Error**: Operation failed, but system can continue with degraded functionality
- **Warning**: Unexpected condition, but operation completed
- **Info**: Normal error recovery or fallback activation

#### By Recovery Strategy
- **Retryable**: Transient errors that should be retried with backoff
- **Non-retryable**: Permanent errors that won't improve with retry
- **Circuit Breaking**: Errors that should trigger circuit breaker activation
- **Escalation**: Errors requiring human intervention or system restart

## Service-Specific Error Handling

### Analysis Service Error Handling

#### Exception Hierarchy
```python
# Custom exception structure
AnalysisServiceError (Base)
├── InvalidAudioFileError
│   ├── UnsupportedFormatError
│   └── CorruptedFileError
├── MetadataExtractionError
├── StorageError
│   └── TransientStorageError (retryable)
├── ConnectionError
│   └── TransientConnectionError (retryable)
├── MessageProcessingError
├── ConfigurationError
└── RetryableError (base for transient errors)
```

#### Circuit Breaker Implementation
```python
# Circuit breaker configuration
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 2      # Successes to close
    timeout_seconds: int = 60       # Time before half-open attempt
    fallback_function: Optional[Callable] = None
```

**States and Transitions:**
- **CLOSED**: Normal operation, failures counted
- **OPEN**: All calls fail fast, fallback used if available
- **HALF_OPEN**: Test calls allowed, success closes circuit

#### Retry Strategies
```python
# Exponential backoff with jitter
def calculate_backoff(attempt: int, base_delay: float = 1.0) -> float:
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.1) * delay
    return min(delay + jitter, 300)  # Max 5 minutes
```

**Error Classification for Retry:**
- **Non-retryable**: `InvalidAudioFileError`, `UnsupportedFormatError`
- **Retryable**: `MetadataExtractionError`, `StorageError`
- **Circuit Breaking**: `ConnectionError`, service timeouts

#### Async Error Handling
```python
# Advanced async error handler with resource cleanup
class AsyncErrorHandler:
    async def execute_with_recovery(self, func, task_id, timeout=300):
        try:
            if self._is_circuit_open():
                raise RuntimeError(f"Circuit breaker open for {task_id}")

            result = await asyncio.wait_for(func(), timeout=timeout)
            self._reset_circuit()
            return result

        except asyncio.TimeoutError:
            await self._cleanup_resources(task_id)
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
        except MemoryError:
            await self._cleanup_memory(task_id)
            raise MemoryError(f"Out of memory processing {task_id}")
        finally:
            await self._ensure_cleanup(task_id)
```

#### Algorithm Error Handling
```python
# BPM detection with fallback algorithms
def detect_bmp_with_fallback(self, audio_path: str) -> Dict[str, Any]:
    try:
        # Primary algorithm: RhythmExtractor2013
        result = self._primary_algorithm(audio_path)
        if result.confidence >= self.confidence_threshold:
            return result
    except Exception as e:
        logger.warning(f"Primary BPM algorithm failed: {e}")

    try:
        # Fallback algorithm: PercivalBmpEstimator
        result = self._fallback_algorithm(audio_path)
        result.needs_review = True  # Flag for manual review
        return result
    except Exception as e:
        logger.error(f"All BPM algorithms failed: {e}")
        return self._default_result(needs_review=True)
```

### Tracklist Service Error Handling

#### Web Scraping Resilience
```python
# Multi-strategy extraction with quality scoring
class ResilientExtractor:
    def extract_with_fallback(self, soup, strategies, field_name):
        best_result = None

        for strategy in strategies:
            try:
                result = strategy.extract(soup)
                if result.quality_score > (best_result.quality_score if best_result else 0):
                    best_result = result
                if result.quality_score >= 0.95:  # High quality, stop trying
                    break
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
                continue

        return best_result or self._empty_result()
```

#### Rate Limiting and Anti-Detection
```python
# Intelligent rate limiting with anti-detection
class RateLimitHandler:
    async def _apply_rate_limit(self) -> None:
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            # Anti-detection jitter
            sleep_time += random.uniform(0, 0.5)
            await asyncio.sleep(sleep_time)

        # Rotate user agent for anti-detection
        self._rotate_user_agent()
```

#### Cache Fallback Strategies
```python
# Multi-layer cache with progressive fallback
class FallbackCache:
    async def get_with_fallback(self, key: str) -> Optional[Any]:
        # Layer 1: Memory cache (fastest)
        if result := self.memory_cache.get(key):
            return result

        # Layer 2: Redis primary
        try:
            if result := await self.redis_primary.get(key):
                self.memory_cache.set(key, result)
                return result
        except ConnectionError:
            logger.warning("Primary Redis unavailable, trying fallback")

        # Layer 3: Redis fallback with extended TTL
        try:
            if result := await self.redis_fallback.get(key):
                return result
        except ConnectionError:
            logger.warning("Fallback Redis unavailable")

        # Layer 4: Archive/backup storage
        return await self.archive_storage.get(key)
```

#### Message Queue Error Handling
```python
# RabbitMQ with automatic retry and dead letter handling
async def process_message(self, message, handler, max_retries=5):
    try:
        await handler(message)
        await message.ack()

    except json.JSONDecodeError:
        # Malformed message - don't retry
        await message.reject(requeue=False)

    except ValidationError:
        # Invalid data - don't retry
        await message.reject(requeue=False)

    except (ConnectionError, TimeoutError) as e:
        # Transient errors - retry with backoff
        retry_count = self._get_retry_count(message.headers)
        if retry_count < max_retries:
            delay = 2 ** retry_count  # Exponential backoff
            await asyncio.sleep(delay)
            await message.reject(requeue=True)
        else:
            # Max retries exceeded - dead letter
            await message.reject(requeue=False)

    except Exception as e:
        # Unexpected errors - log and dead letter
        logger.error(f"Unexpected error processing message: {e}", exc_info=True)
        await message.reject(requeue=False)
```

### File Watcher Service Error Handling

#### File System Error Resilience
```python
# Hash calculation with progressive fallback
def calculate_dual_hashes(self, file_path: Path) -> Tuple[str, str]:
    try:
        # Primary: Calculate from file content
        sha256_hasher = hashlib.sha256()
        xxh128_hasher = xxhash.xxh128()

        with file_path.open("rb") as f:
            while chunk := f.read(8192):
                sha256_hasher.update(chunk)
                xxh128_hasher.update(chunk)

        return sha256_hasher.hexdigest(), xxh128_hasher.hexdigest()

    except (IOError, OSError, PermissionError) as e:
        logger.warning(f"File read failed, using fallback: {e}")

        try:
            # Secondary: Use file metadata
            stat = file_path.stat()
            fallback_data = f"{file_path}{stat.st_size}{stat.st_mtime}".encode()
        except OSError:
            # Tertiary: Use just file path
            fallback_data = str(file_path).encode()

        sha256_fallback = hashlib.sha256(fallback_data).hexdigest()
        xxh128_fallback = xxhash.xxh128(fallback_data).hexdigest()
        return sha256_fallback, xxh128_fallback
```

#### Observer Recovery Mechanisms
```python
# Automatic observer restart with health monitoring
class FileWatcherService:
    def _monitor_observer_health(self):
        while self.running:
            time.sleep(self.health_check_interval)

            if not self.observer.is_alive():
                logger.error("Observer died, attempting restart")

                try:
                    # Stop and clean up dead observer
                    if hasattr(self.observer, 'stop'):
                        self.observer.stop()

                    # Create new observer with same configuration
                    self.observer = Observer()
                    self.observer.schedule(
                        self.event_handler,
                        str(self.scan_path),
                        recursive=True
                    )
                    self.observer.start()

                    logger.info("Observer restarted successfully")

                except Exception as e:
                    logger.error(f"Observer restart failed: {e}")
                    # Could trigger service restart or alert
                    self.running = False
```

#### Batch Processing Error Isolation
```python
# Process files in batches with error isolation
async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
    tasks = []

    for file_path in file_paths:
        task = self._process_single_file(file_path)
        tasks.append(task)

    # Gather with exception handling - isolates failures
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and separate successes from failures
    successes = []
    failures = []

    for path, result in zip(file_paths, results, strict=False):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {path}: {result}")
            failures.append({"path": path, "error": str(result)})
        else:
            successes.append(result)

    # Log batch statistics
    logger.info(f"Batch complete: {len(successes)} successes, {len(failures)} failures")

    return successes
```

## Cross-Cutting Error Handling Patterns

### Correlation ID Tracking
```python
# Maintain correlation IDs across service calls
import uuid
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id')

def set_correlation_id(cid: str = None) -> str:
    if not cid:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid

def get_correlation_id() -> str:
    try:
        return correlation_id.get()
    except LookupError:
        return set_correlation_id()

# Use in all log entries
logger.info("Processing request", correlation_id=get_correlation_id())
```

### Structured Error Context
```python
# Rich error context for debugging
class ErrorContext:
    def __init__(self):
        self.context = {}

    def add_context(self, **kwargs):
        self.context.update(kwargs)
        return self

    def log_error(self, error: Exception, logger):
        logger.error(
            f"Operation failed: {error}",
            correlation_id=get_correlation_id(),
            error_type=type(error).__name__,
            **self.context,
            exc_info=True
        )

# Usage
ctx = ErrorContext()
ctx.add_context(
    file_path="/path/to/file",
    operation="bmp_detection",
    attempt=3
)
ctx.log_error(error, logger)
```

### Health Check Integration
```python
# Health checks with error rate monitoring
class HealthMonitor:
    def __init__(self, error_threshold=0.1, window_seconds=300):
        self.error_threshold = error_threshold
        self.window_seconds = window_seconds
        self.error_log = deque()
        self.total_operations = 0

    def record_error(self):
        self.error_log.append(time.time())
        self._cleanup_old_errors()

    def record_success(self):
        self.total_operations += 1

    def is_healthy(self) -> bool:
        self._cleanup_old_errors()
        error_count = len(self.error_log)

        if self.total_operations == 0:
            return True  # No operations yet

        error_rate = error_count / max(self.total_operations, 1)
        return error_rate <= self.error_threshold
```

## Error Handling Best Practices

### 1. Exception Design
```python
# Well-designed exceptions with context
class ServiceError(Exception):
    """Base exception for service errors."""

    def __init__(self, message: str, error_code: str = None, **context):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context
        self.timestamp = datetime.utcnow()
        self.correlation_id = get_correlation_id()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }
```

### 2. Resource Management
```python
# Context managers for guaranteed cleanup
class ResourceManager:
    def __enter__(self):
        self.resource = self._acquire_resource()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.resource, 'cleanup'):
            try:
                self.resource.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

        # Return False to propagate exceptions
        return False

# Async context manager
class AsyncResourceManager:
    async def __aenter__(self):
        self.resource = await self._acquire_resource_async()
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._cleanup_resource_async(self.resource)
        except Exception as e:
            logger.error(f"Async cleanup failed: {e}")
        return False
```

### 3. Graceful Degradation
```python
# Service operation with graceful degradation
class ServiceWithFallback:
    def process_with_fallback(self, data):
        # Try primary processing
        try:
            return self.primary_processor.process(data)
        except ServiceUnavailableError:
            logger.warning("Primary service unavailable, using fallback")
            return self.fallback_processor.process(data)
        except Exception as e:
            logger.error(f"Primary processing failed: {e}")
            # Try fallback for unexpected errors too
            try:
                result = self.fallback_processor.process(data)
                result['degraded'] = True  # Flag degraded service
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                # Return minimal viable result
                return self._minimal_result(data)
```

### 4. Monitoring Integration
```python
# Metrics collection for error rates
from prometheus_client import Counter, Histogram, Gauge

# Metrics
error_counter = Counter('service_errors_total', 'Total errors', ['service', 'error_type'])
response_time = Histogram('service_response_time_seconds', 'Response time')
circuit_breaker_state = Gauge('circuit_breaker_state', 'Circuit breaker state', ['service'])

# Decorator for automatic metrics collection
def monitor_errors(service_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_counter.labels(
                    service=service_name,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                response_time.observe(time.time() - start_time)
        return wrapper
    return decorator
```

## Testing Error Handling

### 1. Error Injection Testing
```python
# Test error scenarios with injection
class ErrorInjector:
    def __init__(self):
        self.failure_rate = 0.0
        self.failure_types = []

    def set_failure_rate(self, rate: float, error_types: List[Exception]):
        self.failure_rate = rate
        self.failure_types = error_types

    def maybe_fail(self):
        if random.random() < self.failure_rate:
            error_type = random.choice(self.failure_types)
            raise error_type("Injected error for testing")

# Usage in tests
@pytest.fixture
def error_injector():
    return ErrorInjector()

def test_service_resilience(error_injector):
    error_injector.set_failure_rate(0.3, [ConnectionError, TimeoutError])

    # Test service behavior under failure conditions
    results = []
    for _ in range(100):
        try:
            result = service.process_with_injection(error_injector)
            results.append(result)
        except Exception:
            pass

    # Verify graceful degradation
    assert len(results) >= 60  # At least 60% success rate
```

### 2. Circuit Breaker Testing
```python
def test_circuit_breaker_behavior():
    breaker = CircuitBreaker(failure_threshold=3, timeout=5)

    # Trigger circuit breaker opening
    for _ in range(3):
        with pytest.raises(Exception):
            breaker.call(failing_function)

    # Verify circuit is open
    assert breaker.state == CircuitBreakerState.OPEN

    # Wait for timeout and test half-open state
    time.sleep(6)
    with pytest.raises(Exception):
        breaker.call(failing_function)  # Should attempt call

    assert breaker.state == CircuitBreakerState.OPEN  # Back to open
```

## Conclusion

Effective error handling is crucial for building resilient microservices. The patterns documented here provide:

1. **Comprehensive error classification** with appropriate recovery strategies
2. **Circuit breaker protection** to prevent cascading failures
3. **Intelligent retry mechanisms** with exponential backoff and jitter
4. **Graceful degradation** to maintain service availability
5. **Rich error context** for effective debugging and monitoring
6. **Resource cleanup** to prevent resource leaks
7. **Health monitoring** integration for operational visibility

These patterns should be consistently applied across all services while adapting to domain-specific requirements and failure modes.
