# File Watcher Service

Intelligent file monitoring and metadata extraction service for the Tracktion music library management system.

## Overview

The File Watcher service provides real-time monitoring of audio file collections with dual-hash fingerprinting, metadata extraction, and message queue integration. It features both synchronous and asynchronous implementations for different performance requirements, supporting high-throughput scenarios with concurrent file processing.

## Architecture

### Core Components

#### Primary Services
- **FileWatcherService** (`main.py`) - Primary synchronous service implementation
- **AsyncFileWatcherService** (`async_file_watcher.py`) - High-performance async implementation
- **TracktionEventHandler** (`watchdog_handler.py`) - File system event processing
- **AsyncFileEventHandler** (`async_file_watcher.py`) - Async event processing with concurrency control

#### Data Processing
- **FileScanner** (`file_scanner.py`) - Directory scanning and file discovery
- **AsyncMetadataExtractor** (`async_metadata_extractor.py`) - Audio metadata extraction with caching
- **Hash Utilities** (`hash_utils.py`) - Dual-hash fingerprinting (SHA256 + XXHash128)

#### Messaging
- **MessagePublisher** (`message_publisher.py`) - Synchronous RabbitMQ integration
- **AsyncMessagePublisher** (`async_message_publisher.py`) - Async RabbitMQ with batch processing

### Architecture Pattern

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   File System      │    │   Event Processing   │    │   Message Queue     │
│                     │    │                      │    │                     │
│ ┌─────────────────┐ │    │ ┌──────────────────┐ │    │ ┌─────────────────┐ │
│ │  Watchdog       │ │───▶│ │ Event Handlers   │ │───▶│ │ RabbitMQ        │ │
│ │  Observer       │ │    │ │ • Sync/Async     │ │    │ │ • Topic Exchange│ │
│ └─────────────────┘ │    │ │ • File Filtering │ │    │ │ • Routing Keys  │ │
│                     │    │ └──────────────────┘ │    │ └─────────────────┘ │
│ ┌─────────────────┐ │    │ ┌──────────────────┐ │    │                     │
│ │  Directory      │ │    │ │ Metadata         │ │    │ ┌─────────────────┐ │
│ │  Scanner        │ │    │ │ Extraction       │ │    │ │ Event Types:    │ │
│ └─────────────────┘ │    │ │ • Audio Tags     │ │    │ │ • file.created  │ │
└─────────────────────┘    │ │ • Duration/Fmt   │ │    │ │ • file.modified │ │
                           │ └──────────────────┘ │    │ │ • file.deleted  │ │
                           │ ┌──────────────────┐ │    │ │ • file.moved    │ │
                           │ │ Hash Calculation │ │    │ │ • file.renamed  │ │
                           │ │ • SHA256         │ │    │ └─────────────────┘ │
                           │ │ • XXHash128      │ │    └─────────────────────┘
                           │ └──────────────────┘ │
                           └──────────────────────┘
```

## Features

### File Monitoring Capabilities

#### Supported Audio Formats
- **Lossless**: FLAC, WAV, WAVE
- **Compressed**: MP3, OGG, OGA (Vorbis), AAC, OPUS
- **Apple Formats**: M4A, M4B, M4P, M4V, M4R, MP4
- **Microsoft**: WMA

#### Event Detection
- **File Operations**: Created, modified, deleted, moved, renamed
- **Recursive Monitoring**: Watches entire directory trees
- **Real-time Processing**: Immediate event handling with watchdog library
- **Batch Processing**: Efficient handling of bulk operations

#### Performance Features
- **Dual-Hash Fingerprinting**: SHA256 (cryptographic) + XXHash128 (fast)
- **Concurrent Processing**: Configurable semaphore-based concurrency control
- **Metadata Caching**: Smart caching with file modification tracking
- **Batch Operations**: Grouped processing for improved throughput

### High-Performance Async Implementation

#### Concurrency Control
- **Semaphore-based Limiting**: Configurable concurrent file processing (default: 100)
- **Async I/O**: Non-blocking file operations with aiofiles
- **Thread Pool**: CPU-bound metadata extraction in dedicated threads
- **Batch Processing**: Configurable batch sizes (default: 100 files)

#### Resource Management
- **Connection Resilience**: Robust RabbitMQ reconnection with exponential backoff
- **Memory Efficiency**: Streaming hash calculation with 8KB chunks
- **Cache Management**: Intelligent metadata caching with size limits
- **Graceful Shutdown**: Proper cleanup of async tasks and connections

### Metadata Extraction

#### Audio Metadata
- **Technical Properties**: Duration, bitrate, sample rate, channels, format
- **Tag Information**: Title, artist, album, date, genre, track number
- **File Properties**: Size, modification time, path information
- **Error Handling**: Graceful fallback with basic metadata on extraction failure

#### Caching Strategy
- **Cache Keys**: File path + modification time + size
- **Performance**: Avoids re-extraction of unchanged files
- **Memory Management**: Configurable cache size limits
- **Invalidation**: Automatic cache invalidation on file changes

## Configuration

### Environment Variables

#### Core Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data/music` | Root directory to monitor for audio files |
| `INSTANCE_ID` | `auto-generated` | Unique identifier for service instance |
| `MAX_CONCURRENT_FILES` | `100` | Maximum concurrent file processing (async only) |

#### RabbitMQ Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_HOST` | `localhost` | RabbitMQ server hostname |
| `RABBITMQ_PORT` | `5672` | RabbitMQ server port |
| `RABBITMQ_USER` | `guest` | RabbitMQ username |
| `RABBITMQ_PASS` | `guest` | RabbitMQ password |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | Complete connection URL (overrides individual settings) |

#### Legacy Compatibility
| Variable | Description |
|----------|-------------|
| `FILE_WATCHER_SCAN_PATH` | Legacy alias for `DATA_DIR` (deprecated) |

### Message Queue Integration

#### Exchange Configuration
- **Exchange Name**: `file_events`
- **Exchange Type**: Topic
- **Durability**: Persistent messages for reliability

#### Routing Keys
- **file.created** - New file detection
- **file.modified** - File content/metadata changes
- **file.deleted** - File removal
- **file.moved** - File relocation between directories
- **file.renamed** - File name changes within same directory

#### Message Format
```json
{
  "event_type": "created|modified|deleted|moved|renamed",
  "file_path": "/absolute/path/to/file.mp3",
  "timestamp": "2024-01-01T12:00:00Z",
  "instance_id": "abc12345",
  "sha256_hash": "a1b2c3d4...",
  "xxh128_hash": "e5f6g7h8...",
  "old_path": "/old/path/file.mp3",
  "metadata": {
    "duration": 180.5,
    "bitrate": 320000,
    "sample_rate": 44100,
    "channels": 2,
    "format": "audio/mpeg",
    "tags": {
      "title": "Song Title",
      "artist": "Artist Name",
      "album": "Album Name"
    }
  }
}
```

## Usage

### Local Development

#### Synchronous Version (Standard)
```bash
# Install dependencies
uv sync --dev

# Set environment variables
export DATA_DIR="/path/to/music/library"
export RABBITMQ_HOST="localhost"

# Run the service
uv run python -m src.main
```

#### Asynchronous Version (High Performance)
```bash
# Run async implementation
uv run python -m src.async_file_watcher

# With custom concurrency settings
export MAX_CONCURRENT_FILES=200
uv run python -m src.async_file_watcher
```

### Docker Deployment

#### Build Image
```bash
docker build -t tracktion/file-watcher .
```

#### Run Container
```bash
docker run -d \
  --name file-watcher \
  -v /host/music/path:/data/music:ro \
  -e RABBITMQ_HOST=rabbitmq \
  -e RABBITMQ_USER=tracktion \
  -e RABBITMQ_PASS=secure_password \
  -e MAX_CONCURRENT_FILES=150 \
  --restart unless-stopped \
  tracktion/file-watcher
```

#### Docker Compose Integration
```yaml
version: '3.8'
services:
  file-watcher:
    build: ./services/file_watcher
    environment:
      - DATA_DIR=/data/music
      - RABBITMQ_HOST=rabbitmq
      - RABBITMQ_USER=tracktion
      - RABBITMQ_PASS=secure_password
      - MAX_CONCURRENT_FILES=100
      - INSTANCE_ID=file-watcher-1
    volumes:
      - /host/music/library:/data/music:ro
    depends_on:
      - rabbitmq
    restart: unless-stopped
    networks:
      - tracktion-network
```

### Production Considerations

#### Performance Tuning
- **Concurrency**: Adjust `MAX_CONCURRENT_FILES` based on available CPU cores and I/O capacity
- **Memory**: Monitor metadata cache size for large libraries (>100K files)
- **Storage**: Use SSDs for improved hash calculation performance
- **Network**: Ensure stable RabbitMQ connectivity for high-throughput scenarios

#### Monitoring
- **Health Checks**: Observer thread health monitoring with automatic restart
- **Metrics**: Structured logging with instance IDs for distributed deployments
- **Error Handling**: Graceful degradation when RabbitMQ is unavailable
- **Resource Usage**: CPU and memory monitoring for large file collections

#### Scaling
- **Multiple Instances**: Use unique `INSTANCE_ID` for each deployment
- **Directory Partitioning**: Monitor different subdirectories with separate instances
- **Load Balancing**: RabbitMQ handles message distribution automatically
- **Horizontal Scaling**: Deploy multiple containers with shared RabbitMQ

## Testing

### Unit Tests
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/test_async_metadata_extractor.py
uv run pytest tests/unit/test_performance_benchmarks.py
uv run pytest tests/unit/test_edge_cases.py

# Coverage report
uv run pytest --cov=src --cov-report=html
```

### Performance Testing
```bash
# Benchmark hash calculation performance
uv run python -m src.benchmark_hashing

# Multi-instance stress testing
uv run python -m src.stress_test_multi_instance
```

### Integration Testing
```bash
# Test with local RabbitMQ
docker run -d --name test-rabbitmq -p 5672:5672 rabbitmq:3-management
uv run pytest tests/integration/
docker rm -f test-rabbitmq
```

## Development

### Code Quality
```bash
# Pre-commit checks (MANDATORY before commits)
pre-commit run --all-files

# Type checking
uv run mypy src/

# Linting and formatting
uv run ruff check src/
uv run ruff format src/
```

### Dependencies
Core dependencies managed via `pyproject.toml`:
- **watchdog**: File system event monitoring
- **aio-pika**: Async RabbitMQ integration
- **mutagen**: Audio metadata extraction
- **xxhash**: Fast non-cryptographic hashing
- **structlog**: Structured logging
- **aiofiles**: Async file I/O
- **pydantic**: Data validation and serialization

### Architecture Decisions

#### Dual Implementation Strategy
- **Synchronous**: Simple deployment, lower resource usage, easier debugging
- **Asynchronous**: High throughput, concurrent processing, better scalability
- **Choice**: Select based on library size and performance requirements

#### Hash Algorithm Selection
- **SHA256**: Cryptographic integrity, deduplication, security
- **XXHash128**: Performance, reduced CPU usage, fast comparison
- **Dual Strategy**: Best of both worlds for different use cases

#### Metadata Caching
- **Smart Invalidation**: File modification time + size-based cache keys
- **Memory Efficiency**: Configurable cache limits prevent memory bloat
- **Performance**: Avoids re-extraction of unchanged files

## Contributing

### Development Setup
1. Clone repository and navigate to service directory
2. Install dependencies: `uv sync --dev`
3. Set up pre-commit hooks: `pre-commit install`
4. Run tests: `uv run pytest`
5. Start development with proper environment configuration

### Coding Standards
- **Type Hints**: Full type annotation required
- **Error Handling**: Comprehensive exception handling with structured logging
- **Testing**: Unit tests required for new features
- **Documentation**: Docstrings for all public methods and classes
- **Performance**: Benchmark performance-critical code paths

### Submission Guidelines
- All pre-commit checks must pass
- Minimum 80% test coverage
- Performance impact assessment for core changes
- Documentation updates for API changes
- Integration testing with RabbitMQ

---

For questions or support, refer to the main Tracktion project documentation or open an issue in the project repository.
