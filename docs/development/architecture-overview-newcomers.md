# Architecture Overview for Newcomers

## Welcome to Tracktion!

This document provides a high-level introduction to Tracktion's architecture designed specifically for new team members. It focuses on understanding the system from a developer's perspective, emphasizing how components work together and where you'll likely be making changes.

## What is Tracktion?

Tracktion is an **automated music library management system** that helps users organize, analyze, and manage their music collections. Think of it as an intelligent system that:

- **Analyzes** audio files to extract metadata (BPM, key, mood, genre)
- **Manages** tracklists and provides intelligent track matching
- **Watches** file systems for changes and automatically processes new music
- **Organizes** files and provides catalog management
- **Notifies** users about processing results and system events

## High-Level Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  External APIs  │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │      API Gateway           │
                    │   (Load Balancer)          │
                    └─────────────┬──────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
    ┌───────▼─────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │ Analysis Service│  │Tracklist Service│  │File Watcher     │
    │ (Audio AI)      │  │ (Playlist Mgmt) │  │ (File Monitor)  │
    └───────┬─────────┘  └────────┬────────┘  └────────┬────────┘
            │                     │                    │
    ┌───────▼─────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │Cataloging Svc   │  │File Rename Svc  │  │Notification Svc │
    │ (Library Mgmt)  │  │ (File Organizer)│  │ (User Updates)  │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     Message Bus            │
                    │      (RabbitMQ)            │
                    └─────────────┬──────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
  ┌───────▼─────┐       ┌─────────▼─────┐       ┌─────────▼─────┐
  │ PostgreSQL  │       │     Redis     │       │    Neo4j      │
  │ (Metadata)  │       │   (Cache)     │       │ (Relationships)│
  └─────────────┘       └───────────────┘       └───────────────┘
```

### Key Architectural Principles

1. **Microservices Architecture**: Each service has a specific responsibility
2. **Event-Driven Communication**: Services communicate via message queues
3. **Database per Service**: Each service owns its data
4. **Horizontal Scalability**: Services can be scaled independently
5. **Fault Tolerance**: Services handle failures gracefully

## Core Services Deep Dive

### 1. Analysis Service 🎵
**Purpose**: AI-powered audio analysis and feature extraction

**What it does**:
- Extracts BPM (tempo) using multiple algorithms
- Detects musical key and scale
- Analyzes mood and genre characteristics
- Performs audio quality assessment

**Key Technologies**:
- **Essentia**: Audio analysis library
- **Librosa**: Audio processing
- **TensorFlow**: Machine learning models
- **NumPy**: Numerical computing

**When you'll work here**:
- Adding new audio analysis algorithms
- Improving accuracy of existing detectors
- Optimizing performance for large audio files
- Integrating new ML models

**Entry Points**:
- `services/analysis_service/src/main.py` - Service startup
- `services/analysis_service/src/bpm_detector.py` - BPM analysis
- `services/analysis_service/src/key_detector.py` - Key detection
- `services/analysis_service/src/mood_analyzer.py` - Mood analysis

### 2. Tracklist Service 📝
**Purpose**: Playlist and tracklist management with intelligent matching

**What it does**:
- Creates and manages tracklists/playlists
- Provides intelligent track matching and recommendations
- Handles track metadata and relationships
- Manages user preferences and history

**Key Technologies**:
- **FastAPI**: Web framework
- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation
- **Fuzzy matching algorithms**

**When you'll work here**:
- Building playlist recommendation engines
- Improving track matching algorithms
- Adding social features (sharing, collaboration)
- Integrating with external music services

**Entry Points**:
- `services/tracklist_service/src/main.py` - Service startup
- `services/tracklist_service/src/services/matching_service.py` - Track matching
- `services/tracklist_service/src/models/` - Database models
- `services/tracklist_service/src/api/` - REST endpoints

### 3. File Watcher 👀
**Purpose**: File system monitoring and change detection

**What it does**:
- Monitors directories for new, modified, or deleted files
- Triggers analysis workflows for new audio files
- Handles file system events efficiently
- Manages file processing queues

**Key Technologies**:
- **Watchdog**: File system event monitoring
- **asyncio**: Asynchronous processing
- **RabbitMQ**: Message queue integration

**When you'll work here**:
- Optimizing file monitoring performance
- Adding support for new file formats
- Implementing recursive directory scanning
- Handling edge cases (file moves, renames, etc.)

**Entry Points**:
- `services/file_watcher/src/main.py` - Service startup
- `services/file_watcher/src/watcher.py` - File monitoring logic
- `services/file_watcher/src/event_handler.py` - Event processing

### 4. Cataloging Service 📚
**Purpose**: Music library management and organization

**What it does**:
- Manages music library metadata
- Provides search and filtering capabilities
- Handles duplicate detection
- Manages file organization policies

### 5. File Rename Service ✏️
**Purpose**: Intelligent file organization and renaming

**What it does**:
- Renames files based on metadata
- Organizes files into directory structures
- Handles file conflicts and duplicates
- Maintains file integrity during moves

### 6. Notification Service 📢
**Purpose**: User notifications and system alerts

**What it does**:
- Sends processing completion notifications
- Handles error alerts and system status
- Manages user preferences for notifications
- Integrates with external notification services

## Data Architecture

### Database Strategy: Database per Service

Each service owns its data, promoting independence and scalability:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Analysis Service│  │Tracklist Service│  │File Watcher     │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │PostgreSQL   │ │  │ │PostgreSQL   │ │  │ │PostgreSQL   │ │
│ │- audio_files│ │  │ │- playlists  │ │  │ │- watch_dirs │ │
│ │- analysis   │ │  │ │- tracks     │ │  │ │- file_events│ │
│ │- features   │ │  │ │- matches    │ │  │ │- process_log│ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Shared Databases

#### Redis (Caching & Sessions)
- **Analysis results caching**: Avoid re-processing files
- **User sessions**: Web application state
- **Rate limiting**: API throttling
- **Temporary data**: Short-lived processing data

#### Neo4j (Relationships & Graph Data)
- **Music relationships**: Artist → Album → Track connections
- **User behavior**: Listening patterns and preferences
- **Recommendation graphs**: Similar tracks and users
- **Metadata relationships**: Genre hierarchies, collaborations

## Communication Patterns

### Message-Driven Architecture

Services communicate primarily through **RabbitMQ** message queues, not direct HTTP calls:

```
File Added → File Watcher → [Queue] → Analysis Service → [Queue] → Cataloging Service
                                  ↘                    ↙
                                   Notification Service
```

### Message Flow Example

1. **File Detection**: File Watcher detects new MP3 file
2. **Analysis Request**: Sends message to `audio.analysis.requests` queue
3. **Processing**: Analysis Service picks up message and processes audio
4. **Results**: Publishes results to `audio.analysis.completed` queue
5. **Cataloging**: Cataloging Service updates music library
6. **Notification**: User receives processing completion notification

### Queue Structure
```
# Analysis Workflows
audio.analysis.requests     → Analysis Service
audio.analysis.completed    → Cataloging Service
audio.analysis.failed       → Notification Service

# File Operations
file.operations.rename      → File Rename Service
file.operations.move        → File Rename Service
file.operations.completed   → File Watcher

# User Notifications
notifications.email         → Notification Service
notifications.push          → Notification Service
```

## Development Workflow Integration

### How Changes Flow Through the System

Understanding this helps you know where to make changes and how to test them:

#### Scenario: Adding New Audio Feature Detection

1. **Analysis Service**: Implement new feature extractor
   ```python
   # services/analysis_service/src/new_feature_detector.py
   def detect_energy_level(audio: np.ndarray) -> float:
       return calculate_rms_energy(audio)
   ```

2. **Message Schema**: Update analysis result format
   ```python
   # shared/core_types/analysis_types.py
   class AnalysisResult:
       bpm: float
       key: str
       energy_level: float  # New field
   ```

3. **Database Migration**: Add storage for new feature
   ```sql
   -- alembic migration
   ALTER TABLE audio_features ADD COLUMN energy_level FLOAT;
   ```

4. **API Updates**: Expose new feature in API
   ```python
   # services/tracklist_service/src/api/tracks.py
   @router.get("/tracks/{track_id}/features")
   def get_track_features(track_id: int):
       return {
           "bpm": track.bpm,
           "key": track.key,
           "energy_level": track.energy_level  # New field
       }
   ```

5. **Testing**: Add tests at each layer
   ```python
   # tests/unit/analysis_service/test_new_feature_detector.py
   def test_energy_detection():
       audio = generate_test_audio()
       energy = detect_energy_level(audio)
       assert 0.0 <= energy <= 1.0
   ```

## Key Design Patterns

### 1. Service Layer Pattern
Each service separates concerns into layers:
```
Controllers (API endpoints)
    ↓
Services (Business logic)
    ↓
Repositories (Data access)
    ↓
Models (Data structures)
```

### 2. Event Sourcing
Important state changes are recorded as events:
```python
# Instead of just updating state
track.bpm = 128

# We record what happened
publish_event("track.bpm.updated", {
    "track_id": track.id,
    "old_bpm": 120,
    "new_bpm": 128,
    "confidence": 0.95,
    "algorithm": "rhythm_extractor"
})
```

### 3. Circuit Breaker Pattern
Services handle external failures gracefully:
```python
@circuit_breaker(failure_threshold=5, timeout=30)
async def call_external_api():
    # If this fails 5 times, circuit opens
    # and fails fast for 30 seconds
    pass
```

## Common Development Patterns

### Error Handling
```python
# Services use structured error handling
try:
    result = analyze_audio(file_path)
except AudioProcessingError as e:
    logger.error("Analysis failed",
                file=file_path,
                error=str(e))

    # Publish failure event
    publish_error_event("analysis.failed", {
        "file_path": file_path,
        "error": str(e),
        "retry_count": retry_count
    })

    raise
```

### Configuration Management
```python
# All services use environment-based configuration
@dataclass
class AnalysisConfig:
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
    confidence_threshold: float = float(os.getenv("BPM_CONFIDENCE_THRESHOLD", "0.7"))
    enable_gpu: bool = os.getenv("ENABLE_GPU", "false").lower() == "true"
```

### Async Processing
```python
# Most I/O operations are async
async def process_audio_file(file_path: str):
    async with aiofiles.open(file_path, 'rb') as f:
        audio_data = await f.read()

    # CPU-intensive work in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, analyze_audio, audio_data
    )

    return result
```

## Where You'll Make Changes

### Common Development Scenarios

#### 1. **New Audio Analysis Algorithm**
- **Primary**: `services/analysis_service/`
- **Secondary**: Update message schemas, add tests
- **Impact**: Low (isolated to one service)

#### 2. **New API Endpoint**
- **Primary**: Relevant service's `src/api/` directory
- **Secondary**: Update OpenAPI specs, add tests
- **Impact**: Medium (affects client integrations)

#### 3. **Database Schema Changes**
- **Primary**: Service's `alembic/` migrations
- **Secondary**: Update models, add migration tests
- **Impact**: High (affects data persistence)

#### 4. **New Service Integration**
- **Primary**: Multiple services
- **Secondary**: Message queue setup, shared types
- **Impact**: High (affects system architecture)

### File Structure Navigation

```
tracktion/
├── services/                    # ← Most development happens here
│   ├── analysis_service/        # ← Audio processing algorithms
│   ├── tracklist_service/       # ← Playlist and matching logic
│   ├── file_watcher/            # ← File monitoring logic
│   └── ...
├── shared/                      # ← Common code and types
│   ├── database/                # ← Database connections
│   ├── messaging/               # ← Message queue utilities
│   └── core_types/              # ← Shared data structures
├── tests/                       # ← Test code
│   ├── unit/                    # ← Fast, isolated tests
│   ├── integration/             # ← Service integration tests
│   └── fixtures/                # ← Test data and helpers
└── docs/                        # ← Documentation (including this file!)
```

## Performance Characteristics

### Scaling Bottlenecks to Watch For

#### 1. **Audio Processing (CPU-bound)**
- **Problem**: BPM detection is computationally expensive
- **Solution**: Scale Analysis Service horizontally
- **Monitoring**: CPU usage, processing queue length

#### 2. **Database Queries (I/O-bound)**
- **Problem**: Complex track matching queries
- **Solution**: Add indexes, optimize queries, use read replicas
- **Monitoring**: Query performance, connection pool usage

#### 3. **Message Queue Processing**
- **Problem**: Queue backlog during high file ingestion
- **Solution**: Scale consumer services, partition queues
- **Monitoring**: Queue depth, message processing rate

### Typical Processing Times

| Operation | Typical Duration | Scaling Factor |
|-----------|-----------------|----------------|
| BPM Detection | 2-5 seconds | Audio file length |
| Key Detection | 1-3 seconds | Audio complexity |
| Mood Analysis | 3-8 seconds | Model complexity |
| File Organization | 0.1-0.5 seconds | File system speed |
| Track Matching | 0.01-0.1 seconds | Catalog size |

## Security Considerations

### Authentication & Authorization
- **JWT tokens**: For API authentication
- **Role-based access**: Admin, user, service roles
- **API keys**: For service-to-service communication
- **Rate limiting**: Prevent API abuse

### Data Protection
- **Sensitive data**: User credentials, API keys stored encrypted
- **Audio files**: Access controlled by user ownership
- **Metadata**: Public vs. private track information
- **Logs**: Sanitized to avoid leaking sensitive data

## Monitoring and Observability

### What We Monitor

#### Application Metrics
- **Processing queues**: Message backlog and throughput
- **Analysis accuracy**: Confidence scores and user feedback
- **API performance**: Response times and error rates
- **Resource usage**: CPU, memory, disk usage per service

#### Business Metrics
- **Files processed**: Daily/weekly processing volume
- **User activity**: Active users and engagement
- **Analysis quality**: User corrections and feedback
- **System health**: Service uptime and availability

### Logging Strategy

Each service uses **structured logging** with consistent fields:
```python
logger.info("Audio analysis started",
    service="analysis_service",
    operation="bmp_detection",
    file_id=file_id,
    file_size=file_size,
    user_id=user_id,
    trace_id=trace_id
)
```

## Getting Started as a Developer

### 1. **Start Small**
- Fix bugs in existing services
- Add simple features to familiar areas
- Write tests for uncovered code

### 2. **Understand Data Flow**
- Trace a file from upload to analysis completion
- Follow message flows through queues
- Understand how services communicate

### 3. **Read the Code**
- Start with service `main.py` files
- Look at API endpoint definitions
- Examine database models and migrations

### 4. **Set Up Development Environment**
- Follow `docs/development/getting-started.md`
- Run services locally and test interactions
- Use debugging tools to trace execution

### 5. **Contribute Gradually**
- Start with documentation improvements
- Add unit tests for existing code
- Implement small features or bug fixes
- Gradually take on larger architectural changes

## Common Pitfalls for New Developers

### 1. **Trying to Change Too Much**
- **Problem**: Making changes across multiple services without understanding dependencies
- **Solution**: Start with isolated changes within a single service

### 2. **Ignoring Message Semantics**
- **Problem**: Changing message formats breaks other services
- **Solution**: Use versioned message schemas and backward compatibility

### 3. **Not Testing Audio Processing**
- **Problem**: Audio algorithms are complex and edge cases are common
- **Solution**: Always test with real audio files and edge cases

### 4. **Assuming Synchronous Processing**
- **Problem**: Expecting immediate results from async operations
- **Solution**: Embrace event-driven patterns and async processing

### 5. **Not Understanding Service Boundaries**
- **Problem**: Adding functionality to the wrong service
- **Solution**: Follow single responsibility principle for services

## Next Steps

Now that you understand the architecture:

1. **Set up your environment**: Follow the getting started guide
2. **Pick your first task**: Look for "good first issue" labels
3. **Understand a specific service**: Deep dive into one service's code
4. **Make your first change**: Start with a small, isolated improvement
5. **Join the team discussions**: Participate in architecture decisions

Remember: **Every expert was once a beginner**. Don't hesitate to ask questions, and use this document as a reference as you grow into the Tracktion architecture!

## Additional Resources

- **System Architecture**: `docs/architecture/system-overview.md`
- **API Documentation**: `docs/api/`
- **Development Setup**: `docs/development/getting-started.md`
- **Common Tasks**: `docs/development/common-tasks-playbook.md`
- **Code Style Guide**: `docs/development/code-style-guide.md`

Welcome to the team! 🎵
