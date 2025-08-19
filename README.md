# Tracktion

Automated music library management system with intelligent cataloging and analysis.

## 📊 Project Status

**Current Phase**: MVP Development
**Completion**: 66% (2 of 3 epics complete)
**Latest Update**: 2025-08-19

### Completed Features ✅
- **Infrastructure**: Docker containerization, multi-database setup, message queue
- **Audio Analysis**: BPM detection, key detection, mood analysis, genre classification
- **Metadata Extraction**: Complete support for MP3, FLAC, WAV, M4A formats
- **Caching & Performance**: Redis integration, parallel processing, optimized pipelines

### In Progress 🚧
- **Epic 3**: Tracklist Management (0% - Not Started)
- **File Renaming**: Pattern-based automatic renaming

### Documentation
- [📋 Project Status Report](docs/PROJECT_STATUS_REPORT.md) - Detailed progress and metrics
- [📖 Product Requirements](docs/prd.md) - Complete product specification
- [🏗️ Architecture Document](docs/architecture.md) - Technical architecture

## Overview

Tracktion is a microservices-based application designed to automatically catalog, analyze, and manage digital music collections. It uses advanced metadata extraction, graph-based analysis, and external data integration to organize your music library.

## Architecture

- **Microservices Architecture**: Decoupled services communicating via RabbitMQ
- **Polyglot Persistence**: PostgreSQL for relational data, Neo4j for graph analysis, Redis for caching
- **Containerized Deployment**: Docker-based deployment with docker-compose orchestration
- **Event-Driven Processing**: Asynchronous message-based workflows

## Prerequisites

- Docker and Docker Compose
- Python 3.13+ (for local development)
- uv (Python package manager)

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd tracktion
```

2. Copy the environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start all services:
```bash
docker-compose up -d
```

4. Verify services are running:
```bash
docker-compose ps
```

5. Check logs:
```bash
docker-compose logs -f
```

## Development Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install development dependencies:
```bash
uv pip install -e . --all-extras
```

3. Install pre-commit hooks:
```bash
uv run pre-commit install
```

4. Set up databases (see Database Setup section below)

5. Run tests:
```bash
uv run pytest
```

## Development Standards

**IMPORTANT**: This project enforces strict quality standards. Please review:
- [Coding Standards](docs/architecture/coding-standards.md) - Mandatory coding conventions and pre-commit requirements
- [Development Workflow](docs/development/development-workflow.md) - Required workflow for implementing features
- [Test Strategy](docs/architecture/test-strategy-and-standards.md) - Testing requirements and standards

### Quality Requirements
- ✅ All pre-commit hooks MUST pass before committing
- ✅ All unit tests MUST pass after each task
- ✅ All integration tests MUST pass before marking a story as done
- ✅ Minimum 80% code coverage for new code
- ✅ Type hints required for all functions

## Services

- **file_watcher**: Monitors directories for new audio files
- **cataloging_service**: Catalogs files in PostgreSQL
- **analysis_service**: Comprehensive audio analysis and metadata extraction
  - **Format Support**: MP3, FLAC, WAV, M4A with complete metadata extraction
  - **BPM Detection**: Multi-algorithm tempo detection with temporal analysis
  - **Musical Key Detection**: Dual-algorithm validation (major/minor with confidence)
  - **Mood Analysis**: TensorFlow-based mood dimension scoring
  - **Genre Classification**: Discogs EffNet models (15 categories)
  - **Additional Features**: Danceability, energy, valence, voice/instrumental classification
  - **Performance**: Redis caching, parallel processing, <1.5s full analysis
- **tracklist_service**: Retrieves tracklists from external sources

## Project Structure

```
tracktion/
├── services/           # Microservices
├── shared/            # Shared libraries and utilities
├── infrastructure/    # Docker and deployment configuration
├── tests/            # Test suites
└── docs/             # Documentation
```

## Features

### 🎵 BPM Detection (Story 2.3)

Advanced tempo detection system with high accuracy and performance optimization:

#### Core Capabilities
- **Multi-Algorithm Detection**: RhythmExtractor2013 (primary) with PercivalBpmEstimator fallback
- **Temporal Analysis**: Time-based analysis for variable tempo tracks with stability scoring
- **Confidence Scoring**: Normalized confidence values (0-1) with consensus validation
- **Format Support**: MP3, WAV, FLAC, M4A, OGG, WMA, AAC

#### Performance Features
- **Redis Caching**: Intelligent caching with versioned keys and TTL management
- **Streaming Support**: Memory-efficient processing for large files (>50MB)
- **Parallel Processing**: Configurable worker threads for batch processing
- **Memory Management**: Per-process memory limits with monitoring

#### Testing
- **Integration Tests**: Comprehensive end-to-end pipeline validation
- **Synthetic Audio**: Generated test files with known BPM characteristics
- **Edge Case Handling**: Silence, noise, corrupted files, variable tempo

#### Configuration
```bash
# Quick setup
export TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.7
export TRACKTION_CACHE_REDIS_HOST=localhost
export TRACKTION_PERFORMANCE_PARALLEL_WORKERS=4

# Run BPM detection tests
uv run python tests/run_integration_tests.py
```

#### Documentation
- [📖 BPM Detection Guide](docs/stories/story-2.3-bpm-detection-documentation.md)
- [🔧 Configuration Reference](docs/configuration/bpm-detection-config.md)
- [🚀 API Documentation](docs/api/bpm-detection-api.md)

### 🎹 Musical Key Detection (Story 2.4)

Advanced key detection with dual-algorithm validation for high accuracy:

#### Core Capabilities
- **Dual Algorithm Validation**: Primary KeyExtractor with HPCP-based validation
- **Confidence Scoring**: Agreement-based confidence boosting (up to 1.2x)
- **Scale Detection**: Major and minor scale identification
- **Alternative Key Suggestions**: When algorithms disagree

#### Performance
- **Processing Time**: ~300ms per track
- **Accuracy**: >80% target accuracy
- **Test Coverage**: 97% code coverage

### 🎭 Mood & Genre Analysis (Story 2.4)

Deep learning-based mood analysis and genre classification:

#### Mood Analysis
- **Mood Dimensions**: Happy, sad, aggressive, relaxed, acoustic, electronic, party
- **Danceability Score**: 0-1 scale for mix compatibility
- **Energy & Valence**: Musical energy and positivity metrics
- **Voice/Instrumental**: Classification with confidence scores

#### Genre Classification
- **15 Genre Categories**: Blues, Classical, Electronic, Folk, Hip Hop, Jazz, Latin, Pop, Reggae, Rock, and more
- **Ensemble Voting**: Multiple models for improved accuracy
- **Confidence Scores**: Per-genre confidence for reliable classification

#### Technical Implementation
- **TensorFlow Models**: Pre-trained models from Essentia repository
- **Model Manager**: Lazy loading and caching for optimal performance
- **Processing Time**: ~800ms for complete mood/genre analysis
- **Test Coverage**: 90% for mood analyzer, 92% for model manager

## Database Setup

### PostgreSQL Database

The system uses PostgreSQL 17 for storing recording metadata and file information. Database migrations are managed using Alembic.

1. Ensure PostgreSQL is running via Docker:
```bash
docker-compose -f infrastructure/docker-compose.yaml up -d postgres
```

2. Run database migrations:
```bash
uv run alembic upgrade head
```

3. To create a new migration after model changes:
```bash
uv run alembic revision --autogenerate -m "Description of changes"
```

4. To rollback migrations:
```bash
uv run alembic downgrade -1  # Rollback one migration
uv run alembic downgrade base  # Rollback all migrations
```

### Neo4j Database

The system uses Neo4j for graph-based relationship analysis between recordings, metadata, and tracklists.

1. Ensure Neo4j is running via Docker:
```bash
docker-compose -f infrastructure/docker-compose.yaml up -d neo4j
```

2. Access Neo4j Browser (for debugging):
   - URL: http://localhost:7474
   - Username: neo4j
   - Password: (check your .env file)

3. Initialize Neo4j constraints and indexes:
```bash
uv run python scripts/initialize_neo4j.py
```

### Database Models

The system uses three core models:

- **Recording**: Represents audio files with paths, hashes, and timestamps
- **Metadata**: Key-value pairs associated with recordings (artist, album, BPM, etc.)
- **Tracklist**: Track information for mixes and compilations stored as JSONB

### Environment Variables

Required database environment variables in `.env`:

```bash
# PostgreSQL
DATABASE_URL=postgresql://tracktion:tracktion@localhost:5432/tracktion

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit
```

Run integration tests:
```bash
uv run pytest tests/integration
```

Run with coverage report:
```bash
uv run pytest --cov=shared.core_types.src --cov-report=term-missing
```

## Contributing

1. Follow the coding standards defined in `docs/architecture/coding-standards.md`
2. Ensure all tests pass before submitting changes
3. Use pre-commit hooks to maintain code quality

## License

[License information to be added]
