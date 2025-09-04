# Getting Started Guide for New Developers

## Welcome to Tracktion!

This guide will help you get up and running with the Tracktion project as quickly as possible. Tracktion is an automated music library management system built with Python microservices.

## Quick Overview

Tracktion is a comprehensive music management system that:
- Automatically analyzes audio files for BPM, key, and mood
- Manages tracklists and provides intelligent matching
- Watches file systems for changes
- Provides catalog management and file organization
- Offers notification services and user management

## Prerequisites

Before you begin, ensure you have:

### Required Software
- **Python 3.11 or higher** - The project uses modern Python features
- **uv** - Python package and project manager (replaces pip/virtualenv)
- **Git** - Version control
- **Docker & Docker Compose** - For running databases and services
- **Pre-commit** - Code quality and consistency (installed via uv)

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for audio processing)
- **Storage**: At least 10GB free space
- **OS**: Linux, macOS, or Windows (Linux/macOS preferred)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/tracktion.git
cd tracktion
```

### 2. Install uv (if not already installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 3. Set Up the Development Environment
```bash
# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks (CRITICAL - zero tolerance policy)
uv run pre-commit install

# Verify pre-commit is working
uv run pre-commit run --all-files
```

### 4. Set Up Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables (see Configuration section below)
```

### 5. Start External Services
```bash
# Start databases and external services
docker-compose up -d postgres redis neo4j rabbitmq

# Verify services are running
docker-compose ps
```

### 6. Run Database Migrations
```bash
# Run Alembic migrations for PostgreSQL
uv run alembic upgrade head

# Verify database connection
uv run python -c "from shared.database import get_engine; print('Database OK')"
```

### 7. Verify Installation
```bash
# Run quick tests to ensure everything works
uv run pytest tests/unit/test_basic_functionality.py -v

# Start a service to test
cd services/analysis_service
uv run python src/main.py
```

## Essential Configuration

### Environment Variables
Create a `.env` file in the project root with these essential settings:

```bash
# Database Configuration
POSTGRES_URL=postgresql://tracktion:password@localhost:5432/tracktion
REDIS_URL=redis://localhost:6379/0
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# RabbitMQ Configuration
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# Application Configuration
LOG_LEVEL=INFO
DEBUG=true

# Audio Processing Configuration
AUDIO_SAMPLE_RATE=44100
MAX_FILE_SIZE=100MB
TEMP_DIR=/tmp/tracktion

# Service Ports (for development)
ANALYSIS_SERVICE_PORT=8001
TRACKLIST_SERVICE_PORT=8002
FILE_WATCHER_PORT=8003
```

### Service-Specific Configuration
Each service has its own configuration requirements detailed in their README files:
- `services/analysis_service/README.md`
- `services/tracklist_service/README.md`
- `services/file_watcher/README.md`

## Development Workflow

### Daily Development Commands

```bash
# Always use uv run for Python commands
uv run pytest                    # Run tests
uv run python src/main.py        # Start a service
uv run pre-commit run --all-files # Check code quality

# Never use these commands directly:
# python, python3, pip, pytest (without uv run)
```

### Code Quality (Zero Tolerance Policy)

**CRITICAL**: The project maintains a zero-tolerance policy for code quality violations.

```bash
# Before ANY commit, ALWAYS run:
uv run pre-commit run --all-files

# Fix ALL issues reported by pre-commit
# Re-run until ALL checks pass
# Only then commit your changes

# NEVER use these flags:
# --no-verify, SKIP environment variable
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/           # Unit tests only
uv run pytest tests/integration/    # Integration tests
uv run pytest -m "not slow"        # Skip slow tests

# Run tests with coverage
uv run pytest --cov=services --cov-report=html
```

## Project Structure Understanding

```
tracktion/
├── services/                    # Microservices
│   ├── analysis_service/        # Audio analysis (BPM, key, mood)
│   ├── tracklist_service/       # Tracklist management
│   ├── file_watcher/            # File system monitoring
│   ├── cataloging_service/      # Music catalog management
│   ├── file_rename_service/     # File organization
│   └── notification_service/    # User notifications
├── shared/                      # Shared libraries
│   ├── database/                # Database connections
│   ├── messaging/               # RabbitMQ utilities
│   └── core_types/              # Common type definitions
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data
└── docs/                        # Documentation
    ├── architecture/            # System architecture
    ├── api/                     # API documentation
    ├── development/             # Developer guides
    └── operations/              # Operational guides
```

## Your First Contribution

### 1. Pick a Good First Issue
Look for issues labeled:
- `good-first-issue`
- `documentation`
- `testing`
- `refactoring`

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Follow existing code patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all pre-commit checks pass

### 4. Submit a Pull Request
- Use clear, descriptive commit messages
- Reference related issues
- Provide detailed PR description
- Ensure CI/CD passes

## Common Development Tasks

### Adding a New Service
1. Copy an existing service structure
2. Update configuration and dependencies
3. Implement service-specific logic
4. Add comprehensive tests
5. Update documentation
6. Add to Docker Compose

### Working with Audio Files
```python
# Always use the configured sample rate
from shared.config import get_config
config = get_config()
sample_rate = config.audio_sample_rate

# Use proper error handling for audio processing
try:
    audio_data = load_audio_file(filepath)
    result = process_audio(audio_data)
except AudioProcessingError as e:
    logger.error(f"Audio processing failed: {e}")
    return None
```

### Database Operations
```python
# Use async database operations
from shared.database import get_async_engine

async def get_track_info(track_id: int):
    async with get_async_engine().begin() as conn:
        result = await conn.execute(
            text("SELECT * FROM tracks WHERE id = :id"),
            {"id": track_id}
        )
        return result.fetchone()
```

## Getting Help

### Documentation Resources
- **Architecture Overview**: `docs/architecture/system-overview.md`
- **API Documentation**: `docs/api/`
- **Configuration Guide**: `docs/setup/configuration-guide.md`
- **Testing Guide**: `docs/development/testing-guide.md`

### Team Communication
- Use GitHub Issues for bugs and feature requests
- Tag team members for code reviews
- Check existing documentation before asking questions
- Include error logs and reproduction steps when reporting issues

### Debugging Resources
- **Logging**: All services use structured logging with configurable levels
- **Monitoring**: Local development includes basic monitoring setup
- **Testing**: Comprehensive test suite with fixtures and mocks

## Common Pitfalls to Avoid

### 1. Command Usage
```bash
# ❌ Wrong - never use these
python src/main.py
pip install package
pytest tests/

# ✅ Correct - always use uv run
uv run python src/main.py
uv add package
uv run pytest tests/
```

### 2. Code Quality
```bash
# ❌ Wrong - skipping quality checks
git commit --no-verify

# ✅ Correct - always validate
uv run pre-commit run --all-files
git commit
```

### 3. Import Patterns
```python
# ❌ Wrong - conditional imports
try:
    import essentia
except ImportError:
    essentia = None

# ✅ Correct - direct imports with proper dependency management
import essentia  # Listed in pyproject.toml
```

### 4. Configuration
```python
# ❌ Wrong - hardcoded values
sample_rate = 44100

# ✅ Correct - use configuration
from shared.config import get_config
sample_rate = get_config().audio_sample_rate
```

## Next Steps

Once you're set up:

1. **Read the Architecture Guide**: `docs/architecture/system-overview.md`
2. **Review Coding Standards**: `docs/development/code-style-guide.md`
3. **Understand Testing**: `docs/development/testing-guide.md`
4. **Learn Git Workflow**: `docs/development/git-workflow.md` (coming next)
5. **Review PR Process**: `docs/development/pr-review-process.md` (coming next)

Welcome to the team! We're excited to have you contribute to Tracktion.
