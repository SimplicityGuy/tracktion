# Common Tasks Playbook

## Overview

This playbook provides step-by-step instructions for common development tasks in the Tracktion project. Each task includes commands, troubleshooting tips, and best practices to help new developers become productive quickly.

## Daily Development Tasks

### Setting Up Development Environment

#### Initial Setup (First Time)
```bash
# 1. Clone repository
git clone https://github.com/your-org/tracktion.git
cd tracktion

# 2. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Set up Python environment and dependencies
uv sync

# 4. Install pre-commit hooks (MANDATORY)
uv run pre-commit install

# 5. Create environment file
cp .env.example .env
# Edit .env with your configuration

# 6. Start external services
docker-compose up -d

# 7. Run database migrations
uv run alembic upgrade head

# 8. Verify installation
uv run pytest tests/unit/test_basic_functionality.py -v
```

#### Daily Startup
```bash
# 1. Update repository
git pull origin develop

# 2. Update dependencies if needed
uv sync

# 3. Start external services
docker-compose up -d

# 4. Check service health
docker-compose ps
docker-compose logs --tail=10

# 5. Run basic tests
uv run pytest tests/unit/ -x --tb=short
```

### Working with Services

#### Starting Individual Services
```bash
# Analysis Service (audio processing)
cd services/analysis_service
uv run python src/main.py

# Tracklist Service (playlist management)
cd services/tracklist_service
uv run python src/main.py

# File Watcher (file system monitoring)
cd services/file_watcher
uv run python src/main.py

# All services in parallel (development mode)
./scripts/start-dev-services.sh
```

#### Service Health Checks
```bash
# Check service endpoints
curl http://localhost:8001/health  # Analysis Service
curl http://localhost:8002/health  # Tracklist Service
curl http://localhost:8003/health  # File Watcher

# Check database connections
uv run python -c "
from shared.database import get_engine
engine = get_engine()
with engine.connect() as conn:
    result = conn.execute('SELECT 1')
    print('Database OK')
"

# Check RabbitMQ
docker-compose exec rabbitmq rabbitmqctl status
docker-compose exec rabbitmq rabbitmqctl list_queues
```

## Development Workflow Tasks

### Creating a New Feature

#### 1. Planning Phase
```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Document your feature (optional but recommended)
echo "# Feature: Your Feature Name

## Description
Brief description of the feature

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
" > docs/features/your-feature.md
```

#### 2. Development Phase
```bash
# Make your changes
# Edit code files...

# Run tests frequently
uv run pytest tests/unit/relevant_service/ -v

# Check code quality before committing
uv run pre-commit run --all-files

# Commit your changes
git add .
git commit -m "feat: add your feature

Detailed description of what the feature does
and why it's needed."
```

#### 3. Integration Phase
```bash
# Run full test suite
uv run pytest tests/ -v

# Update documentation if needed
# Edit relevant README files, API docs, etc.

# Push feature branch
git push -u origin feature/your-feature-name

# Create pull request (see PR template)
```

### Working with Audio Processing

#### Adding New Audio Analysis Algorithm
```bash
# 1. Create algorithm module
touch services/analysis_service/src/new_algorithm.py

# 2. Basic algorithm structure
cat > services/analysis_service/src/new_algorithm.py << 'EOF'
"""
New audio analysis algorithm implementation.
"""

import numpy as np
import essentia.standard as es
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class NewAlgorithm:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio and return results.

        Args:
            audio: Audio data as numpy array

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting new algorithm analysis",
                   audio_length=len(audio))

        try:
            # Your algorithm implementation here
            result = {"success": True, "value": 0.0}

            logger.info("New algorithm analysis completed",
                       result=result)
            return result

        except Exception as e:
            logger.error("New algorithm analysis failed",
                        error=str(e), exc_info=True)
            raise
EOF

# 3. Create tests
mkdir -p tests/unit/analysis_service/
cat > tests/unit/analysis_service/test_new_algorithm.py << 'EOF'
"""Tests for new audio analysis algorithm."""

import numpy as np
import pytest
from services.analysis_service.src.new_algorithm import NewAlgorithm

class TestNewAlgorithm:
    def setup_method(self):
        self.algorithm = NewAlgorithm()

    def test_analyze_basic(self):
        # Create test audio
        audio = np.random.rand(44100).astype(np.float32)

        result = self.algorithm.analyze(audio)

        assert "success" in result
        assert result["success"] is True
EOF

# 4. Test the algorithm
uv run pytest tests/unit/analysis_service/test_new_algorithm.py -v
```

#### Testing with Real Audio Files
```bash
# Create test audio directory
mkdir -p tests/fixtures/audio/

# Add test audio files (small, royalty-free samples)
# Place files like: tests/fixtures/audio/test_120bpm.wav

# Create integration test
cat > tests/integration/test_audio_processing.py << 'EOF'
"""Integration tests with real audio files."""

import pytest
from pathlib import Path
from services.analysis_service.src.new_algorithm import NewAlgorithm

class TestAudioIntegration:
    @pytest.fixture
    def audio_files(self):
        """Get available test audio files."""
        audio_dir = Path("tests/fixtures/audio")
        if not audio_dir.exists():
            pytest.skip("No audio test files available")

        return list(audio_dir.glob("*.wav"))

    def test_real_audio_files(self, audio_files):
        algorithm = NewAlgorithm()

        for audio_file in audio_files:
            # Load audio file with essentia
            loader = es.MonoLoader(filename=str(audio_file))
            audio = loader()

            # Test algorithm
            result = algorithm.analyze(audio)
            assert result["success"] is True
EOF
```

### Database Operations

#### Creating Database Migrations
```bash
# Generate migration for new table/changes
uv run alembic revision --autogenerate -m "Add new table for feature X"

# Review generated migration file
ls alembic/versions/
# Edit the newest file to verify changes

# Apply migration
uv run alembic upgrade head

# Rollback if needed
uv run alembic downgrade -1
```

#### Adding New Database Models
```bash
# 1. Create model in appropriate service
cat > services/your_service/src/models/new_model.py << 'EOF'
"""Database model for new feature."""

from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class NewModel(Base):
    __tablename__ = 'new_table'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<NewModel(id={self.id}, name='{self.name}')>"
EOF

# 2. Add to main models init
echo "from .new_model import NewModel" >> services/your_service/src/models/__init__.py

# 3. Generate and apply migration
uv run alembic revision --autogenerate -m "Add NewModel table"
uv run alembic upgrade head
```

#### Database Queries and Operations
```bash
# Interactive database session
uv run python -c "
from shared.database import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    # Run your queries
    result = conn.execute(text('SELECT COUNT(*) FROM tracks'))
    print('Total tracks:', result.scalar())

    # Example complex query
    result = conn.execute(text('''
        SELECT service_name, COUNT(*) as count
        FROM analysis_results
        GROUP BY service_name
    '''))

    for row in result:
        print(f'{row.service_name}: {row.count}')
"
```

### Message Queue Operations

#### Working with RabbitMQ
```bash
# Check queue status
docker-compose exec rabbitmq rabbitmqctl list_queues name messages consumers

# Purge queue (development only)
docker-compose exec rabbitmq rabbitmqctl purge_queue audio.analysis.requests

# Send test message
uv run python -c "
import json
import pika

connection = pika.BlockingConnection(
    pika.URLParameters('amqp://guest:guest@localhost:5672/')
)
channel = connection.channel()

test_message = {
    'file_path': '/path/to/test.mp3',
    'recording_id': 'test-123',
    'analysis_type': 'full'
}

channel.basic_publish(
    exchange='',
    routing_key='audio.analysis.requests',
    body=json.dumps(test_message)
)

print('Test message sent')
connection.close()
"

# Monitor message consumption
docker-compose logs -f rabbitmq
```

### Testing Tasks

#### Running Tests
```bash
# Quick unit tests
uv run pytest tests/unit/ -v

# Specific service tests
uv run pytest tests/unit/analysis_service/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Tests with coverage
uv run pytest --cov=services --cov-report=html tests/

# Performance benchmarks
uv run pytest --benchmark-only tests/

# Parallel test execution (faster)
uv run pytest -n auto tests/unit/
```

#### Debugging Test Failures
```bash
# Run failed tests with detailed output
uv run pytest --tb=long --capture=no -v tests/test_file.py::test_function

# Run single test with debugging
uv run pytest -s --pdb tests/test_file.py::test_function

# Run tests with logging
uv run pytest --log-cli-level=DEBUG tests/

# Re-run only failed tests
uv run pytest --lf -v
```

### Code Quality Tasks

#### Pre-commit Hooks (MANDATORY)
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff
uv run pre-commit run mypy

# Update hook versions
uv run pre-commit autoupdate

# Skip hook temporarily (emergency only - NOT recommended)
SKIP=mypy git commit -m "emergency fix"
```

#### Manual Code Quality Checks
```bash
# Linting with ruff
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues

# Type checking with mypy
uv run mypy services/
uv run mypy services/analysis_service/src/

# Format code
uv run ruff format .

# Check specific file
uv run ruff check services/analysis_service/src/bpm_detector.py
uv run mypy services/analysis_service/src/bpm_detector.py
```

## Debugging and Troubleshooting Tasks

### Service Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs service_name

# Check port conflicts
lsof -i :8001  # Analysis service port
lsof -i :8002  # Tracklist service port

# Check environment variables
uv run python -c "
import os
required_vars = ['POSTGRES_URL', 'REDIS_URL', 'RABBITMQ_URL']
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f'{var}: {value[:20]}...')
    else:
        print(f'{var}: NOT SET')
"

# Test database connection
uv run python -c "
from shared.database import get_engine
try:
    engine = get_engine()
    with engine.connect():
        print('Database connection OK')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

#### Performance Issues
```bash
# Profile service startup
uv run python -m cProfile services/analysis_service/src/main.py

# Monitor resource usage
htop  # or top
docker stats

# Check database query performance
uv run python -c "
import time
from shared.database import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    start = time.time()
    result = conn.execute(text('SELECT COUNT(*) FROM tracks'))
    duration = time.time() - start
    print(f'Query took {duration:.3f}s, result: {result.scalar()}')
"
```

### Audio Processing Issues

#### Audio File Problems
```bash
# Test audio file loading
uv run python -c "
import essentia.standard as es
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else 'test.mp3'

try:
    loader = es.MonoLoader(filename=file_path)
    audio = loader()

    print(f'File: {file_path}')
    print(f'Duration: {len(audio) / 44100:.2f}s')
    print(f'Samples: {len(audio)}')
    print(f'Max amplitude: {max(abs(audio)):.3f}')

    if len(audio) == 0:
        print('ERROR: Empty audio file')
    elif max(abs(audio)) < 0.001:
        print('WARNING: Very quiet audio')

except Exception as e:
    print(f'ERROR loading audio: {e}')
" /path/to/audio/file.mp3
```

#### BPM Detection Issues
```bash
# Debug BPM detection
uv run python -c "
from services.analysis_service.src.bmp_detector import BPMDetector
import sys

detector = BPMDetector()
file_path = sys.argv[1] if len(sys.argv) > 1 else 'test.mp3'

try:
    result = detector.detect_bpm_with_confidence(file_path)
    print(f'BPM: {result[\"bmp\"]}')
    print(f'Confidence: {result[\"confidence\"]:.3f}')
    print(f'Needs review: {result[\"needs_review\"]}')
except Exception as e:
    print(f'BPM detection failed: {e}')
" /path/to/audio/file.mp3
```

## Maintenance Tasks

### Dependency Management

#### Updating Dependencies
```bash
# Check for outdated packages
uv tree --outdated

# Update specific dependency
uv add "package-name>=new.version"

# Update all dependencies (careful!)
uv lock --upgrade

# Sync environment with updated lock file
uv sync

# Test after updates
uv run pytest tests/unit/ -x
```

#### Security Updates
```bash
# Check for security vulnerabilities
uv audit

# Update specific vulnerable package
uv add "vulnerable-package>=secure.version"

# Generate security report
uv audit --format json > security-report.json
```

### Database Maintenance

#### Database Cleanup
```bash
# Clean up old analysis results (example)
uv run python -c "
from shared.database import get_engine
from sqlalchemy import text
from datetime import datetime, timedelta

engine = get_engine()
cutoff_date = datetime.utcnow() - timedelta(days=30)

with engine.connect() as conn:
    result = conn.execute(
        text('DELETE FROM analysis_results WHERE created_at < :cutoff'),
        {'cutoff': cutoff_date}
    )
    print(f'Deleted {result.rowcount} old analysis results')
    conn.commit()
"
```

#### Database Backups (Development)
```bash
# Backup database
docker-compose exec postgres pg_dump -U tracktion tracktion_dev > backup.sql

# Restore database
docker-compose exec -T postgres psql -U tracktion tracktion_dev < backup.sql
```

### Log Management

#### Viewing Logs
```bash
# Service logs
tail -f logs/analysis_service.log
tail -f logs/tracklist_service.log

# Docker service logs
docker-compose logs -f analysis_service
docker-compose logs --tail=100 rabbitmq

# System logs (Linux)
journalctl -u tracktion-analysis-service -f
```

#### Log Analysis
```bash
# Find errors in logs
grep -i error logs/*.log
grep -i exception logs/*.log

# Count log levels
awk '{print $3}' logs/service.log | sort | uniq -c

# Recent errors
grep -i error logs/service.log | tail -20
```

## Deployment Preparation Tasks

### Pre-deployment Checklist
```bash
# 1. Run full test suite
uv run pytest tests/ -v --cov=services

# 2. Check code quality
uv run pre-commit run --all-files

# 3. Verify migrations
uv run alembic check
uv run alembic current
uv run alembic heads

# 4. Build documentation
# If using MkDocs or similar
mkdocs build

# 5. Security check
uv audit

# 6. Performance benchmarks
uv run pytest --benchmark-only tests/benchmarks/
```

### Environment Configuration
```bash
# Generate production environment template
cat > .env.production << 'EOF'
# Database
POSTGRES_URL=postgresql://user:pass@prod-db:5432/tracktion
REDIS_URL=redis://prod-redis:6379/0
NEO4J_URL=bolt://prod-neo4j:7687

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
SECRET_KEY=your-production-secret-key

# Service URLs
ANALYSIS_SERVICE_URL=https://analysis.your-domain.com
TRACKLIST_SERVICE_URL=https://tracklist.your-domain.com
EOF

# Validate production config
uv run python -c "
import os
os.environ.update({line.split('=')[0]: line.split('=')[1]
                  for line in open('.env.production').read().strip().split('\n')
                  if '=' in line})

# Test configuration
from shared.config import get_config
config = get_config()
print('Config validation passed')
"
```

## Emergency Procedures

### Service Recovery
```bash
# Emergency service restart
docker-compose restart analysis_service

# Check service health after restart
curl -f http://localhost:8001/health || echo "Service unhealthy"

# Emergency rollback (if needed)
git checkout previous-working-commit
docker-compose down
docker-compose up -d
```

### Database Recovery
```bash
# Emergency database connection reset
docker-compose restart postgres

# Check database integrity
docker-compose exec postgres psql -U tracktion -c "
SELECT schemaname, tablename, n_tup_ins as inserts, n_tup_upd as updates
FROM pg_stat_user_tables ORDER BY n_tup_ins DESC LIMIT 10;
"
```

### Quick Fixes
```bash
# Clear all caches
redis-cli FLUSHALL
rm -rf __pycache__/ .pytest_cache/ .mypy_cache/

# Reset development environment
docker-compose down -v  # WARNING: destroys data
docker-compose up -d
uv run alembic upgrade head

# Emergency log level change
export LOG_LEVEL=DEBUG  # Or add to .env
```

## Learning and Development Tasks

### Code Exploration
```bash
# Understand service structure
find services/ -name "*.py" | head -20
tree services/analysis_service/src/

# Find examples of patterns
grep -r "class.*Exception" services/
grep -r "@router\." services/
grep -r "async def" services/ | head -10

# Understand dependencies
uv tree
uv show package-name
```

### Contributing Tasks
```bash
# Find good first issues (if using GitHub)
gh issue list --label "good first issue"

# Review recent changes
git log --oneline -10
git show HEAD

# Find areas needing help
grep -r "TODO" services/
grep -r "FIXME" services/
```

This playbook should help new developers quickly become productive with common tasks. Each section provides practical, executable commands with context and troubleshooting guidance.
