# Development Environment Setup

## Overview

This guide provides detailed instructions for setting up a complete development environment for Tracktion. It covers everything from system requirements to advanced development tools configuration.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended for audio processing)
- **RAM**: Minimum 8GB, 16GB recommended (audio analysis is memory-intensive)
- **Storage**: At least 20GB free space (10GB for dependencies, 10GB for data)
- **Audio**: Sound card or audio interface (for audio testing)

### Operating System Support
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+ (recommended)
- **macOS**: 10.15+ with Xcode Command Line Tools
- **Windows**: Windows 10+ with WSL2 (recommended) or native PowerShell

## Core Dependencies Installation

### 1. Python 3.11+ Installation

#### Linux (Ubuntu/Debian)
```bash
# Add deadsnakes PPA for latest Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install system dependencies
sudo apt install build-essential libffi-dev libssl-dev \
    libasound2-dev libsndfile1-dev ffmpeg
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.11 ffmpeg

# Using pyenv (alternative)
pyenv install 3.11.7
pyenv global 3.11.7
```

#### Windows
```powershell
# Using Chocolatey (recommended)
choco install python311 ffmpeg

# Or download from python.org and install FFmpeg manually
```

### 2. uv Installation and Configuration

uv is the modern Python package manager that replaces pip, virtualenv, and more.

#### Installation
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

#### Configuration
```bash
# Configure uv for optimal performance
uv config set global.index-strategy unsafe-first-match
uv config set global.resolution-strategy highest

# Verify installation
uv --version
uv python list  # Should show Python 3.11+
```

### 3. Docker and Docker Compose

#### Linux
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose (if not included)
sudo apt install docker-compose-plugin
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from docker.com
```

#### Windows
```powershell
# Install Docker Desktop
choco install docker-desktop

# Or download from docker.com
```

### 4. Git Configuration

```bash
# Install Git
sudo apt install git  # Linux
brew install git       # macOS
choco install git      # Windows

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase true
git config --global core.autocrlf input  # Linux/macOS
git config --global core.autocrlf true   # Windows
```

## Project Setup

### 1. Repository Clone and Initialization
```bash
# Clone the repository
git clone https://github.com/your-org/tracktion.git
cd tracktion

# Initialize the development environment
uv sync --dev  # Install all dependencies including development tools

# Verify Python environment
uv run python --version  # Should be 3.11+
```

### 2. Pre-commit Hooks Installation
```bash
# Install pre-commit hooks (MANDATORY)
uv run pre-commit install

# Install commit message hooks
uv run pre-commit install --hook-type commit-msg

# Test installation
uv run pre-commit run --all-files
```

### 3. Environment Configuration

#### Create Environment File
```bash
# Copy the example environment file
cp .env.example .env

# Edit the environment file
nano .env  # or your preferred editor
```

#### Essential Environment Variables
```bash
# Database Configuration
POSTGRES_URL=postgresql://tracktion:dev_password@localhost:5432/tracktion_dev
POSTGRES_TEST_URL=postgresql://tracktion:dev_password@localhost:5432/tracktion_test
REDIS_URL=redis://localhost:6379/0
REDIS_TEST_URL=redis://localhost:6379/1
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dev_password
NEO4J_DATABASE=tracktion_dev

# Message Queue Configuration
RABBITMQ_URL=amqp://tracktion:dev_password@localhost:5672/tracktion_dev

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true
SECRET_KEY=your-development-secret-key-here

# Audio Processing Configuration
AUDIO_SAMPLE_RATE=44100
AUDIO_CHUNK_SIZE=4096
MAX_FILE_SIZE=100MB
TEMP_DIR=/tmp/tracktion
AUDIO_CACHE_SIZE=1000

# Service Configuration
ANALYSIS_SERVICE_HOST=localhost
ANALYSIS_SERVICE_PORT=8001
TRACKLIST_SERVICE_HOST=localhost
TRACKLIST_SERVICE_PORT=8002
FILE_WATCHER_HOST=localhost
FILE_WATCHER_PORT=8003

# External API Configuration (optional)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
MUSICBRAINZ_USER_AGENT=tracktion-dev/1.0
```

## External Services Setup

### 1. Start Database Services
```bash
# Start all external services
docker-compose up -d

# Verify services are running
docker-compose ps
docker-compose logs  # Check for any startup errors
```

### 2. Database Initialization

#### PostgreSQL Setup
```bash
# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U tracktion

# Run database migrations
uv run alembic upgrade head

# Create test database
uv run python scripts/create_test_db.py
```

#### Neo4j Setup
```bash
# Wait for Neo4j to be ready (check logs)
docker-compose logs neo4j

# Initialize Neo4j schema (if needed)
uv run python scripts/init_neo4j.py
```

#### Redis Setup
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping  # Should return PONG
```

### 3. RabbitMQ Setup
```bash
# Access RabbitMQ management interface
# http://localhost:15672 (guest/guest)

# Create development exchanges and queues
uv run python scripts/setup_rabbitmq.py
```

## Development Tools Setup

### 1. Code Quality Tools Verification
```bash
# Verify all tools are working
uv run ruff --version        # Code linter and formatter
uv run mypy --version        # Type checker
uv run pytest --version     # Test runner
uv run pre-commit --version  # Git hooks manager

# Test all tools on the codebase
uv run ruff check .
uv run mypy .
uv run pytest tests/unit/ --dry-run
```

### 2. Audio Processing Dependencies

#### Install Audio Libraries
```bash
# The audio processing libraries are installed via uv
# Verify they work:
uv run python -c "import essentia; print('Essentia OK')"
uv run python -c "import librosa; print('Librosa OK')"
uv run python -c "import soundfile; print('SoundFile OK')"
```

#### Test Audio Processing
```bash
# Run audio processing tests
uv run pytest tests/unit/analysis_service/test_audio_processing.py -v
```

## Validation and Testing

### 1. Environment Validation Script
Create a validation script to verify setup:

```bash
# Run comprehensive environment validation
uv run python scripts/validate_environment.py
```

### 2. Integration Tests
```bash
# Run integration tests to verify all services work together
uv run pytest tests/integration/ --verbose

# Run specific service tests
uv run pytest tests/integration/test_analysis_service.py
```

### 3. Performance Benchmarks
```bash
# Run performance benchmarks to ensure optimal setup
uv run pytest tests/benchmarks/ --benchmark-only
```

## Troubleshooting

### Common Issues

#### Python Environment Issues
```bash
# Issue: Wrong Python version
uv python install 3.11.7
uv python pin 3.11.7

# Issue: Package installation failures
uv cache clean
uv sync --reinstall

# Issue: Import errors
uv run python -c "import sys; print(sys.path)"
```

#### Docker Issues
```bash
# Issue: Services not starting
docker-compose down
docker system prune -f
docker-compose up -d

# Issue: Permission errors on Linux
sudo chown -R $USER:$USER ~/.docker
```

#### Audio Processing Issues
```bash
# Issue: Audio library not found
# Linux
sudo apt install libasound2-dev libsndfile1-dev

# macOS
brew install libsndfile

# Issue: FFmpeg not found
which ffmpeg  # Should return a path
```

#### Database Connection Issues
```bash
# Issue: PostgreSQL connection refused
docker-compose logs postgres
docker-compose restart postgres

# Issue: Database doesn't exist
docker-compose exec postgres createdb -U tracktion tracktion_dev
```

### Performance Optimization

#### System Optimization
```bash
# Increase file descriptor limits (Linux)
ulimit -n 65536

# Optimize Docker memory (adjust in Docker Desktop or daemon.json)
# Recommended: 4GB+ for development
```

#### Python Optimization
```bash
# Use uv's built-in caching
export UV_CACHE_DIR=~/.cache/uv

# Enable Python optimizations for production testing
export PYTHONOPTIMIZE=1
```

## Advanced Configuration

### 1. Development with Multiple Services
```bash
# Start all services in development mode
scripts/dev-start-all.sh

# Or start services individually
cd services/analysis_service && uv run python src/main.py &
cd services/tracklist_service && uv run python src/main.py &
```

### 2. Hot Reloading Setup
```bash
# Use watchdog for automatic restart on code changes
uv add --dev watchdog
uv run watchmedo auto-restart --directory=services/ --pattern="*.py" --recursive -- python services/analysis_service/src/main.py
```

### 3. Debugging Configuration
```bash
# Enable debug logging for all services
export LOG_LEVEL=DEBUG
export DEBUG=true

# Enable SQL query logging
export SQLALCHEMY_ECHO=true
```

## Security Considerations

### Development Security
- **Never commit `.env` files** with real credentials
- **Use strong passwords** for development databases
- **Rotate secrets regularly** even in development
- **Limit network exposure** of development services

### Local Network Configuration
```bash
# Bind services only to localhost in development
export BIND_HOST=127.0.0.1

# Use development-specific ports
export PORT_OFFSET=0  # Production uses 8000+, dev uses 8001+
```

## Next Steps

After completing environment setup:

1. **Verify Installation**: Run `uv run python scripts/validate_environment.py`
2. **Run Tests**: Execute `uv run pytest tests/` to ensure everything works
3. **Review Architecture**: Read `docs/architecture/system-overview.md`
4. **Configure IDE**: Follow `docs/development/ide-configuration.md`
5. **Learn Workflow**: Study `docs/development/git-workflow.md`

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review service-specific README files
3. Check Docker Compose logs: `docker-compose logs`
4. Create an issue with error logs and system information
5. Ask team members for help with complex setup issues
