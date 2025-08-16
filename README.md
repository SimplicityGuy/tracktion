# Tracktion

Automated music library management system with intelligent cataloging and analysis.

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
docker-compose -f infrastructure/docker-compose.yaml up -d
```

4. Verify services are running:
```bash
docker-compose -f infrastructure/docker-compose.yaml ps
```

5. Check logs:
```bash
docker-compose -f infrastructure/docker-compose.yaml logs -f
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
pre-commit install
```

4. Run tests:
```bash
pytest
```

## Services

- **file_watcher**: Monitors directories for new audio files
- **cataloging_service**: Catalogs files in PostgreSQL
- **analysis_service**: Analyzes audio metadata and stores in Neo4j
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

## Testing

Run unit tests:
```bash
pytest tests/unit
```

Run integration tests:
```bash
pytest tests/integration
```

## Contributing

1. Follow the coding standards defined in `docs/architecture/coding-standards.md`
2. Ensure all tests pass before submitting changes
3. Use pre-commit hooks to maintain code quality

## License

[License information to be added]
