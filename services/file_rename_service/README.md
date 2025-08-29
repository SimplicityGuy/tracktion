# File Rename Service

Standalone service for intelligent file renaming with ML-powered pattern recognition.

## Features

- Pattern-based file renaming
- Machine learning model integration
- User feedback incorporation
- Batch processing capabilities
- RESTful API endpoints
- RabbitMQ message queue integration

## Setup

1. Install dependencies:
   ```bash
   uv pip install -e .
   ```

2. Run database migrations:
   ```bash
   uv run alembic upgrade head
   ```

3. Start the service:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## API Documentation

When running in debug mode, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run tests with:
```bash
uv run pytest
```
