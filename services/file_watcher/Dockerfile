FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# Copy application code
COPY src/ ./src/

# Set default data directory environment variable
ENV DATA_DIR=/data/music

# Create data directory with appropriate permissions
RUN mkdir -p ${DATA_DIR} && \
    chmod 755 ${DATA_DIR}

# Declare volume for data directory (optional, for clarity)
VOLUME ["${DATA_DIR}"]

# Create non-root user and set permissions
RUN useradd -m -r appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser ${DATA_DIR}

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["python", "-m", "src.main"]
