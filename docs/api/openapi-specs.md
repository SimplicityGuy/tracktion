# OpenAPI/Swagger Specifications

This document provides the OpenAPI specifications for all Tracktion services with REST APIs.

## Analysis Service OpenAPI Specification

The Analysis Service uses FastAPI and automatically generates OpenAPI documentation.

### Accessing the OpenAPI Specification

- **Swagger UI**: http://localhost:8001/v1/docs
- **ReDoc**: http://localhost:8001/v1/redoc
- **OpenAPI JSON**: http://localhost:8001/v1/openapi.json

### OpenAPI Configuration

```python
app = FastAPI(
    title="Analysis Service API",
    description="Async API for music analysis and processing",
    version="1.0.0",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json",
)
```

### Key OpenAPI Features

- **Authentication**: API Key in header (`X-API-Key`)
- **Rate Limiting**: Headers for rate limit status
- **Request/Response Models**: Pydantic models for validation
- **Error Responses**: Standardized error schemas
- **File Upload Support**: Multipart form data for audio files

## Tracklist Service OpenAPI Specification

The Tracklist Service also uses FastAPI with comprehensive OpenAPI documentation.

### Accessing the OpenAPI Specification

- **Swagger UI**: http://localhost:8002/v1/docs
- **ReDoc**: http://localhost:8002/v1/redoc
- **OpenAPI JSON**: http://localhost:8002/v1/openapi.json

### OpenAPI Configuration

```python
app = FastAPI(
    title="Tracklist Service API",
    description="Tracklist search, import, and CUE generation API",
    version="1.0.0",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json",
)
```

## OpenAPI Schema Examples

### Analysis Service Schema Sample

```yaml
openapi: 3.0.0
info:
  title: Analysis Service API
  description: Async API for music analysis and processing
  version: 1.0.0

servers:
  - url: http://localhost:8001/v1
    description: Development server
  - url: https://api.tracktion.com/v1
    description: Production server

security:
  - ApiKeyAuth: []

components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    Recording:
      type: object
      properties:
        id:
          type: string
          format: uuid
        file_path:
          type: string
        status:
          type: string
          enum: [pending, processing, completed, failed, invalid]
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    AnalysisResult:
      type: object
      properties:
        recording_id:
          type: string
          format: uuid
        bpm:
          type: number
          format: float
        key:
          type: string
        mood_data:
          type: object
        genre_data:
          type: object
        confidence_scores:
          type: object

    Error:
      type: object
      properties:
        error:
          type: string
        message:
          type: string
        correlation_id:
          type: string
        timestamp:
          type: string
          format: date-time

paths:
  /health:
    get:
      summary: Health Check
      description: Check service health status
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  service:
                    type: string
                  timestamp:
                    type: number

  /recordings:
    post:
      summary: Submit Recording for Analysis
      description: Submit an audio file for analysis
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                metadata:
                  type: object
      responses:
        '201':
          description: Recording submitted successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Recording'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /recordings/{recording_id}/analysis:
    get:
      summary: Get Analysis Results
      description: Get analysis results for a recording
      parameters:
        - name: recording_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Analysis results retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResult'
        '404':
          description: Recording not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
```

### Tracklist Service Schema Sample

```yaml
openapi: 3.0.0
info:
  title: Tracklist Service API
  description: Tracklist search, import, and CUE generation API
  version: 1.0.0

servers:
  - url: http://localhost:8002/v1
    description: Development server

components:
  schemas:
    Tracklist:
      type: object
      properties:
        id:
          type: string
          format: uuid
        title:
          type: string
        artist:
          type: string
        source_url:
          type: string
        tracks:
          type: array
          items:
            $ref: '#/components/schemas/Track'
        created_at:
          type: string
          format: date-time

    Track:
      type: object
      properties:
        position:
          type: integer
        title:
          type: string
        artist:
          type: string
        start_time:
          type: string
        duration:
          type: string
        bpm:
          type: number
        key:
          type: string

    SearchQuery:
      type: object
      properties:
        query:
          type: string
        filters:
          type: object
        limit:
          type: integer
          minimum: 1
          maximum: 100
          default: 20
        offset:
          type: integer
          minimum: 0
          default: 0

paths:
  /search:
    post:
      summary: Search Tracklists
      description: Search for tracklists and tracks
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SearchQuery'
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      $ref: '#/components/schemas/Tracklist'
                  total:
                    type: integer
                  limit:
                    type: integer
                  offset:
                    type: integer

  /tracklist/import:
    post:
      summary: Import Tracklist
      description: Import tracklist from external source
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                source_url:
                  type: string
                  format: uri
                import_options:
                  type: object
      responses:
        '202':
          description: Import started
          content:
            application/json:
              schema:
                type: object
                properties:
                  task_id:
                    type: string
                  status:
                    type: string
                  estimated_completion:
                    type: string
                    format: date-time

  /cue/generate:
    post:
      summary: Generate CUE Sheet
      description: Generate CUE sheet from tracklist
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                tracklist_id:
                  type: string
                  format: uuid
                format:
                  type: string
                  enum: [standard, cdj, traktor, serato, rekordbox]
                options:
                  type: object
      responses:
        '200':
          description: CUE sheet generated
          content:
            text/plain:
              schema:
                type: string
            application/json:
              schema:
                type: object
                properties:
                  cue_content:
                    type: string
                  format:
                    type: string
                  metadata:
                    type: object
```

## Rate Limiting Headers

All APIs include standardized rate limiting headers:

```yaml
X-RateLimit-Limit: 100        # Requests per time window
X-RateLimit-Remaining: 95     # Remaining requests
X-RateLimit-Reset: 1609459200 # Reset timestamp
Retry-After: 60               # Seconds to wait when rate limited
```

## Authentication Schemes

### API Key Authentication

```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for authentication
```

### JWT Bearer Authentication

```yaml
components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT token for authenticated requests
```

## Error Response Standards

All APIs use consistent error response format:

```yaml
components:
  schemas:
    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: Error code or type
        message:
          type: string
          description: Human-readable error message
        correlation_id:
          type: string
          description: Request correlation ID for tracing
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
        details:
          type: object
          description: Additional error details
```

## Response Headers

Standard response headers across all APIs:

```yaml
headers:
  X-Request-ID:
    description: Unique request identifier
    schema:
      type: string
  X-Process-Time:
    description: Request processing time in milliseconds
    schema:
      type: number
  X-RateLimit-Limit:
    description: Request limit per time window
    schema:
      type: integer
  X-RateLimit-Remaining:
    description: Remaining requests in current window
    schema:
      type: integer
```

## Webhook Specifications

### Notification Webhooks

```yaml
components:
  schemas:
    WebhookPayload:
      type: object
      properties:
        event:
          type: string
          enum: [analysis.completed, tracklist.imported, cue.generated]
        data:
          type: object
        timestamp:
          type: string
          format: date-time
        webhook_id:
          type: string
        signature:
          type: string
          description: HMAC signature for verification
```

## Testing with OpenAPI

### Swagger UI Features

- **Interactive Testing**: Try API calls directly from the documentation
- **Authentication**: Enter API keys or tokens for testing
- **Request/Response Validation**: Real-time schema validation
- **Code Generation**: Generate client code in multiple languages

### ReDoc Features

- **Clean Documentation**: Professional API documentation presentation
- **Search Functionality**: Search through endpoints and schemas
- **Code Samples**: Multiple language examples
- **Download Options**: Export as PDF or print

## Integration Examples

### Python Client Generation

```bash
# Install OpenAPI client generator
uv pip install openapi-python-client

# Generate Analysis Service client
uv run openapi-python-client generate \
  --url http://localhost:8001/v1/openapi.json \
  --output-path ./clients/analysis_client

# Generate Tracklist Service client
uv run openapi-python-client generate \
  --url http://localhost:8002/v1/openapi.json \
  --output-path ./clients/tracklist_client
```

### JavaScript/TypeScript Client

```bash
# Install OpenAPI generator
npm install @openapitools/openapi-generator-cli

# Generate TypeScript client
npx openapi-generator-cli generate \
  -i http://localhost:8001/v1/openapi.json \
  -g typescript-axios \
  -o ./clients/analysis-client-ts
```

## API Versioning Strategy

### URL-based Versioning

- Current: `/v1/endpoint`
- Future: `/v2/endpoint`

### Header-based Versioning

```yaml
Accept: application/vnd.tracktion.v1+json
API-Version: v1
```

### Deprecation Strategy

```yaml
headers:
  Sunset:
    description: API version sunset date
    schema:
      type: string
      format: date
  Deprecation:
    description: API deprecation notice
    schema:
      type: string
```

This comprehensive OpenAPI specification provides complete API documentation and enables automatic client generation, testing, and integration for all Tracktion services.
