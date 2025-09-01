# File Rename Service API Implementation

## Overview
This document summarizes the API endpoints implemented for the File Rename Service proposal system.

## Endpoints Implemented

### 1. POST /rename/propose
**Purpose**: Generate single file rename proposal
**Description**: Generates a rename proposal for a single file using ML models and templates
**Request**: `RenameProposalRequest`
**Response**: `RenameProposalResponse`
**Features**:
- ML-powered proposal generation
- Template-based alternatives
- Confidence scoring
- Caching support (30-minute TTL)
- Comprehensive error handling

### 2. POST /rename/propose/batch
**Purpose**: Generate batch rename proposals
**Description**: Processes multiple files in parallel for rename proposals
**Request**: `BatchProposalRequest`
**Response**: `BatchProposalResponse`
**Features**:
- Parallel processing (configurable concurrency)
- Batch success/failure tracking
- Performance metrics
- Individual file metadata support
- Comprehensive error reporting

### 3. GET /rename/templates
**Purpose**: Get naming templates
**Description**: Retrieve available naming templates for a user
**Query Parameters**: `user_id`, optional `search`
**Response**: `list[TemplateResponse]`
**Features**:
- User-specific template filtering
- Search functionality
- Usage statistics
- Active/inactive template filtering

### 4. POST /rename/templates
**Purpose**: Save custom template
**Description**: Create a new custom naming template
**Request**: `TemplateRequest`
**Response**: `TemplateResponse`
**Features**:
- Template pattern validation
- User association
- Usage tracking
- Pattern syntax checking

### 5. POST /rename/validate
**Purpose**: Validate proposed filenames
**Description**: Check proposed filenames for conflicts and validation issues
**Request**: `ValidateRequest`
**Response**: `ValidateResponse`
**Features**:
- Filename validity checking
- Conflict detection
- Optional filesystem checks
- Resolution suggestions
- Batch validation support

## Key Features

### Error Handling
- Comprehensive HTTP status codes (400, 422, 500)
- Detailed error messages with context
- Validation error handling
- Exception propagation with proper logging

### Performance Optimization
- Response caching (simple in-memory cache)
- Parallel batch processing
- Configurable concurrency limits
- Performance metrics tracking

### Integration Points
- **Proposal Generator**: ML-powered rename suggestions
- **Batch Processor**: Parallel file processing
- **Template Manager**: User-defined naming templates
- **Conflict Resolver**: Filename conflict detection and resolution
- **Cache System**: Simple in-memory caching for performance

### OpenAPI Documentation
- Comprehensive endpoint documentation
- Request/response schemas
- Example payloads
- Parameter descriptions
- Error response documentation

## Dependencies
- **FastAPI**: Web framework
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM (via dependency injection)
- **Custom Components**: Proposal generator, batch processor, template manager

## File Structure
```
api/
├── rename_routes.py     # Main API route handlers
└── schemas.py          # Extended with new request/response models

app/
├── cache/
│   ├── __init__.py
│   └── simple_cache.py  # Simple in-memory caching
└── main.py             # Updated to include proposal router
```

## Configuration
The API is automatically included in the main FastAPI application with:
- CORS middleware support
- Exception handlers
- OpenAPI documentation
- Health check endpoints

## Usage Examples

### Single Proposal
```bash
curl -X POST "http://localhost:8000/rename/propose" \
  -H "Content-Type: application/json" \
  -d '{
    "original_name": "track01.mp3",
    "metadata": {"artist": "Artist", "title": "Song"}
  }'
```

### Batch Proposals
```bash
curl -X POST "http://localhost:8000/rename/propose/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "filenames": ["track01.mp3", "track02.mp3"],
    "max_concurrent": 10
  }'
```

### Validation
```bash
curl -X POST "http://localhost:8000/rename/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "proposed_names": ["Artist - Song.mp3"],
    "existing_files": ["existing.mp3"]
  }'
```

## Notes
- All endpoints include comprehensive logging
- Response caching improves performance for repeated requests
- Batch processing supports configurable concurrency
- Template management includes usage tracking
- Validation provides conflict resolution suggestions
