# Epic 10: Missing Components

## Epic Overview
**Epic ID:** EPIC-10
**Epic Name:** Missing Components
**Priority:** High
**Dependencies:** Epic 1-9
**Estimated Effort:** 2 weeks

## Business Value
Completing missing components ensures:
- Full system functionality as originally designed
- Discord notification channels for better user reach
- Complete service implementation per architecture
- No gaps in the data processing pipeline
- Professional Discord alert capabilities

## Technical Scope

### Core Requirements
1. **Cataloging Service Implementation**
   - Complete missing cataloging_service
   - Implement all designed functionality
   - Database operations for recording catalog
   - Message queue integration
   - API endpoints for catalog management

2. **Enhanced Notification System**
   - Add Discord integration
   - Maintain existing Discord support
   - Configurable notification preferences
   - Template system for notifications

3. **Other Missing Implementation**
   - Find "In real implementation, submit to message queue for processing" and implement
   - Any other TODO comments and address them
   - Find any other missing implementations or placeholders and implement them

   **Specific Missing Implementations Found:**

   a) **Analysis Service API Endpoints** (20 occurrences)
      - recordings.py: Submit to message queue, fetch from database, query operations
      - metadata.py: Database operations, extraction message queue, enrichment workflow
      - analysis.py: Processing queue submissions, database fetches
      - tracklist.py: Database operations, CUE parser integration, processing queue
      - streaming.py: Actual file reading, database file path retrieval

   b) **TODO Items Requiring Implementation** (28 items)
      - File rename proposal: Metadata repository integration (5 occurrences)
      - Notification sending via RabbitMQ in main.py
      - Search implementation with filters in cataloging service
      - FileLifecycleService for cleanup in cataloging service
      - Authentication in parser admin
      - Operation history storage in parser admin
      - Tracklist repository integration in CUE generation API
      - Job status tracking in CUE generation API
      - File retrieval from storage in CUE generation API
      - Audio duration detection from actual files (2 occurrences)
      - Webhook implementations in alert manager
      - Version mismatch fix in sync_event_consumer
      - Validation and conversion logic in CUE generation handler
      - Audio service integration in matching service
      - Job tracking implementation with proper UUIDs
      - CUE content generation and caching
      - Metadata extraction using CUE parser

   c) **Placeholder/Mock/Stub Implementations**
      - Mock data returns that need real implementations
      - Placeholder logic in error handlers and processors
      - Dummy data generation that needs real data sources
      - Stub implementations in scrapers and parsers

### Technical Considerations

#### Cataloging Service
- Service was designed but never implemented
- Critical for recording management
- Interfaces with PostgreSQL for storage
- Handles file metadata persistence

#### Notification Enhancements
- Currently Discord-only in tracklist_service
- Need pluggable notification architecture
- Support for Discord channel configuration
- User preference management

#### Missing Implementation Details
- **Analysis Service**: 20 API endpoints returning mock data instead of real implementations
- **Message Queue Integration**: Multiple endpoints need RabbitMQ integration
- **Database Operations**: Many fetch/query operations are stubbed
- **Audio Processing**: Audio duration detection and file analysis are placeholders
- **Job Tracking**: UUID generation and job status tracking incomplete
- **Authentication**: Parser admin lacks proper authentication
- **Metadata Integration**: File rename proposal needs metadata repository

### User Stories

#### Story 10.1: Implement Cataloging Service
**As a** system managing music files
**I want** a dedicated cataloging service
**So that** all recordings are properly tracked and managed

**Acceptance Criteria:**
- Service structure created following project standards
- Database operations implemented (CRUD for recordings)
- RabbitMQ consumer for catalog events
- API endpoints for catalog queries
- Proper error handling and logging
- Docker container configured
- Unit tests with 80% coverage

#### Story 10.2: Unified Discord Notification Architecture
**As a** system requiring notifications
**I want** a unified Discord notification architecture
**So that** we have reliable, maintainable, and extensible Discord alerting

**Acceptance Criteria:**
- Discord webhook implementation
- Channel-specific webhook routing
- Abstract notification interface
- Retry logic for failed notifications
- Notification history logging
- Template system for consistent messaging
- Rate limiting compliance
- Error handling for failed sends
- Configuration via environment variables
- Documentation for setup

#### Story 10.3: Complete Analysis Service API Implementation
**As a** system processing audio files
**I want** fully functional API endpoints
**So that** all operations work with real data instead of mocks

**Acceptance Criteria:**
- All 20 "In real implementation" comments replaced with working code
- Database operations implemented for recordings, metadata, analysis
- Message queue submissions working for all endpoints
- File streaming from actual storage locations
- Proper error handling for all operations
- Integration tests for all endpoints

#### Story 10.4: Implement Missing TODO Items
**As a** development team
**I want** all TODO items completed
**So that** the system has no incomplete features

**Acceptance Criteria:**
- Metadata repository integrated with file rename proposal
- Job tracking with proper UUID generation
- Audio duration detection from actual files
- Authentication implemented in parser admin
- Operation history storage working
- Version mismatch in sync_event_consumer resolved
- All validation and conversion logic implemented

#### Story 10.5: Replace Mock/Placeholder Implementations
**As a** system in production
**I want** all mock data replaced with real implementations
**So that** the system works with actual data

**Acceptance Criteria:**
- All mock data returns replaced with database queries
- Placeholder error handling replaced with proper recovery logic
- Dummy data generation replaced with real data sources
- Stub scrapers updated with actual selectors
- All placeholder comments removed or implemented

## Implementation Approach

### Phase 1: Cataloging Service & Core Implementations (Week 1)
1. Create cataloging service structure
2. Implement database models and operations
3. Set up RabbitMQ consumers
4. Create API endpoints
5. Complete Analysis Service API implementations
6. Replace all mock/placeholder code
7. Add comprehensive tests
8. Docker configuration

### Phase 2: Notification System & TODO Completion (Week 2)
1. Design unified Discord notification architecture
2. Implement Discord webhook integration
3. Create channel-specific routing
4. Add retry logic and rate limiting
5. Complete all TODO items
6. Implement job tracking and metadata integration
7. Testing and documentation

## Service Specifications

### Cataloging Service
```
services/cataloging_service/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   ├── repositories/
│   ├── consumers/
│   ├── api/
│   └── utils/
├── tests/
├── Dockerfile
└── pyproject.toml
```

### Notification System Architecture
```python
# Abstract interface
class NotificationChannel(ABC):
    async def send(self, message: Message) -> Result

# Implementations
class DiscordChannel(NotificationChannel)
```

## Configuration Requirements

### Discord Configuration
- DISCORD_WEBHOOK_URL
- DISCORD_BOT_TOKEN (optional)
- DISCORD_DEFAULT_CHANNEL

### Notification Preferences
- User-level channel preferences
- Event-type routing rules
- Priority-based channel selection

## Testing Strategy
- Unit tests for all new code
- Integration tests for service communication
- Discord testing with mock endpoints
- End-to-end notification flow tests
- Load testing for cataloging service
- Verification that no mock data remains in production paths
- Testing of all previously stubbed endpoints
- Validation of message queue integrations

## Success Metrics
- Cataloging service processing 1000+ files/minute
- Discord message delivery <2s
- Discord notification channel working reliably
- Zero message loss under normal operation
- 80% test coverage for new code

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Discord rate limiting | Low | Implement rate limiting, message batching |
| Cataloging service performance | High | Optimize database queries, add caching |
| Notification spam | Medium | Implement throttling, user preferences |

## Dependencies
- Discord API access
- PostgreSQL for cataloging data
- RabbitMQ for message passing
- Existing Discord implementation

## Definition of Done
- [ ] Cataloging service fully implemented and tested
- [ ] Discord integration complete and documented
- [ ] Unified notification architecture in place
- [ ] All 20 "In real implementation" comments replaced with working code
- [ ] All 28 TODO items completed and removed
- [ ] All mock/placeholder implementations replaced
- [ ] Message queue integrations working for all endpoints
- [ ] Database operations implemented for all services
- [ ] Job tracking system with proper UUIDs implemented
- [ ] Authentication implemented where needed
- [ ] All tests passing (80% coverage)
- [ ] Documentation updated
- [ ] Docker containers configured
- [ ] Integration with existing services verified
- [ ] Performance targets met
- [ ] No placeholder, mock, or stub code in production paths
