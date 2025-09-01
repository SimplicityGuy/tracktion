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
- Multiple notification channels for better user reach
- Complete service implementation per architecture
- No gaps in the data processing pipeline
- Professional multi-channel alert capabilities

## Technical Scope

### Core Requirements
1. **Cataloging Service Implementation**
   - Complete missing cataloging_service
   - Implement all designed functionality
   - Database operations for recording catalog
   - Message queue integration
   - API endpoints for catalog management

2. **Enhanced Notification System**
   - Add email notification support
   - Add Slack integration
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
      - Email and webhook implementations in alert manager
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
- Support for multiple simultaneous channels
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

#### Story 10.2: Add Email Notifications
**As a** user of the system
**I want** to receive email notifications
**So that** I'm alerted through my preferred channel

**Acceptance Criteria:**
- Email service integration (SMTP)
- HTML and text email templates
- Configurable email settings
- Email queuing for reliability
- Delivery status tracking
- Unsubscribe mechanism
- Test mode for development

#### Story 10.3: Add Slack Integration
**As a** team using Slack
**I want** notifications in Slack channels
**So that** the team stays informed of system events

**Acceptance Criteria:**
- Slack webhook support
- Slack app integration option
- Channel configuration
- Message formatting for Slack
- Rate limiting compliance
- Error handling for failed sends
- Documentation for setup

#### Story 10.4: Refactor Notification Architecture
**As a** system sending notifications
**I want** a unified notification system
**So that** adding new channels is easy

**Acceptance Criteria:**
- Abstract notification interface
- Plugin architecture for channels
- Notification preference management
- Retry logic for failed notifications
- Notification history logging
- Template system for consistent messaging
- Configuration via environment variables

#### Story 10.5: Complete Analysis Service API Implementation
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

#### Story 10.6: Implement Missing TODO Items
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

#### Story 10.7: Replace Mock/Placeholder Implementations
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

### Phase 1: Cataloging Service & Missing Implementations (Week 1)
1. Create service structure
2. Implement database models and operations
3. Set up RabbitMQ consumers
4. Create API endpoints
5. Add comprehensive tests
6. Docker configuration
7. Complete Analysis Service API implementations
8. Replace all mock/placeholder code

### Phase 2: Notification System & TODO Completion (Week 2)
1. Design notification architecture
2. Refactor existing Discord integration
3. Implement email support
4. Add Slack integration
5. Create preference management
6. Testing and documentation
7. Complete all TODO items
8. Implement job tracking and metadata integration

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
class EmailChannel(NotificationChannel)
class SlackChannel(NotificationChannel)
class DiscordChannel(NotificationChannel)
```

## Configuration Requirements

### Email Configuration
- SMTP_HOST
- SMTP_PORT
- SMTP_USERNAME
- SMTP_PASSWORD
- FROM_EMAIL
- EMAIL_TEMPLATES_PATH

### Slack Configuration
- SLACK_WEBHOOK_URL
- SLACK_BOT_TOKEN (optional)
- SLACK_DEFAULT_CHANNEL

### Notification Preferences
- User-level channel preferences
- Event-type routing rules
- Priority-based channel selection

## Testing Strategy
- Unit tests for all new code
- Integration tests for service communication
- Email testing with mock SMTP
- Slack testing with mock endpoints
- End-to-end notification flow tests
- Load testing for cataloging service
- Verification that no mock data remains in production paths
- Testing of all previously stubbed endpoints
- Validation of message queue integrations

## Success Metrics
- Cataloging service processing 1000+ files/minute
- Email delivery rate >95%
- Slack message delivery <2s
- All notification channels working simultaneously
- Zero message loss under normal operation
- 80% test coverage for new code

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Email delivery issues | Medium | Use reliable email service, implement retry |
| Slack rate limiting | Low | Implement rate limiting, message batching |
| Cataloging service performance | High | Optimize database queries, add caching |
| Notification spam | Medium | Implement throttling, user preferences |

## Dependencies
- Email service provider (SMTP)
- Slack API access
- PostgreSQL for cataloging data
- RabbitMQ for message passing
- Existing Discord implementation

## Definition of Done
- [ ] Cataloging service fully implemented and tested
- [ ] Email notifications working and configurable
- [ ] Slack integration complete and documented
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
