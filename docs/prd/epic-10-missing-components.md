# Epic 10: Missing Components

## Epic Overview
**Epic ID:** EPIC-10
**Epic Name:** Missing Components
**Priority:** High
**Dependencies:** Epic 1-9
**Estimated Effort:** 4 weeks (2 weeks completed: 10.1-10.5, 2 weeks remaining: 10.6-10.8)

## Business Value
Completing missing components ensures:
- Full system functionality as originally designed
- Discord notification channels for better user reach
- Complete service implementation per architecture
- No gaps in the data processing pipeline
- Professional Discord alert capabilities
- **Updated**: Complete elimination of all "will be implemented" technical debt
- **Updated**: Production-ready system with no placeholder or stub code
- **Updated**: Fully functional job tracking and management system
- **Updated**: Complete authentication and authorization across all services
- **Updated**: Robust file lifecycle management and cleanup
- **Updated**: Real-time performance monitoring and optimization

## Technical Scope

### Core Requirements

**Note**: Stories 10.1 (Cataloging Service), 10.2 (Discord Notifications), and 10.3 (Analysis Service APIs) have been completed. The following requirements address remaining "will be implemented" items found in the current codebase.

4. **Current "Will Be Implemented" Items**
   Based on current codebase analysis, the following implementations are still needed:

   a) **Analysis Service Repository Implementation**
      - `services/analysis_service/src/repositories.py:314`: AsyncAnalysisResultRepository needs implementation once AnalysisResult model is ready
      - Complete the analysis result storage and retrieval system

   b) **Active TODO Items Requiring Implementation** (Current findings)
      - `services/analysis_service/src/main.py:457`: Notification sending via RabbitMQ
      - `services/tracklist_service/src/services/cue_integration.py:385`: Metadata extraction using CUE parser
      - `services/tracklist_service/src/services/cue_generation_service.py:139`: Job tracking with proper UUIDs
      - `services/tracklist_service/src/services/audio_validation_service.py:128`: Actual audio duration detection
      - `services/tracklist_service/src/messaging/cue_generation_handler.py:242`: Validation logic implementation
      - `services/tracklist_service/src/messaging/cue_generation_handler.py:272`: Conversion logic implementation
      - `services/tracklist_service/src/messaging/cue_generation_handler.py:286`: Actual tracklist retrieval from database
      - `services/tracklist_service/src/admin/parser_admin.py:44`: Proper authentication
      - `services/tracklist_service/src/admin/parser_admin.py:637`: Operation history storage and retrieval
      - `services/cataloging_service/src/message_consumer.py:238`: FileLifecycleService for cleanup
      - `services/file_rename_service/api/routers.py:58`: Actual pattern analysis logic
      - `services/file_rename_service/api/routers.py:111`: Actual proposal generation logic
      - `services/tracklist_service/src/api/cue_generation_api.py:92`: Tracklist repository integration
      - `services/tracklist_service/src/api/cue_generation_api.py:372`: Job status tracking
      - `services/tracklist_service/src/api/cue_generation_api.py:403`: File retrieval from storage
      - `services/cataloging_service/src/async_catalog_service.py:264`: Search with filters implementation

   c) **Placeholder/Stub Implementations Requiring Real Logic**
      - `services/tracklist_service/src/services/cue_integration.py:37,53`: CUE handler imports and logic
      - `services/tracklist_service/src/services/storage_service.py:295,309,314,319,324`: Storage service implementations
      - `services/tracklist_service/src/services/audio_validation_service.py:20,129`: Audio validation implementations
      - `services/tracklist_service/src/messaging/cue_generation_handler.py:75,285`: Tracklist retrieval logic
      - `services/tracklist_service/src/messaging/simple_handler.py:20`: Message handler stub
      - `services/analysis_service/src/graceful_shutdown.py:296`: Queue status implementation
      - `services/tracklist_service/src/api/search_api.py:237`: Search integration placeholder
      - `services/tracklist_service/src/api/tracklist_api.py:80`: Tracklist API implementation
      - `services/tracklist_service/src/api/cue_generation_api.py:71`: Database connection placeholder
      - `services/tracklist_service/src/api/import_endpoints.py:47`: Reconnection logic
      - `services/tracklist_service/src/utils/integrity_validator.py:17,19`: Database setup placeholders
      - `services/tracklist_service/src/admin/parser_admin.py:49`: Token validation
      - `services/tracklist_service/src/optimization/performance_optimizer.py:214,487`: Queue status and worker count
      - `services/tracklist_service/src/security/abuse_detector.py:387`: Abuse detection logic
      - `services/file_rename_service/app/main.py:21`: File rename service placeholders
      - `services/file_rename_service/app/proposal/cache.py:74,155`: Redis cache implementations

   d) **Documentation References for Future Implementation**
      - `docs/prd/epic-2-metadata-analysis-naming.md:160`: File renaming implementation pending UI
      - Manual CUE generation endpoints requiring full implementation

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

#### Story 10.6: Complete Remaining "Will Be Implemented" Items
**As a** development team
**I want** all remaining "will be implemented" comments and placeholders addressed
**So that** the system has no incomplete functionality

**Acceptance Criteria:**
- AsyncAnalysisResultRepository fully implemented with AnalysisResult model
- All active TODO items from current codebase scan completed
- All placeholder/stub implementations replaced with production code
- RabbitMQ notification system fully functional
- CUE integration components fully implemented
- Audio validation using actual file analysis
- Job tracking with proper UUID generation
- Authentication and authorization in all admin interfaces
- File lifecycle management and cleanup services
- Real database connections replacing all placeholders
- Performance optimization components with actual queue management
- Search functionality with proper filtering
- All Redis cache implementations completed
- File rename service with actual pattern analysis and proposal logic

#### Story 10.7: Analysis Service Repository Implementation
**As a** system storing analysis results
**I want** the AsyncAnalysisResultRepository fully implemented
**So that** analysis results are properly persisted and retrieved

**Acceptance Criteria:**
- AnalysisResult model defined with proper schema
- AsyncAnalysisResultRepository methods implemented
- Database operations for CRUD analysis results
- Integration with existing analysis workflows
- Proper error handling and logging
- Unit tests with 80% coverage
- Database migrations for new tables

#### Story 10.8: Complete Job Tracking and UUID Management
**As a** system processing jobs
**I want** proper job tracking with real UUIDs
**So that** all operations can be monitored and managed

**Acceptance Criteria:**
- Real UUID generation replacing placeholder UUIDs
- Job status tracking in CUE generation service
- Job tracking in analysis service workflows
- Database schema for job management
- API endpoints for job status queries
- Proper job lifecycle management
- Integration with notification system for job completion

## Implementation Approach

**Note**: Phases 1-2 have been completed (Stories 10.1-10.5). The following phases address remaining implementation work.

### Phase 3: Current "Will Be Implemented" Items (Week 3)
1. Complete Analysis Service repository implementation
2. Implement all active TODO items from current codebase scan
3. Replace remaining placeholder/stub implementations
4. Complete job tracking with proper UUIDs
5. Implement authentication and authorization systems
6. Complete file lifecycle management
7. Testing and validation

### Phase 4: Final Integration & Production Readiness (Week 4)
1. Complete all Redis cache implementations
2. Implement real database connections everywhere
3. Complete performance optimization components
4. Implement search functionality with filtering
5. Complete file rename service logic
6. Final integration testing
7. Performance validation
8. Documentation updates

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

### Completed Items (Stories 10.1-10.5)
- [x] Cataloging service fully implemented and tested
- [x] Discord integration complete and documented
- [x] Unified notification architecture in place
- [x] Analysis Service API endpoints implemented (Story 10.3)
- [x] Initial TODO items completed (Story 10.4)
- [x] Initial mock/placeholder implementations replaced (Story 10.5)

### Remaining Items (Stories 10.6-10.8)
- [ ] AsyncAnalysisResultRepository fully implemented with AnalysisResult model
- [ ] All current "will be implemented" comments addressed:
  - [ ] `services/analysis_service/src/repositories.py:314` - Repository implementation
  - [ ] `services/analysis_service/src/main.py:457` - RabbitMQ notification sending
  - [ ] `services/tracklist_service/src/services/cue_integration.py:385` - CUE parser metadata extraction
  - [ ] All remaining TODO items from current codebase scan completed
- [ ] All placeholder/stub implementations replaced with production code:
  - [ ] Storage service implementations (5 placeholders)
  - [ ] Audio validation implementations (2 placeholders)
  - [ ] CUE handler logic implementations
  - [ ] Database connection placeholders replaced
  - [ ] Authentication and authorization systems
  - [ ] File lifecycle management systems
- [ ] Job tracking system with proper UUIDs implemented everywhere
- [ ] Real database connections replacing all placeholders
- [ ] Performance optimization with actual queue management
- [ ] Search functionality with proper filtering implemented
- [ ] All Redis cache implementations completed
- [ ] File rename service with actual pattern analysis and proposal logic
- [ ] All tests passing (80% coverage)
- [ ] Documentation updated for new implementations
- [ ] Integration with existing services verified
- [ ] Performance targets met
- [ ] No "will be implemented", TODO, placeholder, mock, or stub code in production paths
