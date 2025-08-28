# Epic 7: File Watcher Fixes

## Epic Overview
**Epic ID:** EPIC-7
**Epic Name:** File Watcher Fixes and Lifecycle Management
**Priority:** Critical
**Dependencies:** None (foundational fixes)
**Estimated Effort:** 1.5 weeks (expanded scope for lifecycle handling)

## Business Value
Fixing the file_watcher service ensures:
- Proper file system monitoring using industry-standard watchdog library
- Complete lifecycle tracking (creation, modification, deletion, moves)
- Configurable data directory support for flexible deployment
- Dual hashing implementation (SHA256 + XXH128) for integrity and performance
- Support for multiple concurrent instances monitoring different directories
- Proper handling of file removals and moves throughout the system
- Stable foundation before async refactoring (Epic 8)

## Technical Scope

### Core Requirements
1. **Watchdog Implementation**
   - MUST properly implement watchdog library (currently imported but unused)
   - Replace any custom file watching logic with watchdog observers
   - Configure appropriate event handlers for ALL file events:
     - File creation (new files added)
     - File modification (existing files changed)
     - File deletion (files removed)
     - File moves (files relocated to different directory)
     - File renames (files renamed in same directory)
   - Implement proper observer lifecycle management
   - Ensure all events are propagated through the system with proper event type

2. **Docker Configuration**
   - Add environment variable or volume mount for data directory specification
   - Update Dockerfile to support configurable watch paths
   - Ensure proper permissions for mounted directories
   - Document deployment configuration options

3. **Hashing Implementation**
   - Add XXH128 using xxhash Python library alongside SHA256
   - Calculate BOTH SHA256 and XXH128 hashes for all files
   - Store both hash values in database
   - SHA256 for integrity verification, XXH128 for fast lookups
   - No hash calculation for deleted files (cleanup only)

4. **Multi-Instance Support**
   - Ensure multiple file_watcher services can run simultaneously
   - Each instance monitors different data directories
   - Proper instance identification in logs and messages
   - No conflicts in message queue or database operations

5. **Downstream System Updates**
   - Update message queue events to include event type:
     - created: new file to process
     - modified: re-process existing file
     - deleted: remove file and all related data
     - moved: update file path in all records
     - renamed: update filename in all records
   - Modify cataloging_service to handle all lifecycle events:
     - Deletions: remove from database
     - Moves/Renames: update file path/name
   - Update analysis_service to cleanup analysis data for deleted files
   - Ensure tracklist_service handles removed/renamed recordings
   - Add cascade deletion logic where appropriate
   - Implement soft-delete option for recovery scenarios
   - Maintain referential integrity across all services

### Technical Considerations

#### Current Issues
- Watchdog library imported but not utilized
- No way to specify data directory in Docker container
- Using SHA256 instead of faster XXH128 hashing
- Unclear if multiple instances can coexist

#### Implementation Strategy
- Fix watchdog implementation first (most critical)
- Then Docker configuration
- Then hashing algorithm
- Finally validate multi-instance support

### User Stories

#### Story 7.1: Implement Watchdog Library
**As a** system administrator
**I want** the file_watcher to use the watchdog library properly
**So that** file monitoring is reliable and efficient

**Acceptance Criteria:**
- Watchdog observers are properly initialized
- All file events are captured (create, modify, delete, move)
- Observer threads are managed correctly
- Graceful shutdown of observers
- Unit tests for watchdog event handlers

#### Story 7.2: Configure Docker Data Directory
**As a** DevOps engineer
**I want** to specify the data directory when running the container
**So that** I can monitor different directories in different deployments

**Acceptance Criteria:**
- Environment variable DATA_DIR is respected
- Volume mounts work correctly
- Default directory if not specified
- Clear documentation of configuration options
- Container starts successfully with custom directories

#### Story 7.3: Implement Dual Hashing (SHA256 + XXH128)
**As a** system processing large files
**I want** both SHA256 and fast XXH128 hashing
**So that** I have cryptographic integrity AND efficient duplicate detection

**Acceptance Criteria:**
- xxhash Python library is properly integrated
- Both SHA256 and XXH128 hashes are calculated correctly
- Database stores both hash values
- Performance impact is acceptable
- Both hashes are used appropriately

#### Story 7.4: Validate Multi-Instance Support
**As a** system architect
**I want** to run multiple file_watcher instances
**So that** I can monitor multiple directories simultaneously

**Acceptance Criteria:**
- Multiple containers can run without conflicts
- Each instance has unique identification
- Message queue handles multiple publishers
- No database conflicts
- Proper logging identifies instance source

#### Story 7.5: Handle File Lifecycle Events System-Wide
**As a** system maintaining data consistency
**I want** file deletions, moves, and renames to be handled throughout the system
**So that** the database stays in sync with the file system

**Acceptance Criteria:**
- File deletion events trigger database cleanup
- File move events update file paths in database
- File rename events update filenames in database
- Cascading deletes remove related metadata and analysis
- Soft-delete option available for recovery
- All downstream services handle lifecycle events properly
- No orphaned data remains after file removal
- Referential integrity maintained across all services

## Implementation Approach

### Phase 1: Watchdog Implementation (Days 1-2)
1. Review current file watching implementation
2. Properly integrate watchdog library
3. Implement ALL event handlers (create, modify, delete, move)
4. Add comprehensive tests
5. Validate with various file operations

### Phase 2: Docker Configuration (Day 3)
1. Update Dockerfile with configurable paths
2. Add environment variable handling
3. Update docker-compose examples
4. Test various mount configurations
5. Document deployment options

### Phase 3: Dual Hashing Implementation (Day 4)
1. Integrate xxhash library
2. Implement dual hash calculation (SHA256 + XXH128)
3. Verify database schema supports both
4. Performance benchmarking
5. Update all hash-related code

### Phase 4: Multi-Instance Validation (Day 5)
1. Deploy multiple instances
2. Test concurrent operations
3. Verify no conflicts
4. Performance testing
5. Documentation updates

### Phase 5: Downstream System Updates (Days 6-7)
1. Update message queue event structure
2. Modify cataloging_service for deletions
3. Update analysis_service for cleanup
4. Implement cascade deletion logic
5. Test end-to-end file lifecycle

## Success Metrics
- Watchdog library actively monitoring files
- Docker containers configurable via environment variables
- XXH128 hashing showing performance improvement
- Multiple instances running successfully in test environment
- All unit tests passing
- No regression in existing functionality

## Testing Strategy
- Unit tests for each component
- Integration tests with file operations
- Docker configuration tests
- Multi-instance stress testing
- Performance benchmarking before/after

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Watchdog compatibility issues | High | Test on target OS versions |
| Breaking changes for existing deployments | Medium | Provide migration guide |
| Performance regression | Low | Benchmark before changes |
| Multi-instance conflicts | Medium | Thorough concurrent testing |

## Dependencies
- Python watchdog library (already in requirements)
- xxhash Python library (to be added)
- Docker environment for testing
- Test data directories

## Definition of Done
- [ ] Watchdog library properly implemented and monitoring files
- [ ] Docker container accepts data directory configuration
- [ ] XXH128 hashing implemented and tested
- [ ] Multi-instance support validated
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] No critical issues in production readiness testing
