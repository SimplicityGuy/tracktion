# Epic 7: File Watcher Fixes

## Epic Overview
**Epic ID:** EPIC-7
**Epic Name:** File Watcher Fixes
**Priority:** Critical
**Dependencies:** None (foundational fixes)
**Estimated Effort:** 1 week

## Business Value
Fixing the file_watcher service ensures:
- Proper file system monitoring using industry-standard watchdog library
- Configurable data directory support for flexible deployment
- Correct file hashing implementation for duplicate detection
- Support for multiple concurrent instances monitoring different directories
- Stable foundation before async refactoring (Epic 8)

## Technical Scope

### Core Requirements
1. **Watchdog Implementation**
   - MUST properly implement watchdog library (currently imported but unused)
   - Replace any custom file watching logic with watchdog observers
   - Configure appropriate event handlers for file creation, modification, deletion
   - Implement proper observer lifecycle management

2. **Docker Configuration**
   - Add environment variable or volume mount for data directory specification
   - Update Dockerfile to support configurable watch paths
   - Ensure proper permissions for mounted directories
   - Document deployment configuration options

3. **Hashing Implementation**
   - Replace SHA256 with XXH128 using xxhash Python library
   - Implement proper XXH128 hash calculation for files
   - Update database schema if needed for hash storage
   - Maintain backward compatibility or provide migration path

4. **Multi-Instance Support**
   - Ensure multiple file_watcher services can run simultaneously
   - Each instance monitors different data directories
   - Proper instance identification in logs and messages
   - No conflicts in message queue or database operations

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

#### Story 7.3: Implement XXH128 Hashing
**As a** system processing large files
**I want** fast XXH128 hashing instead of SHA256
**So that** file processing is more efficient

**Acceptance Criteria:**
- xxhash Python library is properly integrated
- XXH128 hashes are calculated correctly
- Database schema updated if needed
- Performance improvement measurable
- Backward compatibility handled

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

## Implementation Approach

### Phase 1: Watchdog Implementation (Days 1-2)
1. Review current file watching implementation
2. Properly integrate watchdog library
3. Implement event handlers
4. Add comprehensive tests
5. Validate with various file operations

### Phase 2: Docker Configuration (Day 3)
1. Update Dockerfile with configurable paths
2. Add environment variable handling
3. Update docker-compose examples
4. Test various mount configurations
5. Document deployment options

### Phase 3: Hashing Upgrade (Days 4-5)
1. Integrate xxhash library
2. Implement XXH128 calculation
3. Update any affected schemas
4. Performance benchmarking
5. Migration strategy if needed

### Phase 4: Multi-Instance Validation (Day 6)
1. Deploy multiple instances
2. Test concurrent operations
3. Verify no conflicts
4. Performance testing
5. Documentation updates

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
