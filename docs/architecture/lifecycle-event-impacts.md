# File Lifecycle Event Impacts - Architecture Document

## Overview
This document describes the system-wide impacts of implementing complete file lifecycle tracking (creation, modification, deletion, moves, and renames) in the tracktion system. This is a critical architectural change introduced in Epic 7 that affects all services.

## Event Types and Definitions

### File Events
1. **Created**: New file appears in watched directory
2. **Modified**: Existing file content changes
3. **Deleted**: File is removed from watched directory
4. **Moved**: File is relocated to a different directory
5. **Renamed**: File name changes but stays in same directory

## Service-by-Service Impact Analysis

### file_watcher Service
**Changes Required:**
- Implement watchdog library event handlers for all event types
- Add event_type field to all outgoing messages
- Include old_path for move/rename events
- Ensure events are published for directories recursively

**Message Format:**
```json
{
  "event_type": "created|modified|deleted|moved|renamed",
  "file_path": "/current/path/to/file.mp3",
  "old_path": "/previous/path/to/file.mp3",  // only for moved/renamed
  "timestamp": "2025-08-28T10:30:00Z",
  "instance_id": "watcher1",
  "sha256_hash": "...",  // only for created/modified
  "xxh128_hash": "..."   // only for created/modified
}
```

### cataloging_service
**Changes Required:**
- Parse event_type from incoming messages
- Implement handlers for each event type:
  - **created**: Insert new recording (current behavior)
  - **modified**: Update hash values and timestamp
  - **deleted**: Remove recording and cascade to metadata
  - **moved**: Update file_path in recordings table
  - **renamed**: Update file_name in recordings table

**Database Operations:**
```sql
-- Delete handler
DELETE FROM recordings WHERE file_path = ?;

-- Move handler
UPDATE recordings SET file_path = ? WHERE file_path = ?;

-- Rename handler
UPDATE recordings SET file_name = ? WHERE file_path = ?;
```

### analysis_service
**Changes Required:**
- Handle lifecycle events for analysis data
- Clean up Neo4j nodes and relationships
- Clear Redis cache entries

**Event Handlers:**
- **deleted**:
  - Remove Recording node from Neo4j
  - Remove all related Metadata nodes
  - Clear Redis cache keys for file
- **moved/renamed**:
  - Update Recording node properties
  - Invalidate Redis cache (re-cache with new path)

**Neo4j Operations:**
```cypher
-- Delete handler
MATCH (r:Recording {uuid: $uuid})
DETACH DELETE r;

-- Move/rename handler
MATCH (r:Recording {uuid: $uuid})
SET r.file_path = $new_path;
```

### tracklist_service
**Changes Required:**
- Handle recordings that no longer exist
- Update references for moved/renamed files
- Regenerate CUE files with updated paths

**Event Handlers:**
- **deleted**: Mark tracklist as orphaned or delete
- **moved/renamed**: Update recording reference and CUE file

### file_rename_service (Epic 9 - Future)
**Considerations:**
- Must handle renamed files that it previously processed
- Update learning model with new names
- Track rename history for pattern learning

## Database Schema Modifications

### PostgreSQL Changes
```sql
-- Add soft-delete support
ALTER TABLE recordings
ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE;

-- Add index for soft-delete queries
CREATE INDEX idx_recordings_deleted_at
ON recordings(deleted_at)
WHERE deleted_at IS NULL;

-- Add triggers for cascade operations
CREATE OR REPLACE FUNCTION cascade_delete_recording()
RETURNS TRIGGER AS $$
BEGIN
  -- Cleanup related data
  DELETE FROM metadata WHERE recording_id = OLD.id;
  DELETE FROM tracklists WHERE recording_id = OLD.id;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER recording_delete_cascade
BEFORE DELETE ON recordings
FOR EACH ROW
EXECUTE FUNCTION cascade_delete_recording();
```

### Neo4j Changes
- Implement cleanup queries for orphaned nodes
- Add timestamp properties for soft-delete tracking

## Message Queue Considerations

### RabbitMQ Topics
- Consider separate queues for different event types
- Implement dead letter queue for failed operations
- Add message versioning for backward compatibility

### Message Flow
```
file_watcher → lifecycle_events_exchange →
  ├→ cataloging_queue (all events)
  ├→ analysis_queue (created/modified/deleted)
  └→ tracklist_queue (deleted/moved/renamed)
```

## Error Handling and Recovery

### Failure Scenarios
1. **Partial Update Failure**: Use transactions to ensure atomicity
2. **Service Unavailable**: Queue messages for retry
3. **Data Inconsistency**: Implement reconciliation job
4. **Lost Events**: Add event sourcing for replay capability

### Recovery Mechanisms
- Soft-delete allows recovery of accidentally deleted files
- Event log maintains history of all operations
- Reconciliation job compares filesystem with database
- Manual cleanup tools for orphaned data

## Performance Considerations

### Optimization Strategies
- Batch delete operations for better performance
- Use async processing for non-critical cleanup
- Implement caching for frequently accessed paths
- Index database columns used in lifecycle queries

### Expected Load
- File creations: 100-1000 per hour
- File modifications: 10-100 per hour
- File deletions: 10-50 per hour
- File moves/renames: 1-10 per hour

## Migration Strategy

### Rollout Plan
1. **Phase 1**: Deploy file_watcher with event types (backward compatible)
2. **Phase 2**: Update cataloging_service to handle new events
3. **Phase 3**: Update analysis_service and tracklist_service
4. **Phase 4**: Enable soft-delete functionality
5. **Phase 5**: Add monitoring and alerting

### Backward Compatibility
- Services must handle both old and new message formats
- Gradual migration using feature flags
- Maintain legacy handlers during transition

## Monitoring and Alerting

### Key Metrics
- Event processing rate by type
- Failed cascade operations
- Orphaned records count
- Soft-deleted records count
- Event processing latency

### Alerts
- High rate of delete events (potential data loss)
- Failed cascade operations
- Growing orphaned data
- Message queue backlog

## Testing Strategy

### Test Scenarios
1. Single file lifecycle (create → modify → rename → delete)
2. Bulk operations (1000+ files)
3. Concurrent operations on same file
4. Service failure during lifecycle operation
5. Recovery from soft-delete
6. Cascade deletion verification

### Integration Tests
```python
def test_file_deletion_cascade():
    # Create file and wait for processing
    create_file("test.mp3")
    wait_for_processing()

    # Verify data exists
    assert recording_exists("test.mp3")
    assert metadata_exists("test.mp3")
    assert neo4j_node_exists("test.mp3")

    # Delete file
    delete_file("test.mp3")
    wait_for_processing()

    # Verify cascade deletion
    assert not recording_exists("test.mp3")
    assert not metadata_exists("test.mp3")
    assert not neo4j_node_exists("test.mp3")
```

## Security Considerations

- Validate file paths to prevent directory traversal
- Audit log all delete operations
- Implement role-based access for recovery operations
- Encrypt sensitive file paths in messages

## Future Enhancements

1. **Event Sourcing**: Store all events for complete history
2. **Batch Operations**: Handle bulk moves/renames efficiently
3. **Smart Cleanup**: ML-based identification of orphaned data
4. **File Versioning**: Track file modifications over time
5. **Undo Operations**: Allow reversal of recent operations

## Conclusion

Implementing complete file lifecycle tracking is essential for maintaining data consistency in the tracktion system. This change affects all services and requires careful coordination during implementation. The benefits include:
- Accurate file system synchronization
- No orphaned data
- Recovery capabilities through soft-delete
- Complete audit trail of file operations
- Foundation for future enhancements

The implementation should follow the phased approach outlined in Epic 7, with careful testing at each stage to ensure system stability.
