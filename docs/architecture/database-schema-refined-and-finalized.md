# Database Schema (Refined and Finalized)

This section translates our conceptual data models into concrete database schemas for the chosen technologies: PostgreSQL and Neo4j. This design ensures data integrity, efficiency, and scalability for all project requirements.

### **PostgreSQL Schema (Relational Data)**

This schema handles the structured data, including the core `Recording` and `Tracklist` information. It has been refined for better performance and robustness.

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE recordings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    sha256_hash VARCHAR(64) UNIQUE,
    xxh128_hash VARCHAR(32) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tracklists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id UUID REFERENCES recordings(id),
    source VARCHAR(255) NOT NULL,
    cue_file_path TEXT,
    tracks JSONB
);

CREATE TABLE metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id UUID REFERENCES recordings(id),
    key VARCHAR(255) NOT NULL,
    value TEXT NOT NULL
);

CREATE INDEX idx_metadata_recording_id ON metadata (recording_id);
CREATE INDEX idx_metadata_key ON metadata (key);
```

**Rationale:** The refined PostgreSQL schema now includes a UUID generator extension and more specific data types for certain fields. It also adds indexes on the `metadata` table to optimize queries, which is a key consideration for performance and scalability. This design is highly extensible and robust, making it suitable for long-term use.

### **Neo4j Conceptual Schema (Refined and Finalized)**

Neo4j will be used to store and query rich, relationship-based data. This refined schema provides a more explicit blueprint for implementation, including properties on nodes and relationships.

```text
(Recording:Recording { uuid: '...' })-[:HAS_METADATA]->(Metadata:Metadata { key: 'bpm', value: '128' })
(Recording:Recording { uuid: '...' })-[:HAS_TRACKLIST]->(Tracklist:Tracklist { source: '1001tracklists.com' })
(Tracklist:Tracklist { uuid: '...' })-[:CONTAINS_TRACK { start_time: '0:00' }]->(Track:Track { title: '...', artist: '...' })
```

**Rationale:** This conceptual schema is now more detailed, specifying key properties that will be stored on nodes and relationships. This clarity is essential for a developer to accurately implement the graph database and for enabling the powerful, complex queries that are a core part of the project's long-term vision.
