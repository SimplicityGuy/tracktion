# Service Interaction Diagrams

This document provides visual representations of how the different services in the Tracktion ecosystem interact with each other.

## High-Level System Architecture

```mermaid
graph TB
    FW[File Watcher Service] --> MQ[Message Queue<br/>RabbitMQ]
    MQ --> AS[Analysis Service]
    MQ --> CS[Cataloging Service]
    MQ --> FRS[File Rename Service]
    MQ --> NS[Notification Service]

    AS --> DB[(PostgreSQL<br/>Database)]
    AS --> NEO[(Neo4j<br/>Graph DB)]
    AS --> CACHE[(Redis Cache)]

    CS --> DB
    FRS --> DB

    TS[Tracklist Service] --> DB
    TS --> CACHE
    TS --> MQ

    AS --> API[Analysis API]
    TS --> TSAPI[Tracklist API]

    API --> CLIENT[Client Applications]
    TSAPI --> CLIENT

    NS --> DISCORD[Discord Webhooks]
    NS --> CACHE
```

## Message Flow Architecture

```mermaid
sequenceDiagram
    participant FW as File Watcher
    participant MQ as RabbitMQ
    participant AS as Analysis Service
    participant CS as Cataloging Service
    participant FRS as File Rename Service
    participant NS as Notification Service
    participant DB as Database

    FW->>MQ: File Detected Event
    Note over MQ: Route to appropriate queues

    MQ->>AS: Analysis Request
    AS->>AS: Extract Metadata + Audio Analysis
    AS->>DB: Store Analysis Results
    AS->>MQ: Analysis Complete Event

    MQ->>CS: Catalog Request
    CS->>DB: Update Catalog
    CS->>MQ: Catalog Complete Event

    MQ->>FRS: Rename Proposal Request
    FRS->>DB: Generate Rename Proposal
    FRS->>MQ: Proposal Generated Event

    MQ->>NS: Send Notification
    NS->>NS: Process & Rate Limit
    NS-->>Discord: Send Notification
```

## Service Dependencies

```mermaid
graph LR
    subgraph "Core Infrastructure"
        MQ[RabbitMQ]
        DB[(PostgreSQL)]
        NEO[(Neo4j)]
        REDIS[(Redis)]
    end

    subgraph "Processing Services"
        FW[File Watcher] --> MQ
        AS[Analysis Service] --> MQ
        AS --> DB
        AS --> NEO
        AS --> REDIS

        CS[Cataloging Service] --> MQ
        CS --> DB

        FRS[File Rename Service] --> MQ
        FRS --> DB
    end

    subgraph "User-Facing Services"
        TS[Tracklist Service] --> DB
        TS --> REDIS
        TS --> MQ
    end

    subgraph "Support Services"
        NS[Notification Service] --> MQ
        NS --> REDIS
    end
```

## Data Flow Diagram

```mermaid
flowchart TD
    A[Audio File Added/Modified] --> FW[File Watcher Service]
    FW --> |File Event| MQ{Message Queue}

    MQ --> |analysis_queue| AS[Analysis Service]
    MQ --> |catalog_queue| CS[Cataloging Service]
    MQ --> |rename_queue| FRS[File Rename Service]
    MQ --> |notification_queue| NS[Notification Service]

    AS --> |Metadata| PGDB[(PostgreSQL)]
    AS --> |Relationships| NGDB[(Neo4j)]
    AS --> |Cache Results| RDS[(Redis)]

    CS --> |Catalog Entry| PGDB
    FRS --> |Rename Proposal| PGDB

    TS[Tracklist Service] --> |Query Data| PGDB
    TS --> |Cache Searches| RDS

    NS --> |Notifications| EXT[External Services<br/>Discord, Email]

    API[REST API] --> TS
    API --> AS

    WEB[Web Interface] --> API
    CLIENT[Client Apps] --> API
```

## Service Communication Patterns

### 1. Event-Driven Architecture
```mermaid
graph LR
    FW[File Watcher] --> |file.created| MQ[Message Queue]
    MQ --> |file.analyze| AS[Analysis Service]
    AS --> |analysis.complete| MQ
    MQ --> |catalog.update| CS[Cataloging Service]
    MQ --> |rename.generate| FRS[File Rename Service]
    MQ --> |notify.user| NS[Notification Service]
```

### 2. Request-Response Pattern
```mermaid
graph LR
    CLIENT[Client] --> |HTTP Request| API[API Gateway]
    API --> |Query| TS[Tracklist Service]
    TS --> |Result| API
    API --> |HTTP Response| CLIENT
```

### 3. Background Processing
```mermaid
graph TB
    MQ[Message Queue] --> |Batch Messages| AS[Analysis Service]
    AS --> |Process Batch| WORKER[Worker Processes]
    WORKER --> |Store Results| DB[(Database)]
    WORKER --> |Cache Results| CACHE[(Redis)]
```

## Integration Points

### Analysis Service Integration
- **Input**: File metadata from File Watcher
- **Processing**: BPM detection, key analysis, mood analysis
- **Output**: Analysis results to database, completion events to queue
- **APIs**: REST API for querying analysis results

### Tracklist Service Integration
- **Input**: User search requests, catalog data
- **Processing**: Web scraping, search indexing, CUE generation
- **Output**: Search results, tracklist data, CUE files
- **APIs**: REST API for search and tracklist operations

### File Watcher Integration
- **Input**: File system events
- **Processing**: File monitoring, metadata extraction, hash generation
- **Output**: File events to message queue
- **APIs**: None (internal service)

### Cataloging Service Integration
- **Input**: File and analysis events from queue
- **Processing**: Catalog organization, metadata consolidation
- **Output**: Updated catalog in database
- **APIs**: Internal database queries

### File Rename Service Integration
- **Input**: Analysis completion events
- **Processing**: Generate rename proposals based on metadata
- **Output**: Rename proposals in database
- **APIs**: Proposal management endpoints

### Notification Service Integration
- **Input**: Events from all services via queue
- **Processing**: Rate limiting, formatting, delivery
- **Output**: Notifications to external services (Discord, email)
- **APIs**: Admin endpoints for notification management

## Error Handling & Circuit Breakers

```mermaid
graph TB
    AS[Analysis Service] --> |Request| DB[(Database)]
    DB --> |Response| AS

    AS --> |Failure| CB{Circuit Breaker}
    CB --> |OPEN| FW[Fallback Workflow]
    CB --> |CLOSED| RETRY[Retry Logic]
    CB --> |HALF-OPEN| TEST[Test Request]

    FW --> QUEUE[Message Queue]
    RETRY --> DB
    TEST --> DB
```

## Monitoring & Health Checks

```mermaid
graph LR
    HC[Health Check Endpoint] --> AS[Analysis Service]
    HC --> TS[Tracklist Service]
    HC --> FW[File Watcher]
    HC --> CS[Cataloging Service]
    HC --> FRS[File Rename Service]
    HC --> NS[Notification Service]

    AS --> |Status| MON[Monitoring Dashboard]
    TS --> |Status| MON
    FW --> |Status| MON
    CS --> |Status| MON
    FRS --> |Status| MON
    NS --> |Status| MON

    MON --> |Alerts| NS
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end

    subgraph "Application Layer"
        AS1[Analysis Service #1]
        AS2[Analysis Service #2]
        TS1[Tracklist Service #1]
        TS2[Tracklist Service #2]
        FW[File Watcher Service]
        NS[Notification Service]
    end

    subgraph "Message Queue"
        MQ[RabbitMQ Cluster]
    end

    subgraph "Data Layer"
        DB[PostgreSQL Primary]
        DBR[PostgreSQL Replica]
        NEO[Neo4j Cluster]
        REDIS[Redis Cluster]
    end

    LB --> AS1
    LB --> AS2
    LB --> TS1
    LB --> TS2

    AS1 --> MQ
    AS2 --> MQ
    TS1 --> MQ
    TS2 --> MQ
    FW --> MQ
    NS --> MQ

    AS1 --> DB
    AS1 --> NEO
    AS1 --> REDIS
    AS2 --> DB
    AS2 --> NEO
    AS2 --> REDIS

    TS1 --> DB
    TS1 --> REDIS
    TS2 --> DB
    TS2 --> REDIS

    DB --> DBR
```

## Message Queue Topics & Routing

### Exchange Configuration
- **Main Exchange**: `tracktion_exchange` (topic)
- **Dead Letter Exchange**: `tracktion_dlx` (direct)

### Routing Keys
```
file.created        -> analysis_queue, catalog_queue
file.modified       -> analysis_queue, catalog_queue
file.deleted        -> catalog_queue, cleanup_queue
analysis.complete   -> catalog_queue, rename_queue, notification_queue
catalog.updated     -> notification_queue
rename.generated    -> notification_queue
error.*            -> notification_queue, dlq_queue
```

### Queue Configuration
```mermaid
graph LR
    EX[tracktion_exchange] --> |file.*| AQ[analysis_queue]
    EX --> |file.*| CQ[catalog_queue]
    EX --> |analysis.*| RQ[rename_queue]
    EX --> |*.*| NQ[notification_queue]
    EX --> |error.*| DLQ[dead_letter_queue]

    AQ --> AS[Analysis Service]
    CQ --> CS[Cataloging Service]
    RQ --> FRS[File Rename Service]
    NQ --> NS[Notification Service]
```

## Security Architecture

```mermaid
graph TB
    CLIENT[Client Applications] --> |HTTPS| LB[Load Balancer]
    LB --> |Internal Network| API[API Gateway]

    API --> |JWT Auth| AS[Analysis Service]
    API --> |JWT Auth| TS[Tracklist Service]

    AS --> |Encrypted| DB[(Database)]
    TS --> |Encrypted| DB

    AS --> |TLS| MQ[Message Queue]
    TS --> |TLS| MQ
    FW --> |TLS| MQ

    MQ --> |Internal| NS[Notification Service]
    NS --> |Webhook Auth| EXT[External Services]

    subgraph "Security Layer"
        FW2[Firewall]
        VPN[VPN Gateway]
        IAM[Identity Management]
        CERT[Certificate Manager]
    end
```

This comprehensive set of diagrams shows how all services in the Tracktion ecosystem interact, communicate, and depend on each other for a cohesive music analysis and management system.
