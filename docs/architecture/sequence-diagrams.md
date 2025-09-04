# Sequence Diagrams for Key Workflows

This document provides detailed sequence diagrams for critical workflows in the Tracktion system.

## Table of Contents

1. [File Processing Workflow](#file-processing-workflow)
2. [Audio Analysis Workflow](#audio-analysis-workflow)
3. [Tracklist Import Workflow](#tracklist-import-workflow)
4. [CUE Generation Workflow](#cue-generation-workflow)
5. [Search and Discovery Workflow](#search-and-discovery-workflow)
6. [Error Handling Workflow](#error-handling-workflow)
7. [Notification Workflow](#notification-workflow)

## File Processing Workflow

Complete workflow from file detection to catalog completion.

```mermaid
sequenceDiagram
    participant User
    participant FS as File System
    participant FW as File Watcher
    participant MQ as RabbitMQ
    participant AS as Analysis Service
    participant CS as Cataloging Service
    participant FRS as File Rename Service
    participant NS as Notification Service
    participant DB as PostgreSQL
    participant Neo as Neo4j
    participant Redis

    User->>FS: Add/Modify Audio File
    FS-->>FW: File System Event

    FW->>FW: Generate File Hash
    FW->>FW: Extract Basic Metadata
    FW->>MQ: Publish file.created Event

    Note over MQ: Route to Multiple Queues

    MQ->>AS: Analysis Request
    MQ->>CS: Catalog Request

    AS->>DB: Check if Already Processed
    alt File Not Processed
        AS->>AS: Extract Full Metadata
        AS->>AS: Perform Audio Analysis
        AS->>DB: Store Recording & Metadata
        AS->>Neo: Create Recording Node
        AS->>Redis: Cache Analysis Results
        AS->>MQ: analysis.complete Event
    else File Already Processed
        AS->>Redis: Return Cached Results
    end

    MQ->>FRS: Rename Proposal Request
    FRS->>DB: Generate Rename Proposal
    FRS->>MQ: proposal.generated Event

    CS->>DB: Update Catalog
    CS->>Neo: Update Relationships
    CS->>MQ: catalog.updated Event

    MQ->>NS: Success Notification
    NS->>NS: Format Notification
    NS->>User: Discord/Email Alert

    Note over User: File Processing Complete
```

## Audio Analysis Workflow

Detailed audio analysis process with multiple algorithms.

```mermaid
sequenceDiagram
    participant AS as Analysis Service
    participant MM as Model Manager
    participant BPM as BPM Detector
    participant KEY as Key Detector
    participant MOOD as Mood Analyzer
    participant GENRE as Genre Classifier
    participant DB as PostgreSQL
    participant Redis
    participant MQ as RabbitMQ

    AS->>Redis: Check Analysis Cache
    alt Cache Hit
        Redis-->>AS: Return Cached Results
    else Cache Miss
        AS->>MM: Load Required Models
        MM->>MM: Initialize TensorFlow Models
        MM-->>AS: Models Ready

        par Parallel Analysis
            AS->>BPM: Analyze BPM
            BPM->>BPM: Multi-Algorithm Detection
            BPM->>BPM: Temporal Analysis
            BPM-->>AS: BPM Results
        and
            AS->>KEY: Analyze Musical Key
            KEY->>KEY: Primary Algorithm (EDMA)
            KEY->>KEY: Validation Algorithm (HPCP)
            KEY->>KEY: Consensus Check
            KEY-->>AS: Key Results
        and
            AS->>MOOD: Analyze Mood Dimensions
            MOOD->>MM: Use Mood Model
            MOOD->>MOOD: Calculate Mood Scores
            MOOD-->>AS: Mood Results
        and
            AS->>GENRE: Classify Genre
            GENRE->>MM: Use EffNet Model
            GENRE->>GENRE: Multi-Genre Classification
            GENRE-->>AS: Genre Results
        end

        AS->>AS: Combine Analysis Results
        AS->>AS: Calculate Confidence Scores

        AS->>DB: Store Analysis Results
        AS->>Redis: Cache Results (TTL: 24h)
        AS->>MQ: analysis.complete Event
    end
```

## Tracklist Import Workflow

Web scraping and tracklist import process.

```mermaid
sequenceDiagram
    participant User
    participant TS as Tracklist Service
    participant WS as Web Scraper
    participant Cache as Redis Cache
    participant Parser as Content Parser
    participant Validator as Data Validator
    participant DB as PostgreSQL
    participant MQ as RabbitMQ
    participant NS as Notification Service

    User->>TS: Import Tracklist Request
    TS->>Cache: Check URL Cache
    alt Cache Hit (< 2 hours)
        Cache-->>TS: Return Cached Tracklist
        TS-->>User: Cached Results
    else Cache Miss or Expired
        TS->>TS: Validate Source URL
        TS->>WS: Scrape Tracklist Page

        WS->>WS: Check robots.txt
        WS->>WS: Apply Rate Limiting
        WS->>WS: Fetch Page Content
        WS->>WS: Handle JavaScript Rendering
        WS-->>TS: Raw HTML Content

        TS->>Parser: Parse Tracklist Data
        Parser->>Parser: Extract Track Information
        Parser->>Parser: Parse Timestamps
        Parser->>Parser: Extract Artist/Title
        Parser-->>TS: Structured Track Data

        TS->>Validator: Validate Track Data
        Validator->>Validator: Check Required Fields
        Validator->>Validator: Validate Time Formats
        Validator->>Validator: Detect Duplicates
        Validator-->>TS: Validated Tracklist

        alt Validation Successful
            TS->>DB: Store Tracklist
            TS->>Cache: Cache Results (TTL: 2h)
            TS->>MQ: tracklist.imported Event
            MQ->>NS: Success Notification
            TS-->>User: Import Successful
        else Validation Failed
            TS->>MQ: tracklist.import_failed Event
            MQ->>NS: Error Notification
            TS-->>User: Import Failed (Errors)
        end
    end
```

## CUE Generation Workflow

CUE sheet generation with format support.

```mermaid
sequenceDiagram
    participant User
    participant TS as Tracklist Service
    participant CUE as CUE Generator
    participant Formatter as Format Handler
    participant Validator as CUE Validator
    participant FS as File System
    participant DB as PostgreSQL

    User->>TS: Generate CUE Request
    TS->>DB: Fetch Tracklist Data
    DB-->>TS: Tracklist with Tracks

    TS->>CUE: Generate CUE Sheet
    CUE->>CUE: Validate Track Timing
    CUE->>CUE: Calculate Track Durations
    CUE->>CUE: Detect Timing Gaps

    alt Valid Timing
        CUE->>Formatter: Format for Target Platform

        alt Standard CUE Format
            Formatter->>Formatter: Generate Standard CUE
        else CDJ Format
            Formatter->>Formatter: Generate CDJ-Compatible CUE
        else Traktor Format
            Formatter->>Formatter: Generate Traktor CUE
        else Serato Format
            Formatter->>Formatter: Generate Serato CUE
        else Rekordbox Format
            Formatter->>Formatter: Generate Rekordbox CUE
        end

        Formatter-->>CUE: Formatted CUE Content

        CUE->>Validator: Validate Generated CUE
        Validator->>Validator: Check CUE Syntax
        Validator->>Validator: Validate Track References
        Validator->>Validator: Check Time Consistency

        alt Validation Successful
            Validator-->>CUE: CUE Valid
            CUE->>FS: Save CUE File
            CUE->>DB: Update Generation History
            CUE-->>TS: CUE Generated Successfully
            TS-->>User: CUE Download Link
        else Validation Failed
            Validator-->>CUE: Validation Errors
            CUE-->>TS: Generation Failed
            TS-->>User: Error Response
        end
    else Invalid Timing
        CUE-->>TS: Timing Validation Failed
        TS-->>User: Error Response (Fix Timing)
    end
```

## Search and Discovery Workflow

Advanced search with caching and ranking.

```mermaid
sequenceDiagram
    participant User
    participant TS as Tracklist Service
    participant Search as Search Engine
    participant Cache as Redis Cache
    participant DB as PostgreSQL
    participant Neo as Neo4j
    participant Ranker as Result Ranker
    participant WS as Web Scraper

    User->>TS: Search Query
    TS->>TS: Parse and Normalize Query
    TS->>Cache: Check Search Cache

    alt Cache Hit (< 15 min)
        Cache-->>TS: Cached Results
        TS-->>User: Search Results
    else Cache Miss
        TS->>Search: Execute Search

        par Parallel Search
            Search->>DB: Search Local Tracklists
            DB-->>Search: Local Results
        and
            Search->>Neo: Graph-Based Search
            Neo-->>Search: Related Results
        and
            Search->>WS: External Source Search
            WS-->>Search: External Results
        end

        Search->>Ranker: Combine and Rank Results
        Ranker->>Ranker: Apply Relevance Scoring
        Ranker->>Ranker: Boost Verified Results
        Ranker->>Ranker: Apply User Preferences
        Ranker-->>Search: Ranked Results

        Search->>Cache: Cache Results (TTL: 15m)
        Search-->>TS: Final Results
        TS-->>User: Search Results
    end

    User->>TS: Select Result for Details
    TS->>DB: Fetch Full Tracklist
    TS->>Cache: Cache Detailed View
    TS-->>User: Detailed Tracklist
```

## Error Handling Workflow

Comprehensive error handling and retry logic.

```mermaid
sequenceDiagram
    participant Service
    participant ErrorHandler as Error Handler
    participant RetryManager as Retry Manager
    participant CircuitBreaker as Circuit Breaker
    participant MQ as RabbitMQ
    participant NS as Notification Service
    participant DB as PostgreSQL
    participant Logs as Logging System

    Service->>Service: Execute Operation
    Service-->>ErrorHandler: Operation Failed

    ErrorHandler->>ErrorHandler: Classify Error Type

    alt Retryable Error
        ErrorHandler->>CircuitBreaker: Check Circuit State

        alt Circuit Closed
            CircuitBreaker-->>ErrorHandler: Allow Retry
            ErrorHandler->>RetryManager: Schedule Retry
            RetryManager->>RetryManager: Apply Exponential Backoff
            RetryManager->>Service: Retry Operation

            alt Retry Successful
                Service-->>ErrorHandler: Success
                ErrorHandler->>CircuitBreaker: Record Success
                ErrorHandler->>Logs: Log Recovery
            else Retry Failed
                Service-->>ErrorHandler: Still Failing
                ErrorHandler->>RetryManager: Check Max Retries

                alt Max Retries Not Reached
                    RetryManager->>Service: Retry Again
                else Max Retries Reached
                    RetryManager-->>ErrorHandler: Give Up
                    ErrorHandler->>CircuitBreaker: Record Failure
                    ErrorHandler->>MQ: Send to Dead Letter Queue
                    ErrorHandler->>NS: Critical Alert
                    ErrorHandler->>Logs: Log Failure
                end
            end
        else Circuit Open
            CircuitBreaker-->>ErrorHandler: Block Request
            ErrorHandler->>MQ: Send to Dead Letter Queue
            ErrorHandler->>NS: Circuit Open Alert
            ErrorHandler->>Logs: Log Circuit Open
        end
    else Non-Retryable Error
        ErrorHandler->>DB: Update Record Status
        ErrorHandler->>NS: Error Notification
        ErrorHandler->>Logs: Log Error Details
    end
```

## Notification Workflow

Multi-channel notification system with rate limiting.

```mermaid
sequenceDiagram
    participant Service
    participant NS as Notification Service
    participant RateLimit as Rate Limiter
    participant Formatter as Message Formatter
    participant Discord
    participant Email as Email Service
    participant Queue as Notification Queue
    participant History as Redis History

    Service->>NS: Send Notification Request
    NS->>NS: Classify Notification Type
    NS->>RateLimit: Check Rate Limits

    alt Rate Limit OK
        RateLimit-->>NS: Allow Notification
        NS->>Formatter: Format Message

        Formatter->>Formatter: Apply Template
        Formatter->>Formatter: Add Context Info
        Formatter->>Formatter: Format for Channels
        Formatter-->>NS: Formatted Messages

        par Multi-Channel Delivery
            NS->>Discord: Send Discord Message
            Discord-->>NS: Delivery Confirmation
        and
            NS->>Email: Send Email Alert
            Email-->>NS: Email Status
        end

        NS->>History: Log Notification
        NS->>RateLimit: Update Rate Counters
        NS-->>Service: Notification Sent
    else Rate Limit Exceeded
        RateLimit-->>NS: Rate Limited
        NS->>Queue: Queue for Later
        Queue->>Queue: Schedule Delivery
        NS-->>Service: Notification Queued

        Note over Queue: Wait for Rate Limit Reset
        Queue->>NS: Retry Notification
        NS->>RateLimit: Check Limits Again

        alt Limits Now OK
            NS->>Formatter: Process Queued Message
            NS->>Discord: Send Delayed Message
            NS->>History: Log Delayed Delivery
        else Still Rate Limited
            NS->>Queue: Re-queue Message
        end
    end
```

## Performance Optimization Workflows

### Caching Strategy

```mermaid
sequenceDiagram
    participant Client
    participant Service
    participant L1 as L1 Cache (Memory)
    participant L2 as L2 Cache (Redis)
    participant DB as Database

    Client->>Service: Request Data
    Service->>L1: Check Memory Cache

    alt L1 Hit
        L1-->>Service: Return Cached Data
        Service-->>Client: Fast Response (< 1ms)
    else L1 Miss
        Service->>L2: Check Redis Cache

        alt L2 Hit
            L2-->>Service: Return Cached Data
            Service->>L1: Store in Memory Cache
            Service-->>Client: Fast Response (< 10ms)
        else L2 Miss
            Service->>DB: Query Database
            DB-->>Service: Database Results
            Service->>L2: Cache in Redis (TTL)
            Service->>L1: Cache in Memory (Short TTL)
            Service-->>Client: Response (< 100ms)
        end
    end
```

### Background Processing

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Queue as Task Queue
    participant Worker1 as Worker 1
    participant Worker2 as Worker 2
    participant DB as Database
    participant WebSocket

    Client->>API: Submit Long-Running Task
    API->>Queue: Enqueue Task
    API->>DB: Create Task Record
    API-->>Client: Task ID + WebSocket URL

    Client->>WebSocket: Connect for Updates

    Queue->>Worker1: Assign Task
    Worker1->>DB: Update Task Status (Processing)
    Worker1->>WebSocket: Progress Update (25%)

    Worker1->>Worker1: Process Data
    Worker1->>WebSocket: Progress Update (50%)

    Worker1->>Worker1: Continue Processing
    Worker1->>WebSocket: Progress Update (75%)

    Worker1->>DB: Store Results
    Worker1->>DB: Update Task Status (Complete)
    Worker1->>WebSocket: Final Update (100%)

    WebSocket-->>Client: Task Complete Notification
    Client->>API: Retrieve Results
    API->>DB: Fetch Task Results
    API-->>Client: Final Results
```

These sequence diagrams provide detailed views of the most critical workflows in the Tracktion system, showing the interactions between services, error handling, performance optimizations, and user experience flows.
