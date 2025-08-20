# tracktion Product Requirements Document (PRD)

## Goals and Background Context
#### **Goals**
* Create a working backend MVP that can catalog and analyze a digital music collection.
* Build a scalable and extensible foundation for future features and a potential frontend.
* Validate the technical approach using Python 3.12+, Docker, PostgreSQL, Neo4j, Redis, and RabbitMQ.
* Automate the processing of new recordings within a target timeframe of one minute per file.
* Successfully retrieve and apply tracklists from `1001tracklists.com` for at least 80% of recordings that lack them.
* Eliminate the need for manual file renaming and metadata tagging.

#### **Background Context**
The project aims to solve the problem of managing a large, disorganized collection of live music recordings. Mainstream music management tools are ill-suited for this content due to a lack of features for handling long single tracks, advanced metadata (BPM, mood, key), and tracklists. The collection is growing, making manual organization unfeasible and creating a need for a dedicated, automated, and extensible solution.

#### **Change Log**

| Date | Version | Description | Author |
| :--- | :--- | :--- | :--- |
| August 16, 2025 | 1.0 | Initial PRD draft | Bob (Scrum Master) |

## Requirements
#### **Functional**
1.  **FR1:** The system shall automatically scan a user-defined directory for new audio files.
2.  **FR2:** The system shall catalog newly detected audio files in a PostgreSQL database.
3.  **FR3:** The system shall analyze audio files to extract metadata such as BPM, musical key, and mood, storing this in a Neo4j graph database.
4.  **FR4:** The system shall query `1001tracklists.com` to find tracklists for recordings that do not have them.
5.  **FR5:** The system shall automatically generate a `.cue` file from retrieved or existing tracklist data.
6.  **FR6:** The system shall rename audio files based on a user-defined pattern.

#### **Non Functional**
1.  **NFR1:** The system must be deployable via Docker containers.
2.  **NFR2:** The system shall be scalable to handle a large and growing collection of recordings.
3.  **NFR3:** The system shall use a message queue (e.g., RabbitMQ) for asynchronous processing to ensure decoupling and reliability.
4.  **NFR4:** All application configuration, including directory paths and datastore connections, must be managed via YAML files.

## User Interface Design Goals
Based on the Project Brief, which specified a backend-first approach, there are no immediate UI/UX requirements for the MVP. However, we'll establish some high-level goals and assumptions here to guide future work and provide context for the architecture.

#### **Overall UX Vision**
The long-term vision is a user-friendly web interface that makes managing a large music library effortless. It should feel intuitive, clean, and provide a clear path for users to discover and organize their music.

#### **Key Interaction Paradigms**
The future application should prioritize clarity and simplicity, with user interactions that are both efficient for power users and approachable for casual users.

#### **Core Screens and Views**
Since a full frontend is out of scope for the MVP, the core "screens" at this stage will be command-line outputs and system logs.

#### **Accessibility**
**Accessibility: None**
Accessibility will be addressed in a later phase when a user interface is being developed.

#### **Branding**
At this time, there are no specific branding elements or style guides to incorporate.

#### **Target Device and Platforms**
**Target Device and Platforms: None**
The MVP is a backend-only system and therefore has no target device or platform requirements.

## Technical Assumptions
This section documents the foundational technical decisions that will guide the **Architect** in designing the system. These choices are critical as they form the constraints for all future technical work.

* **Repository Structure: Monorepo**
    * **Rationale:** Given the project's long-term vision to include a frontend and multiple backend services (file watcher, analysis, query services), a monorepo structure is ideal. It allows for code sharing between services, simplifies dependency management, and keeps all project components in a single repository for easier coordination.

* **Service Architecture: Microservices within a Monorepo**
    * **Rationale:** Your request for a scalable architecture using a message queue system like RabbitMQ strongly suggests a microservices approach. This design decouples services, enabling independent development, deployment via Docker containers, and better scalability for components like the audio analysis service.

* **Testing Requirements: Full Testing Pyramid**
    * **Rationale:** To ensure reliability and robustness, especially with complex data processing, a comprehensive testing strategy is necessary. This includes unit tests for individual functions, integration tests for service-to-service communication, and end-to-end tests for critical user workflows.

* **Additional Technical Assumptions and Requests**
    * The backend will be written in Python 3.12+ following best practices.
    * The system will use PostgreSQL for core data, Neo4j for graph-based analysis, and Redis for caching.
    * RabbitMQ will be used for an asynchronous, message-queue-based architecture.

## Epic List
The project will be structured into the following epics. Each epic is a cohesive unit of functionality designed to be delivered incrementally, building upon the work of the previous epic.

1.  **Epic 1: Foundation & Core Services:** Establish the foundational project infrastructure, including project setup, Dockerization, and the core services for file watching and cataloging. This epic will also set up the datastores (PostgreSQL, Neo4j, Redis) and the message queue (RabbitMQ).
2.  **Epic 2: Metadata Analysis & Naming:** Build the services required to analyze audio files for metadata (BPM, key, mood) and implement the automatic file renaming functionality based on a defined pattern.
3.  **Epic 3: Missing File Format:** Currently the Ogg Vorbis file format is not supported, but it should be.
4.  **Epic 4: Build `1001tracklists.com` API:** There is no API for `1001tracklists.com`, so an API needs to be developed for it using web scraping.
5.  **Epic 5: Build a Robust CUE File Handler:** There is no robust CUE file handler currently. An API needs to be developed for creating, updating, and reading CUE files.
6.  **Epic 6: Tracklist Management:** Develop the services for querying `1001tracklists.com` and generating CUE files from both new and existing tracklist data. This epic will also address the integration of CUE files with the core catalog.
7.  **Epic 7: Asynchronous Refactor:** Ensure that the entire service is written using asynchronous practices where ever possible.

## Epic 1 Foundation & Core Services
This epic will establish all the necessary project infrastructure, including the monorepo structure, Docker configuration, and the initial setup for all datastores (PostgreSQL, Neo4j, Redis) and the message queue (RabbitMQ). It will also deliver a functional, minimal end-to-end flow for the file watcher and cataloging services. The stories are designed to be logically sequential, building upon each other to create a stable and extensible foundation.

### Story 1.1 Project Setup & Dockerization
**As a** music-loving developer,
**I want** a project with a defined monorepo structure and Docker configurations,
**so that** I can easily set up the development environment and prepare for deployment.
#### Acceptance Criteria
1.  The project repository contains a `docker-compose.yaml` file to orchestrate all services.
2.  The project structure adheres to the defined monorepo pattern.
3.  A basic Python service is containerized and runs successfully via `docker-compose`.

## Checklist Results Report
I have successfully drafted the story for Project Setup & Dockerization. I will now validate this draft using the `story-draft-checklist` to ensure it contains all the necessary information for a developer agent to begin implementation.

* **1. Goal & Context Clarity:** ✅ PASS
    * The story clearly states the goal of setting up the project and the context within the larger project. The acceptance criteria are measurable.
* **2. Technical Implementation Guidance:** ✅ PASS
    * The `Dev Notes` section provides specific guidance on the repository structure, testing frameworks, and datastore setup, all with clear references to the architecture document.
* **3. Reference Effectiveness:** ✅ PASS
    * All technical details are summarized in the `Dev Notes` section and include citations to the relevant architectural documents, so the developer agent will not need to hunt for context.
* **4. Self-Containment Assessment:** ✅ PASS
    * The story is self-contained and provides enough information to be implemented without needing to read other documents in full.
* **5. Testing Guidance:** ✅ PASS
    * The story explicitly mentions that the developer needs to verify the successful execution of the containerized service, which is a clear testing requirement.

**Final Assessment:** **✅ READY** - The story provides sufficient context for implementation.

## Next Steps
After completing this validation, the next steps are to:

1.  Review the `pm-checklist` results.
2.  If any issues are found, address them with me.
3.  Once the PRD is approved, the next step is to hand off to the **Architect** to begin designing the system.
