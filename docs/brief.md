# Project Brief: tracktion

## Executive Summary
This document outlines the project plan for **tracktion**, a Python-based backend system designed to manage, catalog, and enrich a large collection of digital music recordings, primarily live DJ mixes and concert recordings. The core problem is the lack of a structured, automated way to manage a growing digital music library, resulting in a disorganized collection that is difficult to search and enjoy.

The solution is an automated system that will perform the following functions:
* **Cataloging:** A file watcher will monitor specified directories for new content and automatically add it to the catalog.
* **Metadata Analysis:** Recordings will be analyzed to capture key characteristics such as BPM, mood, and key, with the ability to expand these characteristics in the future.
* **Naming Standardization:** Recordings will be automatically renamed based on a consistent, user-defined pattern.
* **Tracklist Integration:** For recordings with existing tracklists, the system will create a sidecar cue file. For those without, it will query the website `1001tracklists.com` to find and generate the tracklist.

This project will be built using modern Python practices (3.12+), with a focus on a scalable, containerized architecture using Docker. Key datastores will include PostgreSQL for core data, Neo4j for powerful graph-based analysis, and Redis for caching. A message queue system, such as RabbitMQ, will be used to ensure scalability and decoupling of services. The initial focus is on the backend to create a robust and functional foundation for a future frontend application.

## Problem Statement
This project addresses the significant challenges of managing a large digital library of live music recordings, a category of content that is often poorly handled by mainstream music management applications. The current state is one of disorganization, with thousands of individual files lacking consistent metadata, making it impossible to effectively search, categorize, and discover music within the collection.

The primary pain points are:

* **Lack of Standardization:** File names, artists, and event data are inconsistent, preventing automated sorting and a cohesive library experience.
* **Missing Metadata:** Key characteristics like BPM, musical key, and mood are often absent, which are essential for music organization, particularly for DJ use.
* **Manual Effort:** Manually identifying tracklists for live mixes is a time-consuming and often impossible task, leaving many recordings without proper segmentation or context.
* **Limited Tools:** Existing digital music management solutions, which are typically designed for commercially released albums and singles, do not support the unique attributes of live recordings, nor do they offer the extensibility needed to add new analytical characteristics.

The urgency of solving this problem is increasing as the collection grows, making manual intervention unfeasible and hindering the ability to enjoy the full value of the music library. The core problem is not merely a technical one of storage, but a user experience problem rooted in the lack of effective tools tailored to this specific type of content.

## Proposed Solution
The **tracktion** project will address the identified problem by creating a robust, extensible backend system specifically designed for managing live music recordings. The core concept is an automated "Vibe CEO" system that handles the heavy lifting of data management and analysis, freeing the user to focus on enjoying their music.

Key differentiators from existing solutions include:
* **Specialized Focus:** Unlike general music management tools, this solution is tailored to the unique needs of live DJ mixes and concert recordings, including support for long-form audio and advanced metadata like BPM and key.
* **Extensible Architecture:** The system's design will allow for the seamless addition of new data points for analysis, such as "mood" or "genre" detection, via a modular architecture.
* **Automated Workflow:** The file-watcher and analysis pipeline will automate the entire process from discovery to enrichment, eliminating the need for manual organization.
* **External Data Integration:** The ability to query `1001tracklists.com` directly for missing tracklists provides a significant advantage over tools that rely solely on embedded metadata.

The high-level vision is to create a dynamic, self-organizing digital music library that not only catalogs and standardizes content but also enriches it with valuable, searchable metadata. This solution will succeed where others have failed by embracing the unique nature of live recordings and providing a dedicated, automated, and extensible platform.

## Target Users
**Primary User Segment: Digital Music Enthusiast / DJ**
This segment consists of individuals who have a large, actively growing personal library of digital music, with a significant portion being live recordings and DJ mixes. They are technically proficient enough to manage their files but lack an effective, automated system for organization. Their primary goal is to efficiently catalog their collection, quickly find specific tracks, and discover new listening experiences based on metadata like BPM or mood.

**Secondary User Segment: The music-loving developer**
This segment represents the user who is able to provide a code base, and is able to install the project with some direction. They are technical and have an expansive knowledge of the genre, but they are not the primary target of the project itself, rather the project is for their own personal use. This user may provide additional functionality or modules, so this should be considered.

## Goals & Success Metrics
#### **Business Objectives**
* **Create a functional MVP:** Deliver a working backend system that can perform all core functions outlined in the problem statement within the initial development phase.
* **Establish a reusable foundation:** Build a scalable architecture that can be extended for future features and a potential frontend without significant rework.
* **Validate the technical approach:** Prove that the chosen technologies (Python 3.12+, Docker, PostgreSQL, Neo4j, Redis, RabbitMQ) can successfully integrate to support the project's goals.

#### **User Success Metrics**
* **Increase cataloging speed:** Automatically process new recordings within a target timeframe of one minute per file.
* **Improve data accuracy:** Successfully retrieve and apply accurate tracklists from `1001tracklists.com` for at least 80% of recordings that lack them.
* **Reduce manual effort:** Eliminate the need for manual file renaming and metadata tagging for new content.

#### **Key Performance Indicators (KPIs)**
* **Average processing time per file:** The average time from file detection by the watcher to final cataloging and analysis. Target: < 1 minute.
* **Tracklist retrieval success rate:** The percentage of recordings successfully matched and enriched with a tracklist from `1001tracklists.com`. Target: > 80%.
* **System uptime:** The percentage of time the backend services are operational. Target: > 99.9%.

## MVP Scope
The Minimum Viable Product (MVP) for **tracktion** will focus exclusively on the core backend services necessary to manage and enrich the music library. This initial phase will deliver a fully functional, automated system without a graphical user interface. The user will be able to run the system via command-line tools or a containerized environment to achieve the core objectives.

#### **Core Features (Must Have)**
* **File Watcher:** A service that monitors designated directories for new music files.
* **Cataloging Service:** A service that identifies and logs new recordings in the primary data store (PostgreSQL).
* **Metadata Analysis Service:** A service that analyzes audio files to extract key metadata (BPM, key, mood) and stores this information in Neo4j.
* **Cue File Generation:** A service that creates standardized cue files from existing tracklists.
* **Tracklist Query Service:** A service that queries `1001tracklists.com` for missing tracklists.
* **Message Queue Integration:** A system using RabbitMQ to orchestrate and scale the various services.
* **YAML-based Configuration:** All service settings (directory paths, database connections) will be managed via YAML files.
* **Docker Deployment:** The entire system will be packaged in Docker containers for easy deployment.

#### **Out of Scope for MVP**
* **Frontend Application:** A user-facing web or desktop application for interacting with the catalog.
* **User Authentication:** No user login or access control is required in this phase.
* **Advanced Music Discovery:** Features such as collaborative filtering or personalized recommendations are deferred.
* **Detailed Reporting:** Dashboards or complex analytics beyond basic success metrics are not part of the MVP.

#### **MVP Success Criteria**
The MVP is considered successful when all core backend services are operational, can be deployed via Docker, and can process and enrich a music library automatically and reliably. The system's ability to process new files and retrieve tracklists from the external source must meet the defined KPIs.

## Post-MVP Vision
This section outlines the long-term vision for the **tracktion** project beyond the initial MVP. It provides a strategic roadmap without committing to specific timelines or features.

* **Phase 2 Features:**
    * **Frontend Application:** Develop a user-friendly web interface for visual interaction with the music catalog.
    * **Advanced Metadata:** Integrate more sophisticated analysis tools for mood, genre, and track-matching.
    * **User Management:** Implement user profiles, authentication, and personalized library features.
    * **Integration with Streaming Services:** Explore integration with platforms like Spotify or SoundCloud for enhanced discovery.

* **Long-term Vision:**
    * To evolve **tracktion** from a personal tool into a platform for sharing and discovering live music sets. This could include building a community around the platform, allowing users to share their curated mixes and tracklists.

* **Expansion Opportunities:**
    * **Mobile Application:** Develop native iOS and Android applications for on-the-go music management.
    * **External API:** Expose a public API for developers to build third-party applications on top of the **tracktion** platform.
    * **Machine Learning:** Utilize machine learning models to predict tracklists, automate mood tagging, and create personalized recommendations.

## Technical Considerations
This section documents initial thoughts and preferences regarding the project's technical stack. These are not final decisions but will guide the next step of the project, which is architectural design.

* **Platform Requirements:**
    * **Target Platforms:** The project's initial focus is on the backend, which will be deployable via **Docker containers**.
    * **Browser/OS Support:** Not applicable for the MVP.
    * **Performance Requirements:** The system should be able to process new recordings efficiently, with a focus on quick analysis and cataloging.

* **Technology Preferences:**
    * **Backend:** Python 3.12 or higher, following modern best practices (e.g., using `uv`, `ruff`).
    * **Database:** PostgreSQL for structured data and Neo4j for graph-based relationships.
    * **Caching:** Redis for high-speed caching.
    * **Messaging:** RabbitMQ for an asynchronous, message-queue-based architecture.
    * **Hosting/Infrastructure:** To be determined later, but the Docker-based deployment should make it platform-agnostic.

* **Architecture Considerations:**
    * **Repository Structure:** To be determined based on the final architecture.
    * **Service Architecture:** A scalable, decoupled architecture using a message queue system is the primary goal.
    * **Integration Requirements:** The primary integration will be with the external website `1001tracklists.com` via a web scraper or API.
    * **Security/Compliance:** To be defined in the full architecture document, but best practices for data integrity and secure connections will be a priority.

## Constraints & Assumptions
This section outlines the known limitations, constraints, and key assumptions for the project, ensuring that expectations are realistic and risks are identified early.

* **Constraints**
    * **Budget:** This is a personal project, so development effort should be managed to avoid significant financial costs associated with cloud services or excessive model token usage.
    * **Timeline:** There is no hard deadline, but the project should prioritize a quick path to a functional MVP for the core backend services.
    * **Resources:** The project relies on the development effort of a single individual, the "music-loving developer."
    * **Technical:** The project must adhere to the specified technology stack (Python 3.12+, Docker, PostgreSQL, Neo4j, Redis, RabbitMQ) and be deployable in a containerized environment.

* **Key Assumptions**
    * **Audio Analysis:** It is assumed that existing open-source libraries or a queryable service can accurately and reliably extract metadata such as BPM, key, and mood from digital music files.
    * **External Data:** The service `1001tracklists.com` is a reliable and accessible source for a significant number of the missing tracklists.
    * **File Formats:** The digital music collection primarily consists of common audio formats (e.g., MP3, FLAC) that can be processed by available libraries.

## Risks & Open Questions
This section proactively identifies potential challenges and uncertainties, allowing for a strategic approach to mitigate them.

* **Key Risks**
    * **External Data Source Changes:** `1001tracklists.com` could change its website structure or API, breaking the tracklist query service.
    * **Audio Analysis Accuracy:** The chosen open-source libraries may not be consistently accurate in extracting metadata like mood or key from a diverse collection of recordings.
    * **Scalability Bottlenecks:** While a message queue is planned, unexpected performance issues could arise with large-scale processing of a huge library of music files.

* **Open Questions**
    * What is the precise, established naming pattern for file and folder organization?
    * Are there any specific audio file formats that need to be prioritized or excluded from the analysis?
    * What are the specific definitions of "mood" or other extensible characteristics that should be captured from the music?

* **Areas Needing Further Research**
    * Investigate the robustness and long-term viability of different open-source audio analysis libraries.
    * Research best practices for building a resilient web scraper that can handle changes to the target website's structure.
    * Explore the most efficient way to store and query the graph data in Neo4j to support future complex analytical features.

## Appendices
This section is reserved for any supplementary information, such as research summaries or stakeholder input. As we have completed the core sections of the brief, this section is currently empty.

## Next Steps
#### **Immediate Actions**
1.  Review and confirm the full Project Brief document.
2.  Address the open questions and areas for further research identified in the "Risks & Open Questions" section.

#### **PM Handoff**
This Project Brief provides the full context for tracktion. The next logical step is to create a comprehensive Product Requirements Document (PRD).
