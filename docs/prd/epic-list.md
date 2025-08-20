# Epic List
The project will be structured into the following epics. Each epic is a cohesive unit of functionality designed to be delivered incrementally, building upon the work of the previous epic.

1.  **Epic 1: Foundation & Core Services:** Establish the foundational project infrastructure, including project setup, Dockerization, and the core services for file watching and cataloging. This epic will also set up the datastores (PostgreSQL, Neo4j, Redis) and the message queue (RabbitMQ).
2.  **Epic 2: Metadata Analysis & Naming:** Build the services required to analyze audio files for metadata (BPM, key, mood) and implement the automatic file renaming functionality based on a defined pattern.
3.  **Epic 3: Missing File Format:** Currently the Ogg Vorbis file format is not supported, but it should be.
4.  **Epic 4: Build `1001tracklists.com` API:** There is no API for `1001tracklists.com`, so an API needs to be developed for it using web scraping.
5.  **Epic 5: Build a Robust CUE File Handler:** There is no robust CUE file handler currently. An API needs to be developed for creating, updating, and reading CUE files.
6.  **Epic 6: Tracklist Management:** Develop the services for querying `1001tracklists.com` and generating CUE files from both new and existing tracklist data. This epic will also address the integration of CUE files with the core catalog.
7.  **Epic 7: Asynchronous Refactor:** Ensure that the entire service is written using asynchronous practices where ever possible.
