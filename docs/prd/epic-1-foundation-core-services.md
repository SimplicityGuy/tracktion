# Epic 1 Foundation & Core Services
This epic will establish all the necessary project infrastructure, including the monorepo structure, Docker configuration, and the initial setup for all datastores (PostgreSQL, Neo4j, Redis) and the message queue (RabbitMQ). It will also deliver a functional, minimal end-to-end flow for the file watcher and cataloging services. The stories are designed to be logically sequential, building upon each other to create a stable and extensible foundation.

## Story 1.1 Project Setup & Dockerization
**As a** music-loving developer,
**I want** a project with a defined monorepo structure and Docker configurations,
**so that** I can easily set up the development environment and prepare for deployment.
### Acceptance Criteria
1.  The project repository contains a `docker-compose.yaml` file to orchestrate all services.
2.  The project structure adheres to the defined monorepo pattern.
3.  A basic Python service is containerized and runs successfully via `docker-compose`.
