# Technical Assumptions
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
