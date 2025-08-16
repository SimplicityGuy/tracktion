# Tech Stack

The technology stack has been updated to include an Object-Relational Mapper (ORM) and a database migration tool.

| Category | Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13 | Primary development language | Latest stable version for optimal performance and security. |
| **Runtime** | N/A | N/A | Serverless functions | N/A, as this is a containerized microservices project, not a serverless one. |
| **Framework** | N/A | N/A | Web framework | N/A, as the MVP has no web interface. |
| **Database** | PostgreSQL | 17 | Primary data store | Latest version for enhanced performance, security, and features. |
| **Graph DB** | Neo4j | 5.26 | Graph data store | Latest community edition for graph analysis and relationship modeling. |
| **Cache** | Redis | 7.4 | Caching service | Latest stable version for high-speed key-value storage and caching. |
| **Messaging** | RabbitMQ | 4.0 | Message queue system | Latest version for improved performance and reliability in message handling. |
| **ORM** | SQLAlchemy | Latest | Database access | A powerful ORM that simplifies data access and provides a consistent interface to the PostgreSQL database. |
| **Migrations**| Alembic | Latest | Database schema management | A lightweight and robust migration tool that works seamlessly with SQLAlchemy. |
| **Deployment**| Docker | Latest | Containerization | Ensures a consistent and reproducible deployment environment for all services. |
| **Tooling** | uv, ruff | Latest | Dependency management, linting | Modern, high-performance tools for managing dependencies and maintaining code quality. |
