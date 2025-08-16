# Source Tree

This section defines the project's folder structure, which is a crucial blueprint for AI agents and developers. A well-organized structure ensures consistency and clarity. Based on our decision to use a **microservices architecture** within a **monorepo**, I have drafted the following project folder structure.

```plaintext
tracktion/
├── services/
│   ├── file_watcher/
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── cataloging_service/
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── analysis_service/
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── tracklist_service/
│       ├── src/
│       ├── Dockerfile
│       └── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── prd.md
│   └── project-brief.md
├── shared/
│   ├── core_types/
│   │   └── pyproject.toml
│   ├── docker/
│   ├── utils/
│   └── pyproject.toml
├── infrastructure/
│   ├── docker-compose.yaml
│   └── rabbitmq/
│       └── definitions.json
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── README.md
└── pyproject.toml
```

**Rationale:** The proposed structure organizes the project logically by service within a `services/` directory, which is a common pattern for monorepos. A `shared/` directory is included for reusable code and configuration, preventing duplication. The `infrastructure/` folder centralizes deployment-related files, such as the `docker-compose.yaml` file, while each service retains its own `Dockerfile` for independent containerization. This structure provides a clear, scalable, and maintainable foundation for the project.
