# Coding Standards

These standards are mandatory for all AI agents and developers working on the **tracktion** project. They define project-specific conventions to ensure consistency and prevent common errors. As this is a Python-based project using `uv` and `ruff`, we will assume standard best practices are followed and focus only on the critical rules and conventions needed for our specific architecture.

### **Core Standards**

  * **Languages & Runtimes:** Python 3.12+ (compatible with current stable ecosystem)
  * **Package Management:** All Python package management MUST use `uv`:
    - Use `uv pip install` instead of `pip install` or `pip3 install`
    - Use `uv run python` instead of `python` or `python3`
    - Use `uv run` for any Python tools (pytest, mypy, ruff, etc.)
    - Use `uv venv` for virtual environment creation
    - Use `uv pip compile` for dependency resolution
  * **Style & Linting:** `ruff` (latest version) will be used for linting and formatting, configured via `pyproject.toml`.
    - Execute via: `uv run ruff check` and `uv run ruff format`
  * **Static Type Checking:** `mypy` (latest version) will be used for static type checking to ensure code correctness and maintainability.
    - Execute via: `uv run mypy`
  * **Code Hooks:** `pre-commit` hooks will be used to enforce code standards before commits are made.
    - Update hooks using: `uv run pre-commit autoupdate --freeze`
    - This ensures reproducible builds with exact commit hashes
    - Run updates regularly for security and bug fixes
    - **MANDATORY**: All pre-commit hooks MUST pass before any code is committed
    - Run pre-commit on all files: `uv run pre-commit run --all-files`
    - Pre-commit checks include: ruff linting, ruff formatting, mypy type checking, and file hygiene
  * **Import Sorting:** `isort` (integrated in ruff) will be used to automatically sort Python imports.
  * **Line Length:** The maximum line length for all code is set to 120 characters.
  * **Dependency Policy:** Always use the latest stable versions of all dependencies and update regularly for security and performance.

### **Pre-Commit Workflow**

Before committing any code:

1. **Stage your changes:** `git add <files>`
2. **Run pre-commit:** `uv run pre-commit run --all-files`
3. **Fix any issues:** Address all errors and warnings
4. **Re-run until clean:** Repeat steps 2-3 until all checks pass
5. **Run tests:** `uv run pytest tests/unit/ -v`
6. **Commit:** Only commit when all checks and tests pass

**Pre-commit hooks include:**
- `ruff`: Python linting and error checking
- `ruff-format`: Python code formatting
- `mypy`: Static type checking
- `trailing-whitespace`: Remove trailing whitespace
- `end-of-file-fixer`: Ensure files end with newline
- `check-yaml`: Validate YAML syntax
- `check-json`: Validate JSON syntax
- `check-toml`: Validate TOML syntax
- `check-added-large-files`: Prevent large files from being committed
- `check-merge-conflict`: Check for merge conflict markers
- `debug-statements`: Check for debugger imports
- `mixed-line-ending`: Check for mixed line endings

### **Python Execution Standards**

All Python-related commands MUST use `uv` as the package and environment manager:

  * **Package Installation:**
    - ❌ NEVER: `pip install package`, `pip3 install package`
    - ✅ ALWAYS: `uv pip install package`

  * **Python Execution:**
    - ❌ NEVER: `python script.py`, `python3 script.py`
    - ✅ ALWAYS: `uv run python script.py`

  * **Tool Execution:**
    - ❌ NEVER: `pytest`, `mypy`, `ruff`, `alembic`
    - ✅ ALWAYS: `uv run pytest`, `uv run mypy`, `uv run ruff`, `uv run alembic`

  * **Virtual Environments:**
    - ❌ NEVER: `python -m venv`, `virtualenv`
    - ✅ ALWAYS: `uv venv`

  * **Dependency Management:**
    - Use `uv pip compile` for dependency resolution
    - Use `uv pip sync` for dependency installation
    - Use `uv pip freeze` for listing installed packages

### **Testing Requirements**

  * **Test Execution:** Tests MUST be run after completing each task implementation:
    - Unit tests: `uv run pytest tests/unit/ -v`
    - Integration tests: `uv run pytest tests/integration/ -v` (requires services running)
    - All tests: `uv run pytest -v`
  * **Test Coverage:** Maintain minimum 80% code coverage for new code
  * **Story Completion:** A story CANNOT be marked as "Done" until:
    - All unit tests pass
    - All integration tests pass (when services are available)
    - All pre-commit hooks pass
    - Code has been committed with appropriate commit messages
  * **Task Workflow:**
    1. Implement the task
    2. Run relevant unit tests
    3. Run pre-commit checks
    4. Fix any issues
    5. Commit changes
    6. Move to next task

### **Critical Rules**

  * **Configuration:** All connection strings and sensitive configurations must be loaded from environment variables and not be hardcoded.
  * **Datastore Interaction:** Direct database queries outside of the designated ORM/repository layer are prohibited.
  * **Inter-Service Communication:** Services should communicate exclusively via RabbitMQ messages, and not through direct HTTP calls to other services.
  * **CI Process:** The CI/CD pipeline should not duplicate checks that are already enforced by `pre-commit` hooks, such as linting, type checking, or import sorting. The CI pipeline should focus on integration, deployment, and testing.
  * **Code Quality Gates:** No code can be merged or deployed that:
    - Fails any pre-commit hook
    - Has failing unit tests
    - Introduces mypy type errors
    - Violates ruff linting rules
