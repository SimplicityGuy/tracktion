# Coding Standards

These standards are mandatory for all AI agents and developers working on the **tracktion** project. They define project-specific conventions to ensure consistency and prevent common errors. As this is a Python-based project using `uv` and `ruff`, we will assume standard best practices are followed and focus only on the critical rules and conventions needed for our specific architecture.

### **Core Standards**

  * **Languages & Runtimes:** Python 3.13 (always use latest stable version)
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
  * **Import Sorting:** `isort` (integrated in ruff) will be used to automatically sort Python imports.
  * **Line Length:** The maximum line length for all code is set to 120 characters.
  * **Dependency Policy:** Always use the latest stable versions of all dependencies and update regularly for security and performance.

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

### **Critical Rules**

  * **Configuration:** All connection strings and sensitive configurations must be loaded from environment variables and not be hardcoded.
  * **Datastore Interaction:** Direct database queries outside of the designated ORM/repository layer are prohibited.
  * **Inter-Service Communication:** Services should communicate exclusively via RabbitMQ messages, and not through direct HTTP calls to other services.
  * **CI Process:** The CI/CD pipeline should not duplicate checks that are already enforced by `pre-commit` hooks, such as linting, type checking, or import sorting. The CI pipeline should focus on integration, deployment, and testing.
