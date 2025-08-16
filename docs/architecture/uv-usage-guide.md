# UV Usage Guide for Tracktion Project

This document provides comprehensive guidelines for using `uv` as the exclusive Python package and environment manager for the Tracktion project.

## Overview

`uv` is a fast, reliable Python package installer and resolver written in Rust. It is **mandatory** for all Python operations in this project. Direct usage of `pip`, `pip3`, `python`, or `python3` is strictly prohibited.

## Installation

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Common Commands

### Environment Management

#### Create Virtual Environment
```bash
# Create a new virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.13
```

#### Activate Virtual Environment
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Package Management

#### Install Packages
```bash
# Install a single package
uv pip install sqlalchemy

# Install from requirements.txt
uv pip install -r requirements.txt

# Install from pyproject.toml
uv pip install -e .

# Install with extras
uv pip install -e ".[dev]"
```

#### Compile Dependencies
```bash
# Generate requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# Update all dependencies
uv pip compile --upgrade pyproject.toml -o requirements.txt
```

#### Sync Dependencies
```bash
# Install exact versions from requirements.txt
uv pip sync requirements.txt
```

### Running Python and Tools

#### Execute Python
```bash
# Run Python interpreter
uv run python

# Run Python script
uv run python script.py

# Run module
uv run python -m module_name
```

#### Execute Tools
```bash
# Run pytest
uv run pytest

# Run with specific options
uv run pytest tests/unit -v --cov

# Run mypy
uv run mypy src/

# Run ruff
uv run ruff check .
uv run ruff format .

# Run alembic
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "Add new table"

# Run pre-commit
uv run pre-commit run --all-files
uv run pre-commit install
```

## Project-Specific Usage

### Development Workflow

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/tracktion.git
   cd tracktion
   ```

2. **Create Virtual Environment**
   ```bash
   uv venv --python 3.13
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**
   ```bash
   # Install project dependencies
   uv pip install -e ".[dev]"
   
   # Install pre-commit hooks
   uv run pre-commit install
   ```

4. **Run Tests**
   ```bash
   # Run all tests
   uv run pytest
   
   # Run with coverage
   uv run pytest --cov=src --cov-report=html
   ```

5. **Code Quality Checks**
   ```bash
   # Linting
   uv run ruff check .
   
   # Formatting
   uv run ruff format .
   
   # Type checking
   uv run mypy src/
   ```

### Database Operations

```bash
# Run Alembic migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Rollback migration
uv run alembic downgrade -1
```

### Service Operations

```bash
# Start a service
uv run python services/file_watcher/src/main.py

# Run with environment variables
DATABASE_URL=postgresql://... uv run python services/cataloging_service/src/main.py
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v2
  with:
    version: "latest"

- name: Create venv
  run: uv venv

- name: Install dependencies
  run: uv pip install -e ".[dev]"

- name: Run tests
  run: uv run pytest
```

### Docker Integration

```dockerfile
# Install uv in Docker
RUN pip install --no-cache-dir uv

# Use uv for package installation
RUN uv pip install --system --no-cache .
```

## Common Issues and Solutions

### Issue: Command not found
**Solution**: Ensure uv is installed and in PATH
```bash
which uv  # Should show uv location
```

### Issue: Python version mismatch
**Solution**: Specify Python version explicitly
```bash
uv venv --python 3.13
```

### Issue: Package conflicts
**Solution**: Use uv's resolver
```bash
uv pip compile --upgrade pyproject.toml -o requirements.txt
uv pip sync requirements.txt
```

## Migration from pip/pip3

### Command Mapping

| Old Command | New Command |
|------------|-------------|
| `pip install package` | `uv pip install package` |
| `pip3 install package` | `uv pip install package` |
| `python script.py` | `uv run python script.py` |
| `python3 script.py` | `uv run python script.py` |
| `python -m pytest` | `uv run pytest` |
| `pip freeze` | `uv pip freeze` |
| `pip list` | `uv pip list` |

## Best Practices

1. **Always use uv**: Never fall back to pip or python directly
2. **Version pinning**: Use `uv pip compile` to generate locked dependencies
3. **Virtual environments**: Always work within a virtual environment
4. **Consistent commands**: Use the same uv commands across development and CI
5. **Documentation**: Update any documentation that references pip or python commands

## Enforcement

- Pre-commit hooks will check for direct pip/python usage
- CI/CD pipelines will fail if pip or python are used directly
- Code reviews must verify uv usage compliance

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [uv PyPI Page](https://pypi.org/project/uv/)
- [Migration Guide](https://github.com/astral-sh/uv/blob/main/MIGRATION.md)