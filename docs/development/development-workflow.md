# Development Workflow Guide

This guide defines the mandatory development workflow for the **tracktion** project. All developers and AI agents MUST follow this workflow to ensure code quality and consistency.

## Overview

The development workflow enforces quality at every stage through automated checks and testing requirements. No code can be committed or merged that doesn't meet our quality standards.

## Prerequisites

- Python 3.12+ installed
- `uv` package manager installed
- Pre-commit hooks installed: `uv run pre-commit install`
- Virtual environment activated: `uv venv && source .venv/bin/activate`

## Story Implementation Workflow

### 1. Story Setup

Before starting a story:

1. **Review the story document** in `docs/stories/`
2. **Verify all dependencies** are installed: `uv pip sync`
3. **Ensure pre-commit is updated**: `uv run pre-commit autoupdate --freeze`
4. **Run existing tests** to ensure clean baseline: `uv run pytest -v`

### 2. Task Implementation

For each task in the story:

#### Step 1: Implementation
- Write the code according to task requirements
- Follow coding standards in `docs/architecture/coding-standards.md`
- Use type hints for all functions and methods
- Add docstrings to all public functions

#### Step 2: Test Creation
- Write unit tests for new functionality
- Cover edge cases and error conditions
- Ensure tests are in appropriate `tests/unit/` subdirectory

#### Step 3: Test Execution
```bash
# Run specific test file
uv run pytest tests/unit/path/to/test_file.py -v

# Run all unit tests
uv run pytest tests/unit/ -v
```

**REQUIREMENT**: All tests MUST pass before proceeding

#### Step 4: Pre-Commit Checks
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files
```

**REQUIREMENT**: All hooks MUST pass:
- ✅ ruff (linting)
- ✅ ruff-format (formatting)
- ✅ mypy (type checking)
- ✅ File hygiene checks
- ✅ YAML/JSON/TOML validation

#### Step 5: Fix Issues
If any checks fail:
1. Fix the identified issues
2. Re-run tests (Step 3)
3. Re-run pre-commit (Step 4)
4. Repeat until all checks pass

#### Step 6: Commit Changes
```bash
# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "feat: implement [task description]

- Add [specific change 1]
- Add [specific change 2]
- Include tests for [functionality]"
```

### 3. Story Completion

A story can ONLY be marked as "Done" when:

#### Quality Checks ✅
- [ ] All tasks completed
- [ ] All unit tests passing: `uv run pytest tests/unit/ -v`
- [ ] All integration tests passing (if applicable): `uv run pytest tests/integration/ -v`
- [ ] All pre-commit hooks passing: `uv run pre-commit run --all-files`
- [ ] Code coverage ≥80% for new code: `uv run pytest --cov=services --cov=shared --cov-report=term-missing`

#### Documentation ✅
- [ ] Story document updated with completion status
- [ ] Any new features documented
- [ ] API changes documented (if applicable)

#### Version Control ✅
- [ ] All code committed with appropriate messages
- [ ] Commits organized logically (feature, test, docs)
- [ ] Branch ready for merge (if using branches)

## Common Commands Reference

### Testing Commands
```bash
# Run all tests
uv run pytest -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests only
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest --cov=services --cov=shared --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/services/test_file.py -v

# Run tests matching pattern
uv run pytest -k "test_metadata" -v
```

### Pre-Commit Commands
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run

# Update hooks to latest versions
uv run pre-commit autoupdate --freeze

# Run specific hook
uv run pre-commit run mypy --all-files
```

### Code Quality Commands
```bash
# Run linting
uv run ruff check .

# Run formatting
uv run ruff format .

# Run type checking
uv run mypy .

# Fix linting issues automatically
uv run ruff check --fix .
```

## Troubleshooting

### Pre-commit Failures

**Ruff Linting Errors**
```bash
# Auto-fix safe issues
uv run ruff check --fix .

# Show detailed error explanations
uv run ruff check --show-fixes .
```

**Mypy Type Errors**
- Add type hints to functions
- Use `Optional[]` for nullable types
- Add `# type: ignore[error-code]` only as last resort
- Cast SQLAlchemy query results: `cast(Optional[Model], result)`

**Import Order Issues**
```bash
# Ruff will auto-fix most import issues
uv run ruff check --fix .
```

### Test Failures

**Unit Test Failures**
1. Verify test logic is correct
2. Check mocks and fixtures
3. Ensure proper test isolation
4. Run single test for debugging: `uv run pytest path/to/test.py::test_name -vv`

**Integration Test Failures**
1. Ensure services are running: `docker-compose up -d`
2. Check database migrations: `uv run alembic upgrade head`
3. Verify environment variables are set
4. Check service logs: `docker-compose logs <service>`

## Quality Gates Summary

| Stage | Requirement | Command | Must Pass |
|-------|------------|---------|-----------|
| After each task | Unit tests | `uv run pytest tests/unit/ -v` | ✅ Yes |
| After each task | Pre-commit | `uv run pre-commit run --all-files` | ✅ Yes |
| Before commit | Pre-commit | `uv run pre-commit run` | ✅ Yes |
| Story completion | All unit tests | `uv run pytest tests/unit/ -v` | ✅ Yes |
| Story completion | Integration tests | `uv run pytest tests/integration/ -v` | ✅ Yes* |
| Story completion | Coverage ≥80% | `uv run pytest --cov` | ✅ Yes |

*When services are available

## Enforcement

These workflow requirements are enforced through:

1. **Pre-commit hooks**: Prevent commits with failing checks
2. **CI Pipeline**: Blocks merging of PRs with failures
3. **Code Review**: Manual verification of compliance
4. **Story Definition**: Clear "Definition of Done" criteria

**Remember**: Quality is not negotiable. Take the time to fix issues properly rather than working around them.
