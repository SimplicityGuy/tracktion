# Tracktion Project - Important Instructions

## Python Execution
**ALWAYS use `uv run` for Python commands in this project:**
- Tests: `uv run pytest <test_file>`
- Scripts: `uv run python <script>`
- Module execution: `uv run python -m <module>`
- **NEVER** use plain `python`, `python3`, or `pip` commands directly

## Project Structure
This is a monorepo with multiple services under `services/`. The main service being worked on is `analysis_service`.

## Testing
- Unit tests are located in `tests/unit/`
- Always run tests with: `uv run pytest`
- Never use plain `python` or `python3` commands

## Code Quality Standards - CRITICAL REQUIREMENTS

### Pre-commit Hooks - MANDATORY
- **ALWAYS run `pre-commit run --all-files` before ANY commit**
- **NEVER skip pre-commit checks with --no-verify or SKIP flags**
- **ALL pre-commit checks MUST pass before committing**
- **The project is currently clean of all errors - maintaining this state is CRITICAL**

### Pre-commit Workflow
1. Before making any commits, ALWAYS run: `pre-commit run --all-files`
2. Fix ALL issues reported by pre-commit
3. Re-run `pre-commit run --all-files` to verify all fixes
4. Only commit when ALL checks pass
5. If pre-commit hooks fail during commit, fix the issues and try again

### Zero-Tolerance Policy
- **NO commits with failing pre-commit checks**
- **NO use of --no-verify flag**
- **NO use of SKIP environment variable**
- **NO partial fixes - all issues must be resolved**

### Linting and Type Checking
- Pre-commit automatically runs ruff and mypy checks
- **Always fix ALL ruff and mypy errors** - do not ignore or suppress
- Only use pragma/noqa or type:ignore when absolutely necessary
- **When using pragma/ignore, ALWAYS provide a detailed reason:**
  ```python
  # Example - ruff pragma with reason:
  X, y = trainer._prepare_training_data(data)  # noqa: N806 - X is standard ML convention

  # Example - mypy ignore with reason:
  from alembic import context  # type: ignore[attr-defined]  # Alembic adds attributes at runtime
  ```
