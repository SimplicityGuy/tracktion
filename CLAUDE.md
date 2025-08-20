# Tracktion Project - Important Instructions

## Python Execution
**ALWAYS use `uv run` for Python commands in this project:**
- Tests: `uv run pytest <test_file>`
- Scripts: `uv run python <script>`
- Module execution: `uv run python -m <module>`

## Project Structure
This is a monorepo with multiple services under `services/`. The main service being worked on is `analysis_service`.

## Testing
- Unit tests are located in `tests/unit/`
- Always run tests with: `uv run pytest`
- Never use plain `python` or `python3` commands

## Linting and Type Checking
- Run linting with: `uv run ruff check`
- Run type checking with: `uv run mypy`
