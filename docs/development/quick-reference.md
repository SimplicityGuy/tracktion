# Developer Quick Reference

## 🚨 Quality Gates - MUST PASS

### Before EVERY Commit
```bash
# 1. Run tests
uv run pytest tests/unit/ -v

# 2. Run pre-commit
uv run pre-commit run --all-files

# 3. Only commit if ALL checks pass
git add . && git commit -m "your message"
```

### Before Marking Task Complete
- ✅ Implementation done
- ✅ Unit tests written and passing
- ✅ Pre-commit hooks passing
- ✅ Code committed

### Before Marking Story Done
- ✅ All tasks complete
- ✅ `uv run pytest tests/unit/ -v` - ALL PASS
- ✅ `uv run pytest tests/integration/ -v` - ALL PASS (if services running)
- ✅ `uv run pre-commit run --all-files` - ALL PASS
- ✅ Coverage ≥80%: `uv run pytest --cov`
- ✅ All code committed and pushed

## 🛠️ Essential Commands

### Testing
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test
uv run pytest tests/unit/path/to/test.py -v

# Run with coverage
uv run pytest --cov=services --cov=shared --cov-report=term-missing
```

### Code Quality
```bash
# Run ALL pre-commit checks (REQUIRED)
uv run pre-commit run --all-files

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run mypy .
```

### Common Fixes

**Mypy errors:**
```python
# Add type hints
def function(param: str) -> Optional[Dict[str, Any]]:

# Cast query results
from typing import cast
result = cast(Optional[Model], query.first())

# Handle Optional types
if self.attribute:  # Check before use
    self.attribute.method()
```

**Import order (E402):**
```python
# All imports at top of file
# Exception: alembic/env.py, tests/conftest.py
```

**Unused variables (F841):**
```python
# Remove assignment or use underscore
_ = unused_value  # If needed for side effects
```

## ⛔ NEVER DO THIS

```bash
# ❌ NEVER commit without running pre-commit
git commit -m "quick fix"

# ❌ NEVER use pip directly
pip install package

# ❌ NEVER run python directly
python script.py

# ❌ NEVER skip tests
# "I'll test it later"

# ❌ NEVER ignore type errors
# type: ignore  # Without specific error code
```

## ✅ ALWAYS DO THIS

```bash
# ✅ ALWAYS use uv
uv run pytest
uv pip install package
uv run python script.py

# ✅ ALWAYS run pre-commit before committing
uv run pre-commit run --all-files

# ✅ ALWAYS write tests for new code
# Test first, then implement

# ✅ ALWAYS fix issues properly
# Don't work around them
```

## 📋 Commit Checklist

- [ ] Tests written for new functionality
- [ ] All unit tests passing
- [ ] Pre-commit hooks passing
- [ ] Type hints added
- [ ] Docstrings added for public functions
- [ ] No commented-out code
- [ ] No print statements or debugger imports
- [ ] Commit message describes what and why

## 🚀 Quick Start for New Task

```bash
# 1. Pull latest changes
git pull

# 2. Install/update dependencies
uv pip sync

# 3. Update pre-commit
uv run pre-commit autoupdate --freeze

# 4. Run existing tests (ensure clean baseline)
uv run pytest tests/unit/ -v

# 5. Start implementing...
# After implementation:

# 6. Test your changes
uv run pytest tests/unit/your_tests/ -v

# 7. Run quality checks
uv run pre-commit run --all-files

# 8. Fix any issues and repeat 6-7

# 9. Commit when clean
git add .
git commit -m "feat: your feature description"
```

## 📞 Getting Help

1. Check error messages carefully
2. Review [Coding Standards](../architecture/coding-standards.md)
3. Review [Development Workflow](development-workflow.md)
4. Check [Troubleshooting Guide](../troubleshooting/)
5. Look at existing code for patterns

Remember: **Quality > Speed**. It's better to do it right than to do it twice.
