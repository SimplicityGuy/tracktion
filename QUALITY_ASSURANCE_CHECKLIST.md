# Quality Assurance Checklist for Development Workflows

## Pre-Development Setup
- [ ] Confirm `uv` is installed and configured
- [ ] Verify pre-commit hooks are installed: `pre-commit install`
- [ ] Review CLAUDE.md for latest requirements
- [ ] Confirm working in correct service directory (usually `analysis_service`)

## During Development

### Code Changes
- [ ] Using `uv run` for ALL Python commands (no bare `python` or `pip`)
- [ ] Running tests frequently: `uv run pytest tests/unit/`
- [ ] Following existing code patterns and conventions
- [ ] Adding type hints to all new functions/methods
- [ ] Writing docstrings for public APIs

### Before ANY Commit

#### Pre-commit Validation (MANDATORY)
- [ ] Run: `pre-commit run --all-files`
- [ ] ALL checks PASS (no failures, no skips)
- [ ] If failures occur:
  - [ ] Fix ALL reported issues
  - [ ] Re-run `pre-commit run --all-files`
  - [ ] Repeat until ALL checks pass

#### Code Quality Verification
- [ ] Zero mypy errors
- [ ] Zero ruff violations
- [ ] No new `type: ignore` without detailed justification
- [ ] No new `noqa` comments without detailed justification
- [ ] All tests passing: `uv run pytest`

### Commit Standards
- [ ] Commit message follows conventional format
- [ ] Changes are atomic and focused
- [ ] No mixing of feature work with formatting/linting fixes
- [ ] NO use of `--no-verify` flag EVER
- [ ] NO use of `SKIP` environment variable EVER

## Post-Commit Verification
- [ ] Verify commit in log shows clean pre-commit status
- [ ] Run tests one final time to ensure nothing broken
- [ ] Check that project remains in "clean" state

## Red Flags - STOP if any occur:
- ⛔ Temptation to use `--no-verify`
- ⛔ Wanting to "fix it in the next commit"
- ⛔ Thinking "it's just a small change"
- ⛔ Assuming pre-commit will catch it during CI
- ⛔ Using bare `python` instead of `uv run`

## Escalation Protocol
If unable to resolve pre-commit failures:
1. DO NOT commit with failures
2. Document the specific error
3. Seek assistance with error details
4. Learn from the resolution for future

## Quality Gates Summary
✅ **MUST PASS**: All pre-commit checks
✅ **MUST USE**: `uv run` for all Python execution
✅ **MUST MAINTAIN**: Zero-error baseline
✅ **MUST DOCUMENT**: Any pragma/ignore usage with detailed reasons

---
*Remember: We currently have ZERO errors. Every commit must maintain this standard.*
