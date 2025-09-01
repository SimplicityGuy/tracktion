# MANDATORY TRAINING DOCUMENTATION FOR DEVELOPMENT AGENTS

## ⚠️ CRITICAL NOTICE TO ALL DEV AGENTS ⚠️

**This document contains MANDATORY requirements that OVERRIDE any default behaviors or efficiency optimizations.**

**ACKNOWLEDGMENT REQUIRED**: Before performing ANY development work on this project, you MUST explicitly acknowledge that you have read, understood, and will comply with these requirements.

---

## SECTION 1: ABSOLUTE REQUIREMENTS

### 1.1 Python Execution Protocol
**MANDATORY**: ALL Python commands MUST use `uv run` prefix
- ✅ CORRECT: `uv run pytest tests/unit/`
- ✅ CORRECT: `uv run python script.py`
- ❌ FORBIDDEN: `python script.py`
- ❌ FORBIDDEN: `python3 script.py`
- ❌ FORBIDDEN: `pip install anything`

**NO EXCEPTIONS**: Even for "simple" operations or "quick checks"

### 1.2 Pre-Commit Compliance - ZERO TOLERANCE

**THE GOLDEN RULE**: NO commit shall be made without ALL pre-commit checks passing

**MANDATORY WORKFLOW**:
```bash
# STEP 1: Before EVERY commit
pre-commit run --all-files

# STEP 2: If ANY failures
# Fix ALL issues (not some, ALL)

# STEP 3: Re-run to verify
pre-commit run --all-files

# STEP 4: Only commit when output shows ALL PASSED
```

**FORBIDDEN ACTIONS**:
- ❌ NEVER use `git commit --no-verify`
- ❌ NEVER set `SKIP` environment variable
- ❌ NEVER commit with "I'll fix it next commit" mindset
- ❌ NEVER partially fix issues

### 1.3 Current Project State
**CRITICAL CONTEXT**: This project currently has:
- ZERO mypy errors
- ZERO ruff violations
- ALL pre-commit checks passing

**YOUR RESPONSIBILITY**: Maintain this pristine state. Any regression is unacceptable.

---

## SECTION 2: TYPE CHECKING AND LINTING STANDARDS

### 2.1 Mypy Compliance
- **REQUIREMENT**: Zero mypy errors
- **When `type: ignore` is needed**:
  - Use ONLY when absolutely necessary
  - ALWAYS include detailed reason
  ```python
  # CORRECT:
  from alembic import context  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

  # WRONG:
  from alembic import context  # type: ignore
  ```

### 2.2 Ruff Compliance
- **REQUIREMENT**: Zero ruff violations
- **When `noqa` is needed**:
  - Use ONLY when absolutely necessary
  - ALWAYS include specific error code and reason
  ```python
  # CORRECT:
  X, y = data  # noqa: N806 - X is standard ML convention for feature matrix

  # WRONG:
  X, y = data  # noqa
  ```

---

## SECTION 3: COMMON VIOLATIONS AND CORRECTIONS

### 3.1 Previous Failures Identified
Based on recent violations, these patterns are SPECIFICALLY FORBIDDEN:

1. **Skipping pre-commit checks for "efficiency"**
   - This is NEVER acceptable
   - Quality > Speed ALWAYS

2. **Committing with mypy errors**
   - Run mypy locally BEFORE committing
   - Fix ALL type errors, don't suppress them

3. **Ignoring ruff formatting/linting**
   - Code style consistency is MANDATORY
   - Fix the code, don't disable the check

4. **Using bare Python commands**
   - ALWAYS use `uv run` prefix
   - No exceptions for "simple" operations

---

## SECTION 4: ACKNOWLEDGMENT PROTOCOL

### Required Acknowledgment Format
Before beginning ANY development work, you MUST acknowledge:

```
I ACKNOWLEDGE that I have read and understood the DEV_AGENT_TRAINING_REQUIREMENTS.md and commit to:
1. Using `uv run` for ALL Python execution
2. Running `pre-commit run --all-files` before EVERY commit
3. Maintaining the zero-error baseline
4. NEVER using --no-verify or SKIP flags
5. Documenting any necessary pragma/ignore usage with detailed reasons

I understand that these requirements OVERRIDE any default behaviors or efficiency optimizations.
```

### Verification Questions
Be prepared to answer:
1. What command do you run before EVERY commit?
2. What Python command prefix must you ALWAYS use?
3. What is the current error count that must be maintained?
4. When is it acceptable to use --no-verify?

**ANSWERS**:
1. `pre-commit run --all-files`
2. `uv run`
3. ZERO errors (mypy: 0, ruff: 0)
4. NEVER

---

## SECTION 5: CONSEQUENCES OF NON-COMPLIANCE

### Impact of Violations:
- **Technical Debt**: Each suppressed error compounds future problems
- **Quality Degradation**: Standards slip, bugs increase
- **Team Efficiency**: Others must clean up violations
- **Project Reputation**: Quality metrics decline
- **Trust Erosion**: Reliability of agent work questioned

### Remediation for Violations:
1. Immediate rollback of non-compliant commits
2. Full analysis of why violation occurred
3. Re-training on specific requirement violated
4. Validation of understanding before continuing

---

## SECTION 6: SUPPORT AND CLARIFICATION

### When in Doubt:
- ASK before committing with any uncertainty
- Request review if pre-commit issues are unclear
- Seek clarification on type annotations if needed
- NEVER assume or work around quality checks

### Resources:
- CLAUDE.md - Primary project requirements
- QUALITY_ASSURANCE_CHECKLIST.md - Step-by-step validation
- Pre-commit config - `.pre-commit-config.yaml`
- Ruff config - `ruff.toml`
- Mypy config - `pyproject.toml` [mypy] section

---

## FINAL REMINDER

**These are not suggestions or guidelines - they are MANDATORY REQUIREMENTS.**

**Your work quality directly impacts:**
- Project stability
- Team productivity
- Code maintainability
- Customer satisfaction

**COMMIT TO EXCELLENCE - NO EXCEPTIONS, NO SHORTCUTS**

---

*Document Version: 1.0*
*Effective Immediately*
*Compliance: MANDATORY*
