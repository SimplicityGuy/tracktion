# Pragma Usage Guidelines for Tracktion Project

## Purpose
This document establishes the guidelines for using pragma directives (type: ignore, noqa, etc.) in the Tracktion codebase. Our goal is to maintain high code quality while acknowledging legitimate cases where suppressions are necessary.

## Core Principle
**Fix issues, don't hide them.** Pragmas should only be used when:
1. The issue cannot be reasonably fixed
2. The suppression is for a legitimate framework/library limitation
3. The pragma is thoroughly documented with justification

## Current Pragma Statistics (as of 2025-01-02)
- **Total Pragmas**: 209 (reduced from 211)
- **Type ignore**: 153
- **Noqa**: 56 (reduced from 58)
- **Other pragmas**: 0

## Approved Pragma Patterns

### 1. Type: ignore Pragmas

#### SQLAlchemy Queries [no-any-return] ✅ APPROVED
```python
# APPROVED - SQLAlchemy intentionally returns Any for flexibility
return session.query(Recording).filter(Recording.id == recording_id).first()  # type: ignore[no-any-return]  # SQLAlchemy query methods return Any but we know this returns Recording | None
```
**Justification**: SQLAlchemy's design choice for dynamic query flexibility.

#### Alembic Runtime Attributes [attr-defined] ✅ APPROVED
```python
# APPROVED - Alembic adds attributes at runtime
from alembic import context  # type: ignore[attr-defined]  # Alembic adds attributes at runtime
```
**Justification**: Alembic dynamically adds attributes during migration runtime.

#### Missing Type Stubs [import-untyped] ⚠️ CONDITIONAL
```python
# CONDITIONAL - Only if type stubs unavailable
from croniter import croniter  # Previously needed type: ignore but now has type stubs available via types-croniter
```
**Action Required**: Check for available type stubs before adding this pragma.

### 2. Noqa Pragmas

#### Singleton Pattern Global Variables [PLW0603] ✅ APPROVED
```python
# APPROVED - Singleton pattern requires global state
global _instance
_instance = MyService()  # noqa: PLW0603  # Standard singleton pattern for global service
```
**Justification**: Legitimate singleton pattern implementation.

#### FastAPI Depends Pattern [B008] ✅ APPROVED
```python
# APPROVED - FastAPI framework requirement
def endpoint(db: Session = Depends(get_db)):  # noqa: B008  # FastAPI requires Depends in defaults
```
**Justification**: FastAPI's dependency injection requires function calls in defaults.

#### Module Import Position [E402] ⚠️ CONDITIONAL
```python
# CONDITIONAL - Only for necessary path modifications
sys.path.insert(0, str(Path(__file__).parent))
from my_module import something  # noqa: E402  # Path modification required before import
```
**Action Required**: Consider restructuring to avoid path modifications.

#### ML Convention Variable Names [N806] ✅ APPROVED
```python
# APPROVED - Standard ML convention
X, y = prepare_data()  # noqa: N806  # X is standard ML convention for feature matrix
```
**Justification**: Industry-standard machine learning naming conventions.

## Prohibited Pragma Patterns

### ❌ Generic Suppressions
```python
# PROHIBITED - Too broad
from some_module import something  # type: ignore
result = function()  # noqa
```

### ❌ Undocumented Pragmas
```python
# PROHIBITED - No justification
return query.all()  # type: ignore[no-any-return]
```

### ❌ Fixable Issues
```python
# PROHIBITED - Should use specific exception
with pytest.raises(Exception):  # noqa: B017
```

## New Pragma Approval Process

### Step 1: Attempt to Fix
Before adding any pragma:
1. Try to fix the underlying issue
2. Check for available type stubs
3. Consider refactoring the code
4. Consult team if unsure

### Step 2: Justify the Pragma
If a pragma is necessary:
1. Use the most specific error code
2. Add detailed comment explaining:
   - Why the issue cannot be fixed
   - What the actual behavior/type is
   - Any relevant context

### Step 3: Document in PR
When adding pragmas in a PR:
1. List all new pragmas in PR description
2. Provide justification for each
3. Get explicit approval from reviewer

## Pre-commit Integration

The pre-commit hooks will:
1. Flag any new undocumented pragmas
2. Ensure pragma format compliance
3. Track pragma count metrics

## Monitoring and Metrics

### Current Baseline (Must Not Increase Without Approval)
- Type ignore [no-any-return]: 76 (SQLAlchemy)
- Type ignore [attr-defined]: 29 (Alembic)
- Type ignore [import-untyped]: 7 (External libs)
- Noqa PLW0603: 36 (Singletons)
- Noqa B008: 13 (FastAPI)

### Quarterly Review
Every quarter, review:
1. Total pragma count trends
2. New pragma additions
3. Opportunities to remove pragmas
4. Updates to type stubs availability

## Examples of Good Documentation

### Excellent ✅
```python
return session.query(Recording).filter(Recording.id == recording_id).first()  # type: ignore[no-any-return]  # SQLAlchemy query methods return Any but we know this returns Recording | None
```

### Good ✅
```python
global _cache_service  # noqa: PLW0603  # Standard singleton pattern for cache service initialization
```

### Poor ❌
```python
return result  # type: ignore
```

## Enforcement

1. **Pre-commit Hooks**: Automatically check pragma compliance
2. **Code Review**: Reviewers must approve any new pragmas
3. **CI/CD Pipeline**: Track pragma metrics and fail on undocumented pragmas
4. **Regular Audits**: Quarterly review of all pragmas

## Contact

For questions about pragma usage or to request exceptions:
- Create an issue with the `pragma-review` label
- Tag the code quality team in your PR

## Revision History

- 2025-01-02: Initial guidelines created (Story 11.1)
- Fixed 2 B017 suppressions in tests
- Documented all 209 remaining pragmas
