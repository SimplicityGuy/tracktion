# Pragma Audit Catalog - Story 11.1
Generated: 2025-01-02

## Executive Summary
- **Total Pragmas Found: 211**
  - Type ignore comments: 153
  - Noqa comments: 58
  - Pragma directives: 0
  - Pylint suppressions: 0
  - Mypy directives: 0

## Type: ignore Pragmas (153 total)

### By Category:
1. **Database/ORM Related (98 occurrences - 64%)**
   - SQLAlchemy queries returning Any: 76 `[no-any-return]`
   - Alembic runtime attributes: 29 `[attr-defined]`

2. **External Library Integration (14 occurrences - 9%)**
   - Missing type stubs: 7 `[import-untyped]`
   - Third-party quirks: 7 various

3. **Service/API Integration (24 occurrences - 16%)**
   - Cache service returns: `[no-any-return]`
   - Pydantic model serialization: `[no-any-return]`

4. **Analysis/Processing (17 occurrences - 11%)**
   - Audio analysis results: `[assignment]`, `[index]`
   - Circuit breaker decorators: `[attr-defined]`

### Files with High Concentrations:
- `shared/core_types/src/repositories.py`: 16 pragmas
- `shared/core_types/src/async_repositories.py`: 12 pragmas
- `services/analysis_service/src/audio_cache.py`: 6 pragmas
- `services/tracklist_service/src/services/version_service.py`: 4 pragmas

### Fixable Type Ignores:
1. **SQLAlchemy [no-any-return] (76 cases)**
   - Solution: Upgrade to SQLAlchemy 2.0+ with better typing
   - Alternative: Create typed wrapper functions

2. **Import untyped libraries (7 cases)**
   - aiofiles: Install types-aiofiles
   - requests: Install types-requests
   - python-dateutil: Install types-python-dateutil
   - croniter: Create local stub file

3. **Alembic [attr-defined] (29 cases)**
   - These are legitimate - Alembic adds attributes at runtime
   - Keep with proper documentation

## Noqa Pragmas (58 total)

### By Error Code:
1. **PLW0603 (36 occurrences)**: Global variable usage
   - All for legitimate singleton patterns
   - Well-documented with explanations

2. **B008 (13 occurrences)**: Function calls in defaults
   - All for FastAPI Depends() pattern
   - Required by framework - cannot be fixed

3. **E402 (6 occurrences)**: Module import position
   - Path modifications before import
   - Script initialization requirements

4. **B017 (2 occurrences)**: assertRaises(Exception)
   - Test-specific suppressions
   - Consider more specific exception types

5. **N806 (1 occurrence)**: Variable naming
   - ML convention (X for feature matrix)
   - Properly documented

6. **PLW0602 (1 occurrence)**: Global undefined variable
   - Database cleanup pattern

### Fixable Noqa Directives:
1. **B017 in tests (2 cases)**
   - Can use more specific exception types

2. **E402 in scripts (6 cases)**
   - Consider restructuring imports where possible

## Recommendations

### Immediate Actions:
1. Install type stub packages:
   ```bash
   uv add --dev types-aiofiles types-requests types-python-dateutil
   ```

2. Fix B017 test suppressions by using specific exceptions

3. Review E402 suppressions for potential restructuring

### Long-term Actions:
1. Consider SQLAlchemy 2.0 migration for better typing
2. Create local stubs for croniter
3. Document singleton pattern usage in architecture docs

### Keep As-Is (Properly Justified):
1. All PLW0603 for singleton patterns
2. All B008 for FastAPI Depends()
3. Alembic [attr-defined] pragmas
4. N806 for ML convention

## Quality Assessment
✅ All pragmas have detailed explanations
✅ Specific error codes used (not generic)
✅ Legitimate use cases documented
✅ No unnecessary suppressions found
❌ Some fixable issues remain (type stubs, test exceptions)

## Next Steps
1. Install missing type stub packages
2. Fix test exception suppressions
3. Create documentation for acceptable pragma patterns
4. Implement pragma monitoring in CI/CD
