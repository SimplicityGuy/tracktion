# Git Workflow Guide

## Overview

This guide outlines the Git workflow and branching strategy used in the Tracktion project. We follow GitFlow with additional quality requirements including mandatory pre-commit hooks and comprehensive testing.

## Branching Strategy

### Branch Types

#### Main Branches
- **`main`**: Production-ready code, always deployable
- **`develop`**: Integration branch for features, contains latest development changes

#### Supporting Branches
- **Feature branches**: `feature/description` or `feature/ticket-number`
- **Hotfix branches**: `hotfix/version` or `hotfix/critical-issue`
- **Release branches**: `release/version-number`
- **Bugfix branches**: `bugfix/description` for non-critical fixes

### Branch Naming Conventions

```bash
# Feature branches
feature/user-authentication
feature/TRACK-123-bpm-analysis
feature/improve-audio-cache

# Bugfix branches
bugfix/fix-memory-leak
bugfix/TRACK-456-database-connection
bugfix/correct-key-detection

# Hotfix branches
hotfix/1.2.3
hotfix/critical-security-patch
hotfix/database-timeout

# Release branches
release/1.3.0
release/2.0.0-beta
```

## Development Workflow

### 1. Starting New Work

#### Feature Development
```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# First commit to establish branch
git commit --allow-empty -m "feat: start work on your-feature-name

Initial commit for feature branch."
```

#### Bug Fixes
```bash
# For non-critical bugs, branch from develop
git checkout develop
git pull origin develop
git checkout -b bugfix/fix-description

# For critical issues, branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-issue
```

### 2. Development Process

#### Making Commits
```bash
# Stage your changes
git add .

# MANDATORY: Run pre-commit hooks (zero tolerance policy)
uv run pre-commit run --all-files

# Fix ALL issues reported by pre-commit
# Re-run until ALL checks pass
uv run pre-commit run --all-files

# Only commit when ALL pre-commit checks pass
git commit -m "feat: add BPM detection algorithm

- Implement multi-algorithm BPM detection
- Add confidence scoring and validation
- Include comprehensive test coverage
- Update documentation and examples"
```

#### Commit Message Format
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Build process or auxiliary tool changes
- **perf**: Performance improvements
- **ci**: CI/CD changes

**Examples:**
```bash
git commit -m "feat(analysis): add key detection with HPCP algorithm"

git commit -m "fix(database): resolve connection pool exhaustion

- Increase pool size from 10 to 20 connections
- Add connection timeout configuration
- Improve error handling for connection failures
- Add monitoring for pool usage metrics"

git commit -m "docs: update API documentation for authentication endpoints"

git commit -m "refactor(tracklist): extract matching logic to separate service"
```

### 3. Testing and Quality Assurance

#### Before Pushing
```bash
# Run full test suite
uv run pytest tests/ -v

# Check test coverage
uv run pytest --cov=services --cov-report=html

# Run specific service tests
uv run pytest tests/unit/analysis_service/ -v

# Run integration tests
uv run pytest tests/integration/ -v
```

#### Pre-commit Validation (MANDATORY)
```bash
# This MUST pass before every commit
uv run pre-commit run --all-files

# Common checks that must pass:
# ✅ ruff linting and formatting
# ✅ mypy type checking
# ✅ yaml formatting
# ✅ trailing whitespace removal
# ✅ end-of-file-fixer
# ✅ check-merge-conflict
```

### 4. Pushing and Pull Requests

#### Pushing Changes
```bash
# Push feature branch
git push origin feature/your-feature-name

# For first push, set upstream
git push -u origin feature/your-feature-name
```

#### Creating Pull Requests

Use the PR template (see `docs/development/pr-review-process.md`):

```markdown
## Summary
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Test coverage maintained/improved

## Quality Assurance
- [ ] Pre-commit hooks passing
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No ruff violations
- [ ] No mypy errors

## Deployment Notes
Any special deployment considerations or migration steps.
```

### 5. Code Review Process

#### For Authors
```bash
# Ensure your branch is up to date before review
git checkout feature/your-feature
git fetch origin
git rebase origin/develop

# Address review feedback
git add .
uv run pre-commit run --all-files  # MANDATORY
git commit -m "fix: address code review feedback

- Improve error handling in audio processing
- Add missing type annotations
- Update docstrings for clarity"

# Push updates
git push origin feature/your-feature
```

#### For Reviewers
- Check code quality and style compliance
- Verify test coverage and test quality
- Review documentation updates
- Test functionality locally if needed
- Ensure pre-commit hooks are passing

### 6. Merging and Integration

#### Merging to Develop
```bash
# Squash merge for feature branches (recommended)
git checkout develop
git pull origin develop
git merge --squash feature/your-feature
git commit -m "feat: implement your feature summary

Complete implementation of feature with:
- Core functionality implementation
- Comprehensive test coverage
- Documentation updates
- Performance optimizations"

# Delete feature branch
git branch -d feature/your-feature
git push origin --delete feature/your-feature
```

#### Release Process
```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/1.3.0

# Update version numbers and changelog
# Run final testing
uv run pytest tests/ -v
uv run pre-commit run --all-files

# Commit release preparation
git commit -m "chore: prepare release 1.3.0

- Update version numbers
- Update CHANGELOG.md
- Final testing and validation"

# Merge to main
git checkout main
git pull origin main
git merge --no-ff release/1.3.0
git tag -a v1.3.0 -m "Release version 1.3.0"

# Merge back to develop
git checkout develop
git merge --no-ff release/1.3.0

# Push all changes
git push origin main develop --tags
git branch -d release/1.3.0
```

## Git Configuration

### Required Git Configuration
```bash
# Identity
git config --global user.name "Your Name"
git config --global user.email "your.email@company.com"

# Default behaviors
git config --global init.defaultBranch main
git config --global pull.rebase true
git config --global push.default current
git config --global core.autocrlf input  # Linux/Mac
git config --global core.autocrlf true   # Windows

# Helpful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD --stat'
git config --global alias.visual '!gitk'
```

### Project-Specific Git Hooks

Pre-commit hooks are managed by the `pre-commit` tool, but you can also set up additional project-specific hooks:

```bash
# .git/hooks/pre-push (optional additional validation)
#!/bin/bash
echo "Running pre-push validation..."

# Ensure we're not pushing to main directly
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" == "main" ]]; then
  echo "Direct pushes to main are not allowed. Use pull requests."
  exit 1
fi

# Run quick tests
uv run pytest tests/unit/ -x --tb=short
if [ $? -ne 0 ]; then
  echo "Tests failed. Push aborted."
  exit 1
fi

echo "Pre-push validation passed."
```

## Common Git Operations

### Keeping Branches Updated
```bash
# Update feature branch with latest develop
git checkout feature/your-feature
git fetch origin
git rebase origin/develop

# Handle conflicts during rebase
# Edit files to resolve conflicts
git add resolved-file.py
git rebase --continue

# Force push after rebase (be careful)
git push --force-with-lease origin feature/your-feature
```

### Fixing Commit Issues
```bash
# Amend last commit (before pushing)
git add forgotten-file.py
git commit --amend --no-edit

# Fix commit message
git commit --amend -m "fix: corrected commit message"

# Split a large commit
git reset HEAD~1
git add file1.py
git commit -m "feat: add file1 functionality"
git add file2.py
git commit -m "feat: add file2 functionality"
```

### Working with Stashes
```bash
# Save work in progress
git stash push -m "WIP: audio processing improvements"

# List stashes
git stash list

# Apply and remove stash
git stash pop stash@{0}

# Apply without removing
git stash apply stash@{0}
```

## Troubleshooting

### Common Issues

#### Pre-commit Hooks Failing
```bash
# Install pre-commit if missing
uv run pre-commit install

# Update hook versions
uv run pre-commit autoupdate

# Clear cache if hooks behave strangely
uv run pre-commit clean

# Skip a specific hook (emergency only)
SKIP=mypy git commit -m "emergency fix"  # NOT recommended
```

#### Merge Conflicts
```bash
# Configure merge tool
git config --global merge.tool vimdiff

# Start merge conflict resolution
git mergetool

# Or resolve manually and mark as resolved
git add conflicted-file.py
git commit
```

#### Large File Issues
```bash
# Check repository size
git count-objects -vH

# Find large files
git rev-list --objects --all | grep "$(git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -nr | head -10 | awk '{print $1}')"

# Remove large files from history (dangerous)
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD
```

## Best Practices

### Commit Best Practices
- **Atomic commits**: Each commit should represent one logical change
- **Descriptive messages**: Explain why, not just what
- **Regular commits**: Don't let changes accumulate too long
- **Test before committing**: Always run tests and pre-commit hooks

### Branch Management
- **Keep branches short-lived**: Merge frequently to avoid conflicts
- **Regular updates**: Rebase feature branches regularly
- **Clean history**: Use rebase to maintain linear history when appropriate
- **Delete merged branches**: Clean up after successful merges

### Collaboration
- **Communicate changes**: Use descriptive PR descriptions
- **Review thoroughly**: Both code and tests
- **Respond to feedback**: Address review comments promptly
- **Stay updated**: Keep up with team conventions and tools

## Integration with CI/CD

### GitHub Actions Integration
The project uses GitHub Actions for continuous integration:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up uv
        uses: astral-sh/setup-uv@v1
      - name: Install dependencies
        run: uv sync
      - name: Run pre-commit
        run: uv run pre-commit run --all-files
      - name: Run tests
        run: uv run pytest tests/ --cov=services
```

### Quality Gates
- All CI checks must pass before merging
- Code coverage must not decrease
- No security vulnerabilities detected
- Performance benchmarks within acceptable range

## Security Considerations

### Sensitive Data
- **Never commit secrets**: Use `.env` files (gitignored)
- **Rotate leaked secrets**: If accidentally committed, rotate immediately
- **Use environment variables**: For all configuration
- **Scan for secrets**: Use tools like `git-secrets` or `detect-secrets`

### Branch Protection
Configure GitHub branch protection rules:
- Require pull request reviews
- Require status checks to pass
- Require up-to-date branches
- Restrict pushes to main/develop

## Next Steps

After understanding Git workflow:
1. **Practice workflow**: Create a practice feature branch
2. **Configure tools**: Set up git aliases and merge tools
3. **Review PR process**: Read `docs/development/pr-review-process.md`
4. **Understand testing**: Study `docs/development/testing-guide.md`
5. **Learn debugging**: Review `docs/development/debugging-techniques.md`
