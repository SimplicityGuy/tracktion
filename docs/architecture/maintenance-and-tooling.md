# Maintenance and Tooling

This section defines the strategy for managing and updating project dependencies and tooling.

### **Dependency Management Policy**

  * **Version Strategy:** Always use the latest stable versions of all dependencies
  * **Update Frequency:** Review and update dependencies monthly or when security updates are available
  * **Testing:** All updates must pass the full test suite before being committed
  * **Documentation:** Document any breaking changes or migration steps required

### **Pre-commit Hook Management**

Pre-commit hooks ensure code quality and consistency. They must be managed carefully:

  * **Update Command:** `pre-commit autoupdate --freeze`
  * **Frozen Versions:** Always use frozen commit hashes for reproducible builds
  * **Installation:** Run `pre-commit install` after cloning the repository
  * **Manual Run:** Use `pre-commit run --all-files` to check all files
  * **Skip Hooks:** In emergencies only, use `git commit --no-verify`

### **Comprehensive Update Script**

A single, top-level script will be created to manage project maintenance tasks on-demand. This script will perform the following actions:

  * Update all Python dependencies using `uv`
  * Update `pre-commit` hooks using `pre-commit autoupdate --freeze`
  * Synchronize dependencies and linting rules across all services and shared packages
  * Ensure all GitHub workflows are up-to-date and correctly configured
  * Run full test suite to validate all updates

### **Tool Versions**

Current tool versions in use:

  * **Python:** 3.13
  * **Docker:** Latest stable
  * **PostgreSQL:** 17
  * **Neo4j:** 5.26
  * **Redis:** 7.4
  * **RabbitMQ:** 4.0
  * **Ruff:** 0.12.9 (via pre-commit)
  * **MyPy:** 1.17.1 (via pre-commit)
  * **Pre-commit:** 4.0.1+

### **Update Procedures**

#### Updating Python Dependencies
```bash
# Update all dependencies to latest versions
uv pip compile --upgrade pyproject.toml -o requirements.txt
uv pip sync requirements.txt
```

#### Updating Pre-commit Hooks
```bash
# Update hooks with frozen versions
pre-commit autoupdate --freeze

# Test the updated hooks
pre-commit run --all-files

# Commit the changes
git add .pre-commit-config.yaml
git commit -m "chore: update pre-commit hooks to latest versions"
```

#### Updating Docker Images
1. Update image tags in `docker-compose.yaml`
2. Test with `docker-compose build --no-cache`
3. Verify services start correctly
4. Update documentation if needed
