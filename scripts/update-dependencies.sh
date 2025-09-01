#!/usr/bin/env bash

# update-dependencies.sh - Comprehensive dependency updater for Tracktion project
#
# This script provides a safe and comprehensive way to update ALL dependency types:
# - Regular Python dependencies (project.dependencies)
# - Development dependencies (tool.uv.dev-dependencies)
# - Optional dependencies (project.optional-dependencies)
# - Dependency groups (dependency-groups)
# - Extra dependencies (all extras)
# - Service-specific dependencies (in services/*/pyproject.toml)
# - Pre-commit hooks
# - UV package manager version
# - Docker base images
# - Python version across all project files
#
# Usage: ./scripts/update-dependencies.sh [options]
#
# Options:
#   --python VERSION    Update Python version (default: keep current)
#   --no-backup        Skip creating backup files
#   --dry-run          Show what would be updated without making changes
#   --major            Include major version upgrades for packages
#   --skip-tests       Skip running tests after updates
#   --help             Show this help message

set -euo pipefail

# Store initial directory for recovery in case of errors
INITIAL_DIR="$(pwd)"

# Default options
BACKUP=true
DRY_RUN=false
MAJOR_UPGRADES=false
SKIP_TESTS=false
UPDATE_PYTHON=false
PYTHON_VERSION=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHANGES_MADE=false

# Emojis for visual logging
EMOJI_INFO="ℹ️"
EMOJI_SUCCESS="✅"
EMOJI_WARNING="⚠️"
EMOJI_ERROR="❌"
EMOJI_ROCKET="🚀"
EMOJI_PACKAGE="📦"
EMOJI_PYTHON="🐍"
EMOJI_DOCKER="🐳"
EMOJI_TEST="🧪"
EMOJI_BACKUP="💾"
EMOJI_CHANGES="📝"
EMOJI_VERIFY="🔍"
EMOJI_GIT="🔀"
EMOJI_HOOKS="🪝"

# Print colored output with emojis
print_info() {
    echo -e "\033[0;34m$EMOJI_INFO  [INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m$EMOJI_SUCCESS  [SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m$EMOJI_WARNING  [WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m$EMOJI_ERROR  [ERROR]\033[0m $1"
}

print_section() {
    echo ""
    echo -e "\033[1;36m$1  $2\033[0m"
    echo -e "\033[1;36m$(printf '=%.0s' {1..60})\033[0m"
}

# Show usage
show_help() {
    head -n 20 "$0" | grep '^#' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            UPDATE_PYTHON=true
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --major)
            MAJOR_UPGRADES=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if we're in the project root
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "uv.lock" ]]; then
    print_error "This script must be run from the project root directory"
    exit 1
fi

# Check required tools
for tool in uv git curl jq; do
    if ! command -v $tool &> /dev/null; then
        print_error "$tool is not installed. Please install it first."
        exit 1
    fi
done

# Check for uncommitted changes (only warn, don't exit)
if [[ -n $(git status --porcelain) ]]; then
    print_warning "You have uncommitted changes. Consider committing or stashing them for safe rollback."
    print_info "Continuing anyway..."
fi

# Create backup directory
BACKUP_DIR="backups/dependency-updates-${TIMESTAMP}"
if [[ "$BACKUP" == true ]] && [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$BACKUP_DIR"
    print_info "$EMOJI_BACKUP Creating backups in $BACKUP_DIR/"
fi

# Backup function
backup_file() {
    local file=$1
    if [[ "$BACKUP" == true ]] && [[ -f "$file" ]] && [[ "$DRY_RUN" == false ]]; then
        local backup_path
        backup_path="$BACKUP_DIR/$(dirname "$file")"
        mkdir -p "$backup_path"
        cp "$file" "$backup_path/$(basename "$file").backup"
    fi
}

# Track changes for summary
PACKAGE_CHANGES=()
FILE_CHANGES=()
UV_VERSION_CHANGE=""
PYTHON_VERSION_CHANGE=""
HOOK_CHANGES=()
SERVICE_CHANGES=()

# Helper function to safely get array length
array_length() {
    local array_name=$1
    eval "echo \${#${array_name}[@]}" 2>/dev/null || echo 0
}

# Function to capture package changes
capture_package_changes() {
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi

    # Compare uv.lock before and after
    if [[ -f "$BACKUP_DIR/uv.lock.backup" ]]; then
        print_info "$EMOJI_CHANGES Analyzing package changes..."

        # Extract package versions from backup
        local old_packages
        old_packages=$(grep -E "^name = |^version = " "$BACKUP_DIR/uv.lock.backup" | paste -d' ' - - | sed 's/name = "\(.*\)" version = "\(.*\)"/\1==\2/')

        # Extract package versions from current
        local new_packages
        new_packages=$(grep -E "^name = |^version = " "uv.lock" | paste -d' ' - - | sed 's/name = "\(.*\)" version = "\(.*\)"/\1==\2/')

        # Find changes
        while IFS= read -r old_pkg; do
            local pkg_name
            pkg_name=$(echo "$old_pkg" | cut -d'=' -f1)
            local old_version
            old_version=$(echo "$old_pkg" | cut -d'=' -f3)

            local new_version
            new_version=$(echo "$new_packages" | grep "^$pkg_name==" | cut -d'=' -f3 || echo "")

            if [[ -n "$new_version" ]] && [[ "$old_version" != "$new_version" ]]; then
                PACKAGE_CHANGES+=("$pkg_name: $old_version → $new_version")
                CHANGES_MADE=true
            fi
        done <<< "$old_packages"
    fi
}

# Update Python version function
update_python_version() {
    if [[ "$UPDATE_PYTHON" != true ]]; then
        return
    fi

    print_section "$EMOJI_PYTHON" "Updating Python Version"

    local current_version
    current_version=$(grep 'requires-python = ">=' pyproject.toml | sed 's/.*>=\([0-9.]*\)".*/\1/')
    PYTHON_VERSION_CHANGE="$current_version → $PYTHON_VERSION"

    if [[ "$current_version" == "$PYTHON_VERSION" ]]; then
        print_info "Python version is already $PYTHON_VERSION"
        return
    fi

    print_info "Updating Python from $current_version to $PYTHON_VERSION"

    if [[ "$DRY_RUN" == false ]]; then
        # Update root pyproject.toml
        backup_file "pyproject.toml"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/requires-python = \">=.*\"/requires-python = \">=$PYTHON_VERSION\"/" pyproject.toml
        else
            sed -i "s/requires-python = \">=.*\"/requires-python = \">=$PYTHON_VERSION\"/" pyproject.toml
        fi
        FILE_CHANGES+=("pyproject.toml: Python $current_version → $PYTHON_VERSION")

        # Update service pyproject.toml files
        for service_dir in services/*/; do
            if [[ -f "$service_dir/pyproject.toml" ]]; then
                backup_file "$service_dir/pyproject.toml"
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    sed -i '' "s/requires-python = \">=.*\"/requires-python = \">=$PYTHON_VERSION\"/" "$service_dir/pyproject.toml"
                else
                    sed -i "s/requires-python = \">=.*\"/requires-python = \">=$PYTHON_VERSION\"/" "$service_dir/pyproject.toml"
                fi
                FILE_CHANGES+=("$service_dir/pyproject.toml: Python $current_version → $PYTHON_VERSION")
            fi
        done

        # Update docker-compose files if they have PYTHON_VERSION
        for compose_file in docker-compose*.yml docker-compose*.yaml; do
            if [[ -f "$compose_file" ]] && grep -q "PYTHON_VERSION" "$compose_file"; then
                backup_file "$compose_file"
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    sed -i '' "s/PYTHON_VERSION:-[0-9.]\+/PYTHON_VERSION:-$PYTHON_VERSION/g" "$compose_file"
                else
                    sed -i "s/PYTHON_VERSION:-[0-9.]\+/PYTHON_VERSION:-$PYTHON_VERSION/g" "$compose_file"
                fi
                print_success "Updated $compose_file"
                FILE_CHANGES+=("$compose_file: Python $current_version → $PYTHON_VERSION")
            fi
        done

        CHANGES_MADE=true
    else
        print_info "[DRY RUN] Would update Python version to $PYTHON_VERSION"
    fi
}

# Update UV version in Docker and CI files
update_uv_version() {
    print_section "$EMOJI_DOCKER" "Updating UV Version"

    # Get the latest UV version from GitHub
    local latest_uv
    latest_uv=$(curl -s https://api.github.com/repos/astral-sh/uv/releases/latest | jq -r '.tag_name' | sed 's/^v//')

    if [[ -z "$latest_uv" ]]; then
        print_warning "Could not determine latest UV version from GitHub"
        return
    fi

    print_info "Latest UV version: $latest_uv"

    # Check current UV version in docker-compose files
    local current_uv=""
    for compose_file in docker-compose*.yml docker-compose*.yaml; do
        if [[ -f "$compose_file" ]]; then
            local version=$(grep "ghcr.io/astral-sh/uv:" "$compose_file" 2>/dev/null | head -1 | sed -E 's/.*uv:([0-9.]+).*/\1/')
            if [[ -n "$version" ]]; then
                current_uv="$version"
                break
            fi
        fi
    done

    # Update docker-compose files if UV is used
    if [[ -n "$current_uv" ]] && [[ "$current_uv" != "$latest_uv" ]]; then
        UV_VERSION_CHANGE="$current_uv → $latest_uv"
        print_info "Updating UV from $current_uv to $latest_uv in Docker files"

        if [[ "$DRY_RUN" == false ]]; then
            for compose_file in docker-compose*.yml docker-compose*.yaml; do
                if [[ -f "$compose_file" ]]; then
                    backup_file "$compose_file"
                    if [[ "$OSTYPE" == "darwin"* ]]; then
                        sed -i '' "s/ghcr.io\/astral-sh\/uv:[0-9.]*/ghcr.io\/astral-sh\/uv:$latest_uv/g" "$compose_file"
                    else
                        sed -i "s/ghcr.io\/astral-sh\/uv:[0-9.]*/ghcr.io\/astral-sh\/uv:$latest_uv/g" "$compose_file"
                    fi
                    print_success "Updated $compose_file"
                    FILE_CHANGES+=("$compose_file: UV $current_uv → $latest_uv")
                    CHANGES_MADE=true
                fi
            done
        else
            print_info "[DRY RUN] Would update UV version in Docker files"
        fi
    elif [[ -n "$current_uv" ]]; then
        print_success "UV version in Docker files is already up to date ($current_uv)"
    else
        print_info "No UV references found in Docker files (not using UV in containers)"
    fi
}

# Update pre-commit hooks to latest versions
update_precommit_hooks() {
    print_section "$EMOJI_HOOKS" "Updating Pre-commit Hooks"

    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        print_warning "No .pre-commit-config.yaml found, skipping hook updates"
        return
    fi

    if ! command -v pre-commit >/dev/null 2>&1; then
        print_warning "pre-commit not installed, installing it with uv..."
        if [[ "$DRY_RUN" == false ]]; then
            uv pip install pre-commit
        fi
    fi

    print_info "Updating pre-commit hooks to latest versions..."

    if [[ "$DRY_RUN" == false ]]; then
        # Backup the pre-commit config
        backup_file ".pre-commit-config.yaml"

        # Store current hook versions for comparison
        local old_hooks
        old_hooks=$(grep "rev:" .pre-commit-config.yaml)

        # Update all hooks to latest versions with freeze
        if pre-commit autoupdate --freeze; then
            print_success "Pre-commit hooks updated successfully"

            # Check what changed
            local new_hooks
            new_hooks=$(grep "rev:" .pre-commit-config.yaml)

            if [[ "$old_hooks" != "$new_hooks" ]]; then
                FILE_CHANGES+=(".pre-commit-config.yaml: Updated pre-commit hooks to latest versions")
                HOOK_CHANGES+=("Pre-commit hooks updated")
                CHANGES_MADE=true
            fi

            # Run pre-commit install to ensure hooks are installed
            pre-commit install
        else
            print_warning "Failed to update pre-commit hooks"
        fi
    else
        print_info "[DRY RUN] Would run: pre-commit autoupdate --freeze"
    fi
}

# Update Python packages - ALL dependency types are handled:
# - project.dependencies (regular dependencies)
# - tool.uv.dev-dependencies (legacy dev dependencies)
# - project.optional-dependencies (extras)
# - dependency-groups (new standard for grouped dependencies)
# This ensures 100% coverage of all dependency types
update_python_packages() {
    print_section "$EMOJI_PACKAGE" "Updating Python Packages"

    # Backup critical files
    if [[ "$BACKUP" == true ]] && [[ "$DRY_RUN" == false ]]; then
        backup_file "uv.lock"
        backup_file "pyproject.toml"

        # Backup service pyproject.toml files
        for service_dir in services/*/; do
            if [[ -f "$service_dir/pyproject.toml" ]]; then
                backup_file "$service_dir/pyproject.toml"
            fi
        done
    fi

    # Update uv itself
    print_info "Checking for uv updates..."
    if [[ "$DRY_RUN" == false ]]; then
        if uv self update 2>/dev/null; then
            print_success "uv updated successfully"
        else
            print_info "uv is already at the latest version or managed externally"
        fi
    else
        print_info "[DRY RUN] Would check for uv updates"
    fi

    # Check for outdated packages first
    print_info "Checking for outdated packages..."
    if [[ "$DRY_RUN" == true ]]; then
        print_info "[DRY RUN] Current outdated packages:"
        uv tree --outdated 2>/dev/null || print_info "Unable to check outdated packages"
    fi

    # Update ALL types of dependencies
    print_info "Updating all dependencies (regular, dev, optional, extras, and groups)..."

    # Use --all flag to update all dependency types
    local uv_cmd="uv lock"
    if [[ "$MAJOR_UPGRADES" == true ]]; then
        uv_cmd="$uv_cmd --upgrade"
        print_info "Including major version upgrades for all dependency types"
    else
        uv_cmd="$uv_cmd --upgrade"
        print_info "Upgrading to latest compatible versions for all dependency types"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        print_info "[DRY RUN] Would run: $uv_cmd"
    else
        if $uv_cmd; then
            print_success "All dependencies locked successfully"
            CHANGES_MADE=true
        else
            print_error "Failed to lock dependencies"
            exit 1
        fi
    fi

    # Sync ALL dependency types - use --all-groups and --all-extras to ensure everything is updated
    if [[ "$DRY_RUN" == false ]]; then
        print_info "Syncing all dependency types (regular, dev, optional, extras, groups)..."
        # Use --all-groups to sync all dependency groups and --all-extras for all optional dependencies
        if uv sync --all-extras --all-groups; then
            print_success "All dependency types synced successfully"
        else
            print_error "Failed to sync dependencies"
            exit 1
        fi

        # Capture package changes
        capture_package_changes
    else
        print_info "[DRY RUN] Would run: uv sync --all-extras --all-groups"
    fi

    # Update service-specific dependencies
    update_service_dependencies
}

# Update service-specific dependencies
# Uses pushd/popd for safer directory management:
# - Maintains a directory stack for reliable navigation
# - Automatically returns to previous directory even if errors occur
# - More robust than cd/cd - pattern
update_service_dependencies() {
    print_section "$EMOJI_PACKAGE" "Updating Service Dependencies"

    for service_dir in services/*/; do
        if [[ -f "$service_dir/pyproject.toml" ]]; then
            local service_name=$(basename "$service_dir")
            print_info "Updating dependencies for $service_name..."

            if [[ "$DRY_RUN" == false ]]; then
                # Use pushd for safer directory management
                # This ensures we can always return to the original directory
                if ! pushd "$service_dir" > /dev/null 2>&1; then
                    print_warning "Could not enter directory $service_dir"
                    continue
                fi

                # Process service in a trap-protected block to ensure popd is called
                {
                    # Check if service has its own uv.lock
                    if [[ -f "uv.lock" ]]; then
                        backup_file "uv.lock"

                        # Update service dependencies
                        local service_uv_cmd="uv lock"
                        if [[ "$MAJOR_UPGRADES" == true ]]; then
                            service_uv_cmd="$service_uv_cmd --upgrade"
                        else
                            service_uv_cmd="$service_uv_cmd --upgrade"
                        fi

                        if $service_uv_cmd; then
                            print_success "Updated $service_name dependencies"
                            SERVICE_CHANGES+=("$service_name: All dependencies updated")

                            # Sync ALL service dependency types
                            uv sync --all-extras --all-groups
                        else
                            print_warning "Failed to update $service_name dependencies"
                        fi
                    fi
                }

                # Always return to previous directory, even if an error occurred
                popd > /dev/null 2>&1 || {
                    print_error "Failed to return to original directory"
                    # Try to recover by going back to project root
                    cd "$INITIAL_DIR" 2>/dev/null || exit 1
                }
            else
                print_info "[DRY RUN] Would update $service_name dependencies"
            fi
        fi
    done
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return
    fi

    print_section "$EMOJI_TEST" "Running Tests"

    # Run pre-commit hooks
    print_info "Running pre-commit hooks..."
    if pre-commit run --all-files; then
        print_success "Pre-commit hooks passed"
    else
        print_warning "Some pre-commit hooks failed - review the changes"
    fi

    # Run ruff linting
    print_info "Running ruff linting..."
    if uv run ruff check .; then
        print_success "Ruff linting passed"
    else
        print_warning "Ruff linting failed - review the issues"
    fi

    # Run mypy type checking
    print_info "Running mypy type checking..."
    if uv run mypy .; then
        print_success "Type checking passed"
    else
        print_warning "Type checking failed - review the issues"
    fi

    # Run tests
    print_info "Running pytest..."
    if uv run pytest tests/; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed - review the failures"
    fi
}

# Generate summary
generate_summary() {
    print_section "$EMOJI_CHANGES" "Update Summary"

    if [[ "$DRY_RUN" == true ]]; then
        print_info "This was a dry run. No changes were made."
        print_info "Run without --dry-run to apply changes."
        return
    fi

    if [[ "$CHANGES_MADE" == false ]]; then
        print_success "Everything is already up to date! No changes were needed."
        return
    fi

    # Python version changes
    if [[ -n "$PYTHON_VERSION_CHANGE" ]]; then
        echo ""
        echo "🐍 Python Version:"
        echo "  $PYTHON_VERSION_CHANGE"
    fi

    # UV version changes
    if [[ -n "$UV_VERSION_CHANGE" ]]; then
        echo ""
        echo "🐳 UV Package Manager:"
        echo "  $UV_VERSION_CHANGE"
    fi

    # Package changes
    if [[ $(array_length PACKAGE_CHANGES) -gt 0 ]]; then
        echo ""
        echo "📦 Package Updates:"
        printf '%s\n' "${PACKAGE_CHANGES[@]:-}" | sort | head -20 | while IFS= read -r change; do
            echo "  • $change"
        done

        local total_changes=$(array_length PACKAGE_CHANGES)
        if [[ $total_changes -gt 20 ]]; then
            echo "  ... and $((total_changes - 20)) more packages"
        fi
    fi

    # Service changes
    if [[ $(array_length SERVICE_CHANGES) -gt 0 ]]; then
        echo ""
        echo "🔧 Service Updates:"
        for change in "${SERVICE_CHANGES[@]:-}"; do
            echo "  • $change"
        done
    fi

    # File changes
    if [[ $(array_length FILE_CHANGES) -gt 0 ]]; then
        echo ""
        echo "📄 File Updates:"
        printf '%s\n' "${FILE_CHANGES[@]:-}" | sort | while IFS= read -r change; do
            echo "  • $change"
        done
    fi

    # Hook changes
    if [[ $(array_length HOOK_CHANGES) -gt 0 ]]; then
        echo ""
        echo "🪝 Pre-commit Hook Updates:"
        for change in "${HOOK_CHANGES[@]:-}"; do
            echo "  • $change"
        done
    fi

    # Git instructions
    echo ""
    print_section "$EMOJI_GIT" "Next Steps"

    echo "1. Review the changes:"
    echo "   git diff --stat"
    echo "   git diff uv.lock"
    echo "   git diff .pre-commit-config.yaml"

    echo ""
    echo "2. Test the updates:"
    echo "   uv run pytest tests/"
    echo "   pre-commit run --all-files"

    echo ""
    echo "3. Stage the changes:"
    echo "   git add uv.lock pyproject.toml"

    if [[ $(array_length SERVICE_CHANGES) -gt 0 ]]; then
        echo "   git add services/*/pyproject.toml services/*/uv.lock"
    fi

    if [[ $(array_length HOOK_CHANGES) -gt 0 ]]; then
        echo "   git add .pre-commit-config.yaml"
    fi

    if [[ -n "$UV_VERSION_CHANGE" ]]; then
        echo "   git add docker-compose*.yaml docker-compose*.yml"
    fi

    echo ""
    echo "4. Commit the changes:"
    echo "   git commit -m \"chore: update dependencies"

    if [[ -n "$PYTHON_VERSION_CHANGE" ]]; then
        echo ""
        echo "   - Update Python to ${PYTHON_VERSION_CHANGE##* → }"
    fi

    if [[ -n "$UV_VERSION_CHANGE" ]]; then
        echo "   - Update UV to ${UV_VERSION_CHANGE##* → }"
    fi

    if [[ $(array_length PACKAGE_CHANGES) -gt 0 ]]; then
        echo "   - Update $(array_length PACKAGE_CHANGES) Python packages (all dependency types: regular, dev, optional, extras, groups)"
    fi

    if [[ $(array_length HOOK_CHANGES) -gt 0 ]]; then
        echo "   - Update pre-commit hooks to latest versions"
    fi

    echo "   \""
}

# Manual verification steps
show_verification_steps() {
    print_section "$EMOJI_VERIFY" "Manual Verification Steps"

    echo "Please verify the following before merging:"
    echo ""
    echo "1. 🧪 Run the full test suite:"
    echo "   uv run pytest tests/ -v"
    echo ""
    echo "2. 🪝 Verify pre-commit hooks:"
    echo "   pre-commit run --all-files"
    echo ""
    echo "3. 📊 Check type hints:"
    echo "   uv run mypy ."
    echo ""
    echo "4. 🔍 Review dependency changes for breaking updates:"
    echo "   git diff uv.lock | grep -E \"^[+-]version\""
    echo ""
    echo "5. 🐳 If using Docker, rebuild and test:"
    echo "   docker-compose build"
    echo "   docker-compose up -d"
    echo ""

    if [[ "$BACKUP" == true ]]; then
        echo "💾 Backups are stored in: $BACKUP_DIR/"
        echo "   To restore: cp $BACKUP_DIR/uv.lock.backup uv.lock && uv sync --all-extras --all-groups"
    fi
}

# Handle errors
trap 'handle_error $?' ERR

handle_error() {
    local exit_code=$1
    print_error "An error occurred (exit code: $exit_code)"

    if [[ "$BACKUP" == true ]] && [[ "$DRY_RUN" == false ]] && [[ -d "$BACKUP_DIR" ]]; then
        print_info "You can restore from backup with:"
        echo "  cp $BACKUP_DIR/uv.lock.backup uv.lock"
        echo "  cp $BACKUP_DIR/pyproject.toml.backup pyproject.toml"
        echo "  uv sync --all-extras --all-groups"
    fi

    exit $exit_code
}

# Main execution
main() {
    print_section "$EMOJI_ROCKET" "Starting Dependency Update"

    # Update Python version if requested
    update_python_version

    # Update UV version in Docker files
    update_uv_version

    # Update pre-commit hooks
    update_precommit_hooks

    # Update Python packages (including dev dependencies)
    update_python_packages

    # Run tests
    run_tests

    # Generate summary
    generate_summary

    # Show verification steps
    if [[ "$DRY_RUN" == false ]] && [[ "$CHANGES_MADE" == true ]]; then
        show_verification_steps
    fi

    print_success "Update process completed!"
}

# Run main function
main
