#!/bin/bash

# Documentation deployment script for Tracktion
# Supports both local development and production deployment

set -e

# Configuration
DEFAULT_VERSION="dev"
SITE_NAME="Tracktion Documentation"
REMOTE="origin"
BRANCH="gh-pages"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Documentation Deployment Script

Usage: $0 [OPTIONS] [VERSION]

Options:
    -h, --help          Show this help message
    -l, --local         Deploy locally (serve with mike)
    -p, --push          Push to remote repository
    -d, --dry-run       Show what would be deployed without deploying
    -c, --clean         Clean build artifacts before deployment
    -s, --set-default   Set this version as the default
    -a, --alias ALIAS   Add an alias for this version
    -r, --remote NAME   Remote repository name (default: origin)

Version:
    Version to deploy (default: dev)
    Use semantic versioning (e.g., 1.0.0) for releases
    Use 'latest' for the latest release
    Use 'dev' for development version

Examples:
    $0                          # Deploy dev version locally
    $0 1.0.0 --push            # Deploy version 1.0.0 and push to remote
    $0 latest --set-default    # Deploy and set as default version
    $0 --local --alias stable  # Deploy locally with 'stable' alias
    $0 --clean --dry-run       # Show deployment plan with clean build

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed. Please install uv first."
        exit 1
    fi

    # Check if mike is available
    if ! uv run mike --help &> /dev/null; then
        log_error "mike is not installed. Installing..."
        uv pip install mike
    fi

    # Check if mkdocs is available
    if ! uv run mkdocs --help &> /dev/null; then
        log_error "mkdocs is not installed. Please run 'uv sync' to install dependencies."
        exit 1
    fi

    log_success "All dependencies are available"
}

clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf site/
    rm -rf .mkdocs_cache/
    log_success "Build artifacts cleaned"
}

generate_docs() {
    log_info "Generating auto-documentation..."
    if [[ -f "scripts/generate_docs.py" ]]; then
        uv run python scripts/generate_docs.py
        log_success "Auto-documentation generated"
    else
        log_warning "Auto-generation script not found, skipping"
    fi
}

validate_docs() {
    log_info "Validating documentation..."
    uv run mkdocs build --strict --quiet
    log_success "Documentation validation passed"
}

deploy_local() {
    local version=$1
    local alias=$2
    local set_default=$3

    log_info "Deploying documentation locally..."

    # Deploy with mike
    if [[ -n "$alias" ]]; then
        uv run mike deploy --update-aliases "$version" "$alias"
        log_success "Deployed version $version with alias '$alias'"
    else
        uv run mike deploy "$version"
        log_success "Deployed version $version"
    fi

    # Set as default if requested
    if [[ "$set_default" == "true" ]]; then
        uv run mike set-default "$version"
        log_success "Set $version as default version"
    fi

    # Show available versions
    log_info "Available versions:"
    uv run mike list

    # Serve documentation
    log_info "Starting local server..."
    log_success "Documentation available at: http://localhost:8000"
    uv run mike serve
}

deploy_remote() {
    local version=$1
    local alias=$2
    local set_default=$3
    local remote=$4
    local push=$5

    log_info "Deploying documentation for remote deployment..."

    # Deploy with mike
    if [[ -n "$alias" ]]; then
        if [[ "$push" == "true" ]]; then
            uv run mike deploy --push --remote "$remote" --update-aliases "$version" "$alias"
        else
            uv run mike deploy --update-aliases "$version" "$alias"
        fi
        log_success "Deployed version $version with alias '$alias'"
    else
        if [[ "$push" == "true" ]]; then
            uv run mike deploy --push --remote "$remote" "$version"
        else
            uv run mike deploy "$version"
        fi
        log_success "Deployed version $version"
    fi

    # Set as default if requested
    if [[ "$set_default" == "true" ]]; then
        if [[ "$push" == "true" ]]; then
            uv run mike set-default --push --remote "$remote" "$version"
        else
            uv run mike set-default "$version"
        fi
        log_success "Set $version as default version"
    fi

    # Show deployment info
    log_info "Available versions:"
    uv run mike list

    if [[ "$push" == "true" ]]; then
        log_success "Documentation pushed to $remote/$BRANCH"
    else
        log_warning "Documentation built but not pushed. Use --push to deploy to remote."
    fi
}

dry_run() {
    local version=$1
    local alias=$2
    local set_default=$3
    local is_local=$4
    local push=$5

    log_info "=== DRY RUN MODE ==="
    log_info "Would deploy documentation with the following configuration:"
    echo "  Version: $version"
    [[ -n "$alias" ]] && echo "  Alias: $alias"
    echo "  Set as default: $set_default"
    echo "  Local deployment: $is_local"
    echo "  Push to remote: $push"
    echo "  Remote: $REMOTE"
    echo ""

    log_info "Current versions:"
    uv run mike list || log_warning "No versions found"

    log_info "Would run the following commands:"
    if [[ -n "$alias" ]]; then
        if [[ "$is_local" == "true" ]]; then
            echo "  uv run mike deploy --update-aliases $version $alias"
        elif [[ "$push" == "true" ]]; then
            echo "  uv run mike deploy --push --remote $REMOTE --update-aliases $version $alias"
        else
            echo "  uv run mike deploy --update-aliases $version $alias"
        fi
    else
        if [[ "$is_local" == "true" ]]; then
            echo "  uv run mike deploy $version"
        elif [[ "$push" == "true" ]]; then
            echo "  uv run mike deploy --push --remote $REMOTE $version"
        else
            echo "  uv run mike deploy $version"
        fi
    fi

    if [[ "$set_default" == "true" ]]; then
        if [[ "$is_local" == "true" ]]; then
            echo "  uv run mike set-default $version"
        elif [[ "$push" == "true" ]]; then
            echo "  uv run mike set-default --push --remote $REMOTE $version"
        else
            echo "  uv run mike set-default $version"
        fi
    fi

    [[ "$is_local" == "true" ]] && echo "  uv run mike serve"
}

# Main script
main() {
    local version="$DEFAULT_VERSION"
    local is_local="true"
    local push="false"
    local dry_run_mode="false"
    local clean_mode="false"
    local set_default="false"
    local alias=""
    local remote="$REMOTE"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--local)
                is_local="true"
                shift
                ;;
            -p|--push)
                push="true"
                is_local="false"
                shift
                ;;
            -d|--dry-run)
                dry_run_mode="true"
                shift
                ;;
            -c|--clean)
                clean_mode="true"
                shift
                ;;
            -s|--set-default)
                set_default="true"
                shift
                ;;
            -a|--alias)
                alias="$2"
                shift 2
                ;;
            -r|--remote)
                remote="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option $1"
                show_help
                exit 1
                ;;
            *)
                version="$1"
                shift
                ;;
        esac
    done

    log_info "Starting documentation deployment for version: $version"

    # Check dependencies
    check_dependencies

    # Clean if requested
    [[ "$clean_mode" == "true" ]] && clean_build

    # Generate documentation
    generate_docs

    # Validate documentation
    validate_docs

    # Handle dry run
    if [[ "$dry_run_mode" == "true" ]]; then
        dry_run "$version" "$alias" "$set_default" "$is_local" "$push"
        exit 0
    fi

    # Deploy documentation
    if [[ "$is_local" == "true" ]]; then
        deploy_local "$version" "$alias" "$set_default"
    else
        deploy_remote "$version" "$alias" "$set_default" "$remote" "$push"
    fi
}

# Run main function with all arguments
main "$@"
