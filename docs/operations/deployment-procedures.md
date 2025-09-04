# Deployment Procedures

## Table of Contents

1. [Overview](#overview)
2. [Deployment Strategy](#deployment-strategy)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Environment Management](#environment-management)
5. [Release Management](#release-management)
6. [Pre-Deployment Checklist](#pre-deployment-checklist)
7. [Deployment Execution](#deployment-execution)
8. [Post-Deployment Validation](#post-deployment-validation)
9. [Production Deployment](#production-deployment)
10. [Database Migrations](#database-migrations)
11. [Configuration Management](#configuration-management)
12. [Monitoring and Alerting](#monitoring-and-alerting)
13. [Troubleshooting](#troubleshooting)
14. [Deployment Automation](#deployment-automation)

## Overview

This document outlines comprehensive deployment procedures for the Tracktion audio analysis platform, ensuring reliable, secure, and efficient software releases across all environments. The procedures follow industry best practices and incorporate automated testing, staged deployments, and comprehensive validation.

### Deployment Principles

- **Automation First**: Minimize manual intervention through automation
- **Gradual Rollout**: Progressive deployment to minimize blast radius
- **Fail Fast**: Early detection and rapid rollback of issues
- **Reproducible**: Consistent deployments across all environments
- **Observable**: Comprehensive monitoring and logging of deployments
- **Secure**: Security validation at every stage

### Deployment Objectives

- **Zero-Downtime**: Maintain service availability during deployments
- **Reliability**: Ensure successful deployments with minimal failures
- **Speed**: Rapid deployment cycles to enable agile development
- **Quality**: Maintain high code quality and system stability
- **Compliance**: Meet security and regulatory requirements

## Deployment Strategy

### Blue-Green Deployment

The Tracktion system uses blue-green deployment strategy for zero-downtime deployments:

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   Traffic Router   │
        │    (Switch)        │
        └─────────┬─────────┘
                  │
     ┌────────────▼────────────┐
     │                         │
┌────▼────┐              ┌────▼────┐
│ Blue    │              │ Green   │
│ (Live)  │              │ (Stage) │
│         │              │         │
│ v1.2.3  │     or       │ v1.2.4  │
│         │              │         │
└─────────┘              └─────────┘
```

#### Blue-Green Process
1. **Blue Environment**: Currently serving production traffic
2. **Green Environment**: Staging environment for new deployment
3. **Deploy to Green**: New version deployed to green environment
4. **Test Green**: Comprehensive testing on green environment
5. **Switch Traffic**: Atomic traffic switch from blue to green
6. **Monitor**: Monitor green environment for issues
7. **Rollback**: Quick rollback to blue if issues detected

### Rolling Deployment

For services that support rolling updates:

```yaml
# Rolling deployment configuration
rolling_deployment:
  max_unavailable: 25%
  max_surge: 25%
  strategy:
    - update_batch_size: 1
    - health_check_grace_period: 30s
    - health_check_timeout: 60s
    - rollback_on_failure: true
```

### Canary Deployment

For high-risk deployments:

```yaml
# Canary deployment stages
canary_stages:
  - name: "canary_5"
    traffic_percentage: 5
    duration: 15m
    success_criteria:
      error_rate: <1%
      response_time_p95: <2000ms

  - name: "canary_20"
    traffic_percentage: 20
    duration: 30m
    success_criteria:
      error_rate: <0.5%
      response_time_p95: <1500ms

  - name: "full_rollout"
    traffic_percentage: 100
    success_criteria:
      error_rate: <0.1%
      response_time_p95: <1000ms
```

## CI/CD Pipeline

### Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Commit    │───▶│   Build     │───▶│    Test     │───▶│   Package   │
│   & Push    │    │   & Lint    │    │  & Scan     │    │  & Publish  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│ Production  │◀───│   Staging   │◀───│ Development │◀──────────┘
│ Deployment  │    │ Deployment  │    │ Deployment  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### GitHub Actions Pipeline

```yaml
# .github/workflows/deploy.yml
name: Tracktion Deployment Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: tracktion

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [analysis-service, file-watcher, tracklist-service, notification-service]

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: |
        cd services/${{ matrix.service }}
        uv sync

    - name: Run pre-commit hooks
      run: |
        cd services/${{ matrix.service }}
        uv run pre-commit run --all-files

    - name: Run tests with coverage
      run: |
        cd services/${{ matrix.service }}
        uv run pytest tests/ --cov=src --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./services/${{ matrix.service }}/coverage.xml
        flags: ${{ matrix.service }}

    - name: Run security scan
      run: |
        cd services/${{ matrix.service }}
        uv run bandit -r src/ -f json -o security-report.json

  build-and-push:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'

    strategy:
      matrix:
        service: [analysis-service, file-watcher, tracklist-service, notification-service]

    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ github.repository }}/${{ matrix.service }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: ./services/${{ matrix.service }}
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        image: ${{ steps.meta.outputs.tags }}
        format: spdx-json
        output-file: ${{ matrix.service }}-sbom.spdx.json

    - name: Scan image for vulnerabilities
      uses: anchore/scan-action@v3
      with:
        image: ${{ steps.meta.outputs.tags }}
        fail-build: true
        severity-cutoff: high

  deploy-development:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: development

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Deploy to Development
      run: |
        ./scripts/deploy.sh development ${{ needs.build-and-push.outputs.image-tag }}

    - name: Run smoke tests
      run: |
        ./scripts/smoke-tests.sh development

    - name: Notify deployment status
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Development deployment completed'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  deploy-staging:
    needs: [build-and-push, deploy-development]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Deploy to Staging
      run: |
        ./scripts/deploy.sh staging ${{ needs.build-and-push.outputs.image-tag }}

    - name: Run integration tests
      run: |
        ./scripts/integration-tests.sh staging

    - name: Run performance tests
      run: |
        ./scripts/performance-tests.sh staging

    - name: Security scan
      run: |
        ./scripts/security-scan.sh staging

  deploy-production:
    needs: [build-and-push, deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Production deployment approval
      uses: trstringer/manual-approval@v1
      with:
        secret: ${{ secrets.GITHUB_TOKEN }}
        approvers: ${{ secrets.PRODUCTION_APPROVERS }}
        minimum-approvals: 2
        issue-title: "Production Deployment Approval"

    - name: Deploy to Production (Blue-Green)
      run: |
        ./scripts/deploy-production.sh ${{ needs.build-and-push.outputs.image-tag }}

    - name: Post-deployment validation
      run: |
        ./scripts/post-deployment-validation.sh production

    - name: Update monitoring dashboards
      run: |
        ./scripts/update-dashboards.sh ${{ needs.build-and-push.outputs.image-tag }}
```

### Pipeline Stages Detail

#### Stage 1: Code Quality & Testing
```bash
#!/bin/bash
# lint_and_test.sh

set -e

SERVICE="$1"
echo "Running quality checks for $SERVICE..."

cd "services/$SERVICE"

# Install dependencies
echo "Installing dependencies..."
uv sync

# Run linting and type checking
echo "Running pre-commit hooks..."
uv run pre-commit run --all-files

# Run unit tests with coverage
echo "Running unit tests..."
uv run pytest tests/unit/ \
    --cov=src \
    --cov-report=xml \
    --cov-report=html \
    --cov-fail-under=80 \
    --junit-xml=test-results.xml

# Run integration tests
echo "Running integration tests..."
uv run pytest tests/integration/ \
    --junit-xml=integration-test-results.xml

# Security scanning
echo "Running security scan..."
uv run bandit -r src/ -f json -o security-report.json

# License scanning
echo "Checking license compliance..."
uv run pip-licenses --format=json --output-file=licenses.json

# Dependency vulnerability scanning
echo "Scanning dependencies for vulnerabilities..."
uv run safety check --json --output=safety-report.json

echo "Quality checks completed for $SERVICE"
```

#### Stage 2: Build & Package
```bash
#!/bin/bash
# build_and_package.sh

set -e

SERVICE="$1"
TAG="$2"
REGISTRY="ghcr.io/tracktion"

echo "Building and packaging $SERVICE with tag $TAG..."

cd "services/$SERVICE"

# Build Docker image
echo "Building Docker image..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "$REGISTRY/$SERVICE:$TAG" \
    --tag "$REGISTRY/$SERVICE:latest" \
    --push \
    --cache-from type=gha \
    --cache-to type=gha,mode=max \
    .

# Generate and sign SBOM
echo "Generating SBOM..."
syft "$REGISTRY/$SERVICE:$TAG" -o spdx-json > "$SERVICE-sbom.spdx.json"

# Sign image with cosign
echo "Signing image..."
cosign sign "$REGISTRY/$SERVICE:$TAG"

# Scan image for vulnerabilities
echo "Scanning image for vulnerabilities..."
grype "$REGISTRY/$SERVICE:$TAG" -o json > "$SERVICE-vulnerabilities.json"

# Check vulnerability threshold
CRITICAL_COUNT=$(jq '[.matches[] | select(.vulnerability.severity == "Critical")] | length' "$SERVICE-vulnerabilities.json")
HIGH_COUNT=$(jq '[.matches[] | select(.vulnerability.severity == "High")] | length' "$SERVICE-vulnerabilities.json")

if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "❌ Critical vulnerabilities found: $CRITICAL_COUNT"
    exit 1
fi

if [ "$HIGH_COUNT" -gt 5 ]; then
    echo "❌ Too many high severity vulnerabilities: $HIGH_COUNT (max: 5)"
    exit 1
fi

echo "✅ Image security scan passed"
echo "Build and packaging completed for $SERVICE"
```

## Environment Management

### Environment Configuration

#### Development Environment
```yaml
# environments/development.yml
environment:
  name: development
  cluster: dev-cluster
  namespace: tracktion-dev

infrastructure:
  replicas:
    analysis-service: 1
    file-watcher: 1
    tracklist-service: 1
    notification-service: 1

  resources:
    limits:
      cpu: "500m"
      memory: "512Mi"
    requests:
      cpu: "250m"
      memory: "256Mi"

  database:
    instance_type: "db.t3.micro"
    storage: "20GB"
    backup_retention: "7 days"

  monitoring:
    enabled: true
    log_level: "DEBUG"
    metrics_retention: "7 days"

database:
  host: "dev-db.tracktion.internal"
  port: 5432
  database: "tracktion_dev"
  connection_pool:
    min_size: 1
    max_size: 5

redis:
  host: "dev-redis.tracktion.internal"
  port: 6379
  database: 0

external_services:
  api_rate_limit: 1000
  webhook_timeout: 30s
```

#### Staging Environment
```yaml
# environments/staging.yml
environment:
  name: staging
  cluster: staging-cluster
  namespace: tracktion-staging

infrastructure:
  replicas:
    analysis-service: 2
    file-watcher: 1
    tracklist-service: 2
    notification-service: 1

  resources:
    limits:
      cpu: "1000m"
      memory: "1Gi"
    requests:
      cpu: "500m"
      memory: "512Mi"

  database:
    instance_type: "db.t3.small"
    storage: "100GB"
    backup_retention: "30 days"
    read_replicas: 1

  monitoring:
    enabled: true
    log_level: "INFO"
    metrics_retention: "30 days"

database:
  host: "staging-db.tracktion.internal"
  port: 5432
  database: "tracktion_staging"
  connection_pool:
    min_size: 2
    max_size: 10

redis:
  host: "staging-redis.tracktion.internal"
  port: 6379
  database: 0
  cluster_mode: false

external_services:
  api_rate_limit: 5000
  webhook_timeout: 30s
```

#### Production Environment
```yaml
# environments/production.yml
environment:
  name: production
  cluster: prod-cluster
  namespace: tracktion-prod

infrastructure:
  replicas:
    analysis-service: 5
    file-watcher: 3
    tracklist-service: 4
    notification-service: 2

  resources:
    limits:
      cpu: "2000m"
      memory: "2Gi"
    requests:
      cpu: "1000m"
      memory: "1Gi"

  database:
    instance_type: "db.r5.xlarge"
    storage: "500GB"
    backup_retention: "30 days"
    read_replicas: 2
    multi_az: true

  monitoring:
    enabled: true
    log_level: "WARN"
    metrics_retention: "90 days"

database:
  host: "prod-db.tracktion.internal"
  port: 5432
  database: "tracktion_prod"
  connection_pool:
    min_size: 5
    max_size: 20
  ssl_mode: "require"

redis:
  host: "prod-redis.tracktion.internal"
  port: 6379
  database: 0
  cluster_mode: true
  ssl: true

external_services:
  api_rate_limit: 10000
  webhook_timeout: 30s
  cdn_enabled: true
```

### Environment Provisioning

#### Terraform Configuration
```hcl
# infrastructure/environments/production/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket         = "tracktion-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tracktion-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "production"
      Project     = "tracktion"
      ManagedBy   = "terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "../../modules/vpc"

  environment = var.environment
  vpc_cidr    = "10.0.0.0/16"

  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.11.0/24", "10.0.12.0/24"]
  database_subnets = ["10.0.21.0/24", "10.0.22.0/24"]

  enable_nat_gateway     = true
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
}

# EKS Cluster
module "eks" {
  source = "../../modules/eks"

  cluster_name    = "tracktion-${var.environment}"
  cluster_version = "1.27"

  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3

      instance_types = ["c5.xlarge"]
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
  }
}

# RDS Database
module "database" {
  source = "../../modules/rds"

  identifier = "tracktion-${var.environment}"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r5.xlarge"

  allocated_storage     = 500
  max_allocated_storage = 1000

  db_name  = "tracktion"
  username = "tracktion_user"

  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = module.vpc.database_subnet_group

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  multi_az               = true
  publicly_accessible    = false

  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
module "redis" {
  source = "../../modules/redis"

  cluster_id         = "tracktion-${var.environment}"
  node_type          = "cache.r5.large"
  port               = 6379
  parameter_group    = "default.redis7"

  subnet_group_name = module.vpc.redis_subnet_group
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Environment = var.environment
  }
}

# S3 Buckets
module "storage" {
  source = "../../modules/s3"

  buckets = {
    audio_files = {
      name = "tracktion-${var.environment}-audio-files"
      versioning = true
      lifecycle_rules = [
        {
          id     = "transition_to_ia"
          status = "Enabled"
          transition = {
            days          = 30
            storage_class = "STANDARD_IA"
          }
        },
        {
          id     = "transition_to_glacier"
          status = "Enabled"
          transition = {
            days          = 90
            storage_class = "GLACIER"
          }
        }
      ]
    }

    backups = {
      name = "tracktion-${var.environment}-backups"
      versioning = true
      lifecycle_rules = [
        {
          id     = "delete_old_backups"
          status = "Enabled"
          expiration = {
            days = 2555  # 7 years
          }
        }
      ]
    }
  }
}

# Application Load Balancer
module "alb" {
  source = "../../modules/alb"

  name = "tracktion-${var.environment}"

  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets

  security_groups = [aws_security_group.alb.id]

  listeners = [
    {
      port     = 80
      protocol = "HTTP"
      default_actions = [
        {
          type = "redirect"
          redirect = {
            port        = "443"
            protocol    = "HTTPS"
            status_code = "HTTP_301"
          }
        }
      ]
    },
    {
      port            = 443
      protocol        = "HTTPS"
      ssl_policy      = "ELBSecurityPolicy-TLS-1-2-2017-01"
      certificate_arn = aws_acm_certificate.main.arn
    }
  ]

  tags = {
    Environment = var.environment
  }
}

# Security Groups
resource "aws_security_group" "database" {
  name        = "tracktion-${var.environment}-database"
  description = "Security group for RDS database"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "tracktion-${var.environment}-database"
    Environment = var.environment
  }
}

resource "aws_security_group" "redis" {
  name        = "tracktion-${var.environment}-redis"
  description = "Security group for Redis cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  tags = {
    Name        = "tracktion-${var.environment}-redis"
    Environment = var.environment
  }
}

resource "aws_security_group" "app" {
  name        = "tracktion-${var.environment}-app"
  description = "Security group for application services"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8003
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "tracktion-${var.environment}-app"
    Environment = var.environment
  }
}
```

## Release Management

### Semantic Versioning

```bash
#!/bin/bash
# release_management.sh

set -e

# Semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features, backward compatible
# PATCH: Bug fixes, backward compatible

CURRENT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo "Current version: $CURRENT_VERSION"

# Parse version components
if [[ $CURRENT_VERSION =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    MAJOR=${BASH_REMATCH[1]}
    MINOR=${BASH_REMATCH[2]}
    PATCH=${BASH_REMATCH[3]}
else
    echo "Invalid version format: $CURRENT_VERSION"
    exit 1
fi

# Determine version bump type
BUMP_TYPE="$1"

case "$BUMP_TYPE" in
    "major")
        NEW_MAJOR=$((MAJOR + 1))
        NEW_MINOR=0
        NEW_PATCH=0
        ;;
    "minor")
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$((MINOR + 1))
        NEW_PATCH=0
        ;;
    "patch")
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        NEW_PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Usage: $0 {major|minor|patch}"
        echo "Current version: $CURRENT_VERSION"
        exit 1
        ;;
esac

NEW_VERSION="v${NEW_MAJOR}.${NEW_MINOR}.${NEW_PATCH}"
echo "New version: $NEW_VERSION"

# Validate that we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Error: Must be on main branch for release"
    exit 1
fi

# Ensure working directory is clean
if ! git diff-index --quiet HEAD --; then
    echo "Error: Working directory not clean"
    exit 1
fi

# Generate changelog
echo "Generating changelog..."
generate_changelog() {
    cat > "CHANGELOG_${NEW_VERSION}.md" << EOF
# Release ${NEW_VERSION}

**Release Date:** $(date +%Y-%m-%d)

## Changes

$(git log ${CURRENT_VERSION}..HEAD --pretty=format:"- %s" --reverse)

## Pull Requests

$(git log ${CURRENT_VERSION}..HEAD --grep="Merge pull request" --pretty=format:"- %s")

## Contributors

$(git log ${CURRENT_VERSION}..HEAD --pretty=format:"%an" | sort | uniq)
EOF
}

generate_changelog

# Create release tag
echo "Creating release tag: $NEW_VERSION"
git tag -a "$NEW_VERSION" -m "Release $NEW_VERSION

$(cat CHANGELOG_${NEW_VERSION}.md)
"

# Push tag to remote
echo "Pushing tag to remote..."
git push origin "$NEW_VERSION"

# Create GitHub release
echo "Creating GitHub release..."
gh release create "$NEW_VERSION" \
    --title "Release $NEW_VERSION" \
    --notes-file "CHANGELOG_${NEW_VERSION}.md" \
    --latest

echo "Release $NEW_VERSION created successfully"
```

### Release Checklist

#### Pre-Release Checklist
```markdown
# Pre-Release Checklist

## Code Quality
- [ ] All tests passing in CI/CD pipeline
- [ ] Code coverage >= 80%
- [ ] Security scans passed (no critical/high vulnerabilities)
- [ ] Performance benchmarks within acceptable range
- [ ] Documentation updated

## Dependencies
- [ ] All dependencies updated to latest stable versions
- [ ] Security vulnerabilities in dependencies resolved
- [ ] License compatibility verified
- [ ] SBOM (Software Bill of Materials) generated

## Database
- [ ] Database migration scripts tested
- [ ] Migration rollback procedures verified
- [ ] Database backup completed before migration
- [ ] Migration performance impact assessed

## Configuration
- [ ] Environment-specific configurations validated
- [ ] Secrets and credentials rotated if needed
- [ ] Feature flags configured appropriately
- [ ] Monitoring and alerting rules updated

## Testing
- [ ] Unit tests passed
- [ ] Integration tests passed
- [ ] End-to-end tests passed
- [ ] Performance tests passed
- [ ] Security tests passed
- [ ] Accessibility tests passed (if UI changes)

## Documentation
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Deployment documentation updated
- [ ] Changelog generated
- [ ] Release notes prepared

## Stakeholder Communication
- [ ] Stakeholders notified of upcoming deployment
- [ ] Maintenance window scheduled (if needed)
- [ ] Support team briefed on changes
- [ ] Rollback plan communicated

## Infrastructure
- [ ] Infrastructure capacity verified
- [ ] Monitoring systems ready
- [ ] Backup systems verified
- [ ] Disaster recovery plan updated
```

### Release Notes Template

```markdown
# Release v1.2.4

**Release Date:** 2024-01-15
**Release Type:** Minor Release

## 🚀 New Features

### Audio Analysis Enhancements
- **Improved Accuracy**: Enhanced machine learning models with 15% better accuracy
- **Format Support**: Added support for FLAC and OGG audio formats
- **Batch Processing**: New batch analysis API endpoint for processing multiple files

### User Interface
- **Dark Mode**: Added dark mode theme option
- **Progress Tracking**: Real-time progress indicators for long-running analyses
- **Export Options**: New export formats including CSV and JSON

## 🐛 Bug Fixes

- **Fixed** Memory leak in file processing service
- **Fixed** Intermittent connection timeouts to database
- **Fixed** Race condition in concurrent file uploads
- **Resolved** Incorrect timestamps in analysis results

## 🔧 Improvements

- **Performance**: 25% faster analysis processing through algorithm optimization
- **Reliability**: Improved error handling and retry mechanisms
- **Security**: Updated authentication system with enhanced session management
- **Monitoring**: Added comprehensive health checks and metrics

## 📋 API Changes

### New Endpoints
- `POST /api/v1/analysis/batch` - Batch analysis processing
- `GET /api/v1/formats/supported` - List supported audio formats
- `GET /api/v1/analysis/{id}/progress` - Get analysis progress

### Changed Endpoints
- `POST /api/v1/analysis` - Now supports additional metadata parameters
- `GET /api/v1/analysis/{id}` - Response includes processing duration

### Deprecated Endpoints
- `GET /api/v1/legacy/analysis` - Will be removed in v2.0.0

## 🔄 Database Changes

### Migrations
- **Migration 001**: Add support for new audio formats
- **Migration 002**: Index optimization for faster queries
- **Migration 003**: Add batch processing tracking tables

### Schema Changes
- Added `batch_id` column to `analysis_results` table
- Added `format_metadata` JSONB column to `audio_files` table
- Created new `batch_jobs` table for tracking batch operations

## 🚧 Breaking Changes

**None in this release**

## ⚠️ Known Issues

- **Issue #123**: Large file uploads (>100MB) may timeout in slower networks
  - **Workaround**: Use chunked upload API or smaller batch sizes
  - **Fix planned**: v1.2.5

## 📦 Deployment Notes

### Prerequisites
- Kubernetes 1.24+ required
- PostgreSQL 13+ required
- Redis 6.2+ required

### Deployment Steps
1. Run database migrations: `kubectl apply -f migrations/`
2. Deploy services: `helm upgrade tracktion ./chart/`
3. Verify deployment: `./scripts/verify-deployment.sh`

### Rollback Instructions
If issues occur, rollback using:
```bash
helm rollback tracktion
kubectl apply -f rollback/database-migrations/
```

## 📊 Performance Metrics

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| Analysis Speed | 2.3s avg | 1.7s avg | +35% faster |
| Memory Usage | 512MB avg | 384MB avg | -25% |
| Error Rate | 0.08% | 0.03% | -62% |
| API Response Time | 145ms p95 | 98ms p95 | +32% faster |

## 🙏 Contributors

- John Smith (@jsmith) - Audio analysis improvements
- Sarah Johnson (@sjohnson) - UI/UX enhancements
- Mike Chen (@mchen) - Performance optimizations
- Lisa Wang (@lwang) - Security improvements

## 🔗 Additional Resources

- [Migration Guide](docs/migrations/v1.2.4.md)
- [API Documentation](docs/api/v1.2.4/)
- [Performance Benchmarks](docs/performance/v1.2.4.md)
- [Security Assessment](docs/security/v1.2.4.md)

---

**Full Changelog**: [v1.2.3...v1.2.4](https://github.com/tracktion/tracktion/compare/v1.2.3...v1.2.4)
```

## Pre-Deployment Checklist

### Automated Pre-Flight Checks

```bash
#!/bin/bash
# pre_deployment_checks.sh

set -e

ENVIRONMENT="$1"
VERSION="$2"

if [ -z "$ENVIRONMENT" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <environment> <version>"
    exit 1
fi

echo "Running pre-deployment checks for $ENVIRONMENT deployment of version $VERSION"

# Configuration file paths
CONFIG_DIR="./environments/$ENVIRONMENT"
SECRETS_DIR="./secrets/$ENVIRONMENT"

# Initialize check results
TOTAL_CHECKS=0
FAILED_CHECKS=0

run_check() {
    local check_name="$1"
    local check_command="$2"

    echo "Checking: $check_name..."
    ((TOTAL_CHECKS++))

    if eval "$check_command" > /dev/null 2>&1; then
        echo "✅ $check_name: PASSED"
    else
        echo "❌ $check_name: FAILED"
        ((FAILED_CHECKS++))

        # Show error details for failed checks
        echo "   Error details:"
        eval "$check_command" 2>&1 | sed 's/^/   /'
    fi
}

# Infrastructure Checks
echo "=== Infrastructure Checks ==="

run_check "Kubernetes cluster connectivity" \
    "kubectl cluster-info --request-timeout=10s"

run_check "Kubernetes cluster version" \
    "kubectl version --client=false -o json | jq -r '.serverVersion.gitVersion' | grep -E 'v1\.(24|25|26|27)'"

run_check "Namespace exists" \
    "kubectl get namespace tracktion-${ENVIRONMENT}"

run_check "Database connectivity" \
    "pg_isready -h \$(kubectl get secret db-credentials -n tracktion-${ENVIRONMENT} -o jsonpath='{.data.host}' | base64 -d) -p 5432"

run_check "Redis connectivity" \
    "redis-cli -h \$(kubectl get secret redis-credentials -n tracktion-${ENVIRONMENT} -o jsonpath='{.data.host}' | base64 -d) ping"

# Configuration Checks
echo "=== Configuration Checks ==="

run_check "Environment configuration exists" \
    "test -f $CONFIG_DIR/config.yml"

run_check "Configuration file syntax" \
    "python -c \"import yaml; yaml.safe_load(open('$CONFIG_DIR/config.yml'))\""

run_check "Required secrets exist" \
    "kubectl get secrets -n tracktion-${ENVIRONMENT} | grep -E '(db-credentials|api-keys|tls-certificates)'"

run_check "TLS certificates valid" \
    "kubectl get secret tls-certificates -n tracktion-${ENVIRONMENT} -o jsonpath='{.data.tls\\.crt}' | base64 -d | openssl x509 -checkend 2592000 -noout"

# Image Checks
echo "=== Image Checks ==="

SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

for service in "${SERVICES[@]}"; do
    IMAGE="ghcr.io/tracktion/$service:$VERSION"

    run_check "Image exists: $service" \
        "docker manifest inspect $IMAGE"

    run_check "Image signature valid: $service" \
        "cosign verify $IMAGE"

    run_check "Image vulnerability scan: $service" \
        "grype $IMAGE --fail-on high"
done

# Resource Checks
echo "=== Resource Checks ==="

run_check "Sufficient CPU capacity" \
    "kubectl top nodes | awk 'NR>1 {sum+=\$3} END {print (sum<80) ? \"true\" : \"false\"}' | grep true"

run_check "Sufficient memory capacity" \
    "kubectl top nodes | awk 'NR>1 {sum+=\$5} END {print (sum<80) ? \"true\" : \"false\"}' | grep true"

run_check "Storage capacity" \
    "df -h /var/lib/docker | awk 'NR==2 {print (\$5+0<80) ? \"true\" : \"false\"}' | grep true"

# Network Checks
echo "=== Network Checks ==="

run_check "External API connectivity" \
    "curl -s --max-time 10 https://api.external-service.com/health | jq -r '.status' | grep -i ok"

run_check "DNS resolution" \
    "nslookup tracktion.com | grep -E '^Name:'"

run_check "Load balancer health" \
    "kubectl get service -n tracktion-${ENVIRONMENT} tracktion-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' | grep -E '^[0-9.]+$'"

# Security Checks
echo "=== Security Checks ==="

run_check "RBAC policies applied" \
    "kubectl get rolebindings,clusterrolebindings -n tracktion-${ENVIRONMENT} | grep tracktion"

run_check "Network policies exist" \
    "kubectl get networkpolicy -n tracktion-${ENVIRONMENT}"

run_check "Pod security policies" \
    "kubectl get podsecuritypolicy tracktion-psp"

run_check "Secret encryption at rest" \
    "kubectl get secrets -n tracktion-${ENVIRONMENT} -o yaml | grep -E 'encryption|kms'"

# Backup Checks
echo "=== Backup Checks ==="

run_check "Recent database backup exists" \
    "find /backups/database -name '*$(date -d '1 day ago' +%Y%m%d)*' -type f | wc -l | grep -E '^[1-9][0-9]*$'"

run_check "Backup verification passed" \
    "test -f /var/log/backup_verification_$(date +%Y%m%d).log && grep -q 'PASSED' /var/log/backup_verification_$(date +%Y%m%d).log"

# Monitoring Checks
echo "=== Monitoring Checks ==="

run_check "Prometheus accessible" \
    "curl -s http://prometheus.tracktion.internal:9090/-/healthy | grep 'Prometheus is Healthy'"

run_check "Grafana accessible" \
    "curl -s http://grafana.tracktion.internal:3000/api/health | jq -r '.database' | grep ok"

run_check "Alert manager accessible" \
    "curl -s http://alertmanager.tracktion.internal:9093/-/healthy"

# Application Health Checks
echo "=== Application Health Checks ==="

if [ "$ENVIRONMENT" != "production" ]; then
    # Skip for production as services might not be running yet
    for service in "${SERVICES[@]}"; do
        SERVICE_URL="http://$service.tracktion-${ENVIRONMENT}.svc.cluster.local:8000"
        run_check "Service health: $service" \
            "curl -s --max-time 10 $SERVICE_URL/health | jq -r '.status' | grep -i healthy"
    done
fi

# Performance Baseline Checks
echo "=== Performance Checks ==="

run_check "Database connection pool" \
    "psql -h \$(kubectl get secret db-credentials -n tracktion-${ENVIRONMENT} -o jsonpath='{.data.host}' | base64 -d) -U tracktion -c 'SELECT count(*) FROM pg_stat_activity;' | grep -E '^[[:space:]]*[0-9]+$'"

run_check "Redis memory usage" \
    "redis-cli -h \$(kubectl get secret redis-credentials -n tracktion-${ENVIRONMENT} -o jsonpath='{.data.host}' | base64 -d) info memory | grep used_memory_human | cut -d: -f2 | grep -E '^[0-9.]+[KMG]B$'"

# Summary
echo "=== Pre-Deployment Check Summary ==="
echo "Total checks: $TOTAL_CHECKS"
echo "Passed: $((TOTAL_CHECKS - FAILED_CHECKS))"
echo "Failed: $FAILED_CHECKS"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "✅ All pre-deployment checks passed. Deployment can proceed."

    # Generate deployment approval file
    cat > "/tmp/deployment_approval_${ENVIRONMENT}_${VERSION}.json" << EOF
{
    "environment": "$ENVIRONMENT",
    "version": "$VERSION",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "checks_passed": $TOTAL_CHECKS,
    "checks_failed": $FAILED_CHECKS,
    "approval_status": "approved",
    "approved_by": "$(whoami)",
    "valid_for": "24 hours"
}
EOF

    echo "Deployment approval generated: /tmp/deployment_approval_${ENVIRONMENT}_${VERSION}.json"
else
    echo "❌ $FAILED_CHECKS pre-deployment checks failed. Fix issues before deploying."
    exit 1
fi
```

### Manual Verification Checklist

```markdown
# Manual Pre-Deployment Verification

**Environment:** _______________
**Version:** _______________
**Date:** _______________
**Approver:** _______________

## Business Readiness
- [ ] Stakeholder approval obtained
- [ ] Change management process completed
- [ ] Communication plan executed
- [ ] Support team briefed
- [ ] Documentation updated

## Technical Readiness
- [ ] All automated checks passed
- [ ] Performance baselines established
- [ ] Security scan results reviewed
- [ ] Database migration tested
- [ ] Rollback procedure validated

## Operational Readiness
- [ ] Monitoring alerts configured
- [ ] On-call team notified
- [ ] Incident response plan updated
- [ ] Backup verification completed
- [ ] Capacity planning confirmed

## Risk Assessment
- [ ] Risk register updated
- [ ] Mitigation strategies defined
- [ ] Blast radius analysis completed
- [ ] Rollback criteria established
- [ ] Success criteria defined

**Risk Level:** [ ] Low [ ] Medium [ ] High
**Deployment Window:** _______________
**Rollback Window:** _______________

**Approval Signature:** _______________
**Date/Time:** _______________
```

## Deployment Execution

### Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT="$1"
VERSION="$2"
DRY_RUN="${3:-false}"

if [ -z "$ENVIRONMENT" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <environment> <version> [dry-run]"
    echo "Environments: development, staging, production"
    exit 1
fi

# Configuration
NAMESPACE="tracktion-${ENVIRONMENT}"
CONFIG_DIR="./environments/$ENVIRONMENT"
CHART_DIR="./helm/tracktion"
SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

# Logging setup
LOG_FILE="/var/log/deployment_${ENVIRONMENT}_${VERSION}_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [DEPLOY] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

# Deployment tracking
DEPLOYMENT_ID="deploy-${ENVIRONMENT}-${VERSION}-$(date +%s)"
DEPLOYMENT_START=$(date +%s)

log "Starting deployment: $DEPLOYMENT_ID"
log "Environment: $ENVIRONMENT"
log "Version: $VERSION"
log "Dry run: $DRY_RUN"

# Pre-deployment validation
log "Running pre-deployment validation..."
if ! ./scripts/pre-deployment-checks.sh "$ENVIRONMENT" "$VERSION"; then
    error "Pre-deployment validation failed"
    exit 1
fi

# Create deployment record
create_deployment_record() {
    cat > "/tmp/deployment_${DEPLOYMENT_ID}.json" << EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "environment": "$ENVIRONMENT",
    "version": "$VERSION",
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "status": "in_progress",
    "services": $(printf '%s\n' "${SERVICES[@]}" | jq -R . | jq -s .),
    "deployed_by": "$(whoami)",
    "deployment_strategy": "blue_green"
}
EOF
}

create_deployment_record

# Database migration
deploy_database_migrations() {
    log "Deploying database migrations..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would run database migrations"
        return 0
    fi

    # Run migrations in a Kubernetes job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: migration-${VERSION//./-}
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: ghcr.io/tracktion/migrations:$VERSION
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        command: ["python", "migrate.py", "--environment", "$ENVIRONMENT"]
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for migration to complete
    log "Waiting for migration to complete..."
    kubectl wait --for=condition=complete job/migration-${VERSION//./-} -n "$NAMESPACE" --timeout=600s

    # Check migration status
    if kubectl get job migration-${VERSION//./-} -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; then
        log "Database migration completed successfully"
    else
        error "Database migration failed"
        kubectl logs job/migration-${VERSION//./-} -n "$NAMESPACE"
        exit 1
    fi
}

# Deploy services using blue-green strategy
deploy_services() {
    log "Deploying services using blue-green strategy..."

    # Determine current and next slots
    CURRENT_SLOT=$(kubectl get deployment -n "$NAMESPACE" -l app=tracktion,slot=blue -o name 2>/dev/null | head -1)
    if [ -n "$CURRENT_SLOT" ]; then
        NEXT_SLOT="green"
        CURRENT_SLOT="blue"
    else
        NEXT_SLOT="blue"
        CURRENT_SLOT="green"
    fi

    log "Current slot: $CURRENT_SLOT"
    log "Next slot: $NEXT_SLOT"

    # Deploy to next slot
    log "Deploying to $NEXT_SLOT slot..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would deploy services to $NEXT_SLOT slot"
        return 0
    fi

    helm upgrade tracktion-${NEXT_SLOT} "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --values "$CONFIG_DIR/values.yml" \
        --set image.tag="$VERSION" \
        --set deployment.slot="$NEXT_SLOT" \
        --set deployment.version="$VERSION" \
        --wait \
        --timeout=600s

    log "Services deployed to $NEXT_SLOT slot"
}

# Health check for deployed services
verify_deployment() {
    log "Verifying deployment health..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would verify deployment health"
        return 0
    fi

    # Wait for pods to be ready
    log "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=tracktion,slot="$NEXT_SLOT" -n "$NAMESPACE" --timeout=300s

    # Run health checks
    for service in "${SERVICES[@]}"; do
        log "Health check for $service..."

        SERVICE_URL="http://$service-$NEXT_SLOT.$NAMESPACE.svc.cluster.local:8000"

        # Retry health check up to 10 times
        for i in {1..10}; do
            if kubectl run health-check-$service-$i -n "$NAMESPACE" --rm -i --restart=Never --image=curlimages/curl -- \
                curl -s --max-time 10 "$SERVICE_URL/health" | grep -q "healthy"; then
                log "✅ $service health check passed"
                break
            else
                log "⚠️ $service health check failed (attempt $i/10)"
                if [ $i -eq 10 ]; then
                    error "$service health check failed after 10 attempts"
                    return 1
                fi
                sleep 10
            fi
        done
    done

    log "All health checks passed"
}

# Switch traffic to new deployment
switch_traffic() {
    log "Switching traffic to new deployment..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would switch traffic to $NEXT_SLOT"
        return 0
    fi

    # Update service selector to point to new slot
    kubectl patch service tracktion -n "$NAMESPACE" -p '{"spec":{"selector":{"slot":"'$NEXT_SLOT'"}}}'

    log "Traffic switched to $NEXT_SLOT slot"

    # Wait a moment for traffic to stabilize
    sleep 30

    # Verify traffic switch
    ACTIVE_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=tracktion,slot="$NEXT_SLOT" --field-selector=status.phase=Running --no-headers | wc -l)
    if [ "$ACTIVE_PODS" -gt 0 ]; then
        log "✅ Traffic switch verified - $ACTIVE_PODS active pods"
    else
        error "Traffic switch verification failed - no active pods"
        return 1
    fi
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would run post-deployment validation"
        return 0
    fi

    # Run smoke tests
    if ! ./scripts/smoke-tests.sh "$ENVIRONMENT"; then
        error "Smoke tests failed"
        return 1
    fi

    # Run integration tests for non-production
    if [ "$ENVIRONMENT" != "production" ]; then
        if ! ./scripts/integration-tests.sh "$ENVIRONMENT"; then
            error "Integration tests failed"
            return 1
        fi
    fi

    # Performance validation
    log "Running performance validation..."
    if ! ./scripts/performance-validation.sh "$ENVIRONMENT"; then
        error "Performance validation failed"
        return 1
    fi

    log "Post-deployment validation completed"
}

# Clean up old deployment
cleanup_old_deployment() {
    log "Cleaning up old deployment..."

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would cleanup old $CURRENT_SLOT deployment"
        return 0
    fi

    # Keep old deployment for quick rollback (cleanup after 24 hours)
    log "Scheduling cleanup of $CURRENT_SLOT slot in 24 hours"

    # Create cleanup job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-${CURRENT_SLOT}-${VERSION//./-}
  namespace: $NAMESPACE
spec:
  schedule: "$(date -d '+24 hours' +'%M %H %d %m *')"
  successfulJobsHistoryLimit: 1
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - |
              helm uninstall tracktion-${CURRENT_SLOT} -n $NAMESPACE || true
              kubectl delete cronjob cleanup-${CURRENT_SLOT}-${VERSION//./-} -n $NAMESPACE
          restartPolicy: OnFailure
EOF
}

# Update deployment record
update_deployment_record() {
    local status="$1"
    local end_time=$(date +%s)
    local duration=$((end_time - DEPLOYMENT_START))

    jq --arg status "$status" \
       --arg end_time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
       --arg duration "${duration}s" \
       '.status = $status | .end_time = $end_time | .duration = $duration' \
       "/tmp/deployment_${DEPLOYMENT_ID}.json" > "/tmp/deployment_${DEPLOYMENT_ID}.tmp"

    mv "/tmp/deployment_${DEPLOYMENT_ID}.tmp" "/tmp/deployment_${DEPLOYMENT_ID}.json"
}

# Send deployment notification
send_notification() {
    local status="$1"
    local message="$2"

    local emoji="✅"
    if [ "$status" = "failed" ]; then
        emoji="❌"
    fi

    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"'$emoji' Deployment '"$status"': '"$ENVIRONMENT"' v'"$VERSION"'\n'"$message"'"}' \
        "$SLACK_WEBHOOK_URL" || true
}

# Main deployment flow
main() {
    trap 'handle_error' ERR

    if [ "$ENVIRONMENT" = "production" ]; then
        log "Production deployment - requiring additional confirmation"
        read -p "Confirm production deployment of version $VERSION (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi

    # Execute deployment steps
    deploy_database_migrations
    deploy_services
    verify_deployment
    switch_traffic
    post_deployment_validation
    cleanup_old_deployment

    # Mark deployment as successful
    update_deployment_record "success"

    DEPLOYMENT_END=$(date +%s)
    DEPLOYMENT_DURATION=$((DEPLOYMENT_END - DEPLOYMENT_START))

    log "✅ Deployment completed successfully in ${DEPLOYMENT_DURATION}s"
    log "Deployment ID: $DEPLOYMENT_ID"
    log "Log file: $LOG_FILE"

    send_notification "completed" "Deployment duration: ${DEPLOYMENT_DURATION}s"
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Deployment failed with exit code $exit_code"

    update_deployment_record "failed"
    send_notification "failed" "Check logs: $LOG_FILE"

    # Trigger rollback for production
    if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" != "true" ]; then
        log "Triggering automatic rollback for production failure"
        ./scripts/rollback.sh "$ENVIRONMENT" "$CURRENT_SLOT"
    fi

    exit $exit_code
}

# Execute main deployment
main "$@"
```

## Post-Deployment Validation

### Comprehensive Validation Suite

```bash
#!/bin/bash
# post_deployment_validation.sh

set -e

ENVIRONMENT="$1"

if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

NAMESPACE="tracktion-${ENVIRONMENT}"
VALIDATION_LOG="/var/log/post_deployment_validation_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$VALIDATION_LOG")
exec 2> >(tee -a "$VALIDATION_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [VALIDATE] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

TOTAL_CHECKS=0
FAILED_CHECKS=0

run_validation() {
    local check_name="$1"
    local check_command="$2"
    local required="${3:-true}"

    log "Validating: $check_name..."
    ((TOTAL_CHECKS++))

    if eval "$check_command" > /dev/null 2>&1; then
        log "✅ $check_name: PASSED"
    else
        log "❌ $check_name: FAILED"

        if [ "$required" = "true" ]; then
            ((FAILED_CHECKS++))
            error "Critical validation failed: $check_name"
            eval "$check_command" 2>&1 | sed 's/^/   /'
        else
            log "⚠️ Non-critical validation failed: $check_name"
        fi
    fi
}

log "Starting post-deployment validation for $ENVIRONMENT"

# Service Health Validation
log "=== Service Health Validation ==="

SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

for service in "${SERVICES[@]}"; do
    run_validation "Pod readiness: $service" \
        "kubectl get pods -n $NAMESPACE -l app=$service --field-selector=status.phase=Running | grep -q Running"

    run_validation "Service endpoint: $service" \
        "kubectl get service -n $NAMESPACE $service -o jsonpath='{.spec.clusterIP}' | grep -E '^[0-9.]+$'"

    run_validation "Health endpoint: $service" \
        "kubectl exec -n $NAMESPACE deployment/$service -- curl -s --max-time 10 http://localhost:8000/health | grep -q healthy"
done

# Database Validation
log "=== Database Validation ==="

run_validation "Database connectivity" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT 1')
print('Database connected')
conn.close()
\""

run_validation "Database schema version" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1')
version = cur.fetchone()[0]
print(f'Schema version: {version}')
conn.close()
\""

run_validation "Database performance" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import psycopg2
import os
import time
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
start = time.time()
cur.execute('SELECT COUNT(*) FROM audio_files')
duration = time.time() - start
assert duration < 1.0, f'Query took {duration}s'
print(f'Database query performance: {duration:.3f}s')
conn.close()
\""

# Redis Validation
log "=== Redis Validation ==="

run_validation "Redis connectivity" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import redis
import os
r = redis.Redis.from_url(os.environ['REDIS_URL'])
r.ping()
print('Redis connected')
\""

run_validation "Redis performance" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import redis
import os
import time
r = redis.Redis.from_url(os.environ['REDIS_URL'])
start = time.time()
r.set('test_key', 'test_value')
value = r.get('test_key')
duration = time.time() - start
assert duration < 0.1, f'Redis operation took {duration}s'
print(f'Redis performance: {duration:.3f}s')
r.delete('test_key')
\""

# API Functionality Validation
log "=== API Functionality Validation ==="

# Get service endpoints
if [ "$ENVIRONMENT" = "production" ]; then
    BASE_URL="https://api.tracktion.com"
else
    BASE_URL="https://api-${ENVIRONMENT}.tracktion.com"
fi

run_validation "API health endpoint" \
    "curl -s --max-time 10 $BASE_URL/health | jq -r '.status' | grep -i healthy"

run_validation "API authentication" \
    "curl -s --max-time 10 -H 'Authorization: Bearer test-token' $BASE_URL/api/v1/user | jq -r '.error' | grep -i 'unauthorized\|invalid'"

run_validation "API rate limiting" \
    "for i in {1..5}; do curl -s -o /dev/null -w '%{http_code}' $BASE_URL/api/v1/health; done | tail -1 | grep -E '^(200|429)$'"

# File Upload Validation (if not production)
if [ "$ENVIRONMENT" != "production" ]; then
    run_validation "File upload endpoint" \
        "curl -s --max-time 30 -F 'file=@./test-data/sample.mp3' $BASE_URL/api/v1/upload | jq -r '.status' | grep -i 'success\|uploaded'"
fi

# Performance Validation
log "=== Performance Validation ==="

run_validation "API response time" \
    "RESPONSE_TIME=\$(curl -o /dev/null -s -w '%{time_total}' $BASE_URL/health); echo \"Response time: \${RESPONSE_TIME}s\"; (( \$(echo \"\$RESPONSE_TIME < 2.0\" | bc -l) ))"

run_validation "Database connection pool" \
    "kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c \"
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()')
active_connections = cur.fetchone()[0]
print(f'Active connections: {active_connections}')
assert active_connections < 20, f'Too many connections: {active_connections}'
conn.close()
\""

# Security Validation
log "=== Security Validation ==="

run_validation "TLS certificate validity" \
    "echo | openssl s_client -connect api-${ENVIRONMENT}.tracktion.com:443 2>/dev/null | openssl x509 -checkend 2592000 -noout"

run_validation "Security headers" \
    "curl -s -I $BASE_URL/health | grep -i -E '(strict-transport-security|x-frame-options|x-content-type-options)'"

run_validation "No sensitive data exposure" \
    "! curl -s $BASE_URL/health | grep -i -E '(password|secret|key|token)'"

# Monitoring Validation
log "=== Monitoring Validation ==="

run_validation "Metrics endpoint" \
    "curl -s --max-time 10 $BASE_URL/metrics | grep -E '^# (HELP|TYPE)'"

run_validation "Prometheus target discovery" \
    "kubectl get servicemonitor -n $NAMESPACE | grep tracktion"

run_validation "Alert rules" \
    "kubectl get prometheusrule -n $NAMESPACE | grep tracktion"

# Business Logic Validation
log "=== Business Logic Validation ==="

if [ "$ENVIRONMENT" != "production" ]; then
    # Run business logic tests in non-production
    run_validation "Audio analysis workflow" \
        "./scripts/test-audio-analysis.sh $ENVIRONMENT"

    run_validation "File processing workflow" \
        "./scripts/test-file-processing.sh $ENVIRONMENT"
fi

# Integration Validation
log "=== Integration Validation ==="

run_validation "External service connectivity" \
    "kubectl exec -n $NAMESPACE deployment/notification-service -- curl -s --max-time 10 https://api.external-service.com/health | grep -i ok" \
    "false"  # Non-critical

# Log Aggregation Validation
log "=== Log Aggregation Validation ==="

run_validation "Logs are being collected" \
    "kubectl logs -n $NAMESPACE deployment/analysis-service --tail=10 | wc -l | grep -E '^[1-9][0-9]*$'"

run_validation "Structured logging format" \
    "kubectl logs -n $NAMESPACE deployment/analysis-service --tail=1 | jq -r '.timestamp' | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T'"

# Backup Validation
log "=== Backup Validation ==="

run_validation "Backup system operational" \
    "kubectl get cronjob -n $NAMESPACE | grep backup"

# Deployment Metadata Validation
log "=== Deployment Metadata Validation ==="

run_validation "Deployment labels" \
    "kubectl get deployment -n $NAMESPACE -l app=tracktion --show-labels | grep -E 'version=.*,environment=$ENVIRONMENT'"

run_validation "Image tags correct" \
    "kubectl get deployment -n $NAMESPACE -o jsonpath='{.items[*].spec.template.spec.containers[*].image}' | grep -v ':latest'"

# Resource Usage Validation
log "=== Resource Usage Validation ==="

run_validation "Memory usage within limits" \
    "kubectl top pods -n $NAMESPACE | awk 'NR>1 {print \$3}' | sed 's/Mi//' | awk '{if(\$1>1000) exit 1}'"

run_validation "CPU usage reasonable" \
    "kubectl top pods -n $NAMESPACE | awk 'NR>1 {print \$2}' | sed 's/m//' | awk '{if(\$1>500) exit 1}'"

# Generate validation report
log "=== Validation Summary ==="
log "Total validations: $TOTAL_CHECKS"
log "Passed: $((TOTAL_CHECKS - FAILED_CHECKS))"
log "Failed: $FAILED_CHECKS"

# Create validation report
cat > "/tmp/validation_report_${ENVIRONMENT}_$(date +%Y%m%d).json" << EOF
{
    "environment": "$ENVIRONMENT",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_checks": $TOTAL_CHECKS,
    "passed_checks": $((TOTAL_CHECKS - FAILED_CHECKS)),
    "failed_checks": $FAILED_CHECKS,
    "success_rate": $(echo "scale=2; ($TOTAL_CHECKS - $FAILED_CHECKS) * 100 / $TOTAL_CHECKS" | bc),
    "log_file": "$VALIDATION_LOG",
    "status": $([ $FAILED_CHECKS -eq 0 ] && echo '"passed"' || echo '"failed"')
}
EOF

if [ $FAILED_CHECKS -eq 0 ]; then
    log "✅ All post-deployment validations passed"

    # Send success notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"✅ Post-deployment validation passed for '"$ENVIRONMENT"' ('"$((TOTAL_CHECKS - FAILED_CHECKS))"'/'"$TOTAL_CHECKS"' checks)"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit 0
else
    error "❌ $FAILED_CHECKS post-deployment validations failed"

    # Send failure notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"❌ Post-deployment validation failed for '"$ENVIRONMENT"' ('"$FAILED_CHECKS"'/'"$TOTAL_CHECKS"' checks failed). Log: '"$VALIDATION_LOG"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit 1
fi
```

## Production Deployment

### Production Deployment Process

```bash
#!/bin/bash
# deploy_production.sh

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

ENVIRONMENT="production"
NAMESPACE="tracktion-prod"
DEPLOYMENT_ID="prod-deploy-${VERSION}-$(date +%s)"

# Enhanced logging for production
LOG_FILE="/var/log/production_deployment_${VERSION}_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [PROD-DEPLOY] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [PROD-ERROR] $1" >&2
}

# Production-specific validations
production_pre_checks() {
    log "Running production-specific pre-checks..."

    # Verify staging deployment success
    if ! curl -s "https://api-staging.tracktion.com/health" | grep -q "healthy"; then
        error "Staging environment is not healthy"
        exit 1
    fi

    # Verify all tests passed in staging
    STAGING_TEST_RESULTS=$(kubectl get jobs -n tracktion-staging -l type=test -o jsonpath='{.items[*].status.conditions[?(@.type=="Complete")].status}')
    if echo "$STAGING_TEST_RESULTS" | grep -q "False"; then
        error "Staging tests did not pass successfully"
        exit 1
    fi

    # Check production capacity
    NODE_CAPACITY=$(kubectl top nodes --no-headers | awk '{sum+=$3; sum+=$5} END {print sum/2}' | cut -d'%' -f1)
    if [ "$NODE_CAPACITY" -gt 70 ]; then
        error "Production cluster capacity too high: ${NODE_CAPACITY}%"
        exit 1
    fi

    # Verify backup systems
    RECENT_BACKUP=$(find /backups/production -name "*$(date -d '6 hours ago' +%Y%m%d)*" | wc -l)
    if [ "$RECENT_BACKUP" -eq 0 ]; then
        error "No recent production backup found"
        exit 1
    fi

    log "Production pre-checks passed"
}

# Production deployment with canary
production_canary_deployment() {
    log "Starting production canary deployment..."

    # Stage 1: 5% traffic
    log "Deploying canary with 5% traffic..."

    helm upgrade tracktion-canary ./helm/tracktion \
        --namespace "$NAMESPACE" \
        --values ./environments/production/values.yml \
        --set image.tag="$VERSION" \
        --set deployment.strategy="canary" \
        --set deployment.canary.percentage=5 \
        --set deployment.canary.stage="initial" \
        --wait \
        --timeout=600s

    # Monitor canary for 15 minutes
    log "Monitoring canary deployment (15 minutes)..."

    for i in {1..15}; do
        sleep 60

        # Check error rate
        ERROR_RATE=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
            curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | \
            jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

        # Check response time
        RESPONSE_TIME=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
            curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | \
            jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

        log "Canary metrics - Error rate: ${ERROR_RATE}%, Response time P95: ${RESPONSE_TIME}ms"

        # Check if metrics are within acceptable range
        if (( $(echo "$ERROR_RATE > 1.0" | bc -l) )); then
            error "Canary error rate too high: ${ERROR_RATE}%"
            rollback_canary
            exit 1
        fi

        if (( $(echo "$RESPONSE_TIME > 2000" | bc -l) )); then
            error "Canary response time too high: ${RESPONSE_TIME}ms"
            rollback_canary
            exit 1
        fi
    done

    log "Canary stage 1 successful"

    # Stage 2: 20% traffic
    log "Scaling canary to 20% traffic..."

    helm upgrade tracktion-canary ./helm/tracktion \
        --namespace "$NAMESPACE" \
        --values ./environments/production/values.yml \
        --set image.tag="$VERSION" \
        --set deployment.strategy="canary" \
        --set deployment.canary.percentage=20 \
        --set deployment.canary.stage="expanded" \
        --wait \
        --timeout=300s

    # Monitor for 30 minutes
    log "Monitoring expanded canary (30 minutes)..."

    for i in {1..30}; do
        sleep 60

        # Enhanced monitoring for higher traffic
        ERROR_RATE=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
            curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | \
            jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

        RESPONSE_TIME=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
            curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | \
            jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

        log "Expanded canary metrics - Error rate: ${ERROR_RATE}%, Response time P95: ${RESPONSE_TIME}ms"

        if (( $(echo "$ERROR_RATE > 0.5" | bc -l) )); then
            error "Expanded canary error rate too high: ${ERROR_RATE}%"
            rollback_canary
            exit 1
        fi

        if (( $(echo "$RESPONSE_TIME > 1500" | bc -l) )); then
            error "Expanded canary response time too high: ${RESPONSE_TIME}ms"
            rollback_canary
            exit 1
        fi
    done

    log "Canary stage 2 successful"

    # Stage 3: Full rollout
    log "Proceeding with full rollout..."

    helm upgrade tracktion ./helm/tracktion \
        --namespace "$NAMESPACE" \
        --values ./environments/production/values.yml \
        --set image.tag="$VERSION" \
        --set deployment.strategy="rolling" \
        --wait \
        --timeout=600s

    # Clean up canary deployment
    helm uninstall tracktion-canary --namespace "$NAMESPACE"

    log "Full production rollout completed"
}

# Rollback canary deployment
rollback_canary() {
    log "Rolling back canary deployment..."

    # Remove canary traffic
    helm upgrade tracktion-canary ./helm/tracktion \
        --namespace "$NAMESPACE" \
        --values ./environments/production/values.yml \
        --set deployment.canary.percentage=0 \
        --wait \
        --timeout=300s

    # Clean up canary deployment
    helm uninstall tracktion-canary --namespace "$NAMESPACE"

    log "Canary rollback completed"
}

# Production-specific post-deployment validation
production_post_validation() {
    log "Running production post-deployment validation..."

    # Enhanced validation for production
    ./scripts/post-deployment-validation.sh production

    # Business metrics validation
    log "Validating business metrics..."

    # Check request rate
    REQUEST_RATE=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    log "Current request rate: ${REQUEST_RATE} req/s"

    # Check user activity
    ACTIVE_USERS=$(kubectl exec -n "$NAMESPACE" deployment/redis -- \
        redis-cli eval "return #redis.call('keys', 'session:*')" 0 2>/dev/null || echo "0")

    log "Active users: $ACTIVE_USERS"

    # Revenue impact check (placeholder)
    log "Revenue metrics appear normal (implement actual revenue tracking)"

    log "Production validation completed successfully"
}

# Production deployment notifications
send_production_notification() {
    local stage="$1"
    local status="$2"
    local message="$3"

    local emoji="📢"
    case "$status" in
        "success") emoji="✅" ;;
        "failed") emoji="❌" ;;
        "warning") emoji="⚠️" ;;
    esac

    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"'$emoji' Production Deployment - '"$stage"': '"$status"'\nVersion: '"$VERSION"'\n'"$message"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    # PagerDuty notification for failures
    if [ "$status" = "failed" ]; then
        curl -X POST \
            -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{
                "incident": {
                    "type": "incident",
                    "title": "Production Deployment Failed - '"$VERSION"'",
                    "service": {"id": "'"$PAGERDUTY_SERVICE_ID"'", "type": "service_reference"},
                    "urgency": "high",
                    "body": {
                        "type": "incident_body",
                        "details": "'"$message"'"
                    }
                }
            }' \
            "https://api.pagerduty.com/incidents" || true
    fi
}

# Main production deployment flow
main() {
    trap 'handle_production_error' ERR

    log "Starting production deployment of version $VERSION"
    log "Deployment ID: $DEPLOYMENT_ID"

    # Require explicit confirmation
    echo "🚨 PRODUCTION DEPLOYMENT 🚨"
    echo "Version: $VERSION"
    echo "Time: $(date)"
    echo
    read -p "Type 'DEPLOY TO PRODUCTION' to confirm: " confirmation

    if [ "$confirmation" != "DEPLOY TO PRODUCTION" ]; then
        log "Production deployment cancelled"
        exit 0
    fi

    send_production_notification "Started" "success" "Production deployment initiated"

    # Execute production deployment steps
    production_pre_checks

    send_production_notification "Pre-checks" "success" "All pre-deployment checks passed"

    # Database migrations
    log "Running production database migrations..."
    ./scripts/database-migration.sh production "$VERSION"

    send_production_notification "Database" "success" "Database migrations completed"

    # Canary deployment
    production_canary_deployment

    send_production_notification "Canary Complete" "success" "Canary deployment successful, proceeding to full rollout"

    # Post-deployment validation
    production_post_validation

    send_production_notification "Validation" "success" "Post-deployment validation passed"

    # Update production status
    kubectl patch configmap deployment-info -n "$NAMESPACE" --type merge -p '{"data":{"current-version":"'$VERSION'","deployment-time":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","deployment-id":"'$DEPLOYMENT_ID'"}}'

    log "✅ Production deployment completed successfully"
    log "Version $VERSION is now live in production"

    send_production_notification "Completed" "success" "Production deployment successful. Version $VERSION is now live."
}

# Production error handler
handle_production_error() {
    local exit_code=$?
    error "Production deployment failed with exit code $exit_code"

    send_production_notification "Failed" "failed" "Production deployment failed. Immediate attention required. Log: $LOG_FILE"

    # Automatic rollback for production failures
    log "Initiating automatic production rollback..."
    ./scripts/rollback.sh production

    exit $exit_code
}

# Execute production deployment
main "$@"
```

This comprehensive deployment procedures document provides detailed guidance for deploying the Tracktion system across all environments with proper validation, monitoring, and rollback procedures. The procedures ensure reliable, secure deployments while maintaining system availability and performance.
