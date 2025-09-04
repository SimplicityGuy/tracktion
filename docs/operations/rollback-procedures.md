# Rollback Procedures

## Table of Contents

1. [Overview](#overview)
2. [Rollback Strategy](#rollback-strategy)
3. [Rollback Types](#rollback-types)
4. [Pre-Rollback Assessment](#pre-rollback-assessment)
5. [Service Rollback](#service-rollback)
6. [Database Rollback](#database-rollback)
7. [Configuration Rollback](#configuration-rollback)
8. [Emergency Rollback](#emergency-rollback)
9. [Rollback Validation](#rollback-validation)
10. [Communication During Rollback](#communication-during-rollback)
11. [Post-Rollback Activities](#post-rollback-activities)
12. [Rollback Automation](#rollback-automation)
13. [Testing Rollback Procedures](#testing-rollback-procedures)

## Overview

This document provides comprehensive rollback procedures for the Tracktion audio analysis platform, ensuring rapid recovery from deployment issues, bugs, or system failures. These procedures are designed to minimize downtime and restore service functionality as quickly as possible.

### Rollback Principles

- **Speed Over Perfection**: Fast rollback to restore service, detailed analysis later
- **Data Preservation**: Protect data integrity during rollback operations
- **Communication First**: Clear communication throughout rollback process
- **Documentation**: Record all rollback actions for post-incident analysis
- **Validation**: Verify system functionality after rollback completion

### When to Rollback

#### Immediate Rollback Triggers
- **Service Unavailability**: Critical services completely down for >5 minutes
- **High Error Rates**: Error rate >5% sustained for >5 minutes
- **Performance Degradation**: Response time >3x baseline for >10 minutes
- **Data Corruption**: Any indication of data loss or corruption
- **Security Breach**: Evidence of security compromise

#### Planned Rollback Triggers
- **Feature Issues**: New features causing user experience problems
- **Integration Failures**: Third-party integrations broken
- **Resource Exhaustion**: System resources consistently over capacity
- **Business Impact**: Negative business metrics or user complaints

### Rollback Objectives

- **Recovery Time Objective (RTO)**: Restore service within 15 minutes
- **Recovery Point Objective (RPO)**: Minimize data loss to <5 minutes
- **Communication Time**: Notify stakeholders within 2 minutes
- **Validation Time**: Confirm rollback success within 5 minutes

## Rollback Strategy

### Deployment Architecture for Rollbacks

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                           │
│                 (Traffic Routing)                          │
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
│ Current │              │Previous │
│Version  │              │Version  │
│(Active) │     ROLLBACK │(Standby)│
│ v1.2.4  │    ──────►   │ v1.2.3  │
│         │              │         │
└─────────┘              └─────────┘
```

### Blue-Green Rollback Strategy

The Tracktion system maintains two identical production environments:

1. **Active Environment**: Currently serving production traffic
2. **Standby Environment**: Previous version kept running for instant rollback

#### Rollback Process
1. **Traffic Switch**: Redirect traffic from active to standby (1-2 minutes)
2. **Validation**: Verify standby environment health (2-3 minutes)
3. **Cleanup**: Scale down failed deployment (5-10 minutes)
4. **Analysis**: Investigate failure cause (post-rollback)

### Rolling Rollback Strategy

For services supporting rolling updates:

```yaml
# Rolling rollback configuration
rolling_rollback:
  max_unavailable: 0%
  max_surge: 25%
  strategy:
    - rollback_batch_size: 1
    - health_check_grace_period: 30s
    - health_check_timeout: 60s
    - abort_on_failure: true
```

## Rollback Types

### Type 1: Traffic Rollback (Fastest)

**Use Case**: Application issues, performance problems
**Time**: 1-2 minutes
**Data Loss**: None

```bash
# Immediate traffic switch
kubectl patch service tracktion -n production \
    --type='json' \
    -p='[{"op": "replace", "path": "/spec/selector/version", "value": "previous"}]'
```

### Type 2: Service Rollback (Fast)

**Use Case**: Service-specific issues, container problems
**Time**: 5-10 minutes
**Data Loss**: Minimal

```bash
# Rollback specific service
helm rollback tracktion-analysis-service -n production
kubectl rollout status deployment/analysis-service -n production
```

### Type 3: Database Rollback (Complex)

**Use Case**: Schema changes, data corruption
**Time**: 15-60 minutes
**Data Loss**: Possible

```bash
# Database point-in-time recovery
./scripts/database-rollback.sh production "2024-01-15 14:30:00"
```

### Type 4: Full System Rollback (Comprehensive)

**Use Case**: Multiple service failures, infrastructure issues
**Time**: 10-30 minutes
**Data Loss**: Possible

```bash
# Complete system rollback
./scripts/full-system-rollback.sh production v1.2.3
```

## Pre-Rollback Assessment

### Rollback Decision Matrix

```bash
#!/bin/bash
# rollback_assessment.sh

SEVERITY="$1"
DURATION="$2"
AFFECTED_SERVICES="$3"

assess_rollback_need() {
    local severity="$1"
    local duration="$2"
    local affected_services="$3"

    echo "=== Rollback Assessment ==="
    echo "Severity: $severity"
    echo "Duration: $duration minutes"
    echo "Affected Services: $affected_services"

    # Critical severity - immediate rollback
    if [ "$severity" = "critical" ]; then
        echo "DECISION: IMMEDIATE ROLLBACK REQUIRED"
        echo "Reason: Critical severity incident"
        return 0
    fi

    # High severity with extended duration
    if [ "$severity" = "high" ] && [ "$duration" -gt 10 ]; then
        echo "DECISION: ROLLBACK RECOMMENDED"
        echo "Reason: High severity incident lasting >10 minutes"
        return 0
    fi

    # Multiple services affected
    SERVICE_COUNT=$(echo "$affected_services" | tr ',' '\n' | wc -l)
    if [ "$SERVICE_COUNT" -gt 2 ]; then
        echo "DECISION: ROLLBACK RECOMMENDED"
        echo "Reason: Multiple services affected ($SERVICE_COUNT services)"
        return 0
    fi

    # Medium severity with long duration
    if [ "$severity" = "medium" ] && [ "$duration" -gt 30 ]; then
        echo "DECISION: CONSIDER ROLLBACK"
        echo "Reason: Medium severity incident lasting >30 minutes"
        return 1
    fi

    echo "DECISION: MONITOR AND FIX"
    echo "Reason: Incident does not meet rollback criteria"
    return 2
}

# Current system health check
check_system_health() {
    echo "=== Current System Health ==="

    # Service health
    HEALTHY_SERVICES=0
    TOTAL_SERVICES=0

    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        ((TOTAL_SERVICES++))
        if kubectl get deployment "$service" -n production -o jsonpath='{.status.readyReplicas}' | grep -q '[1-9]'; then
            echo "✅ $service: Healthy"
            ((HEALTHY_SERVICES++))
        else
            echo "❌ $service: Unhealthy"
        fi
    done

    HEALTH_PERCENTAGE=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
    echo "Overall Health: $HEALTH_PERCENTAGE% ($HEALTHY_SERVICES/$TOTAL_SERVICES services)"

    # Error rate check
    ERROR_RATE=$(kubectl exec -n production deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    echo "Current Error Rate: ${ERROR_RATE}%"

    # Response time check
    RESPONSE_TIME=$(kubectl exec -n production deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    echo "Current Response Time P95: ${RESPONSE_TIME}ms"

    # Recommendation based on metrics
    if [ "$HEALTH_PERCENTAGE" -lt 50 ] || (( $(echo "$ERROR_RATE > 5.0" | bc -l) )); then
        echo "RECOMMENDATION: IMMEDIATE ROLLBACK"
    elif [ "$HEALTH_PERCENTAGE" -lt 75 ] || (( $(echo "$ERROR_RATE > 2.0" | bc -l) )); then
        echo "RECOMMENDATION: CONSIDER ROLLBACK"
    else
        echo "RECOMMENDATION: MONITOR CLOSELY"
    fi
}

# Rollback readiness check
check_rollback_readiness() {
    echo "=== Rollback Readiness Check ==="

    # Check if previous version is available
    PREVIOUS_VERSION=$(kubectl get deployment -n production -l app=tracktion-standby -o jsonpath='{.items[0].metadata.labels.version}' 2>/dev/null)

    if [ -n "$PREVIOUS_VERSION" ]; then
        echo "✅ Previous version available: $PREVIOUS_VERSION"
    else
        echo "❌ No previous version available for rollback"
        return 1
    fi

    # Check standby environment health
    STANDBY_PODS=$(kubectl get pods -n production -l app=tracktion-standby,status=ready --no-headers | wc -l)
    if [ "$STANDBY_PODS" -gt 0 ]; then
        echo "✅ Standby environment ready: $STANDBY_PODS pods"
    else
        echo "❌ Standby environment not ready"
        return 1
    fi

    # Check database rollback capability
    if [ -f "/backups/database/latest_backup.sql" ]; then
        BACKUP_AGE=$(find /backups/database -name "latest_backup.sql" -mtime -1 | wc -l)
        if [ "$BACKUP_AGE" -gt 0 ]; then
            echo "✅ Recent database backup available"
        else
            echo "⚠️ Database backup is older than 24 hours"
        fi
    else
        echo "❌ No database backup available"
    fi

    # Check rollback scripts
    ROLLBACK_SCRIPTS=("rollback.sh" "database-rollback.sh" "service-rollback.sh")
    for script in "${ROLLBACK_SCRIPTS[@]}"; do
        if [ -x "./scripts/$script" ]; then
            echo "✅ Rollback script available: $script"
        else
            echo "❌ Rollback script missing: $script"
        fi
    done

    echo "Rollback readiness assessment completed"
}

# Main assessment
main() {
    if [ $# -ne 3 ]; then
        echo "Usage: $0 <severity> <duration_minutes> <affected_services>"
        echo "Severity: critical, high, medium, low"
        echo "Example: $0 high 15 'analysis-service,tracklist-service'"
        exit 1
    fi

    check_system_health
    echo
    check_rollback_readiness
    echo
    assess_rollback_need "$1" "$2" "$3"
}

main "$@"
```

### Rollback Impact Assessment

```bash
#!/bin/bash
# rollback_impact_assessment.sh

assess_rollback_impact() {
    echo "=== Rollback Impact Assessment ==="

    # Feature impact
    echo "Features that will be reverted:"
    CURRENT_VERSION=$(kubectl get deployment -n production -l app=tracktion -o jsonpath='{.items[0].metadata.labels.version}')
    PREVIOUS_VERSION=$(kubectl get deployment -n production -l app=tracktion-standby -o jsonpath='{.items[0].metadata.labels.version}')

    echo "Current: $CURRENT_VERSION → Previous: $PREVIOUS_VERSION"

    # Check changelog for features that will be lost
    if [ -f "CHANGELOG.md" ]; then
        echo "Features being rolled back:"
        sed -n "/## \[${CURRENT_VERSION}\]/,/## \[${PREVIOUS_VERSION}\]/p" CHANGELOG.md | grep "^-" | head -10
    fi

    # Data impact
    echo
    echo "Data Impact Assessment:"

    # Check for schema changes
    SCHEMA_CHANGES=$(find ./migrations -name "*" -newer "/tmp/last_deployment_${PREVIOUS_VERSION}" 2>/dev/null | wc -l)
    if [ "$SCHEMA_CHANGES" -gt 0 ]; then
        echo "⚠️ Database schema changes detected: $SCHEMA_CHANGES migrations"
        echo "   Database rollback may be required"
    else
        echo "✅ No database schema changes - safe rollback"
    fi

    # Check for data format changes
    echo "Checking for data format changes..."
    # This would check application logs or configuration for data format changes

    # User impact
    echo
    echo "User Impact Assessment:"
    ACTIVE_USERS=$(kubectl exec -n production deployment/redis -- redis-cli eval "return #redis.call('keys', 'session:*')" 0 2>/dev/null || echo "unknown")
    echo "Active users: $ACTIVE_USERS"
    echo "Users will experience:"
    echo "- Brief service interruption (1-2 minutes)"
    echo "- Loss of recent features from $CURRENT_VERSION"
    echo "- Possible need to re-authenticate"

    # Business impact
    echo
    echo "Business Impact Assessment:"
    echo "Revenue impact: Estimated $X per minute of downtime"
    echo "SLA impact: May affect monthly uptime SLA"
    echo "Customer impact: Support tickets may increase"
}

assess_rollback_impact
```

## Service Rollback

### Individual Service Rollback

```bash
#!/bin/bash
# service_rollback.sh

set -e

SERVICE="$1"
ENVIRONMENT="${2:-production}"
TARGET_VERSION="$3"

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service> [environment] [target_version]"
    echo "Services: analysis-service, file-watcher, tracklist-service, notification-service"
    exit 1
fi

NAMESPACE="tracktion-${ENVIRONMENT}"
ROLLBACK_LOG="/var/log/service_rollback_${SERVICE}_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$ROLLBACK_LOG")
exec 2> >(tee -a "$ROLLBACK_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ROLLBACK] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

log "Starting rollback for service: $SERVICE in $ENVIRONMENT"

# Get current deployment info
get_deployment_info() {
    CURRENT_VERSION=$(kubectl get deployment "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown")
    CURRENT_REPLICAS=$(kubectl get deployment "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    READY_REPLICAS=$(kubectl get deployment "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

    log "Current deployment info:"
    log "  Version: $CURRENT_VERSION"
    log "  Replicas: $READY_REPLICAS/$CURRENT_REPLICAS"
}

# Determine target version for rollback
determine_target_version() {
    if [ -n "$TARGET_VERSION" ]; then
        log "Using specified target version: $TARGET_VERSION"
        return
    fi

    # Get previous version from Helm history
    TARGET_VERSION=$(helm history tracktion -n "$NAMESPACE" --max 5 -o json | \
        jq -r '.[1].app_version' 2>/dev/null || echo "")

    if [ -z "$TARGET_VERSION" ]; then
        # Fallback to looking for standby deployment
        TARGET_VERSION=$(kubectl get deployment "${SERVICE}-standby" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "")
    fi

    if [ -z "$TARGET_VERSION" ]; then
        error "Cannot determine target version for rollback"
        exit 1
    fi

    log "Determined target version: $TARGET_VERSION"
}

# Pre-rollback health check
pre_rollback_check() {
    log "Running pre-rollback health check..."

    # Check if service exists
    if ! kubectl get deployment "$SERVICE" -n "$NAMESPACE" >/dev/null 2>&1; then
        error "Service deployment not found: $SERVICE"
        exit 1
    fi

    # Check current health
    CURRENT_HEALTH=$(kubectl get deployment "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
    log "Current service health: $CURRENT_HEALTH"

    # Check if rollback target is available
    if ! docker manifest inspect "ghcr.io/tracktion/${SERVICE}:${TARGET_VERSION}" >/dev/null 2>&1; then
        error "Target image not available: ghcr.io/tracktion/${SERVICE}:${TARGET_VERSION}"
        exit 1
    fi

    log "Pre-rollback checks passed"
}

# Execute service rollback
execute_rollback() {
    log "Executing service rollback..."

    # Method 1: Helm rollback (preferred)
    if helm list -n "$NAMESPACE" | grep -q "tracktion"; then
        log "Using Helm rollback..."

        # Get previous revision
        PREVIOUS_REVISION=$(helm history tracktion -n "$NAMESPACE" --max 2 -o json | jq -r '.[1].revision')

        if [ "$PREVIOUS_REVISION" != "null" ] && [ -n "$PREVIOUS_REVISION" ]; then
            helm rollback tracktion "$PREVIOUS_REVISION" -n "$NAMESPACE" --wait --timeout=300s
            log "Helm rollback completed to revision $PREVIOUS_REVISION"
        else
            error "Cannot determine previous Helm revision"
            exit 1
        fi
    else
        # Method 2: Direct Kubernetes rollback
        log "Using Kubernetes rollback..."
        kubectl rollout undo deployment/"$SERVICE" -n "$NAMESPACE"
        kubectl rollout status deployment/"$SERVICE" -n "$NAMESPACE" --timeout=300s
        log "Kubernetes rollback completed"
    fi
}

# Verify rollback success
verify_rollback() {
    log "Verifying rollback success..."

    # Wait for deployment to be ready
    kubectl wait --for=condition=available deployment/"$SERVICE" -n "$NAMESPACE" --timeout=300s

    # Check new version
    NEW_VERSION=$(kubectl get deployment "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}')
    log "Service rolled back to version: $NEW_VERSION"

    # Health check
    for i in {1..10}; do
        if kubectl exec -n "$NAMESPACE" deployment/"$SERVICE" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log "✅ Service health check passed"
            break
        else
            log "⚠️ Health check failed (attempt $i/10)"
            if [ $i -eq 10 ]; then
                error "Service health check failed after rollback"
                return 1
            fi
            sleep 10
        fi
    done

    # Performance check
    log "Running performance validation..."
    RESPONSE_TIME=$(kubectl exec -n "$NAMESPACE" deployment/"$SERVICE" -- curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/health)
    log "Response time: ${RESPONSE_TIME}s"

    if (( $(echo "$RESPONSE_TIME > 3.0" | bc -l) )); then
        log "⚠️ Response time higher than expected: ${RESPONSE_TIME}s"
    else
        log "✅ Performance validation passed"
    fi

    log "Rollback verification completed successfully"
}

# Update monitoring and alerts
update_monitoring() {
    log "Updating monitoring configuration..."

    # Update deployment labels
    kubectl patch deployment "$SERVICE" -n "$NAMESPACE" --type='json' \
        -p='[{"op": "add", "path": "/metadata/labels/rollback-timestamp", "value": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}]'

    # Send metrics to monitoring system
    curl -X POST http://prometheus-pushgateway:9091/metrics/job/rollback \
        --data-binary "service_rollback_total{service=\"$SERVICE\",environment=\"$ENVIRONMENT\"} 1" 2>/dev/null || true

    log "Monitoring configuration updated"
}

# Send notifications
send_notifications() {
    local status="$1"
    local message="$2"

    local emoji="✅"
    if [ "$status" = "failed" ]; then
        emoji="❌"
    fi

    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"'$emoji' Service Rollback '"$status"': '"$SERVICE"' in '"$ENVIRONMENT"'\nVersion: '"$CURRENT_VERSION"' → '"$TARGET_VERSION"'\n'"$message"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    # PagerDuty for failures
    if [ "$status" = "failed" ]; then
        curl -X POST \
            -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{
                "incident": {
                    "type": "incident",
                    "title": "Service Rollback Failed: '"$SERVICE"'",
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

# Main rollback flow
main() {
    trap 'handle_error' ERR

    get_deployment_info
    determine_target_version
    pre_rollback_check

    send_notifications "started" "Service rollback initiated"

    execute_rollback
    verify_rollback
    update_monitoring

    log "✅ Service rollback completed successfully"
    log "Service: $SERVICE"
    log "Environment: $ENVIRONMENT"
    log "Version: $CURRENT_VERSION → $TARGET_VERSION"
    log "Log file: $ROLLBACK_LOG"

    send_notifications "completed" "Service rollback successful"
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Service rollback failed with exit code $exit_code"

    send_notifications "failed" "Service rollback failed. Check logs: $ROLLBACK_LOG"

    exit $exit_code
}

# Execute rollback
main "$@"
```

### Complete Application Rollback

```bash
#!/bin/bash
# full_application_rollback.sh

set -e

ENVIRONMENT="${1:-production}"
TARGET_VERSION="$2"

NAMESPACE="tracktion-${ENVIRONMENT}"
SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")
ROLLBACK_LOG="/var/log/full_rollback_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$ROLLBACK_LOG")
exec 2> >(tee -a "$ROLLBACK_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [FULL-ROLLBACK] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

log "Starting full application rollback for $ENVIRONMENT"

# Pre-rollback system snapshot
create_system_snapshot() {
    log "Creating system snapshot before rollback..."

    SNAPSHOT_FILE="/tmp/system_snapshot_$(date +%s).json"

    cat > "$SNAPSHOT_FILE" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "services": {
$(for service in "${SERVICES[@]}"; do
    VERSION=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown")
    REPLICAS=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    echo "        \"$service\": {\"version\": \"$VERSION\", \"replicas\": $REPLICAS},"
done | sed '$ s/,$//')
    },
    "database_version": "$(kubectl exec -n $NAMESPACE deployment/analysis-service -- python -c 'import psycopg2; import os; conn = psycopg2.connect(os.environ[\"DATABASE_URL\"]); cur = conn.cursor(); cur.execute(\"SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1\"); print(cur.fetchone()[0]); conn.close()' 2>/dev/null || echo "unknown")"
}
EOF

    log "System snapshot saved: $SNAPSHOT_FILE"
}

# Execute blue-green rollback
execute_blue_green_rollback() {
    log "Executing blue-green rollback..."

    # Determine current and standby slots
    CURRENT_SLOT=$(kubectl get service tracktion -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}' 2>/dev/null || echo "blue")
    STANDBY_SLOT=$([ "$CURRENT_SLOT" = "blue" ] && echo "green" || echo "blue")

    log "Current slot: $CURRENT_SLOT"
    log "Standby slot: $STANDBY_SLOT"

    # Verify standby environment
    STANDBY_PODS=$(kubectl get pods -n "$NAMESPACE" -l slot="$STANDBY_SLOT" --field-selector=status.phase=Running --no-headers | wc -l)

    if [ "$STANDBY_PODS" -eq 0 ]; then
        error "Standby environment not available for rollback"
        return 1
    fi

    log "Standby environment ready: $STANDBY_PODS pods"

    # Health check standby environment
    log "Health checking standby environment..."
    for service in "${SERVICES[@]}"; do
        SERVICE_URL="http://${service}-${STANDBY_SLOT}.${NAMESPACE}.svc.cluster.local:8000"

        if kubectl run health-check-rollback-$service --rm -i --restart=Never --image=curlimages/curl -n "$NAMESPACE" -- \
            curl -f --max-time 10 "$SERVICE_URL/health" >/dev/null 2>&1; then
            log "✅ $service standby health check passed"
        else
            error "$service standby health check failed"
            return 1
        fi
    done

    # Switch traffic to standby
    log "Switching traffic to standby environment..."
    kubectl patch service tracktion -n "$NAMESPACE" --type='json' \
        -p='[{"op": "replace", "path": "/spec/selector/slot", "value": "'$STANDBY_SLOT'"}]'

    # Wait for traffic switch
    sleep 10

    # Verify traffic switch
    ACTIVE_SLOT=$(kubectl get service tracktion -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}')
    if [ "$ACTIVE_SLOT" = "$STANDBY_SLOT" ]; then
        log "✅ Traffic successfully switched to $STANDBY_SLOT"
    else
        error "Traffic switch failed"
        return 1
    fi
}

# Rollback using Helm
execute_helm_rollback() {
    log "Executing Helm rollback..."

    # Get previous revision
    PREVIOUS_REVISION=$(helm history tracktion -n "$NAMESPACE" --max 2 -o json | jq -r '.[1].revision')

    if [ "$PREVIOUS_REVISION" = "null" ] || [ -z "$PREVIOUS_REVISION" ]; then
        error "Cannot determine previous Helm revision"
        return 1
    fi

    log "Rolling back to Helm revision: $PREVIOUS_REVISION"

    # Execute rollback
    helm rollback tracktion "$PREVIOUS_REVISION" -n "$NAMESPACE" --wait --timeout=600s

    log "Helm rollback completed"
}

# Verify full rollback
verify_full_rollback() {
    log "Verifying full application rollback..."

    local failed_checks=0

    # Check all services
    for service in "${SERVICES[@]}"; do
        log "Checking service: $service"

        # Wait for deployment
        if kubectl wait --for=condition=available deployment/"$service" -n "$NAMESPACE" --timeout=300s; then
            log "✅ $service deployment ready"
        else
            error "$service deployment not ready"
            ((failed_checks++))
            continue
        fi

        # Health check
        for i in {1..5}; do
            if kubectl exec -n "$NAMESPACE" deployment/"$service" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
                log "✅ $service health check passed"
                break
            else
                if [ $i -eq 5 ]; then
                    error "$service health check failed after 5 attempts"
                    ((failed_checks++))
                fi
                sleep 10
            fi
        done
    done

    # Overall system health
    if [ $failed_checks -eq 0 ]; then
        log "✅ All services verified successfully"
    else
        error "$failed_checks services failed verification"
        return 1
    fi

    # End-to-end test
    log "Running end-to-end verification..."
    if ./scripts/smoke-tests.sh "$ENVIRONMENT" >/dev/null 2>&1; then
        log "✅ End-to-end verification passed"
    else
        error "End-to-end verification failed"
        return 1
    fi
}

# Generate rollback report
generate_rollback_report() {
    local status="$1"

    REPORT_FILE="/tmp/rollback_report_$(date +%s).md"

    cat > "$REPORT_FILE" << EOF
# Full Application Rollback Report

**Environment:** $ENVIRONMENT
**Date:** $(date)
**Status:** $status
**Duration:** $(($(date +%s) - ROLLBACK_START))s

## Services Rolled Back

$(for service in "${SERVICES[@]}"; do
    VERSION=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown")
    echo "- **$service**: $VERSION"
done)

## Database Status

$(kubectl exec -n "$NAMESPACE" deployment/analysis-service -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute('SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1')
    print('Database schema version:', cur.fetchone()[0])
    conn.close()
except Exception as e:
    print('Database check failed:', str(e))
" 2>/dev/null || echo "Database status: Unknown")

## System Health

$(kubectl get pods -n "$NAMESPACE" --no-headers | awk '{print "- " $1 ": " $3}')

## Next Steps

- [ ] Monitor system performance
- [ ] Investigate root cause of original issue
- [ ] Plan forward fix deployment
- [ ] Update incident documentation

**Log File:** $ROLLBACK_LOG
**Report Generated:** $(date)
EOF

    log "Rollback report generated: $REPORT_FILE"
}

# Main rollback execution
main() {
    trap 'handle_rollback_error' ERR

    ROLLBACK_START=$(date +%s)

    # Send initial notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🔄 Full application rollback initiated for '"$ENVIRONMENT"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    create_system_snapshot

    # Choose rollback method
    if kubectl get service tracktion -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}' >/dev/null 2>&1; then
        # Blue-green rollback
        execute_blue_green_rollback
    else
        # Helm rollback
        execute_helm_rollback
    fi

    verify_full_rollback
    generate_rollback_report "success"

    ROLLBACK_END=$(date +%s)
    ROLLBACK_DURATION=$((ROLLBACK_END - ROLLBACK_START))

    log "✅ Full application rollback completed successfully in ${ROLLBACK_DURATION}s"

    # Success notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"✅ Full application rollback completed for '"$ENVIRONMENT"' in '"$ROLLBACK_DURATION"'s"}' \
        "$SLACK_WEBHOOK_URL" || true
}

# Error handler
handle_rollback_error() {
    local exit_code=$?
    error "Full application rollback failed with exit code $exit_code"

    generate_rollback_report "failed"

    # Failure notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"❌ Full application rollback FAILED for '"$ENVIRONMENT"'. Manual intervention required. Log: '"$ROLLBACK_LOG"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit $exit_code
}

# Execute full rollback
main "$@"
```

## Database Rollback

### Point-in-Time Database Recovery

```bash
#!/bin/bash
# database_rollback.sh

set -e

ENVIRONMENT="${1:-production}"
TARGET_TIME="$2"
ROLLBACK_TYPE="${3:-point_in_time}"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 <environment> <target_time> [rollback_type]"
    echo "Target time format: 'YYYY-MM-DD HH:MM:SS'"
    echo "Rollback types: point_in_time, backup_restore, schema_rollback"
    exit 1
fi

NAMESPACE="tracktion-${ENVIRONMENT}"
DB_BACKUP_DIR="/backups/database"
ROLLBACK_LOG="/var/log/database_rollback_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$ROLLBACK_LOG")
exec 2> >(tee -a "$ROLLBACK_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [DB-ROLLBACK] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

log "Starting database rollback for $ENVIRONMENT to $TARGET_TIME"

# Database connection details
get_db_connection() {
    DB_HOST=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.host}' | base64 -d)
    DB_PORT=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.port}' | base64 -d)
    DB_NAME=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.database}' | base64 -d)
    DB_USER=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.username}' | base64 -d)
    DB_PASSWORD=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.password}' | base64 -d)

    export PGPASSWORD="$DB_PASSWORD"
}

# Create database backup before rollback
create_safety_backup() {
    log "Creating safety backup before rollback..."

    SAFETY_BACKUP="$DB_BACKUP_DIR/safety_backup_$(date +%s).sql"

    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --format=custom \
        --compress=9 \
        --verbose \
        --file="$SAFETY_BACKUP"

    if [ $? -eq 0 ]; then
        log "✅ Safety backup created: $SAFETY_BACKUP"

        # Calculate and store checksum
        sha256sum "$SAFETY_BACKUP" > "${SAFETY_BACKUP}.sha256"
    else
        error "Safety backup failed"
        exit 1
    fi
}

# Point-in-time recovery
point_in_time_recovery() {
    log "Executing point-in-time recovery to $TARGET_TIME..."

    # Stop application services to prevent new connections
    log "Stopping application services..."
    for service in "analysis-service" "file-watcher" "tracklist-service" "notification-service"; do
        kubectl scale deployment "$service" --replicas=0 -n "$NAMESPACE"
    done

    # Wait for connections to close
    sleep 30

    # Terminate existing connections
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '$DB_NAME'
          AND pid <> pg_backend_pid()
          AND state = 'active';"

    # Find base backup for PITR
    TARGET_EPOCH=$(date -d "$TARGET_TIME" +%s)
    BASE_BACKUP=""

    for backup in $(find "$DB_BACKUP_DIR" -name "base_backup_*.tar" | sort); do
        BACKUP_TIME=$(basename "$backup" | sed 's/base_backup_\([0-9]\+\)\.tar/\1/')
        if [ "$BACKUP_TIME" -le "$TARGET_EPOCH" ]; then
            BASE_BACKUP="$backup"
        fi
    done

    if [ -z "$BASE_BACKUP" ]; then
        error "No suitable base backup found for point-in-time recovery"
        exit 1
    fi

    log "Using base backup: $BASE_BACKUP"

    # This is a simplified example - actual PITR requires stopping PostgreSQL
    # and restoring from base backup + WAL files
    log "⚠️ Point-in-time recovery requires database downtime and careful execution"
    log "In production, this would:"
    log "1. Stop PostgreSQL server"
    log "2. Replace data directory with base backup"
    log "3. Configure recovery.conf with target time"
    log "4. Start PostgreSQL for recovery"
    log "5. Promote when recovery reaches target time"

    # For this example, we'll simulate the process
    log "Simulating PITR process..."
    sleep 10

    log "Point-in-time recovery simulation completed"
}

# Backup restore rollback
backup_restore_rollback() {
    log "Executing backup restore rollback..."

    # Find appropriate backup
    TARGET_DATE=$(date -d "$TARGET_TIME" +%Y%m%d)
    BACKUP_FILE=$(find "$DB_BACKUP_DIR" -name "full_backup_*_${TARGET_DATE}*.sql" | head -1)

    if [ -z "$BACKUP_FILE" ]; then
        # Find closest backup
        BACKUP_FILE=$(find "$DB_BACKUP_DIR" -name "full_backup_*.sql" | sort | head -1)
    fi

    if [ -z "$BACKUP_FILE" ]; then
        error "No suitable backup found for restore"
        exit 1
    fi

    log "Using backup file: $BACKUP_FILE"

    # Verify backup integrity
    if [ -f "${BACKUP_FILE}.sha256" ]; then
        if sha256sum -c "${BACKUP_FILE}.sha256"; then
            log "✅ Backup file integrity verified"
        else
            error "Backup file integrity check failed"
            exit 1
        fi
    fi

    # Stop application services
    log "Stopping application services..."
    for service in "analysis-service" "file-watcher" "tracklist-service" "notification-service"; do
        kubectl scale deployment "$service" --replicas=0 -n "$NAMESPACE"
    done

    # Wait for connections to close
    sleep 30

    # Terminate connections and drop/recreate database
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "postgres" -c "
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '$DB_NAME'
          AND pid <> pg_backend_pid();

        DROP DATABASE IF EXISTS \"${DB_NAME}\";
        CREATE DATABASE \"${DB_NAME}\";
    "

    # Restore from backup
    log "Restoring database from backup..."
    pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --verbose \
        --clean \
        --if-exists \
        "$BACKUP_FILE"

    if [ $? -eq 0 ]; then
        log "✅ Database restore completed"
    else
        error "Database restore failed"
        exit 1
    fi
}

# Schema rollback
schema_rollback() {
    log "Executing schema rollback..."

    # Get current schema version
    CURRENT_VERSION=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;" | xargs)

    log "Current schema version: $CURRENT_VERSION"

    # Determine target schema version based on target time
    # This would typically involve checking migration timestamps
    TARGET_SCHEMA_VERSION=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT version FROM schema_migrations
        WHERE created_at <= '$TARGET_TIME'
        ORDER BY version DESC LIMIT 1;" | xargs)

    if [ -z "$TARGET_SCHEMA_VERSION" ]; then
        error "Cannot determine target schema version"
        exit 1
    fi

    log "Target schema version: $TARGET_SCHEMA_VERSION"

    # Find rollback migrations
    ROLLBACK_MIGRATIONS=$(find ./migrations -name "*_rollback.sql" | grep -E "${CURRENT_VERSION}_to_${TARGET_SCHEMA_VERSION}" | sort -r)

    if [ -z "$ROLLBACK_MIGRATIONS" ]; then
        error "No rollback migrations found from $CURRENT_VERSION to $TARGET_SCHEMA_VERSION"
        exit 1
    fi

    # Execute rollback migrations
    for migration in $ROLLBACK_MIGRATIONS; do
        log "Executing rollback migration: $migration"

        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration"; then
            log "✅ Migration executed successfully: $migration"
        else
            error "Migration failed: $migration"
            exit 1
        fi
    done

    # Update schema version
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        UPDATE schema_migrations
        SET version = '$TARGET_SCHEMA_VERSION',
            updated_at = CURRENT_TIMESTAMP
        WHERE version = '$CURRENT_VERSION';"

    log "Schema rollback completed"
}

# Verify database rollback
verify_database_rollback() {
    log "Verifying database rollback..."

    # Test database connectivity
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        log "✅ Database connectivity verified"
    else
        error "Database connectivity test failed"
        return 1
    fi

    # Check schema integrity
    TABLES_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)

    log "Database contains $TABLES_COUNT tables"

    if [ "$TABLES_COUNT" -lt 5 ]; then
        error "Database seems to have too few tables after rollback"
        return 1
    fi

    # Check data integrity
    log "Checking data integrity..."

    # Check for basic data consistency
    AUDIO_FILES_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM audio_files;" | xargs 2>/dev/null || echo "0")

    log "Audio files in database: $AUDIO_FILES_COUNT"

    # Verify foreign key constraints
    FK_VIOLATIONS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM information_schema.table_constraints
        WHERE constraint_type = 'FOREIGN KEY'
          AND constraint_name NOT IN (
              SELECT conname FROM pg_constraint WHERE contype = 'f' AND confrelid IS NOT NULL
          );" | xargs)

    if [ "$FK_VIOLATIONS" -gt 0 ]; then
        error "Foreign key constraint violations detected: $FK_VIOLATIONS"
        return 1
    fi

    log "✅ Database integrity verification passed"
}

# Restart application services
restart_application_services() {
    log "Restarting application services..."

    ORIGINAL_REPLICAS=("5" "3" "4" "2")  # production replica counts
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for i in "${!SERVICES[@]}"; do
        service="${SERVICES[$i]}"
        replicas="${ORIGINAL_REPLICAS[$i]}"

        if [ "$ENVIRONMENT" != "production" ]; then
            replicas=1
        fi

        log "Scaling $service to $replicas replicas..."
        kubectl scale deployment "$service" --replicas="$replicas" -n "$NAMESPACE"
    done

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    for service in "${SERVICES[@]}"; do
        kubectl wait --for=condition=available deployment/"$service" -n "$NAMESPACE" --timeout=300s
        log "✅ $service is ready"
    done

    # Health check all services
    log "Running health checks..."
    for service in "${SERVICES[@]}"; do
        for i in {1..5}; do
            if kubectl exec -n "$NAMESPACE" deployment/"$service" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
                log "✅ $service health check passed"
                break
            else
                if [ $i -eq 5 ]; then
                    error "$service health check failed after 5 attempts"
                    return 1
                fi
                sleep 10
            fi
        done
    done

    log "All services are healthy after database rollback"
}

# Main database rollback execution
main() {
    trap 'handle_db_error' ERR

    get_db_connection
    create_safety_backup

    # Send notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🗃️ Database rollback initiated for '"$ENVIRONMENT"' to '"$TARGET_TIME"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    # Execute appropriate rollback type
    case "$ROLLBACK_TYPE" in
        "point_in_time")
            point_in_time_recovery
            ;;
        "backup_restore")
            backup_restore_rollback
            ;;
        "schema_rollback")
            schema_rollback
            ;;
        *)
            error "Unknown rollback type: $ROLLBACK_TYPE"
            exit 1
            ;;
    esac

    verify_database_rollback
    restart_application_services

    log "✅ Database rollback completed successfully"
    log "Environment: $ENVIRONMENT"
    log "Target time: $TARGET_TIME"
    log "Rollback type: $ROLLBACK_TYPE"

    # Success notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"✅ Database rollback completed for '"$ENVIRONMENT"'"}' \
        "$SLACK_WEBHOOK_URL" || true
}

# Error handler
handle_db_error() {
    local exit_code=$?
    error "Database rollback failed with exit code $exit_code"

    # Attempt to restart services even if rollback failed
    log "Attempting to restart services after failure..."
    restart_application_services || true

    # Failure notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"❌ Database rollback FAILED for '"$ENVIRONMENT"'. Manual intervention required. Log: '"$ROLLBACK_LOG"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit $exit_code
}

# Execute database rollback
main "$@"
```

## Configuration Rollback

### Configuration Management Rollback

```bash
#!/bin/bash
# configuration_rollback.sh

set -e

ENVIRONMENT="${1:-production}"
TARGET_CONFIG_VERSION="$2"

if [ -z "$TARGET_CONFIG_VERSION" ]; then
    echo "Usage: $0 <environment> <target_config_version>"
    echo "Example: $0 production v1.2.3"
    exit 1
fi

NAMESPACE="tracktion-${ENVIRONMENT}"
CONFIG_BACKUP_DIR="/backups/configuration"
ROLLBACK_LOG="/var/log/config_rollback_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$ROLLBACK_LOG")
exec 2> >(tee -a "$ROLLBACK_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [CONFIG-ROLLBACK] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
}

log "Starting configuration rollback for $ENVIRONMENT to $TARGET_CONFIG_VERSION"

# Create current configuration backup
backup_current_config() {
    log "Backing up current configuration..."

    CURRENT_BACKUP_DIR="/tmp/config_backup_$(date +%s)"
    mkdir -p "$CURRENT_BACKUP_DIR"

    # Backup Kubernetes ConfigMaps
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/configmaps.yaml"

    # Backup Kubernetes Secrets
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/secrets.yaml"

    # Backup application configuration files
    if [ -d "/app/config" ]; then
        tar -czf "$CURRENT_BACKUP_DIR/app_config.tar.gz" -C /app config/
    fi

    # Backup environment variables
    kubectl get deployment -n "$NAMESPACE" -o yaml | grep -A 100 "env:" > "$CURRENT_BACKUP_DIR/environment_vars.yaml"

    log "Current configuration backed up to: $CURRENT_BACKUP_DIR"
}

# Find target configuration
find_target_config() {
    log "Finding target configuration for version $TARGET_CONFIG_VERSION..."

    # Look for configuration backup
    CONFIG_ARCHIVE=$(find "$CONFIG_BACKUP_DIR" -name "config_${TARGET_CONFIG_VERSION}*.tar.gz" | head -1)

    if [ -z "$CONFIG_ARCHIVE" ]; then
        # Try git tag
        if git show-ref --tags | grep -q "$TARGET_CONFIG_VERSION"; then
            log "Found configuration in git tag: $TARGET_CONFIG_VERSION"
            CONFIG_SOURCE="git:$TARGET_CONFIG_VERSION"
        else
            error "Cannot find configuration for version $TARGET_CONFIG_VERSION"
            exit 1
        fi
    else
        log "Found configuration archive: $CONFIG_ARCHIVE"
        CONFIG_SOURCE="$CONFIG_ARCHIVE"
    fi
}

# Restore configuration from archive
restore_config_from_archive() {
    local archive="$1"

    log "Restoring configuration from archive: $archive"

    TEMP_DIR="/tmp/config_restore_$(date +%s)"
    mkdir -p "$TEMP_DIR"

    # Extract configuration archive
    tar -xzf "$archive" -C "$TEMP_DIR"

    # Restore ConfigMaps
    if [ -f "$TEMP_DIR/configmaps.yaml" ]; then
        log "Restoring ConfigMaps..."
        kubectl apply -f "$TEMP_DIR/configmaps.yaml" -n "$NAMESPACE"
    fi

    # Restore application configuration
    if [ -f "$TEMP_DIR/app_config.tar.gz" ]; then
        log "Restoring application configuration..."
        tar -xzf "$TEMP_DIR/app_config.tar.gz" -C /app/
    fi

    # Clean up
    rm -rf "$TEMP_DIR"

    log "Configuration restored from archive"
}

# Restore configuration from git
restore_config_from_git() {
    local git_ref="$1"

    log "Restoring configuration from git: $git_ref"

    # Create temporary git workspace
    TEMP_DIR="/tmp/config_git_$(date +%s)"
    git clone . "$TEMP_DIR"
    cd "$TEMP_DIR"

    # Checkout target version
    git checkout "$git_ref"

    # Apply Kubernetes configurations
    if [ -d "k8s/environments/$ENVIRONMENT" ]; then
        log "Applying Kubernetes configurations..."
        kubectl apply -f "k8s/environments/$ENVIRONMENT/" -n "$NAMESPACE"
    fi

    # Copy application configurations
    if [ -d "config/$ENVIRONMENT" ]; then
        log "Copying application configurations..."
        cp -r "config/$ENVIRONMENT/"* /app/config/
    fi

    cd - >/dev/null
    rm -rf "$TEMP_DIR"

    log "Configuration restored from git"
}

# Validate configuration
validate_configuration() {
    log "Validating restored configuration..."

    local failed_checks=0

    # Check ConfigMaps exist
    CONFIGMAPS=("app-config" "database-config" "redis-config")
    for cm in "${CONFIGMAPS[@]}"; do
        if kubectl get configmap "$cm" -n "$NAMESPACE" >/dev/null 2>&1; then
            log "✅ ConfigMap exists: $cm"
        else
            error "ConfigMap missing: $cm"
            ((failed_checks++))
        fi
    done

    # Check Secrets exist
    SECRETS=("db-credentials" "api-keys" "tls-certificates")
    for secret in "${SECRETS[@]}"; do
        if kubectl get secret "$secret" -n "$NAMESPACE" >/dev/null 2>&1; then
            log "✅ Secret exists: $secret"
        else
            error "Secret missing: $secret"
            ((failed_checks++))
        fi
    done

    # Validate configuration syntax
    if [ -f "/app/config/app.yml" ]; then
        if python -c "import yaml; yaml.safe_load(open('/app/config/app.yml'))" 2>/dev/null; then
            log "✅ Application configuration syntax valid"
        else
            error "Application configuration syntax invalid"
            ((failed_checks++))
        fi
    fi

    # Check database connection string
    DB_URL=$(kubectl get secret db-credentials -n "$NAMESPACE" -o jsonpath='{.data.url}' | base64 -d 2>/dev/null)
    if [[ "$DB_URL" =~ ^postgresql:// ]]; then
        log "✅ Database URL format valid"
    else
        error "Database URL format invalid"
        ((failed_checks++))
    fi

    if [ $failed_checks -eq 0 ]; then
        log "✅ Configuration validation passed"
    else
        error "Configuration validation failed: $failed_checks issues"
        return 1
    fi
}

# Restart affected services
restart_affected_services() {
    log "Restarting services affected by configuration changes..."

    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        log "Restarting $service..."

        # Rolling restart to pick up new configuration
        kubectl rollout restart deployment/"$service" -n "$NAMESPACE"

        # Wait for rollout to complete
        kubectl rollout status deployment/"$service" -n "$NAMESPACE" --timeout=300s

        log "✅ $service restarted successfully"
    done

    # Health check all services
    log "Running post-restart health checks..."
    for service in "${SERVICES[@]}"; do
        for i in {1..10}; do
            if kubectl exec -n "$NAMESPACE" deployment/"$service" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
                log "✅ $service health check passed"
                break
            else
                if [ $i -eq 10 ]; then
                    error "$service health check failed after restart"
                    return 1
                fi
                sleep 15
            fi
        done
    done
}

# Test configuration functionality
test_configuration() {
    log "Testing configuration functionality..."

    # Test database connectivity
    if kubectl exec -n "$NAMESPACE" deployment/analysis-service -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute('SELECT 1')
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" >/dev/null 2>&1; then
        log "✅ Database connectivity test passed"
    else
        error "Database connectivity test failed"
        return 1
    fi

    # Test Redis connectivity
    if kubectl exec -n "$NAMESPACE" deployment/analysis-service -- python -c "
import redis
import os
try:
    r = redis.Redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" >/dev/null 2>&1; then
        log "✅ Redis connectivity test passed"
    else
        error "Redis connectivity test failed"
        return 1
    fi

    # Test external API configuration
    if kubectl exec -n "$NAMESPACE" deployment/analysis-service -- curl -f --max-time 10 "https://api.external-service.com/health" >/dev/null 2>&1; then
        log "✅ External API connectivity test passed"
    else
        log "⚠️ External API connectivity test failed (non-critical)"
    fi

    log "Configuration functionality tests completed"
}

# Generate rollback report
generate_config_rollback_report() {
    local status="$1"

    REPORT_FILE="/tmp/config_rollback_report_$(date +%s).md"

    cat > "$REPORT_FILE" << EOF
# Configuration Rollback Report

**Environment:** $ENVIRONMENT
**Target Version:** $TARGET_CONFIG_VERSION
**Date:** $(date)
**Status:** $status

## Configuration Changes

### ConfigMaps
$(kubectl get configmaps -n "$NAMESPACE" --show-labels | grep -v NAME)

### Secrets
$(kubectl get secrets -n "$NAMESPACE" --show-labels | grep -v NAME | awk '{print $1 " " $2 " " $3}')

### Services Restarted
$(for service in "analysis-service" "file-watcher" "tracklist-service" "notification-service"; do
    RESTART_TIME=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')
    echo "- $service (revision: $RESTART_TIME)"
done)

## Validation Results

$(kubectl get pods -n "$NAMESPACE" --no-headers | awk '{print "- " $1 ": " $3}')

## Next Steps

- [ ] Monitor application behavior
- [ ] Validate business functionality
- [ ] Check integration endpoints
- [ ] Review application logs

**Log File:** $ROLLBACK_LOG
EOF

    log "Configuration rollback report generated: $REPORT_FILE"
}

# Main configuration rollback execution
main() {
    trap 'handle_config_error' ERR

    backup_current_config
    find_target_config

    # Send notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"⚙️ Configuration rollback initiated for '"$ENVIRONMENT"' to '"$TARGET_CONFIG_VERSION"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    # Restore configuration based on source
    if [[ "$CONFIG_SOURCE" == git:* ]]; then
        restore_config_from_git "${CONFIG_SOURCE#git:}"
    else
        restore_config_from_archive "$CONFIG_SOURCE"
    fi

    validate_configuration
    restart_affected_services
    test_configuration
    generate_config_rollback_report "success"

    log "✅ Configuration rollback completed successfully"

    # Success notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"✅ Configuration rollback completed for '"$ENVIRONMENT"' to '"$TARGET_CONFIG_VERSION"'"}' \
        "$SLACK_WEBHOOK_URL" || true
}

# Error handler
handle_config_error() {
    local exit_code=$?
    error "Configuration rollback failed with exit code $exit_code"

    generate_config_rollback_report "failed"

    # Failure notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"❌ Configuration rollback FAILED for '"$ENVIRONMENT"'. Manual intervention required. Log: '"$ROLLBACK_LOG"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit $exit_code
}

# Execute configuration rollback
main "$@"
```

## Emergency Rollback

### One-Click Emergency Rollback

```bash
#!/bin/bash
# emergency_rollback.sh

set -e

ENVIRONMENT="${1:-production}"
EMERGENCY_TYPE="${2:-auto}"  # auto, manual, partial

# Emergency rollback configuration
EMERGENCY_LOG="/var/log/emergency_rollback_$(date +%Y%m%d_%H%M%S).log"
ROLLBACK_TIMEOUT=300  # 5 minutes maximum

exec 1> >(tee -a "$EMERGENCY_LOG")
exec 2> >(tee -a "$EMERGENCY_LOG" >&2)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [EMERGENCY] $1"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [CRITICAL] $1" >&2
}

log "🚨 EMERGENCY ROLLBACK INITIATED 🚨"
log "Environment: $ENVIRONMENT"
log "Type: $EMERGENCY_TYPE"
log "Timeout: ${ROLLBACK_TIMEOUT}s"

# Immediate notification
send_emergency_alert() {
    local message="$1"

    # Slack with @channel mention
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 EMERGENCY ROLLBACK 🚨\nEnvironment: '"$ENVIRONMENT"'\n'"$message"'\n<!channel>"}' \
        "$SLACK_WEBHOOK_URL" || true

    # PagerDuty critical incident
    curl -X POST \
        -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "incident": {
                "type": "incident",
                "title": "EMERGENCY ROLLBACK - '"$ENVIRONMENT"'",
                "service": {"id": "'"$PAGERDUTY_SERVICE_ID"'", "type": "service_reference"},
                "urgency": "high",
                "body": {
                    "type": "incident_body",
                    "details": "'"$message"'"
                }
            }
        }' \
        "https://api.pagerduty.com/incidents" || true
}

send_emergency_alert "Emergency rollback initiated - all hands on deck required"

# Quick system health assessment
emergency_health_check() {
    log "Running emergency health assessment..."

    NAMESPACE="tracktion-${ENVIRONMENT}"
    CRITICAL_ISSUES=0

    # Check service availability
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")
    DOWN_SERVICES=()

    for service in "${SERVICES[@]}"; do
        READY_PODS=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        DESIRED_PODS=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")

        if [ "$READY_PODS" -eq 0 ] || [ "$READY_PODS" -lt $(($DESIRED_PODS / 2)) ]; then
            DOWN_SERVICES+=("$service")
            ((CRITICAL_ISSUES++))
            log "❌ CRITICAL: $service severely degraded ($READY_PODS/$DESIRED_PODS pods ready)"
        fi
    done

    # Check error rate
    ERROR_RATE=$(kubectl exec -n "$NAMESPACE" deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "100")

    if (( $(echo "$ERROR_RATE > 10.0" | bc -l) )); then
        ((CRITICAL_ISSUES++))
        log "❌ CRITICAL: Error rate extremely high: ${ERROR_RATE}%"
    fi

    # Database connectivity
    if ! kubectl exec -n "$NAMESPACE" deployment/analysis-service -- python -c "
import psycopg2; import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
conn.close()
" 2>/dev/null; then
        ((CRITICAL_ISSUES++))
        log "❌ CRITICAL: Database connectivity lost"
    fi

    log "Emergency assessment: $CRITICAL_ISSUES critical issues detected"

    if [ $CRITICAL_ISSUES -gt 2 ]; then
        log "SEVERITY: CRITICAL - Full system rollback required"
        return 2
    elif [ $CRITICAL_ISSUES -gt 0 ]; then
        log "SEVERITY: HIGH - Partial rollback may be sufficient"
        return 1
    else
        log "SEVERITY: LOW - System appears stable"
        return 0
    fi
}

# Ultra-fast traffic switch rollback
ultra_fast_rollback() {
    log "Executing ultra-fast traffic rollback..."

    NAMESPACE="tracktion-${ENVIRONMENT}"

    # Find standby environment
    CURRENT_SLOT=$(kubectl get service tracktion -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}' 2>/dev/null || echo "blue")
    STANDBY_SLOT=$([ "$CURRENT_SLOT" = "blue" ] && echo "green" || echo "blue")

    log "Current: $CURRENT_SLOT, Standby: $STANDBY_SLOT"

    # Check if standby is available
    STANDBY_PODS=$(kubectl get pods -n "$NAMESPACE" -l slot="$STANDBY_SLOT" --field-selector=status.phase=Running --no-headers | wc -l)

    if [ "$STANDBY_PODS" -gt 0 ]; then
        log "Standby environment available: $STANDBY_PODS pods"

        # Immediate traffic switch
        log "Switching traffic to standby (0-downtime)..."
        kubectl patch service tracktion -n "$NAMESPACE" --type='json' \
            -p='[{"op": "replace", "path": "/spec/selector/slot", "value": "'$STANDBY_SLOT'"}]'

        sleep 5

        # Verify switch
        ACTIVE_SLOT=$(kubectl get service tracktion -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}')
        if [ "$ACTIVE_SLOT" = "$STANDBY_SLOT" ]; then
            log "✅ Traffic switched to $STANDBY_SLOT in <10 seconds"
            return 0
        else
            error "Traffic switch failed"
            return 1
        fi
    else
        log "No standby environment available - falling back to Helm rollback"
        return 1
    fi
}

# Emergency Helm rollback
emergency_helm_rollback() {
    log "Executing emergency Helm rollback..."

    NAMESPACE="tracktion-${ENVIRONMENT}"

    # Get last known good revision
    CURRENT_REVISION=$(helm history tracktion -n "$NAMESPACE" --max 1 -o json | jq -r '.[0].revision')
    PREVIOUS_REVISION=$(helm history tracktion -n "$NAMESPACE" --max 2 -o json | jq -r '.[1].revision')

    log "Current revision: $CURRENT_REVISION"
    log "Previous revision: $PREVIOUS_REVISION"

    # Execute rollback with aggressive timeouts
    timeout "$ROLLBACK_TIMEOUT" helm rollback tracktion "$PREVIOUS_REVISION" -n "$NAMESPACE" --wait --timeout=180s

    if [ $? -eq 0 ]; then
        log "✅ Helm rollback completed"
        return 0
    else
        error "Helm rollback failed or timed out"
        return 1
    fi
}

# Emergency pod restart (last resort)
emergency_pod_restart() {
    log "Executing emergency pod restart (last resort)..."

    NAMESPACE="tracktion-${ENVIRONMENT}"
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        log "Force restarting $service..."

        # Scale down to 0
        kubectl scale deployment "$service" --replicas=0 -n "$NAMESPACE"

        # Wait briefly
        sleep 5

        # Scale back up
        ORIGINAL_REPLICAS=1
        if [ "$ENVIRONMENT" = "production" ]; then
            case "$service" in
                "analysis-service") ORIGINAL_REPLICAS=5 ;;
                "file-watcher") ORIGINAL_REPLICAS=3 ;;
                "tracklist-service") ORIGINAL_REPLICAS=4 ;;
                "notification-service") ORIGINAL_REPLICAS=2 ;;
            esac
        fi

        kubectl scale deployment "$service" --replicas="$ORIGINAL_REPLICAS" -n "$NAMESPACE"

        # Don't wait for all services - parallel restart
    done

    log "All services restarting in parallel..."

    # Wait for at least one service to be ready
    for i in {1..60}; do  # 5 minutes max
        READY_SERVICES=0
        for service in "${SERVICES[@]}"; do
            READY_PODS=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            if [ "$READY_PODS" -gt 0 ]; then
                ((READY_SERVICES++))
            fi
        done

        if [ $READY_SERVICES -ge 2 ]; then
            log "✅ $READY_SERVICES services are ready"
            return 0
        fi

        sleep 5
    done

    error "Emergency pod restart did not restore sufficient services"
    return 1
}

# Quick verification
emergency_verification() {
    log "Running emergency verification..."

    NAMESPACE="tracktion-${ENVIRONMENT}"

    # Quick health check
    HEALTHY_SERVICES=0
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        if kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' | grep -q '[1-9]'; then
            if timeout 10 kubectl exec -n "$NAMESPACE" deployment/"$service" -- curl -f http://localhost:8000/health >/dev/null 2>&1; then
                ((HEALTHY_SERVICES++))
                log "✅ $service: Healthy"
            else
                log "❌ $service: Health check failed"
            fi
        else
            log "❌ $service: No ready pods"
        fi
    done

    if [ $HEALTHY_SERVICES -ge 3 ]; then
        log "✅ Emergency rollback successful: $HEALTHY_SERVICES/4 services healthy"
        return 0
    else
        error "Emergency rollback insufficient: only $HEALTHY_SERVICES/4 services healthy"
        return 1
    fi
}

# Main emergency rollback execution
main() {
    ROLLBACK_START=$(date +%s)

    # Time-boxed execution
    {
        # Run health check to determine severity
        emergency_health_check
        SEVERITY=$?

        case "$SEVERITY" in
            0)
                log "System appears stable - monitoring without rollback"
                send_emergency_alert "System assessment complete - no rollback needed"
                return 0
                ;;
            1)
                log "Attempting quick traffic rollback..."
                if ultra_fast_rollback; then
                    emergency_verification && return 0
                fi

                log "Traffic rollback failed - attempting Helm rollback..."
                if emergency_helm_rollback; then
                    emergency_verification && return 0
                fi
                ;;
            2)
                log "Critical system failure - attempting all recovery methods..."

                # Try all methods in parallel/sequence
                if ultra_fast_rollback && emergency_verification; then
                    return 0
                fi

                if emergency_helm_rollback && emergency_verification; then
                    return 0
                fi

                log "Standard rollbacks failed - executing emergency pod restart..."
                if emergency_pod_restart && emergency_verification; then
                    return 0
                fi
                ;;
        esac

        error "All emergency rollback methods failed"
        return 1

    } &

    ROLLBACK_PID=$!

    # Kill rollback if it takes too long
    sleep "$ROLLBACK_TIMEOUT" && kill $ROLLBACK_PID 2>/dev/null &
    TIMEOUT_PID=$!

    wait $ROLLBACK_PID 2>/dev/null
    ROLLBACK_EXIT_CODE=$?

    kill $TIMEOUT_PID 2>/dev/null

    ROLLBACK_END=$(date +%s)
    ROLLBACK_DURATION=$((ROLLBACK_END - ROLLBACK_START))

    if [ $ROLLBACK_EXIT_CODE -eq 0 ]; then
        log "🎉 EMERGENCY ROLLBACK SUCCESSFUL in ${ROLLBACK_DURATION}s"
        send_emergency_alert "EMERGENCY ROLLBACK SUCCESSFUL in ${ROLLBACK_DURATION}s - System restored"
    else
        error "🚨 EMERGENCY ROLLBACK FAILED after ${ROLLBACK_DURATION}s"
        send_emergency_alert "EMERGENCY ROLLBACK FAILED after ${ROLLBACK_DURATION}s - MANUAL INTERVENTION REQUIRED IMMEDIATELY"

        # Escalate to senior engineers
        curl -X POST \
            -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{
                "incident": {
                    "type": "incident",
                    "title": "CRITICAL: Emergency Rollback Failed - '"$ENVIRONMENT"'",
                    "service": {"id": "'"$PAGERDUTY_SERVICE_ID"'", "type": "service_reference"},
                    "urgency": "high",
                    "escalation_policy": {"id": "'"$SENIOR_ESCALATION_POLICY_ID"'", "type": "escalation_policy_reference"},
                    "body": {
                        "type": "incident_body",
                        "details": "All automated recovery methods have failed. System requires immediate manual intervention. Log: '"$EMERGENCY_LOG"'"
                    }
                }
            }' \
            "https://api.pagerduty.com/incidents" || true
    fi

    return $ROLLBACK_EXIT_CODE
}

# Execute emergency rollback
main "$@"
```

This comprehensive rollback procedures document provides detailed guidance for quickly and safely rolling back deployments, configurations, and database changes in the Tracktion system. The procedures prioritize speed and reliability while maintaining data integrity and system functionality.
