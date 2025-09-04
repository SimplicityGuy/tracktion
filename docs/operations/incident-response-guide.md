# Incident Response Guide

## Table of Contents

1. [Overview](#overview)
2. [Incident Classification](#incident-classification)
3. [Response Team Structure](#response-team-structure)
4. [Escalation Procedures](#escalation-procedures)
5. [Communication Protocols](#communication-protocols)
6. [Initial Response](#initial-response)
7. [Investigation Procedures](#investigation-procedures)
8. [Resolution and Recovery](#resolution-and-recovery)
9. [Post-Incident Activities](#post-incident-activities)
10. [Common Incident Runbooks](#common-incident-runbooks)
11. [Communication Templates](#communication-templates)
12. [Tools and Resources](#tools-and-resources)
13. [Training and Preparedness](#training-and-preparedness)

## Overview

This guide provides comprehensive procedures for responding to incidents affecting the Tracktion audio analysis platform. The goal is to minimize impact on users, restore service quickly, and learn from incidents to prevent future occurrences.

### Incident Response Objectives

- **Rapid Detection**: Identify and acknowledge incidents within 5 minutes
- **Quick Response**: Begin resolution activities within 15 minutes
- **Clear Communication**: Keep stakeholders informed throughout the incident
- **Complete Resolution**: Restore full service functionality
- **Continuous Learning**: Conduct thorough post-incident reviews

### Core Principles

- **Safety First**: Protect data integrity and user privacy
- **Speed Over Perfection**: Fast response to minimize impact
- **Communication is Critical**: Keep everyone informed
- **Document Everything**: Maintain detailed incident records
- **Learn and Improve**: Use incidents to strengthen the system

## Incident Classification

### Severity Levels

#### Severity 1 - Critical (P1)
**Response Time**: Immediate (< 5 minutes)
**Resolution Target**: 1 hour
**Escalation**: Immediate to on-call manager

**Criteria:**
- Complete service unavailability for all users
- Data loss or corruption affecting multiple users
- Security breach with confirmed data exposure
- Major functionality unavailable (>50% of features)
- Revenue-impacting issues (>$10k/hour loss)

**Examples:**
- All services returning 5xx errors
- Database completely inaccessible
- Authentication system down
- Major data loss confirmed
- Active security incident

#### Severity 2 - High (P2)
**Response Time**: 15 minutes
**Resolution Target**: 4 hours
**Escalation**: On-call engineer + team lead

**Criteria:**
- Significant degraded performance (>3x normal response time)
- Key functionality unavailable (20-50% of features)
- Service unavailable for specific user segments
- Moderate data inconsistency
- External integrations failing

**Examples:**
- API response times >10 seconds
- File upload functionality broken
- Search service returning errors
- Payment processing issues
- Third-party service integration failures

#### Severity 3 - Medium (P3)
**Response Time**: 1 hour
**Resolution Target**: 24 hours
**Escalation**: On-call engineer

**Criteria:**
- Minor performance degradation (<3x normal response time)
- Non-critical features unavailable
- Cosmetic issues affecting user experience
- Minor data inconsistencies
- Monitoring alerts indicating potential issues

**Examples:**
- Slow page load times
- Minor UI bugs
- Non-critical API endpoints returning errors
- Email notifications delayed
- Monitoring dashboard issues

#### Severity 4 - Low (P4)
**Response Time**: Next business day
**Resolution Target**: 1 week
**Escalation**: None required

**Criteria:**
- Feature requests
- Documentation issues
- Minor improvements
- Non-user-facing issues
- Planned maintenance items

**Examples:**
- Feature enhancement requests
- Documentation updates
- Code refactoring needs
- Development tool issues
- Internal process improvements

### Impact Assessment Matrix

| Impact Level | User Base | Business Function | Revenue Impact | Example |
|--------------|-----------|-------------------|----------------|---------|
| **Critical** | All users | Core functionality | >$10k/hour | Complete service outage |
| **High** | Major segment | Key features | $1k-10k/hour | Authentication issues |
| **Medium** | Some users | Secondary features | $100-1k/hour | Report generation broken |
| **Low** | Few users | Nice-to-have features | <$100/hour | UI cosmetic issues |

## Response Team Structure

### Incident Commander (IC)
**Primary Role**: Overall incident coordination and decision-making
**Responsibilities:**
- Lead incident response efforts
- Make final decisions on resolution approaches
- Coordinate with stakeholders
- Manage communication flow
- Declare incident resolution

**Skills Required:**
- Strong leadership and decision-making
- Technical knowledge of system architecture
- Experience with incident management
- Excellent communication skills

### Technical Lead (TL)
**Primary Role**: Technical investigation and resolution
**Responsibilities:**
- Diagnose technical root cause
- Coordinate technical team activities
- Implement technical solutions
- Validate fixes and system recovery

**Skills Required:**
- Deep technical expertise in Tracktion systems
- Strong troubleshooting and debugging skills
- Knowledge of system architecture
- Experience with production systems

### Communications Lead (CL)
**Primary Role**: Stakeholder communication and updates
**Responsibilities:**
- Draft and send status updates
- Manage customer communications
- Coordinate with support team
- Handle media inquiries if needed

**Skills Required:**
- Excellent written and verbal communication
- Understanding of business impact
- Experience with crisis communication
- Knowledge of stakeholder needs

### Subject Matter Expert (SME)
**Primary Role**: Domain-specific expertise
**Responsibilities:**
- Provide specialized technical knowledge
- Assist with investigation in specific areas
- Implement specialized fixes
- Review technical decisions

**Examples:**
- Database Administrator for database issues
- Security Engineer for security incidents
- Infrastructure Engineer for system issues
- Frontend Engineer for UI problems

### On-Call Rotation

#### Primary On-Call
- **Coverage**: 24/7 rotation
- **Duration**: 1-week shifts
- **Responsibilities**: First responder, initial triage
- **Escalation**: Can escalate to secondary after 30 minutes

#### Secondary On-Call
- **Coverage**: Backup for primary
- **Duration**: 1-week shifts
- **Responsibilities**: Support primary, take over if needed
- **Escalation**: Can escalate to management after 1 hour

#### Management Escalation
- **Availability**: On-call manager available 24/7
- **Responsibilities**: Resource allocation, external communication
- **Authority**: Can authorize emergency measures

## Escalation Procedures

### Escalation Decision Tree

```
Incident Detected
       │
       ▼
   Severity 1?
   ┌────┴────┐
  Yes        No
   │          │
   ▼          ▼
Immediate   Normal
Escalation  Response
   │          │
   ▼          ▼
All Hands   On-Call
Response   Engineer
   │          │
   ▼          ▼
   ┌──────────┐
   │ Response │
   │ Proceeds │
   └────┬─────┘
        │
        ▼
   Resolution
```

### Escalation Triggers

#### Automatic Escalation
- **15 minutes**: P1 incidents automatically escalate to manager
- **30 minutes**: P2 incidents automatically escalate to team lead
- **1 hour**: Any unresolved P1/P2 escalates to senior management
- **2 hours**: P1 incidents escalate to executive team

#### Manual Escalation Criteria
- On-call engineer requests help
- Technical expertise needed beyond current team
- External vendor engagement required
- Media attention or customer escalation
- Potential legal or compliance implications

### Escalation Contacts

```yaml
# Escalation contact hierarchy
escalation_levels:
  level_1:
    - primary_oncall: "+1-555-0101"
    - secondary_oncall: "+1-555-0102"

  level_2:
    - team_lead: "+1-555-0201"
    - senior_engineer: "+1-555-0202"

  level_3:
    - engineering_manager: "+1-555-0301"
    - product_manager: "+1-555-0302"

  level_4:
    - vp_engineering: "+1-555-0401"
    - cto: "+1-555-0402"

  level_5:
    - ceo: "+1-555-0501"

external_contacts:
  aws_support: "+1-800-AWS-SUPPORT"
  security_firm: "+1-555-SEC-HELP"
  legal_counsel: "+1-555-LAW-HELP"
```

### Escalation Script

```bash
#!/bin/bash
# escalate_incident.sh

INCIDENT_ID="$1"
ESCALATION_LEVEL="$2"
REASON="$3"

if [ -z "$INCIDENT_ID" ] || [ -z "$ESCALATION_LEVEL" ]; then
    echo "Usage: $0 <incident_id> <escalation_level> <reason>"
    echo "Escalation levels: 1-5"
    exit 1
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ESCALATE] $1"
}

log "Escalating incident $INCIDENT_ID to level $ESCALATION_LEVEL"
log "Reason: $REASON"

# Update incident status
curl -X PATCH "https://api.pagerduty.com/incidents/$INCIDENT_ID" \
    -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "incident": {
            "escalation_level": "'$ESCALATION_LEVEL'",
            "escalation_reason": "'$REASON'",
            "escalation_time": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        }
    }'

# Send notifications based on escalation level
case "$ESCALATION_LEVEL" in
    "1"|"2")
        # Notify team leads
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"🚨 Incident Escalated to L'"$ESCALATION_LEVEL"'\nIncident: '"$INCIDENT_ID"'\nReason: '"$REASON"'"}' \
            "$TEAM_SLACK_WEBHOOK"
        ;;
    "3")
        # Notify management
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"⚠️ Management Escalation Required\nIncident: '"$INCIDENT_ID"'\nReason: '"$REASON"'"}' \
            "$MGMT_SLACK_WEBHOOK"
        ;;
    "4"|"5")
        # Notify executives
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"🚨 EXECUTIVE ESCALATION\nIncident: '"$INCIDENT_ID"'\nReason: '"$REASON"'\nImmediate attention required"}' \
            "$EXEC_SLACK_WEBHOOK"
        ;;
esac

log "Escalation notifications sent for level $ESCALATION_LEVEL"
```

## Communication Protocols

### Communication Principles

1. **Early and Often**: Communicate as soon as an incident is confirmed
2. **Clear and Concise**: Use simple, non-technical language for external communications
3. **Accurate Information**: Only share confirmed information
4. **Regular Updates**: Provide updates even if there's no new information
5. **Transparency**: Be honest about impact and timeline

### Communication Channels

#### Internal Communications
- **Slack**: `#incidents` channel for team coordination
- **PagerDuty**: Automated notifications and escalations
- **Conference Bridge**: Voice communication during major incidents
- **Email**: Formal updates to management and stakeholders

#### External Communications
- **Status Page**: Public status updates for customers
- **Customer Support**: Direct communication with affected customers
- **Social Media**: Public communications if needed
- **Press/Media**: Formal statements through PR team

### Communication Timeline

```
Incident Detected
    │
    ▼ (0-5 min)
Internal Alert
    │
    ▼ (5-15 min)
Incident Confirmed
    │
    ▼ (15-30 min)
First External Update
    │
    ▼ (Every 30 min)
Regular Updates
    │
    ▼
Resolution Confirmed
    │
    ▼ (Within 2 hours)
Final Update
    │
    ▼ (Within 24 hours)
Post-Incident Summary
```

### Status Page Updates

#### Status Levels
- **Operational**: All systems functioning normally
- **Degraded Performance**: Some systems experiencing issues
- **Partial Outage**: Some systems unavailable
- **Major Outage**: Significant functionality unavailable
- **Under Maintenance**: Planned maintenance in progress

#### Update Templates
```markdown
# Investigating
We are investigating reports of issues with [affected service]. We will provide updates as we learn more.

# Identified
We have identified the issue affecting [affected service]. [Brief description of problem]. We are working on a fix.

# Monitoring
A fix has been implemented and we are monitoring the results. Service should be restored for most users.

# Resolved
This issue has been resolved. All services are operating normally.
```

## Initial Response

### Incident Detection and Acknowledgment

```bash
#!/bin/bash
# initial_response.sh

INCIDENT_SOURCE="$1"  # alert, manual, customer
INCIDENT_DETAILS="$2"
RESPONDER="$3"

# Generate incident ID
INCIDENT_ID="INC-$(date +%Y%m%d)-$(printf "%04d" $(($(date +%s) % 10000)))"
INCIDENT_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INCIDENT] $1" | tee -a "/var/log/incidents/${INCIDENT_ID}.log"
}

log "Incident $INCIDENT_ID detected via $INCIDENT_SOURCE"
log "Initial details: $INCIDENT_DETAILS"
log "Responder: $RESPONDER"

# Immediate acknowledgment
acknowledge_incident() {
    log "Acknowledging incident in monitoring systems..."

    # Update PagerDuty
    curl -X PUT "https://api.pagerduty.com/incidents/$INCIDENT_ID/acknowledge" \
        -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "acknowledged_by": "'$RESPONDER'",
            "acknowledged_at": "'$INCIDENT_START'"
        }'

    # Create Slack thread
    SLACK_RESPONSE=$(curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 Incident '"$INCIDENT_ID"' - Initial Response\nDetection: '"$INCIDENT_SOURCE"'\nResponder: '"$RESPONDER"'\nDetails: '"$INCIDENT_DETAILS"'"}' \
        "$INCIDENTS_SLACK_WEBHOOK")

    THREAD_ID=$(echo "$SLACK_RESPONSE" | jq -r '.ts')
    echo "$THREAD_ID" > "/tmp/incident_${INCIDENT_ID}_thread"

    log "Incident acknowledged and communication channels established"
}

# Initial triage and assessment
initial_triage() {
    log "Starting initial triage..."

    # System health check
    log "Running system health assessment..."

    # Check service status
    SERVICES_DOWN=0
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        if ! curl -f "http://${service}:8000/health" >/dev/null 2>&1; then
            log "❌ $service: DOWN"
            ((SERVICES_DOWN++))
        else
            log "✅ $service: UP"
        fi
    done

    # Check database connectivity
    if kubectl exec deployment/analysis-service -- python -c "
import psycopg2; import os
conn = psycopg2.connect(os.environ['DATABASE_URL']); conn.close()
" >/dev/null 2>&1; then
        log "✅ Database: ACCESSIBLE"
    else
        log "❌ Database: INACCESSIBLE"
        ((SERVICES_DOWN++))
    fi

    # Determine severity
    if [ $SERVICES_DOWN -eq 0 ]; then
        SEVERITY="P3"
    elif [ $SERVICES_DOWN -le 2 ]; then
        SEVERITY="P2"
    else
        SEVERITY="P1"
    fi

    log "Initial severity assessment: $SEVERITY ($SERVICES_DOWN critical components affected)"
    echo "$SEVERITY" > "/tmp/incident_${INCIDENT_ID}_severity"
}

# Gather initial information
gather_initial_info() {
    log "Gathering initial incident information..."

    INFO_FILE="/tmp/incident_${INCIDENT_ID}_info.json"

    cat > "$INFO_FILE" << EOF
{
    "incident_id": "$INCIDENT_ID",
    "start_time": "$INCIDENT_START",
    "detection_source": "$INCIDENT_SOURCE",
    "initial_responder": "$RESPONDER",
    "initial_description": "$INCIDENT_DETAILS",
    "severity": "$(cat /tmp/incident_${INCIDENT_ID}_severity)",
    "affected_services": [
$(kubectl get pods --no-headers | grep -E "(Error|CrashLoop|Pending)" | awk '{print "\"" $1 "\""}' | paste -sd ',' || echo '""')
    ],
    "error_rate": $(kubectl exec deployment/prometheus -- curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | jq -r '.data.result[0].value[1]' 2>/dev/null || echo '0'),
    "response_time_p95": $(kubectl exec deployment/prometheus -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | jq -r '.data.result[0].value[1]' 2>/dev/null || echo '0')
}
EOF

    log "Initial information gathered and stored in $INFO_FILE"
}

# Establish war room for P1 incidents
establish_war_room() {
    local severity="$1"

    if [ "$severity" = "P1" ]; then
        log "Establishing war room for P1 incident..."

        # Create dedicated Slack channel
        CHANNEL_NAME="incident-${INCIDENT_ID,,}"
        curl -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
            -H "Content-Type: application/json" \
            -d '{"name":"'$CHANNEL_NAME'","purpose":"Incident response for '$INCIDENT_ID'"}' \
            "https://slack.com/api/conversations.create"

        # Start conference bridge
        log "Conference bridge: 1-800-TRACKTION ext. $INCIDENT_ID"

        # Invite key team members
        CORE_TEAM=("john.smith" "sarah.johnson" "mike.chen")
        for member in "${CORE_TEAM[@]}"; do
            curl -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                -d "channel=$CHANNEL_NAME&users=$member" \
                "https://slack.com/api/conversations.invite"
        done

        log "War room established: #$CHANNEL_NAME"
    fi
}

# Main initial response flow
main() {
    # Create incident log directory
    mkdir -p "/var/log/incidents"

    acknowledge_incident
    initial_triage
    gather_initial_info

    SEVERITY=$(cat "/tmp/incident_${INCIDENT_ID}_severity")
    establish_war_room "$SEVERITY"

    # Send initial status update
    send_status_update "$INCIDENT_ID" "investigating" "We are investigating reports of issues affecting some users."

    log "Initial response completed for incident $INCIDENT_ID"
    log "Next steps:"
    log "1. Begin detailed investigation"
    log "2. Escalate if severity P1 or no progress in 30 minutes"
    log "3. Provide updates every 15 minutes (P1) or 30 minutes (P2/P3)"

    echo "$INCIDENT_ID"
}

main "$@"
```

### Rapid Assessment Checklist

```markdown
# 5-Minute Rapid Assessment Checklist

## Service Health (2 minutes)
- [ ] API endpoints responding normally
- [ ] Database connectivity working
- [ ] Critical services running
- [ ] Error rates within normal range
- [ ] Response times acceptable

## Impact Assessment (2 minutes)
- [ ] Number of affected users estimated
- [ ] Core functionality affected?
- [ ] Revenue-generating features impacted?
- [ ] External customer complaints received?
- [ ] Internal team notifications sent?

## Initial Classification (1 minute)
- [ ] Severity level assigned (P1/P2/P3/P4)
- [ ] Incident ID generated
- [ ] Initial responder identified
- [ ] Communication channels established
- [ ] Next steps planned

## Immediate Actions
- [ ] Acknowledge alerts/pages
- [ ] Join incident channel/bridge
- [ ] Begin investigation
- [ ] Notify stakeholders
- [ ] Document findings
```

## Investigation Procedures

### Investigation Methodology

```bash
#!/bin/bash
# investigate_incident.sh

INCIDENT_ID="$1"

if [ -z "$INCIDENT_ID" ]; then
    echo "Usage: $0 <incident_id>"
    exit 1
fi

INVESTIGATION_LOG="/var/log/incidents/${INCIDENT_ID}_investigation.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INVESTIGATE] $1" | tee -a "$INVESTIGATION_LOG"
}

log "Starting investigation for incident $INCIDENT_ID"

# Step 1: Timeline reconstruction
reconstruct_timeline() {
    log "Step 1: Reconstructing incident timeline..."

    TIMELINE_FILE="/tmp/${INCIDENT_ID}_timeline.txt"

    # Get deployment history
    log "Checking recent deployments..."
    kubectl rollout history deployment --all-namespaces | grep -E "$(date -d '24 hours ago' +%Y-%m-%d)" > "$TIMELINE_FILE"

    # Get configuration changes
    log "Checking configuration changes..."
    kubectl get events --all-namespaces --sort-by='.firstTimestamp' | grep -E "$(date -d '6 hours ago' +%Y-%m-%d)" >> "$TIMELINE_FILE"

    # Get application logs around incident time
    log "Analyzing application logs..."
    INCIDENT_TIME=$(cat "/tmp/incident_${INCIDENT_ID}_info.json" | jq -r '.start_time')
    START_TIME=$(date -d "$INCIDENT_TIME - 1 hour" +%Y-%m-%dT%H:%M:%SZ)
    END_TIME=$(date -d "$INCIDENT_TIME + 30 minutes" +%Y-%m-%dT%H:%M:%SZ)

    kubectl logs --since-time="$START_TIME" --until-time="$END_TIME" \
        deployment/analysis-service | grep -E "(ERROR|FATAL|Exception)" >> "$TIMELINE_FILE"

    log "Timeline reconstruction completed: $TIMELINE_FILE"
}

# Step 2: System state analysis
analyze_system_state() {
    log "Step 2: Analyzing current system state..."

    # Resource usage analysis
    log "Checking resource usage..."
    kubectl top nodes | tee -a "$INVESTIGATION_LOG"
    kubectl top pods --all-namespaces | head -20 | tee -a "$INVESTIGATION_LOG"

    # Network connectivity
    log "Testing network connectivity..."
    for service in "analysis-service" "file-watcher" "tracklist-service"; do
        if kubectl exec deployment/"$service" -- curl -f http://database:5432 >/dev/null 2>&1; then
            log "✅ $service → database: OK"
        else
            log "❌ $service → database: FAILED"
        fi
    done

    # Database analysis
    log "Analyzing database state..."
    kubectl exec deployment/analysis-service -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()

    # Check active connections
    cur.execute('SELECT count(*) FROM pg_stat_activity')
    connections = cur.fetchone()[0]
    print(f'Active database connections: {connections}')

    # Check for locks
    cur.execute('SELECT count(*) FROM pg_locks WHERE NOT granted')
    locks = cur.fetchone()[0]
    print(f'Blocking locks: {locks}')

    # Check recent queries
    cur.execute('SELECT query, state, query_start FROM pg_stat_activity WHERE state = \\'active\\' ORDER BY query_start DESC LIMIT 5')
    active_queries = cur.fetchall()
    print(f'Active queries: {len(active_queries)}')

    conn.close()
except Exception as e:
    print(f'Database analysis failed: {e}')
" | tee -a "$INVESTIGATION_LOG"
}

# Step 3: Error pattern analysis
analyze_error_patterns() {
    log "Step 3: Analyzing error patterns..."

    # Application error analysis
    log "Analyzing application errors..."

    # Get recent errors from all services
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        log "Analyzing errors in $service:"
        kubectl logs --tail=1000 deployment/"$service" | \
            grep -E "(ERROR|FATAL|Exception)" | \
            sort | uniq -c | sort -nr | head -10 | \
            tee -a "$INVESTIGATION_LOG"
    done

    # HTTP error analysis
    log "Analyzing HTTP error patterns..."
    kubectl exec deployment/prometheus -- curl -s \
        'http://localhost:9090/api/v1/query_range?query=rate(http_requests_total{status=~"5.."}[5m])&start='$(date -d '2 hours ago' +%s)'&end='$(date +%s)'&step=300' | \
        jq -r '.data.result[] | "\(.metric.service): \(.values[-1][1]) errors/sec"' | \
        tee -a "$INVESTIGATION_LOG"
}

# Step 4: External dependency check
check_external_dependencies() {
    log "Step 4: Checking external dependencies..."

    # Check external API health
    EXTERNAL_APIS=("https://api.external-service.com/health" "https://payments.provider.com/status")

    for api in "${EXTERNAL_APIS[@]}"; do
        if curl -f --max-time 10 "$api" >/dev/null 2>&1; then
            log "✅ External API OK: $api"
        else
            log "❌ External API FAILED: $api"
        fi
    done

    # Check DNS resolution
    log "Testing DNS resolution..."
    kubectl exec deployment/analysis-service -- nslookup database | tee -a "$INVESTIGATION_LOG"

    # Check certificate status
    log "Checking SSL certificates..."
    echo | openssl s_client -connect api.tracktion.com:443 2>/dev/null | \
        openssl x509 -noout -dates | tee -a "$INVESTIGATION_LOG"
}

# Step 5: Performance analysis
analyze_performance() {
    log "Step 5: Analyzing performance metrics..."

    # Response time analysis
    log "Current response time percentiles:"
    kubectl exec deployment/prometheus -- curl -s \
        'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | \
        jq -r '.data.result[0].value[1] + "ms (95th percentile)"' | tee -a "$INVESTIGATION_LOG"

    # Memory usage analysis
    log "Memory usage by service:"
    kubectl top pods --no-headers | awk '{print $1 ": " $3}' | tee -a "$INVESTIGATION_LOG"

    # Disk usage analysis
    log "Disk usage analysis:"
    kubectl exec deployment/analysis-service -- df -h | tee -a "$INVESTIGATION_LOG"
}

# Generate investigation summary
generate_investigation_summary() {
    log "Generating investigation summary..."

    SUMMARY_FILE="/tmp/${INCIDENT_ID}_investigation_summary.md"

    cat > "$SUMMARY_FILE" << EOF
# Investigation Summary: $INCIDENT_ID

**Investigation Time:** $(date)
**Investigator:** $(whoami)

## Key Findings

### Timeline
$(head -20 /tmp/${INCIDENT_ID}_timeline.txt)

### System State
- Resource usage: $(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum/NR "%"}' | head -1) average CPU
- Error rate: $(kubectl exec deployment/prometheus -- curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | jq -r '.data.result[0].value[1]' 2>/dev/null || echo 'N/A')%
- Response time: $(kubectl exec deployment/prometheus -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | jq -r '.data.result[0].value[1]' 2>/dev/null || echo 'N/A')ms

### Root Cause Hypothesis
[To be filled by investigator based on findings]

### Recommended Actions
1. [Action item 1]
2. [Action item 2]
3. [Action item 3]

## Next Steps
- [ ] Implement immediate fixes
- [ ] Test resolution
- [ ] Monitor for stability
- [ ] Plan permanent fix

**Full Investigation Log:** $INVESTIGATION_LOG
EOF

    log "Investigation summary generated: $SUMMARY_FILE"
}

# Main investigation workflow
main() {
    reconstruct_timeline
    analyze_system_state
    analyze_error_patterns
    check_external_dependencies
    analyze_performance
    generate_investigation_summary

    log "Investigation completed for incident $INCIDENT_ID"
    log "Summary available at: /tmp/${INCIDENT_ID}_investigation_summary.md"
    log "Full log available at: $INVESTIGATION_LOG"
}

main "$@"
```

### Root Cause Analysis Framework

#### 5 Whys Analysis
```markdown
# 5 Whys Analysis Template

**Incident:** [Incident ID and brief description]

**Problem Statement:** [What happened?]

**Why 1:** Why did this problem occur?
**Answer:** [First level cause]

**Why 2:** Why did [Answer 1] happen?
**Answer:** [Second level cause]

**Why 3:** Why did [Answer 2] happen?
**Answer:** [Third level cause]

**Why 4:** Why did [Answer 3] happen?
**Answer:** [Fourth level cause]

**Why 5:** Why did [Answer 4] happen?
**Answer:** [Root cause identified]

**Root Cause:** [Final root cause statement]

**Corrective Actions:**
1. [Immediate fix]
2. [Short-term prevention]
3. [Long-term prevention]
```

#### Fishbone Diagram Categories
- **People**: Human factors, training, procedures
- **Process**: Workflows, procedures, policies
- **Environment**: External factors, dependencies
- **Materials**: Tools, resources, infrastructure
- **Methods**: Techniques, approaches, standards
- **Machines**: Systems, hardware, software

## Resolution and Recovery

### Resolution Workflow

```bash
#!/bin/bash
# resolve_incident.sh

INCIDENT_ID="$1"
RESOLUTION_TYPE="$2"  # fix, rollback, workaround

if [ -z "$INCIDENT_ID" ] || [ -z "$RESOLUTION_TYPE" ]; then
    echo "Usage: $0 <incident_id> <resolution_type>"
    echo "Resolution types: fix, rollback, workaround"
    exit 1
fi

RESOLUTION_LOG="/var/log/incidents/${INCIDENT_ID}_resolution.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [RESOLVE] $1" | tee -a "$RESOLUTION_LOG"
}

log "Starting resolution for incident $INCIDENT_ID using $RESOLUTION_TYPE"

# Implement fix based on type
implement_resolution() {
    local resolution_type="$1"

    case "$resolution_type" in
        "fix")
            log "Implementing technical fix..."
            implement_technical_fix
            ;;
        "rollback")
            log "Executing rollback procedure..."
            execute_rollback
            ;;
        "workaround")
            log "Applying temporary workaround..."
            apply_workaround
            ;;
        *)
            log "Unknown resolution type: $resolution_type"
            return 1
            ;;
    esac
}

implement_technical_fix() {
    log "Implementing technical fix based on investigation findings..."

    # Example: Restart failing service
    FAILING_SERVICE=$(kubectl get pods --field-selector=status.phase=Failed --no-headers | head -1 | awk '{print $1}' | sed 's/-[^-]*-[^-]*$//')

    if [ -n "$FAILING_SERVICE" ]; then
        log "Restarting failing service: $FAILING_SERVICE"
        kubectl rollout restart deployment/"$FAILING_SERVICE"
        kubectl rollout status deployment/"$FAILING_SERVICE" --timeout=300s
    fi

    # Example: Clear Redis cache if memory issue
    REDIS_MEMORY=$(kubectl exec deployment/redis -- redis-cli info memory | grep used_memory_human | cut -d: -f2)
    log "Redis memory usage: $REDIS_MEMORY"

    # Add specific fix logic based on incident type
    log "Technical fix implementation completed"
}

execute_rollback() {
    log "Executing rollback to previous stable version..."

    # Get previous version
    PREVIOUS_VERSION=$(helm history tracktion --max 2 -o json | jq -r '.[1].app_version')

    if [ "$PREVIOUS_VERSION" != "null" ]; then
        log "Rolling back to version: $PREVIOUS_VERSION"
        ./scripts/rollback.sh production "$PREVIOUS_VERSION"
    else
        log "No previous version found for rollback"
        return 1
    fi
}

apply_workaround() {
    log "Applying temporary workaround..."

    # Example workarounds
    # Scale up resources
    log "Scaling up critical services..."
    kubectl scale deployment analysis-service --replicas=8
    kubectl scale deployment tracklist-service --replicas=6

    # Enable circuit breakers
    log "Enabling circuit breakers..."
    kubectl patch configmap app-config --type merge -p '{"data":{"circuit_breaker_enabled":"true"}}'

    # Redirect traffic if needed
    log "Configuring traffic redirection..."
    # Implementation specific to incident

    log "Temporary workaround applied"
}

# Verify resolution effectiveness
verify_resolution() {
    log "Verifying resolution effectiveness..."

    # Wait for system to stabilize
    log "Waiting for system stabilization (60 seconds)..."
    sleep 60

    # Check service health
    HEALTHY_SERVICES=0
    SERVICES=("analysis-service" "file-watcher" "tracklist-service" "notification-service")

    for service in "${SERVICES[@]}"; do
        if kubectl get deployment "$service" -o jsonpath='{.status.readyReplicas}' | grep -q '[1-9]'; then
            if curl -f "http://${service}:8000/health" >/dev/null 2>&1; then
                log "✅ $service: Healthy"
                ((HEALTHY_SERVICES++))
            else
                log "❌ $service: Health check failed"
            fi
        else
            log "❌ $service: No ready pods"
        fi
    done

    # Check error rate
    ERROR_RATE=$(kubectl exec deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    log "Current error rate: ${ERROR_RATE}%"

    # Check response time
    RESPONSE_TIME=$(kubectl exec deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    log "Current response time P95: ${RESPONSE_TIME}ms"

    # Determine if resolution is successful
    if [ $HEALTHY_SERVICES -ge 3 ] && (( $(echo "$ERROR_RATE < 2.0" | bc -l) )) && (( $(echo "$RESPONSE_TIME < 3000" | bc -l) )); then
        log "✅ Resolution verification successful"
        return 0
    else
        log "❌ Resolution verification failed"
        return 1
    fi
}

# Monitor stability after resolution
monitor_stability() {
    log "Monitoring system stability post-resolution..."

    MONITORING_DURATION=600  # 10 minutes
    CHECK_INTERVAL=60        # 1 minute
    STABLE_CHECKS=0
    REQUIRED_STABLE_CHECKS=5

    for ((i=0; i<MONITORING_DURATION; i+=CHECK_INTERVAL)); do
        sleep $CHECK_INTERVAL

        # Quick health check
        if verify_resolution >/dev/null 2>&1; then
            ((STABLE_CHECKS++))
            log "Stability check $((i/CHECK_INTERVAL + 1)): ✅ ($STABLE_CHECKS/$REQUIRED_STABLE_CHECKS)"
        else
            STABLE_CHECKS=0
            log "Stability check $((i/CHECK_INTERVAL + 1)): ❌ (reset counter)"
        fi

        if [ $STABLE_CHECKS -ge $REQUIRED_STABLE_CHECKS ]; then
            log "✅ System stability confirmed after $((i/CHECK_INTERVAL + 1)) checks"
            return 0
        fi
    done

    log "⚠️ System stability monitoring completed without confirmation"
    return 1
}

# Update incident status
update_incident_status() {
    local status="$1"
    local message="$2"

    # Update PagerDuty
    curl -X PUT "https://api.pagerduty.com/incidents/$INCIDENT_ID" \
        -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "incident": {
                "status": "'$status'",
                "resolution": "'$message'"
            }
        }'

    # Update status page
    send_status_update "$INCIDENT_ID" "$status" "$message"

    # Notify Slack
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"📋 Incident '"$INCIDENT_ID"' Status Update\nStatus: '"$status"'\nMessage: '"$message"'"}' \
        "$INCIDENTS_SLACK_WEBHOOK"
}

# Main resolution workflow
main() {
    # Send initial resolution notification
    update_incident_status "implementing_fix" "We have identified the issue and are implementing a resolution."

    # Implement the resolution
    if implement_resolution "$RESOLUTION_TYPE"; then
        log "Resolution implementation completed"
    else
        log "Resolution implementation failed"
        update_incident_status "investigating" "Initial resolution attempt failed. Investigating alternative approaches."
        exit 1
    fi

    # Verify the resolution
    if verify_resolution; then
        log "Resolution verification successful"
        update_incident_status "monitoring" "A fix has been implemented and we are monitoring the results."
    else
        log "Resolution verification failed"
        update_incident_status "investigating" "Resolution verification failed. Continuing investigation."
        exit 1
    fi

    # Monitor stability
    if monitor_stability; then
        log "System stability confirmed"
        update_incident_status "resolved" "This incident has been resolved. All services are operating normally."
    else
        log "System stability not confirmed"
        update_incident_status "monitoring" "Resolution appears successful but we continue monitoring for stability."
    fi

    log "Resolution process completed for incident $INCIDENT_ID"
}

main "$@"
```

## Post-Incident Activities

### Post-Incident Review Process

```bash
#!/bin/bash
# post_incident_review.sh

INCIDENT_ID="$1"

if [ -z "$INCIDENT_ID" ]; then
    echo "Usage: $0 <incident_id>"
    exit 1
fi

PIR_DIR="/var/log/incidents/post_incident_reviews"
mkdir -p "$PIR_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [PIR] $1"
}

log "Starting post-incident review for $INCIDENT_ID"

# Collect incident data
collect_incident_data() {
    log "Collecting incident data..."

    INCIDENT_DATA_FILE="$PIR_DIR/${INCIDENT_ID}_data.json"

    # Get basic incident information
    INCIDENT_INFO=$(cat "/tmp/incident_${INCIDENT_ID}_info.json" 2>/dev/null || echo '{}')

    # Get timeline data
    TIMELINE_DATA=$(cat "/tmp/${INCIDENT_ID}_timeline.txt" 2>/dev/null || echo "")

    # Get resolution data
    RESOLUTION_LOG=$(cat "/var/log/incidents/${INCIDENT_ID}_resolution.log" 2>/dev/null || echo "")

    # Calculate incident metrics
    START_TIME=$(echo "$INCIDENT_INFO" | jq -r '.start_time // empty')
    if [ -n "$START_TIME" ]; then
        START_EPOCH=$(date -d "$START_TIME" +%s)
        END_EPOCH=$(date +%s)
        DURATION_MINUTES=$(( (END_EPOCH - START_EPOCH) / 60 ))
    else
        DURATION_MINUTES="unknown"
    fi

    # Estimate user impact
    AFFECTED_USERS=$(kubectl exec deployment/prometheus -- \
        curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total[1h]))' | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "unknown")

    # Create comprehensive incident data
    cat > "$INCIDENT_DATA_FILE" << EOF
{
    "incident_id": "$INCIDENT_ID",
    "basic_info": $INCIDENT_INFO,
    "duration_minutes": $DURATION_MINUTES,
    "affected_users_estimate": "$AFFECTED_USERS",
    "timeline": "$(echo "$TIMELINE_DATA" | sed 's/"/\\"/g')",
    "resolution_log": "$(echo "$RESOLUTION_LOG" | sed 's/"/\\"/g')",
    "review_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    log "Incident data collected: $INCIDENT_DATA_FILE"
}

# Generate PIR template
generate_pir_template() {
    log "Generating post-incident review template..."

    PIR_TEMPLATE="$PIR_DIR/${INCIDENT_ID}_review.md"

    # Get incident details
    START_TIME=$(jq -r '.basic_info.start_time // "unknown"' "$PIR_DIR/${INCIDENT_ID}_data.json")
    SEVERITY=$(jq -r '.basic_info.severity // "unknown"' "$PIR_DIR/${INCIDENT_ID}_data.json")
    DURATION=$(jq -r '.duration_minutes' "$PIR_DIR/${INCIDENT_ID}_data.json")

    cat > "$PIR_TEMPLATE" << EOF
# Post-Incident Review: $INCIDENT_ID

**Date:** $(date +%Y-%m-%d)
**Incident Start:** $START_TIME
**Duration:** $DURATION minutes
**Severity:** $SEVERITY
**Review Participants:** [To be filled]

## Executive Summary

[Brief summary of the incident, its impact, and resolution]

## Incident Details

### What Happened?
[Detailed description of the incident]

### Timeline
| Time | Event | Action Taken | Person |
|------|-------|--------------|--------|
| $START_TIME | Incident detected | [Initial response] | [Name] |
| [Time] | [Event] | [Action] | [Name] |
| [Time] | [Resolution] | [Final action] | [Name] |

### Impact Assessment
- **Users Affected:** [Number/percentage]
- **Services Affected:** [List of services]
- **Revenue Impact:** [Estimated amount]
- **Duration:** $DURATION minutes
- **Customer Complaints:** [Number received]

## Root Cause Analysis

### Primary Root Cause
[Detailed explanation of the root cause]

### Contributing Factors
1. [Factor 1]
2. [Factor 2]
3. [Factor 3]

### Why This Wasn't Caught Earlier
[Analysis of detection and prevention failures]

## Response Analysis

### What Went Well
1. [Positive aspect 1]
2. [Positive aspect 2]
3. [Positive aspect 3]

### What Could Be Improved
1. [Improvement area 1]
2. [Improvement area 2]
3. [Improvement area 3]

### Response Metrics
- **Detection Time:** [Time from start to detection]
- **Response Time:** [Time from detection to response]
- **Resolution Time:** [Time from response to resolution]
- **Communication Time:** [Time from detection to first communication]

## Action Items

### Immediate (Within 1 week)
- [ ] [Action item 1] - Owner: [Name] - Due: [Date]
- [ ] [Action item 2] - Owner: [Name] - Due: [Date]

### Short-term (Within 1 month)
- [ ] [Action item 1] - Owner: [Name] - Due: [Date]
- [ ] [Action item 2] - Owner: [Name] - Due: [Date]

### Long-term (Within 3 months)
- [ ] [Action item 1] - Owner: [Name] - Due: [Date]
- [ ] [Action item 2] - Owner: [Name] - Due: [Date]

## Prevention Measures

### Technical Improvements
1. [Technical change 1]
2. [Technical change 2]
3. [Technical change 3]

### Process Improvements
1. [Process change 1]
2. [Process change 2]
3. [Process change 3]

### Monitoring Improvements
1. [Monitoring change 1]
2. [Monitoring change 2]
3. [Monitoring change 3]

## Lessons Learned

### For Engineering Team
- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

### For Operations Team
- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

### For Organization
- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

## Follow-up Actions

### 30-Day Review
- [ ] Schedule follow-up review
- [ ] Check action item progress
- [ ] Assess effectiveness of changes

### Quarterly Review
- [ ] Include in quarterly incident analysis
- [ ] Update disaster recovery procedures
- [ ] Review and update monitoring

---

**Review Completed By:** [Name]
**Review Date:** $(date +%Y-%m-%d)
**Next Review Date:** [30 days from now]
EOF

    log "PIR template generated: $PIR_TEMPLATE"
}

# Schedule PIR meeting
schedule_pir_meeting() {
    log "Scheduling post-incident review meeting..."

    # Calculate appropriate meeting time (within 24-48 hours)
    MEETING_DATE=$(date -d '+2 days' +%Y-%m-%d)

    # Create calendar event
    cat > "/tmp/${INCIDENT_ID}_pir_meeting.ics" << EOF
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Tracktion//PIR//EN
BEGIN:VEVENT
UID:pir-${INCIDENT_ID}@tracktion.com
DTSTAMP:$(date -u +%Y%m%dT%H%M%SZ)
DTSTART:$(date -d "$MEETING_DATE 14:00" -u +%Y%m%dT%H%M%SZ)
DTEND:$(date -d "$MEETING_DATE 15:00" -u +%Y%m%dT%H%M%SZ)
SUMMARY:Post-Incident Review: $INCIDENT_ID
DESCRIPTION:Review of incident $INCIDENT_ID to identify root causes and improvement opportunities.
LOCATION:Conference Room / Video Call
ORGANIZER:CN=DevOps Team:MAILTO:devops@tracktion.com
ATTENDEE:MAILTO:engineering@tracktion.com
ATTENDEE:MAILTO:product@tracktion.com
ATTENDEE:MAILTO:support@tracktion.com
END:VEVENT
END:VCALENDAR
EOF

    # Send meeting invitation (would integrate with calendar system)
    log "PIR meeting scheduled for $MEETING_DATE"
    log "Calendar file: /tmp/${INCIDENT_ID}_pir_meeting.ics"
}

# Create action items tracking
create_action_items_tracking() {
    log "Setting up action items tracking..."

    ACTION_ITEMS_FILE="$PIR_DIR/${INCIDENT_ID}_action_items.json"

    cat > "$ACTION_ITEMS_FILE" << EOF
{
    "incident_id": "$INCIDENT_ID",
    "created_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "action_items": [
        {
            "id": "AI-${INCIDENT_ID}-001",
            "title": "Review and update monitoring alerts",
            "priority": "high",
            "owner": "engineering",
            "due_date": "$(date -d '+1 week' +%Y-%m-%d)",
            "status": "open",
            "category": "monitoring"
        },
        {
            "id": "AI-${INCIDENT_ID}-002",
            "title": "Improve incident response documentation",
            "priority": "medium",
            "owner": "devops",
            "due_date": "$(date -d '+2 weeks' +%Y-%m-%d)",
            "status": "open",
            "category": "process"
        }
    ]
}
EOF

    log "Action items tracking created: $ACTION_ITEMS_FILE"
}

# Send PIR completion notification
send_pir_notification() {
    log "Sending PIR completion notification..."

    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"📊 Post-Incident Review Ready: '"$INCIDENT_ID"'\n\nPIR document: '"$PIR_DIR/${INCIDENT_ID}_review.md"'\nMeeting scheduled for: '"$(date -d '+2 days' +%Y-%m-%d)"'\n\nPlease review the document before the meeting."}' \
        "$INCIDENTS_SLACK_WEBHOOK"
}

# Main PIR workflow
main() {
    collect_incident_data
    generate_pir_template
    schedule_pir_meeting
    create_action_items_tracking
    send_pir_notification

    log "Post-incident review setup completed for $INCIDENT_ID"
    log "PIR template: $PIR_DIR/${INCIDENT_ID}_review.md"
    log "Meeting scheduled for: $(date -d '+2 days' +%Y-%m-%d')"
}

main "$@"
```

## Common Incident Runbooks

### High Error Rate Runbook

```markdown
# Runbook: High Error Rate

**Trigger:** Error rate >5% for >5 minutes
**Severity:** P2 (P1 if >20% error rate)
**Response Team:** On-call engineer + Backend SME

## Initial Response (0-5 minutes)

1. **Acknowledge Alert**
   ```bash
   # Acknowledge PagerDuty alert
   pd incident ack <incident_id>
   ```

2. **Join Incident Channel**
   - #incidents channel in Slack
   - Incident bridge: 1-800-TRACKTION

3. **Quick Assessment**
   ```bash
   # Check current error rate
   kubectl exec deployment/prometheus -- curl -s \
     'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100'

   # Check which services are affected
   kubectl get pods --field-selector=status.phase!=Running
   ```

## Investigation (5-15 minutes)

1. **Check Recent Changes**
   ```bash
   # Check deployment history
   kubectl rollout history deployment --all-namespaces

   # Check recent configuration changes
   kubectl get events --sort-by='.firstTimestamp' | head -20
   ```

2. **Analyze Error Patterns**
   ```bash
   # Get error breakdown by service
   kubectl logs --tail=1000 deployment/analysis-service | grep ERROR | sort | uniq -c
   kubectl logs --tail=1000 deployment/tracklist-service | grep ERROR | sort | uniq -c
   ```

3. **Check System Resources**
   ```bash
   # Check resource usage
   kubectl top nodes
   kubectl top pods --all-namespaces | head -20
   ```

## Common Causes & Solutions

### Database Connection Issues
**Symptoms:** Database timeout errors, connection pool exhausted
**Solution:**
```bash
# Check database connections
kubectl exec deployment/analysis-service -- python -c "
import psycopg2; import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT count(*) FROM pg_stat_activity')
print('Active connections:', cur.fetchone()[0])
conn.close()
"

# If connection pool exhausted, restart services
kubectl rollout restart deployment/analysis-service
```

### Memory Leaks
**Symptoms:** Out of memory errors, pods being killed
**Solution:**
```bash
# Check memory usage
kubectl top pods --no-headers | sort -k3 -nr | head -10

# Restart high-memory pods
kubectl delete pod <high-memory-pod>
```

### External API Failures
**Symptoms:** Timeout errors to external services
**Solution:**
```bash
# Test external connectivity
curl -f --max-time 10 https://api.external-service.com/health

# Enable circuit breakers if available
kubectl patch configmap app-config --type merge -p '{"data":{"circuit_breaker_enabled":"true"}}'
```

### Configuration Errors
**Symptoms:** Application startup errors, invalid configuration
**Solution:**
```bash
# Check configuration
kubectl get configmap app-config -o yaml

# Rollback to previous configuration if needed
kubectl rollout undo deployment/analysis-service
```

## Escalation Criteria

- Error rate >20% (immediate P1 escalation)
- No improvement after 15 minutes of investigation
- Multiple services affected
- Database connectivity issues

## Resolution Verification

```bash
# Verify error rate has decreased
kubectl exec deployment/prometheus -- curl -s \
  'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100'

# Should be <2% for resolution
```
```

### Service Down Runbook

```markdown
# Runbook: Service Down

**Trigger:** Service health check failing for >2 minutes
**Severity:** P1 (if critical service), P2 (if non-critical)
**Response Team:** On-call engineer + Service SME

## Initial Response (0-2 minutes)

1. **Identify Affected Service**
   ```bash
   # Check which services are down
   kubectl get pods --field-selector=status.phase!=Running

   # Check service status
   kubectl get deployments
   ```

2. **Assess Impact**
   ```bash
   # Check if traffic is being served
   curl -f http://analysis-service:8000/health

   # Check load balancer status
   kubectl get service tracktion -o wide
   ```

## Quick Recovery Attempts (2-10 minutes)

1. **Pod Restart**
   ```bash
   # Get failing pods
   FAILING_PODS=$(kubectl get pods --field-selector=status.phase!=Running --no-headers | awk '{print $1}')

   # Delete failing pods (they will be recreated)
   for pod in $FAILING_PODS; do
     kubectl delete pod $pod
   done
   ```

2. **Deployment Restart**
   ```bash
   # Restart the deployment
   kubectl rollout restart deployment/analysis-service
   kubectl rollout status deployment/analysis-service --timeout=300s
   ```

3. **Resource Check**
   ```bash
   # Check if resource limits are causing issues
   kubectl describe pod <failing-pod-name>

   # Look for resource-related events
   kubectl get events --sort-by='.firstTimestamp' | grep -i -E "(killed|oom|resource)"
   ```

## Deep Investigation (10-30 minutes)

1. **Log Analysis**
   ```bash
   # Check application logs
   kubectl logs deployment/analysis-service --tail=100

   # Check previous pod logs if current pod is new
   kubectl logs deployment/analysis-service --previous --tail=100
   ```

2. **Resource Analysis**
   ```bash
   # Check node resources
   kubectl describe nodes | grep -E "(CPU|Memory)" -A 3

   # Check pod resource usage
   kubectl top pods --all-namespaces | sort -k3 -nr
   ```

3. **Network Connectivity**
   ```bash
   # Check if pod can reach dependencies
   kubectl exec deployment/analysis-service -- nc -zv database 5432
   kubectl exec deployment/analysis-service -- nc -zv redis 6379
   ```

## Common Solutions

### Resource Exhaustion
```bash
# Scale up temporarily
kubectl scale deployment analysis-service --replicas=8

# Or increase resource limits
kubectl patch deployment analysis-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"analysis-service","resources":{"limits":{"memory":"2Gi","cpu":"2000m"}}}]}}}}'
```

### Image Pull Issues
```bash
# Check if image exists
docker manifest inspect ghcr.io/tracktion/analysis-service:latest

# Update image pull policy
kubectl patch deployment analysis-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"analysis-service","imagePullPolicy":"Always"}]}}}}'
```

### Database Connectivity
```bash
# Check database status
kubectl get pods -l app=database

# Test database connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql -h database -U tracktion -c "SELECT 1;"
```

## Rollback Procedure

If service cannot be restored:
```bash
# Rollback to previous version
kubectl rollout undo deployment/analysis-service

# Monitor rollback progress
kubectl rollout status deployment/analysis-service

# Verify rollback success
curl -f http://analysis-service:8000/health
```

## Resolution Criteria

- All pods in Running state
- Health check returning 200
- Service responding to requests
- Error rate <1%
```

### Database Performance Issues

```markdown
# Runbook: Database Performance Issues

**Trigger:** Database response time >5 seconds or connection timeouts
**Severity:** P2 (P1 if >50% of requests affected)
**Response Team:** On-call engineer + DBA

## Initial Assessment (0-5 minutes)

1. **Check Database Connectivity**
   ```bash
   # Test basic connectivity
   kubectl exec deployment/analysis-service -- python -c "
   import psycopg2, os, time
   start = time.time()
   conn = psycopg2.connect(os.environ['DATABASE_URL'])
   print(f'Connection time: {time.time() - start:.2f}s')
   conn.close()"
   ```

2. **Check Current Load**
   ```bash
   # Check active connections
   kubectl exec deployment/analysis-service -- python -c "
   import psycopg2, os
   conn = psycopg2.connect(os.environ['DATABASE_URL'])
   cur = conn.cursor()
   cur.execute('SELECT count(*) FROM pg_stat_activity WHERE state = \'active\'')
   print(f'Active connections: {cur.fetchone()[0]}')
   cur.execute('SELECT count(*) FROM pg_stat_activity')
   print(f'Total connections: {cur.fetchone()[0]}')
   conn.close()"
   ```

## Investigation (5-15 minutes)

1. **Identify Slow Queries**
   ```sql
   -- Connect to database and run
   SELECT query, total_time, calls, mean_time, stddev_time
   FROM pg_stat_statements
   WHERE mean_time > 1000  -- queries taking more than 1 second
   ORDER BY total_time DESC
   LIMIT 10;
   ```

2. **Check for Locks**
   ```sql
   -- Check for blocking queries
   SELECT blocked_locks.pid AS blocked_pid,
          blocked_activity.usename AS blocked_user,
          blocking_locks.pid AS blocking_pid,
          blocking_activity.usename AS blocking_user,
          blocked_activity.query AS blocked_statement,
          blocking_activity.query AS current_statement_in_blocking_process
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted AND blocking_locks.granted;
   ```

3. **Check Resource Usage**
   ```sql
   -- Check database size and usage
   SELECT
       schemaname,
       tablename,
       pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
       n_tup_ins + n_tup_upd + n_tup_del as total_operations
   FROM pg_stat_user_tables
   ORDER BY pg_total_relation_size(tablename::regclass) DESC
   LIMIT 10;
   ```

## Common Solutions

### Too Many Connections
```bash
# Kill idle connections
kubectl exec deployment/analysis-service -- python -c "
import psycopg2, os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = \'idle\' AND query_start < now() - interval \'10 minutes\'')
conn.commit()
conn.close()"

# Restart application services to reset connection pools
kubectl rollout restart deployment/analysis-service
```

### Slow Queries
```sql
-- Kill specific slow query
SELECT pg_terminate_backend(12345); -- use actual PID

-- Update query statistics
ANALYZE;
```

### Lock Contention
```sql
-- Kill blocking query (use actual PID from locks query above)
SELECT pg_terminate_backend(blocking_pid);
```

### High CPU Usage
```bash
# Scale down application load temporarily
kubectl scale deployment analysis-service --replicas=2

# Check for missing indexes
-- Run EXPLAIN ANALYZE on slow queries to identify missing indexes
```

## Escalation Criteria

- Database completely unresponsive for >5 minutes
- More than 75% of application requests timing out
- Evidence of data corruption
- Database resource usage >90% sustained

## Prevention

1. **Query Optimization**
   - Review pg_stat_statements weekly
   - Add indexes for slow queries
   - Optimize application queries

2. **Connection Management**
   - Monitor connection pool usage
   - Set appropriate connection limits
   - Use connection pooling (PgBouncer)

3. **Resource Monitoring**
   - Set up alerts for high CPU/memory usage
   - Monitor disk space
   - Track query performance trends
```

## Communication Templates

### Initial Customer Communication

```markdown
Subject: Service Issue Notification - [Service Name]

Dear [Customer Name],

We are currently investigating reports of performance issues affecting some users of our [Service Name].

**Current Status:** We have identified the issue and our engineering team is actively working on a resolution.

**What happened:** [Brief, non-technical description]

**Impact:** [Description of what customers might experience]

**What we're doing:** Our team is implementing a fix and monitoring the results closely.

**Next update:** We will provide another update within [timeframe] or sooner if we have significant progress to report.

We sincerely apologize for any inconvenience this may cause. You can check our status page at [status.tracktion.com] for the latest updates.

If you have any questions or concerns, please don't hesitate to contact our support team at support@tracktion.com.

Best regards,
The Tracktion Team
```

### Resolution Notification

```markdown
Subject: Service Restored - [Service Name]

Dear [Customer Name],

We're pleased to report that the service issue affecting [Service Name] has been fully resolved.

**Resolution Time:** [Duration]
**Root Cause:** [Brief explanation]
**Services Affected:** [List]
**Data Impact:** No customer data was lost during this incident.

**What we did:**
- [Action 1]
- [Action 2]
- [Action 3]

**What we're doing next:**
We are conducting a thorough post-incident review to understand how this happened and what additional steps we can take to prevent similar issues in the future.

If you continue to experience any issues, please contact our support team immediately at support@tracktion.com.

Thank you for your patience during this incident.

Best regards,
The Tracktion Team
```

### Internal Status Update Template

```markdown
**Incident:** [INCIDENT_ID]
**Time:** [Timestamp]
**Status:** [Investigating/Identified/Monitoring/Resolved]
**Severity:** [P1/P2/P3]

**Current Situation:**
[Brief description of current state]

**Actions Taken:**
- [Action 1] - [Timestamp] - [Owner]
- [Action 2] - [Timestamp] - [Owner]

**Next Steps:**
- [Next action] - [Owner] - [ETA]

**Impact:**
- Affected Users: [Number/Estimate]
- Services Down: [List]
- Error Rate: [Percentage]

**Team:**
- IC: [Name]
- TL: [Name]
- CL: [Name]

**Next Update:** [Time]
```

This comprehensive incident response guide provides detailed procedures, tools, and templates for effectively managing incidents in the Tracktion system. Regular training and practice with these procedures ensures the team is prepared to respond quickly and effectively to any incident that may occur.
