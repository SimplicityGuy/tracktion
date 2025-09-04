# Backup Procedures

## Table of Contents

1. [Overview](#overview)
2. [Backup Strategy](#backup-strategy)
3. [Database Backups](#database-backups)
4. [File System Backups](#file-system-backups)
5. [Configuration Backups](#configuration-backups)
6. [Automated Backup Scripts](#automated-backup-scripts)
7. [Backup Verification](#backup-verification)
8. [Restore Procedures](#restore-procedures)
9. [Backup Monitoring](#backup-monitoring)
10. [Storage Management](#storage-management)
11. [Security Considerations](#security-considerations)
12. [Testing and Validation](#testing-and-validation)
13. [Troubleshooting](#troubleshooting)

## Overview

This document outlines comprehensive backup procedures for the Tracktion system, ensuring data protection, business continuity, and compliance with recovery time objectives (RTO) and recovery point objectives (RPO).

### Backup Objectives

- **Data Protection**: Safeguard against data loss from hardware failures, corruption, or human error
- **Business Continuity**: Enable rapid recovery to minimize downtime
- **Compliance**: Meet regulatory requirements for data retention
- **Version Control**: Maintain historical data for auditing and rollback purposes
- **Disaster Recovery**: Support disaster recovery scenarios

### Recovery Targets

- **RTO (Recovery Time Objective)**: Maximum acceptable downtime
  - Critical services: 30 minutes
  - Non-critical services: 2 hours
  - Development environments: 4 hours

- **RPO (Recovery Point Objective)**: Maximum acceptable data loss
  - Production database: 15 minutes
  - File system: 1 hour
  - Configuration: 24 hours

## Backup Strategy

### Backup Types

#### Full Backup
- **Frequency**: Weekly (Sunday 2:00 AM UTC)
- **Content**: Complete system state
- **Retention**: 4 weeks for production, 2 weeks for staging

#### Incremental Backup
- **Frequency**: Daily (2:00 AM UTC)
- **Content**: Changes since last backup
- **Retention**: 30 days

#### Transaction Log Backup
- **Frequency**: Every 15 minutes
- **Content**: Database transaction logs
- **Retention**: 7 days

#### Configuration Snapshot
- **Frequency**: Before each deployment
- **Content**: Configuration files, environment variables
- **Retention**: 10 snapshots per environment

### Backup Schedule Matrix

| Component | Backup Type | Frequency | Retention | Storage Location |
|-----------|-------------|-----------|-----------|------------------|
| Database | Full | Weekly | 4 weeks | AWS S3 + Local NAS |
| Database | Incremental | Daily | 30 days | AWS S3 |
| Database | Transaction Log | 15 minutes | 7 days | Local + AWS S3 |
| Audio Files | Full | Weekly | 12 weeks | AWS S3 Glacier |
| Audio Files | Incremental | Daily | 30 days | AWS S3 |
| Configurations | Snapshot | Pre-deployment | 10 versions | Git + AWS S3 |
| Logs | Archive | Daily | 90 days | AWS S3 IA |
| User Data | Full | Daily | 30 days | AWS S3 |

## Database Backups

### PostgreSQL Backup Procedures

#### Full Database Backup
```bash
#!/bin/bash
# full_db_backup.sh

set -e

# Configuration
DB_NAME="tracktion"
DB_USER="backup_user"
DB_HOST="localhost"
BACKUP_DIR="/backups/database"
S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/full_backup_${DB_NAME}_${DATE}.sql"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting full database backup for $DB_NAME..."

# Create compressed backup with custom format
pg_dump \
    -h "$DB_HOST" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -f "$BACKUP_FILE" \
    --format=custom \
    --compress=9 \
    --verbose \
    --lock-wait-timeout=30000

# Verify backup file exists and has content
if [[ ! -s "$BACKUP_FILE" ]]; then
    echo "Error: Backup file is empty or doesn't exist"
    exit 1
fi

# Calculate checksum
CHECKSUM=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)
echo "$CHECKSUM" > "${BACKUP_FILE}.sha256"

echo "Backup completed: $BACKUP_FILE (Size: $(du -h "$BACKUP_FILE" | cut -f1))"
echo "Checksum: $CHECKSUM"

# Upload to S3
aws s3 cp "$BACKUP_FILE" "s3://$S3_BUCKET/database/full/" --storage-class STANDARD_IA
aws s3 cp "${BACKUP_FILE}.sha256" "s3://$S3_BUCKET/database/full/"

# Upload to local NAS (if available)
if [[ -d "/mnt/nas/backups" ]]; then
    cp "$BACKUP_FILE" "/mnt/nas/backups/database/"
    cp "${BACKUP_FILE}.sha256" "/mnt/nas/backups/database/"
fi

echo "Backup upload completed successfully"
```

#### Incremental Database Backup
```bash
#!/bin/bash
# incremental_db_backup.sh

set -e

# Configuration
DB_NAME="tracktion"
DB_USER="backup_user"
DB_HOST="localhost"
BACKUP_DIR="/backups/database/incremental"
S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d_%H%M%S)
WAL_ARCHIVE_DIR="/backups/wal_archive"

# Create backup directories
mkdir -p "$BACKUP_DIR" "$WAL_ARCHIVE_DIR"

echo "Starting incremental backup (WAL archiving) for $DB_NAME..."

# Archive WAL files
pg_basebackup \
    -h "$DB_HOST" \
    -U "$DB_USER" \
    -D "${BACKUP_DIR}/base_${DATE}" \
    -F tar \
    -z \
    -P \
    -v \
    -W

# Sync WAL archive to S3
aws s3 sync "$WAL_ARCHIVE_DIR" "s3://$S3_BUCKET/database/wal_archive/" --delete

echo "Incremental backup completed"
```

#### Transaction Log Backup
```bash
#!/bin/bash
# wal_archive.sh - Called by PostgreSQL for WAL archiving

set -e

WAL_FILE="$1"
WAL_PATH="$2"
ARCHIVE_DIR="/backups/wal_archive"
S3_BUCKET="tracktion-backups"

# Copy WAL file to archive
cp "$WAL_PATH" "$ARCHIVE_DIR/"

# Upload to S3 immediately for critical data
aws s3 cp "$ARCHIVE_DIR/$WAL_FILE" "s3://$S3_BUCKET/database/wal_archive/$WAL_FILE"

# Clean up old WAL files (keep 7 days)
find "$ARCHIVE_DIR" -name "*.backup" -mtime +7 -delete
find "$ARCHIVE_DIR" -name "*" -mtime +7 -delete
```

#### PostgreSQL Configuration for Backups
```sql
-- postgresql.conf settings for backup
wal_level = replica
archive_mode = on
archive_command = '/opt/scripts/wal_archive.sh %f %p'
archive_timeout = 900  -- 15 minutes
max_wal_senders = 3
wal_keep_size = 1000   -- Keep 1GB of WAL files
```

### Backup User Setup
```sql
-- Create backup user with minimal permissions
CREATE ROLE backup_user WITH LOGIN PASSWORD 'secure_backup_password';
GRANT CONNECT ON DATABASE tracktion TO backup_user;
GRANT USAGE ON SCHEMA public TO backup_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO backup_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO backup_user;

-- Grant backup permissions
GRANT pg_read_all_settings TO backup_user;
GRANT pg_read_all_stats TO backup_user;
```

## File System Backups

### Audio Files Backup
```bash
#!/bin/bash
# audio_files_backup.sh

set -e

# Configuration
SOURCE_DIR="/app/data/audio_files"
BACKUP_DIR="/backups/audio_files"
S3_BUCKET="tracktion-backups"
GLACIER_BUCKET="tracktion-archive"
DATE=$(date +%Y%m%d)

echo "Starting audio files backup..."

# Create incremental backup using rsync
rsync -avz \
    --delete \
    --backup \
    --backup-dir="$BACKUP_DIR/incremental_$(date +%Y%m%d_%H%M%S)" \
    --exclude='*.tmp' \
    --exclude='*.processing' \
    "$SOURCE_DIR/" \
    "$BACKUP_DIR/current/"

# Create tarball for weekly full backup
if [[ $(date +%u) -eq 7 ]]; then  # Sunday
    echo "Creating weekly full backup archive..."

    ARCHIVE_NAME="audio_files_full_$DATE.tar.gz"
    tar -czf "$BACKUP_DIR/$ARCHIVE_NAME" -C "$SOURCE_DIR" .

    # Upload to S3 Glacier for long-term storage
    aws s3 cp "$BACKUP_DIR/$ARCHIVE_NAME" "s3://$GLACIER_BUCKET/audio_files/" --storage-class GLACIER

    # Calculate and store checksum
    sha256sum "$BACKUP_DIR/$ARCHIVE_NAME" > "$BACKUP_DIR/$ARCHIVE_NAME.sha256"
    aws s3 cp "$BACKUP_DIR/$ARCHIVE_NAME.sha256" "s3://$GLACIER_BUCKET/audio_files/"
fi

# Sync current state to S3 (daily incremental)
aws s3 sync "$BACKUP_DIR/current/" "s3://$S3_BUCKET/audio_files/current/" --delete

# Clean up old incremental backups (keep 30 days)
find "$BACKUP_DIR" -name "incremental_*" -mtime +30 -type d -exec rm -rf {} +

echo "Audio files backup completed"
```

### Application Logs Backup
```bash
#!/bin/bash
# logs_backup.sh

set -e

# Configuration
LOGS_DIR="/app/logs"
BACKUP_DIR="/backups/logs"
S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d)

echo "Starting logs backup..."

# Create logs backup directory
mkdir -p "$BACKUP_DIR"

# Archive and compress logs older than 1 day
find "$LOGS_DIR" -name "*.log" -mtime +1 -not -path "*/current/*" | while read -r logfile; do
    # Get relative path
    relative_path="${logfile#$LOGS_DIR/}"
    archive_path="$BACKUP_DIR/$(dirname "$relative_path")"

    # Create directory structure
    mkdir -p "$archive_path"

    # Compress and move to backup
    gzip -c "$logfile" > "$archive_path/$(basename "$logfile")_$DATE.gz"

    # Upload to S3 with Infrequent Access storage class
    aws s3 cp "$archive_path/$(basename "$logfile")_$DATE.gz" \
        "s3://$S3_BUCKET/logs/$(dirname "$relative_path")/" \
        --storage-class STANDARD_IA

    # Remove original log file after successful backup
    rm "$logfile"
done

# Clean up old archived logs (keep 90 days locally)
find "$BACKUP_DIR" -name "*.gz" -mtime +90 -delete

echo "Logs backup completed"
```

### User Data Backup
```bash
#!/bin/bash
# user_data_backup.sh

set -e

# Configuration
DATA_DIR="/app/data/user_uploads"
BACKUP_DIR="/backups/user_data"
S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting user data backup..."

# Create backup using rsync with hard links for space efficiency
rsync -avH \
    --delete \
    --link-dest="$BACKUP_DIR/previous" \
    "$DATA_DIR/" \
    "$BACKUP_DIR/backup_$DATE/"

# Update symlink to latest backup
ln -sfn "backup_$DATE" "$BACKUP_DIR/current"
ln -sfn "backup_$DATE" "$BACKUP_DIR/previous"

# Upload to S3
aws s3 sync "$BACKUP_DIR/backup_$DATE/" "s3://$S3_BUCKET/user_data/backup_$DATE/"

# Clean up old backups (keep 30 days)
find "$BACKUP_DIR" -name "backup_*" -mtime +30 -type d -exec rm -rf {} +

echo "User data backup completed"
```

## Configuration Backups

### Application Configuration Backup
```bash
#!/bin/bash
# config_backup.sh

set -e

# Configuration
CONFIG_DIRS=("/app/config" "/etc/tracktion" "/opt/tracktion/config")
BACKUP_DIR="/backups/configuration"
S3_BUCKET="tracktion-backups"
GIT_REPO="/backups/config_repo"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting configuration backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Initialize git repo if it doesn't exist
if [[ ! -d "$GIT_REPO/.git" ]]; then
    mkdir -p "$GIT_REPO"
    cd "$GIT_REPO"
    git init
    git config user.name "Backup System"
    git config user.email "backup@tracktion.com"
fi

# Copy configurations to git repo
for config_dir in "${CONFIG_DIRS[@]}"; do
    if [[ -d "$config_dir" ]]; then
        cp -r "$config_dir" "$GIT_REPO/"
    fi
done

# Add environment-specific configurations
cat > "$GIT_REPO/environment_snapshot.txt" << EOF
Backup Date: $DATE
Environment: $(hostname)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Docker Images:
$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" 2>/dev/null || echo "Docker not available")

Environment Variables:
$(env | grep "TRACKTION_" | sort)
EOF

# Commit to git
cd "$GIT_REPO"
git add .
git commit -m "Configuration backup - $DATE" || echo "No changes to commit"

# Create archive
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" -C "$GIT_REPO" .

# Upload to S3
aws s3 cp "$BACKUP_DIR/config_$DATE.tar.gz" "s3://$S3_BUCKET/configuration/"
aws s3 cp "$GIT_REPO/environment_snapshot.txt" "s3://$S3_BUCKET/configuration/snapshots/"

# Push to remote git repository (if configured)
if git remote get-url origin >/dev/null 2>&1; then
    git push origin main
fi

# Clean up old config backups (keep 10 versions)
ls -t "$BACKUP_DIR"/config_*.tar.gz | tail -n +11 | xargs rm -f

echo "Configuration backup completed"
```

### Docker Configuration Backup
```bash
#!/bin/bash
# docker_config_backup.sh

set -e

# Configuration
BACKUP_DIR="/backups/docker"
S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting Docker configuration backup..."

mkdir -p "$BACKUP_DIR"

# Backup Docker Compose files
find /opt/tracktion -name "docker-compose*.yml" -exec cp {} "$BACKUP_DIR/" \;

# Export running containers configuration
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" > "$BACKUP_DIR/running_containers_$DATE.txt"

# Export all images list
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" > "$BACKUP_DIR/images_list_$DATE.txt"

# Backup Docker volumes
docker volume ls -q | while read -r volume; do
    echo "Backing up volume: $volume"
    docker run --rm -v "$volume:/volume_data" -v "$BACKUP_DIR:/backup" \
        alpine tar czf "/backup/volume_${volume}_$DATE.tar.gz" -C /volume_data .
done

# Create complete Docker configuration archive
tar -czf "$BACKUP_DIR/docker_config_$DATE.tar.gz" -C "$BACKUP_DIR" .

# Upload to S3
aws s3 cp "$BACKUP_DIR/docker_config_$DATE.tar.gz" "s3://$S3_BUCKET/docker_config/"

echo "Docker configuration backup completed"
```

## Automated Backup Scripts

### Master Backup Orchestrator
```bash
#!/bin/bash
# master_backup.sh - Orchestrates all backup operations

set -e

# Configuration
SCRIPT_DIR="/opt/tracktion/backup_scripts"
LOG_DIR="/var/log/tracktion_backups"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/master_backup_$DATE.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: Backup failed at step: $1"
    # Send alert
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 Backup Failed: '"$1"'"}' \
        "$SLACK_WEBHOOK_URL" || true
    exit 1
}

log "Starting master backup process..."

# Pre-backup checks
log "Running pre-backup checks..."
if ! command -v pg_dump &> /dev/null; then
    handle_error "PostgreSQL client not available"
fi

if ! aws s3 ls s3://tracktion-backups/ &> /dev/null; then
    handle_error "S3 bucket not accessible"
fi

# Check disk space (require at least 10GB free)
AVAILABLE_SPACE=$(df /backups | awk 'NR==2 {print $4}')
if [[ $AVAILABLE_SPACE -lt 10485760 ]]; then  # 10GB in KB
    handle_error "Insufficient disk space for backups"
fi

# Run backup operations in sequence
BACKUP_OPERATIONS=(
    "database:$SCRIPT_DIR/full_db_backup.sh"
    "audio_files:$SCRIPT_DIR/audio_files_backup.sh"
    "user_data:$SCRIPT_DIR/user_data_backup.sh"
    "configuration:$SCRIPT_DIR/config_backup.sh"
    "logs:$SCRIPT_DIR/logs_backup.sh"
    "docker:$SCRIPT_DIR/docker_config_backup.sh"
)

for operation in "${BACKUP_OPERATIONS[@]}"; do
    operation_name="${operation%:*}"
    script_path="${operation#*:}"

    log "Starting $operation_name backup..."

    if [[ -x "$script_path" ]]; then
        if "$script_path" >> "$LOG_FILE" 2>&1; then
            log "$operation_name backup completed successfully"
        else
            handle_error "$operation_name backup script failed"
        fi
    else
        handle_error "$operation_name backup script not found or not executable: $script_path"
    fi
done

# Run backup verification
log "Running backup verification..."
if "$SCRIPT_DIR/verify_backups.sh" >> "$LOG_FILE" 2>&1; then
    log "Backup verification completed successfully"
else
    log "WARNING: Backup verification failed"
fi

# Cleanup old backups
log "Cleaning up old backup files..."
find /backups -name "*" -mtime +30 -type f -delete

# Generate backup report
BACKUP_SIZE=$(du -sh /backups | cut -f1)
S3_BACKUP_COUNT=$(aws s3 ls s3://tracktion-backups/ --recursive | wc -l)

cat > "$LOG_DIR/backup_report_$DATE.txt" << EOF
Backup Report - $DATE

Summary:
- All backup operations completed successfully
- Total local backup size: $BACKUP_SIZE
- S3 backup files count: $S3_BACKUP_COUNT
- Log file: $LOG_FILE

Operations completed:
$(for op in "${BACKUP_OPERATIONS[@]}"; do echo "✓ ${op%:*}"; done)

Next scheduled backup: $(date -d '+1 day' '+%Y-%m-%d %H:%M:%S')
EOF

# Send success notification
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"✅ Daily backup completed successfully\nSize: '"$BACKUP_SIZE"'\nFiles: '"$S3_BACKUP_COUNT"'"}' \
    "$SLACK_WEBHOOK_URL" || true

log "Master backup process completed successfully"
```

### Cron Configuration
```bash
# /etc/cron.d/tracktion-backups

# Full backup - Weekly (Sunday 2:00 AM)
0 2 * * 0 root /opt/tracktion/backup_scripts/master_backup.sh full

# Incremental backup - Daily (2:00 AM, except Sunday)
0 2 * * 1-6 root /opt/tracktion/backup_scripts/master_backup.sh incremental

# Transaction log backup - Every 15 minutes
*/15 * * * * postgres /opt/tracktion/backup_scripts/wal_archive.sh

# Configuration backup - Before deployments (triggered manually)
# 0 1 * * * root /opt/tracktion/backup_scripts/config_backup.sh

# Backup verification - Daily (6:00 AM)
0 6 * * * root /opt/tracktion/backup_scripts/verify_backups.sh

# Cleanup old backups - Weekly (Monday 3:00 AM)
0 3 * * 1 root /opt/tracktion/backup_scripts/cleanup_backups.sh
```

## Backup Verification

### Verification Script
```bash
#!/bin/bash
# verify_backups.sh

set -e

# Configuration
BACKUP_DIR="/backups"
S3_BUCKET="tracktion-backups"
LOG_FILE="/var/log/tracktion_backups/verification_$(date +%Y%m%d_%H%M%S).log"
DATE=$(date +%Y%m%d)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting backup verification process..."

# Verify database backup integrity
verify_database_backup() {
    log "Verifying database backup integrity..."

    LATEST_BACKUP=$(find "$BACKUP_DIR/database" -name "full_backup_*.sql" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [[ -z "$LATEST_BACKUP" ]]; then
        log "ERROR: No database backup found"
        return 1
    fi

    # Verify checksum
    if [[ -f "${LATEST_BACKUP}.sha256" ]]; then
        if sha256sum -c "${LATEST_BACKUP}.sha256"; then
            log "Database backup checksum verification passed"
        else
            log "ERROR: Database backup checksum verification failed"
            return 1
        fi
    fi

    # Test restore to temporary database
    log "Testing database backup restore..."
    TEST_DB="tracktion_test_restore_$(date +%s)"

    # Create test database
    createdb -U postgres "$TEST_DB"

    # Restore backup
    if pg_restore -U postgres -d "$TEST_DB" "$LATEST_BACKUP" >/dev/null 2>&1; then
        log "Database backup restore test passed"

        # Verify data integrity
        TABLE_COUNT=$(psql -U postgres -d "$TEST_DB" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
        log "Restored database contains $TABLE_COUNT tables"

        # Cleanup test database
        dropdb -U postgres "$TEST_DB"
    else
        log "ERROR: Database backup restore test failed"
        dropdb -U postgres "$TEST_DB" || true
        return 1
    fi
}

# Verify file system backups
verify_filesystem_backups() {
    log "Verifying file system backups..."

    # Check audio files backup
    if [[ -d "$BACKUP_DIR/audio_files/current" ]]; then
        BACKUP_COUNT=$(find "$BACKUP_DIR/audio_files/current" -type f | wc -l)
        SOURCE_COUNT=$(find /app/data/audio_files -type f -name "*.mp3" -o -name "*.wav" -o -name "*.flac" | wc -l)

        if [[ $BACKUP_COUNT -ge $(( SOURCE_COUNT * 95 / 100)) ]]; then  # Allow 5% variance
            log "Audio files backup verification passed ($BACKUP_COUNT files backed up)"
        else
            log "WARNING: Audio files backup may be incomplete (Backup: $BACKUP_COUNT, Source: $SOURCE_COUNT)"
        fi
    else
        log "ERROR: Audio files backup directory not found"
        return 1
    fi

    # Check user data backup
    LATEST_USER_BACKUP=$(find "$BACKUP_DIR/user_data" -name "backup_*" -type d | sort | tail -1)
    if [[ -n "$LATEST_USER_BACKUP" ]]; then
        USER_BACKUP_SIZE=$(du -sb "$LATEST_USER_BACKUP" | cut -f1)
        USER_SOURCE_SIZE=$(du -sb /app/data/user_uploads | cut -f1)

        if [[ $USER_BACKUP_SIZE -ge $(( USER_SOURCE_SIZE * 95 / 100)) ]]; then
            log "User data backup verification passed"
        else
            log "WARNING: User data backup size discrepancy"
        fi
    fi
}

# Verify S3 backups
verify_s3_backups() {
    log "Verifying S3 backups..."

    # Check if recent backups exist in S3
    S3_DB_BACKUPS=$(aws s3 ls "s3://$S3_BUCKET/database/full/" | grep "$(date +%Y%m%d)" | wc -l)
    S3_AUDIO_BACKUPS=$(aws s3 ls "s3://$S3_BUCKET/audio_files/current/" --recursive | wc -l)

    if [[ $S3_DB_BACKUPS -gt 0 ]]; then
        log "S3 database backup found for today"
    else
        log "WARNING: No S3 database backup found for today"
    fi

    if [[ $S3_AUDIO_BACKUPS -gt 100 ]]; then  # Expect at least 100 audio files
        log "S3 audio files backup contains $S3_AUDIO_BACKUPS files"
    else
        log "WARNING: S3 audio files backup may be incomplete"
    fi
}

# Verify backup retention
verify_retention() {
    log "Verifying backup retention policies..."

    # Check database backup retention (should have 4 weeks of full backups)
    FULL_BACKUPS_COUNT=$(find "$BACKUP_DIR/database" -name "full_backup_*.sql" -mtime -28 | wc -l)
    if [[ $FULL_BACKUPS_COUNT -ge 4 ]]; then
        log "Database backup retention policy satisfied ($FULL_BACKUPS_COUNT backups)"
    else
        log "WARNING: Database backup retention may not be satisfied ($FULL_BACKUPS_COUNT backups found)"
    fi

    # Check for old backups that should be cleaned up
    OLD_BACKUPS=$(find "$BACKUP_DIR" -name "*" -mtime +35 -type f | wc -l)
    if [[ $OLD_BACKUPS -eq 0 ]]; then
        log "Backup cleanup is working correctly"
    else
        log "WARNING: $OLD_BACKUPS old backup files found that should be cleaned up"
    fi
}

# Run all verification tests
VERIFICATION_FAILED=false

if ! verify_database_backup; then
    VERIFICATION_FAILED=true
fi

if ! verify_filesystem_backups; then
    VERIFICATION_FAILED=true
fi

if ! verify_s3_backups; then
    VERIFICATION_FAILED=true
fi

verify_retention

# Generate verification report
if [[ $VERIFICATION_FAILED == false ]]; then
    log "✅ All backup verifications passed"
    echo "success" > "$BACKUP_DIR/.verification_status"
else
    log "❌ Some backup verifications failed"
    echo "failed" > "$BACKUP_DIR/.verification_status"

    # Send alert for failed verification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"⚠️ Backup verification failed. Check logs: '"$LOG_FILE"'"}' \
        "$SLACK_WEBHOOK_URL" || true

    exit 1
fi

log "Backup verification completed"
```

## Restore Procedures

### Database Restore
```bash
#!/bin/bash
# restore_database.sh

set -e

# Configuration
USAGE="Usage: $0 <backup_file> [target_database] [restore_time]"

if [[ $# -lt 1 ]]; then
    echo "$USAGE"
    exit 1
fi

BACKUP_FILE="$1"
TARGET_DB="${2:-tracktion_restore}"
RESTORE_TIME="$3"  # Optional: for point-in-time recovery

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting database restore process..."

# Validate backup file
if [[ ! -f "$BACKUP_FILE" ]]; then
    log "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Verify backup file integrity
if [[ -f "${BACKUP_FILE}.sha256" ]]; then
    if ! sha256sum -c "${BACKUP_FILE}.sha256"; then
        log "ERROR: Backup file integrity check failed"
        exit 1
    fi
    log "Backup file integrity verified"
fi

# Create target database
log "Creating target database: $TARGET_DB"
createdb -U postgres "$TARGET_DB"

# Restore from backup
log "Restoring database from backup: $BACKUP_FILE"
pg_restore -U postgres -d "$TARGET_DB" -v --clean --if-exists "$BACKUP_FILE"

# Point-in-time recovery (if specified)
if [[ -n "$RESTORE_TIME" ]]; then
    log "Performing point-in-time recovery to: $RESTORE_TIME"

    # This requires WAL files and specific PostgreSQL configuration
    # Implementation depends on your WAL archiving setup
    log "Point-in-time recovery requires manual intervention - check documentation"
fi

# Verify restore
log "Verifying database restore..."
TABLE_COUNT=$(psql -U postgres -d "$TARGET_DB" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
log "Restored database contains $TABLE_COUNT tables"

# Run basic integrity checks
psql -U postgres -d "$TARGET_DB" -c "
SELECT 'audio_files' as table_name, count(*) as row_count FROM audio_files
UNION ALL
SELECT 'tracklists', count(*) FROM tracklists
UNION ALL
SELECT 'analysis_results', count(*) FROM analysis_results;
"

log "Database restore completed successfully"
log "Restored database: $TARGET_DB"
```

### File System Restore
```bash
#!/bin/bash
# restore_files.sh

set -e

USAGE="Usage: $0 <backup_source> <restore_target> [date]"

if [[ $# -lt 2 ]]; then
    echo "$USAGE"
    exit 1
fi

BACKUP_SOURCE="$1"
RESTORE_TARGET="$2"
RESTORE_DATE="$3"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting file system restore process..."

# Determine backup location
if [[ "$BACKUP_SOURCE" == "s3://"* ]]; then
    log "Restoring from S3: $BACKUP_SOURCE"

    # Create temporary directory
    TEMP_DIR="/tmp/restore_$(date +%s)"
    mkdir -p "$TEMP_DIR"

    # Download from S3
    aws s3 sync "$BACKUP_SOURCE" "$TEMP_DIR/"
    BACKUP_SOURCE="$TEMP_DIR"

elif [[ "$BACKUP_SOURCE" == *.tar.gz ]]; then
    log "Restoring from archive: $BACKUP_SOURCE"

    # Extract archive
    TEMP_DIR="/tmp/restore_$(date +%s)"
    mkdir -p "$TEMP_DIR"
    tar -xzf "$BACKUP_SOURCE" -C "$TEMP_DIR"
    BACKUP_SOURCE="$TEMP_DIR"
fi

# Create restore target directory
mkdir -p "$RESTORE_TARGET"

# Restore files
log "Copying files from $BACKUP_SOURCE to $RESTORE_TARGET"
rsync -avz --progress "$BACKUP_SOURCE/" "$RESTORE_TARGET/"

# Verify restore
SOURCE_COUNT=$(find "$BACKUP_SOURCE" -type f | wc -l)
TARGET_COUNT=$(find "$RESTORE_TARGET" -type f | wc -l)

if [[ $SOURCE_COUNT -eq $TARGET_COUNT ]]; then
    log "File restore verification passed ($TARGET_COUNT files restored)"
else
    log "WARNING: File count mismatch (Source: $SOURCE_COUNT, Target: $TARGET_COUNT)"
fi

# Set appropriate permissions
chown -R tracktion:tracktion "$RESTORE_TARGET"
chmod -R 755 "$RESTORE_TARGET"

# Cleanup temporary files
if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
fi

log "File system restore completed"
```

### Complete System Restore
```bash
#!/bin/bash
# complete_system_restore.sh

set -e

USAGE="Usage: $0 <restore_date> [environment]"

if [[ $# -lt 1 ]]; then
    echo "$USAGE"
    exit 1
fi

RESTORE_DATE="$1"
ENVIRONMENT="${2:-production}"
SCRIPT_DIR="/opt/tracktion/backup_scripts"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "/var/log/tracktion_restore.log"
}

log "Starting complete system restore for date: $RESTORE_DATE"

# Step 1: Stop all services
log "Stopping Tracktion services..."
docker-compose -f /opt/tracktion/docker-compose.yml down

# Step 2: Backup current state (safety measure)
log "Creating safety backup of current state..."
SAFETY_BACKUP_DIR="/backups/safety_$(date +%s)"
mkdir -p "$SAFETY_BACKUP_DIR"
cp -r /app/data "$SAFETY_BACKUP_DIR/"
pg_dump -U postgres tracktion > "$SAFETY_BACKUP_DIR/current_database.sql"

# Step 3: Restore database
log "Restoring database..."
DB_BACKUP=$(find /backups/database -name "full_backup_tracktion_${RESTORE_DATE}*.sql" | head -1)
if [[ -z "$DB_BACKUP" ]]; then
    log "ERROR: No database backup found for date: $RESTORE_DATE"
    exit 1
fi

# Drop and recreate database
dropdb -U postgres tracktion --if-exists
createdb -U postgres tracktion
"$SCRIPT_DIR/restore_database.sh" "$DB_BACKUP" tracktion

# Step 4: Restore file systems
log "Restoring file systems..."

# Restore audio files
AUDIO_BACKUP="/backups/audio_files/backup_${RESTORE_DATE}"
if [[ -d "$AUDIO_BACKUP" ]]; then
    "$SCRIPT_DIR/restore_files.sh" "$AUDIO_BACKUP" "/app/data/audio_files"
else
    log "WARNING: No audio files backup found for date: $RESTORE_DATE"
fi

# Restore user data
USER_BACKUP="/backups/user_data/backup_${RESTORE_DATE}"
if [[ -d "$USER_BACKUP" ]]; then
    "$SCRIPT_DIR/restore_files.sh" "$USER_BACKUP" "/app/data/user_uploads"
else
    log "WARNING: No user data backup found for date: $RESTORE_DATE"
fi

# Step 5: Restore configuration
log "Restoring configuration..."
CONFIG_BACKUP=$(find /backups/configuration -name "config_${RESTORE_DATE}*.tar.gz" | head -1)
if [[ -n "$CONFIG_BACKUP" ]]; then
    tar -xzf "$CONFIG_BACKUP" -C /tmp/config_restore/
    cp -r /tmp/config_restore/config/* /app/config/
    rm -rf /tmp/config_restore
else
    log "WARNING: No configuration backup found for date: $RESTORE_DATE"
fi

# Step 6: Start services
log "Starting Tracktion services..."
docker-compose -f /opt/tracktion/docker-compose.yml up -d

# Step 7: Verify system health
log "Verifying system health..."
sleep 30  # Wait for services to start

HEALTH_CHECK_PASSED=true
for service in analysis-service file-watcher tracklist-service notification-service; do
    if ! curl -f "http://localhost:8000/health" >/dev/null 2>&1; then
        log "ERROR: Health check failed for $service"
        HEALTH_CHECK_PASSED=false
    fi
done

if [[ $HEALTH_CHECK_PASSED == true ]]; then
    log "✅ Complete system restore completed successfully"

    # Send success notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"✅ System restore completed successfully for date: '"$RESTORE_DATE"'"}' \
        "$SLACK_WEBHOOK_URL" || true
else
    log "❌ System restore completed but health checks failed"
    log "Safety backup available at: $SAFETY_BACKUP_DIR"

    # Send failure notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"⚠️ System restore completed but health checks failed. Safety backup: '"$SAFETY_BACKUP_DIR"'"}' \
        "$SLACK_WEBHOOK_URL" || true
fi

log "System restore process completed"
```

## Backup Monitoring

### Monitoring Dashboard Metrics
```python
# backup_metrics.py
from prometheus_client import Counter, Gauge, Histogram
import time

# Backup operation metrics
BACKUP_OPERATIONS_TOTAL = Counter(
    'backup_operations_total',
    'Total backup operations',
    ['operation', 'status']
)

BACKUP_DURATION = Histogram(
    'backup_duration_seconds',
    'Backup operation duration',
    ['operation']
)

BACKUP_SIZE = Gauge(
    'backup_size_bytes',
    'Backup size in bytes',
    ['backup_type']
)

BACKUP_SUCCESS_TIME = Gauge(
    'backup_last_success_timestamp',
    'Timestamp of last successful backup',
    ['operation']
)

# Usage example
def track_backup_operation(operation_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                BACKUP_OPERATIONS_TOTAL.labels(operation=operation_name, status='success').inc()
                BACKUP_SUCCESS_TIME.labels(operation=operation_name).set(time.time())
                return result
            except Exception as e:
                BACKUP_OPERATIONS_TOTAL.labels(operation=operation_name, status='failure').inc()
                raise
            finally:
                BACKUP_DURATION.labels(operation=operation_name).observe(time.time() - start_time)
        return wrapper
    return decorator
```

### Backup Monitoring Alerts
```yaml
# backup_alerts.yml
groups:
  - name: backup_monitoring
    rules:
      - alert: BackupFailed
        expr: increase(backup_operations_total{status="failure"}[1h]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Backup operation failed"
          description: "Backup operation {{ $labels.operation }} has failed"

      - alert: BackupOverdue
        expr: time() - backup_last_success_timestamp > 86400
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Backup is overdue"
          description: "{{ $labels.operation }} backup is overdue by {{ $value | humanizeDuration }}"

      - alert: BackupSizeAnomaly
        expr: |
          (
            backup_size_bytes > 1.5 * avg_over_time(backup_size_bytes[7d])
            or
            backup_size_bytes < 0.5 * avg_over_time(backup_size_bytes[7d])
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Backup size anomaly detected"
          description: "{{ $labels.backup_type }} backup size is {{ $value | humanizeBytes }}, which is significantly different from the 7-day average"
```

## Storage Management

### S3 Lifecycle Policies
```json
{
  "Rules": [
    {
      "ID": "DatabaseBackupLifecycle",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "database/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ],
      "Expiration": {
        "Days": 2555  // 7 years
      }
    },
    {
      "ID": "AudioFilesLifecycle",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "audio_files/"
      },
      "Transitions": [
        {
          "Days": 7,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 2190  // 6 years
      }
    },
    {
      "ID": "LogsLifecycle",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "logs/"
      },
      "Transitions": [
        {
          "Days": 1,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 2555  // 7 years for compliance
      }
    }
  ]
}
```

### Backup Storage Cost Optimization
```bash
#!/bin/bash
# optimize_backup_storage.sh

set -e

S3_BUCKET="tracktion-backups"
DATE=$(date +%Y%m%d)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting backup storage optimization..."

# Analyze storage usage
aws s3 ls "s3://$S3_BUCKET" --recursive --summarize | tail -2

# Move old database backups to cheaper storage class
log "Moving old database backups to Glacier..."
aws s3 ls "s3://$S3_BUCKET/database/" | while read -r line; do
    DATE_PART=$(echo "$line" | awk '{print $1}')
    FILE_NAME=$(echo "$line" | awk '{print $4}')

    # Calculate age in days
    FILE_DATE=$(date -d "$DATE_PART" +%s)
    CURRENT_DATE=$(date +%s)
    AGE_DAYS=$(( (CURRENT_DATE - FILE_DATE) / 86400 ))

    if [[ $AGE_DAYS -gt 30 ]]; then
        aws s3 cp "s3://$S3_BUCKET/database/$FILE_NAME" "s3://$S3_BUCKET/database/$FILE_NAME" \
            --storage-class GLACIER
        log "Moved $FILE_NAME to Glacier (age: $AGE_DAYS days)"
    fi
done

# Compress and deduplicate backups
log "Running deduplication analysis..."

# Generate storage report
aws s3api list-objects-v2 --bucket "$S3_BUCKET" --query 'Contents[?StorageClass==`STANDARD`].[Key,Size,StorageClass]' --output table

log "Storage optimization completed"
```

## Security Considerations

### Backup Encryption
```bash
# Encrypt backup before uploading to S3
encrypt_backup() {
    local input_file="$1"
    local output_file="${input_file}.enc"
    local key_file="/etc/tracktion/backup.key"

    # Generate random key if it doesn't exist
    if [[ ! -f "$key_file" ]]; then
        openssl rand -base64 32 > "$key_file"
        chmod 600 "$key_file"
    fi

    # Encrypt file
    openssl enc -aes-256-cbc -salt -in "$input_file" -out "$output_file" -pass file:"$key_file"

    # Remove unencrypted file
    shred -vfz -n 3 "$input_file"

    echo "$output_file"
}

# Decrypt backup
decrypt_backup() {
    local input_file="$1"
    local output_file="${input_file%.enc}"
    local key_file="/etc/tracktion/backup.key"

    # Decrypt file
    openssl enc -aes-256-cbc -d -in "$input_file" -out "$output_file" -pass file:"$key_file"

    echo "$output_file"
}
```

### Access Control
```bash
# Set up IAM policy for backup user
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::tracktion-backups/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::tracktion-backups"
    }
  ]
}
```

## Testing and Validation

### Backup Testing Schedule
```bash
# Monthly backup restoration test
#!/bin/bash
# test_backup_restore.sh

set -e

TEST_ENV="backup_test"
DATE=$(date +%Y%m%d)
LOG_FILE="/var/log/backup_test_$DATE.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting monthly backup restoration test..."

# Test database restore
log "Testing database backup restore..."
LATEST_BACKUP=$(find /backups/database -name "full_backup_*.sql" -type f | sort | tail -1)
TEST_DB="test_restore_$(date +%s)"

createdb -U postgres "$TEST_DB"
if pg_restore -U postgres -d "$TEST_DB" "$LATEST_BACKUP"; then
    log "✅ Database restore test passed"
    dropdb -U postgres "$TEST_DB"
else
    log "❌ Database restore test failed"
    dropdb -U postgres "$TEST_DB" 2>/dev/null || true
fi

# Test file system restore
log "Testing file system restore..."
TEST_DIR="/tmp/file_restore_test_$(date +%s)"
mkdir -p "$TEST_DIR"

if rsync -av /backups/audio_files/current/ "$TEST_DIR/"; then
    FILE_COUNT=$(find "$TEST_DIR" -type f | wc -l)
    if [[ $FILE_COUNT -gt 0 ]]; then
        log "✅ File system restore test passed ($FILE_COUNT files)"
    else
        log "❌ File system restore test failed (no files found)"
    fi
else
    log "❌ File system restore test failed"
fi

# Cleanup test directory
rm -rf "$TEST_DIR"

# Generate test report
log "Backup restoration test completed"
```

## Troubleshooting

### Common Issues and Solutions

#### Backup Script Failures
```bash
# Debug backup script issues
#!/bin/bash

# Check disk space
df -h /backups

# Check database connectivity
pg_isready -h localhost -p 5432

# Check S3 credentials and access
aws s3 ls s3://tracktion-backups/

# Check log files for errors
tail -n 50 /var/log/tracktion_backups/master_backup_*.log

# Test backup script manually
/opt/tracktion/backup_scripts/full_db_backup.sh
```

#### Restore Failures
```bash
# Troubleshoot restore issues

# Check backup file integrity
sha256sum -c backup_file.sha256

# Verify PostgreSQL service
systemctl status postgresql

# Check database permissions
psql -U postgres -c "\du"

# Test network connectivity for remote restores
ping database_server
telnet database_server 5432
```

#### Storage Issues
```bash
# Resolve storage problems

# Check S3 bucket permissions
aws s3api get-bucket-acl --bucket tracktion-backups

# Verify IAM credentials
aws sts get-caller-identity

# Check local storage space
du -sh /backups/*
find /backups -type f -size +1G

# Clean up old backups manually if needed
find /backups -name "*" -mtime +35 -delete
```

This comprehensive backup procedures document provides detailed guidance for protecting all aspects of the Tracktion system, from databases to configuration files, with automated scripts, monitoring, and recovery procedures to ensure business continuity and data protection.
