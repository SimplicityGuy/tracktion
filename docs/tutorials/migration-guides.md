# Migration Guides

This document provides comprehensive migration guides for upgrading Tracktion services, including database migrations, configuration changes, API updates, and data migration strategies.

## Table of Contents

1. [Database Schema Migrations](#database-schema-migrations)
2. [Configuration Migration](#configuration-migration)
3. [API Version Migration](#api-version-migration)
4. [Data Format Migration](#data-format-migration)
5. [Service Architecture Migration](#service-architecture-migration)
6. [Storage Migration](#storage-migration)
7. [Authentication System Migration](#authentication-system-migration)
8. [Deployment Migration](#deployment-migration)
9. [Rollback Strategies](#rollback-strategies)
10. [Migration Testing](#migration-testing)

## Database Schema Migrations

### 1. Alembic Migration System

```python
"""
Alembic migration example for Tracktion database schema changes
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# Migration: Add enhanced track analysis fields
# Revision ID: 2024_01_15_001
# Revises: 2023_12_20_001

def upgrade():
    """Upgrade to new schema with enhanced analysis fields."""

    # Add new columns to tracks table
    op.add_column('tracks',
        sa.Column('energy_level', sa.Float, nullable=True, comment='Energy level 0.0-1.0'))
    op.add_column('tracks',
        sa.Column('danceability', sa.Float, nullable=True, comment='Danceability score 0.0-1.0'))
    op.add_column('tracks',
        sa.Column('valence', sa.Float, nullable=True, comment='Musical positivity 0.0-1.0'))
    op.add_column('tracks',
        sa.Column('acousticness', sa.Float, nullable=True, comment='Acoustic confidence 0.0-1.0'))
    op.add_column('tracks',
        sa.Column('instrumentalness', sa.Float, nullable=True, comment='Instrumental confidence 0.0-1.0'))
    op.add_column('tracks',
        sa.Column('analysis_version', sa.String(20), nullable=True, comment='Analysis algorithm version'))
    op.add_column('tracks',
        sa.Column('analysis_metadata', postgresql.JSONB, nullable=True, comment='Additional analysis data'))

    # Create new analysis_history table
    op.create_table('analysis_history',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('track_id', sa.Integer(), sa.ForeignKey('tracks.id'), nullable=False),
        sa.Column('analysis_type', sa.String(50), nullable=False),
        sa.Column('analysis_version', sa.String(20), nullable=False),
        sa.Column('results', postgresql.JSONB, nullable=False),
        sa.Column('confidence_score', sa.Float, nullable=True),
        sa.Column('processing_time_ms', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('created_by', sa.String(100), nullable=True),
    )

    # Create indexes for performance
    op.create_index('ix_tracks_energy_level', 'tracks', ['energy_level'])
    op.create_index('ix_tracks_danceability', 'tracks', ['danceability'])
    op.create_index('ix_tracks_analysis_version', 'tracks', ['analysis_version'])
    op.create_index('ix_analysis_history_track_id', 'analysis_history', ['track_id'])
    op.create_index('ix_analysis_history_type_version', 'analysis_history', ['analysis_type', 'analysis_version'])

    # Create composite index for common queries
    op.create_index('ix_tracks_energy_dance_valence', 'tracks',
                   ['energy_level', 'danceability', 'valence'])

    # Add constraints
    op.create_check_constraint(
        'ck_tracks_energy_range', 'tracks',
        sa.and_(sa.column('energy_level') >= 0.0, sa.column('energy_level') <= 1.0)
    )
    op.create_check_constraint(
        'ck_tracks_danceability_range', 'tracks',
        sa.and_(sa.column('danceability') >= 0.0, sa.column('danceability') <= 1.0)
    )

def downgrade():
    """Downgrade to previous schema."""

    # Drop new table
    op.drop_table('analysis_history')

    # Drop indexes
    op.drop_index('ix_tracks_energy_dance_valence', 'tracks')
    op.drop_index('ix_tracks_analysis_version', 'tracks')
    op.drop_index('ix_tracks_danceability', 'tracks')
    op.drop_index('ix_tracks_energy_level', 'tracks')

    # Drop new columns
    op.drop_column('tracks', 'analysis_metadata')
    op.drop_column('tracks', 'analysis_version')
    op.drop_column('tracks', 'instrumentalness')
    op.drop_column('tracks', 'acousticness')
    op.drop_column('tracks', 'valence')
    op.drop_column('tracks', 'danceability')
    op.drop_column('tracks', 'energy_level')

class DatabaseMigrationManager:
    """Manage database migrations with safety checks."""

    def __init__(self, engine, backup_enabled: bool = True):
        self.engine = engine
        self.backup_enabled = backup_enabled

    async def migrate_with_safety(self, target_revision: str = 'head'):
        """Execute migration with safety checks and backup."""

        # 1. Pre-migration backup
        if self.backup_enabled:
            backup_file = await self.create_backup()
            print(f"Database backed up to: {backup_file}")

        # 2. Validate migration
        await self.validate_migration(target_revision)

        # 3. Check for blocking conditions
        await self.check_blocking_conditions()

        # 4. Execute migration with monitoring
        try:
            migration_start = datetime.now()

            # Run Alembic migration
            from alembic.command import upgrade
            from alembic.config import Config

            alembic_cfg = Config("alembic.ini")
            upgrade(alembic_cfg, target_revision)

            migration_duration = datetime.now() - migration_start
            print(f"Migration completed in {migration_duration.total_seconds():.2f} seconds")

            # 5. Post-migration validation
            await self.validate_post_migration()

        except Exception as e:
            print(f"Migration failed: {e}")

            # Attempt rollback if backup exists
            if self.backup_enabled:
                print("Attempting to restore from backup...")
                await self.restore_from_backup(backup_file)

            raise

    async def create_backup(self) -> str:
        """Create database backup before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"tracktion_backup_{timestamp}.sql"

        # Use pg_dump for PostgreSQL
        import subprocess

        cmd = [
            "pg_dump",
            "--host", os.getenv("DB_HOST"),
            "--port", os.getenv("DB_PORT", "5432"),
            "--username", os.getenv("DB_USER"),
            "--dbname", os.getenv("DB_NAME"),
            "--file", backup_file,
            "--verbose",
            "--no-password"
        ]

        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = os.getenv("DB_PASSWORD")

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Backup failed: {result.stderr}")

        return backup_file

    async def validate_migration(self, target_revision: str):
        """Validate migration before execution."""
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from alembic.config import Config

        # Check current revision
        with self.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_rev = context.get_current_revision()

        # Check target revision exists
        alembic_cfg = Config("alembic.ini")
        script_dir = ScriptDirectory.from_config(alembic_cfg)

        if target_revision != 'head':
            try:
                script_dir.get_revision(target_revision)
            except Exception:
                raise ValueError(f"Target revision {target_revision} not found")

        print(f"Migration path: {current_rev} -> {target_revision}")

    async def check_blocking_conditions(self):
        """Check for conditions that would block migration."""

        # Check for active connections
        active_connections = await self.get_active_connections()
        if active_connections > 5:  # Allow some connections
            print(f"Warning: {active_connections} active connections detected")

        # Check available disk space
        disk_usage = await self.check_disk_space()
        if disk_usage > 0.9:  # 90% full
            raise Exception(f"Insufficient disk space: {disk_usage:.1%} used")

        # Check for long-running transactions
        long_transactions = await self.check_long_transactions()
        if long_transactions:
            raise Exception(f"Long-running transactions detected: {long_transactions}")

    async def validate_post_migration(self):
        """Validate database state after migration."""

        # Check table existence and structure
        await self.verify_table_structure()

        # Run data integrity checks
        await self.verify_data_integrity()

        # Check index performance
        await self.verify_index_performance()

        print("Post-migration validation successful")
```

### 2. Data Migration Scripts

```python
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class DataMigrator:
    """Handle data migration between schema versions."""

    def __init__(self, source_engine, target_engine=None):
        self.source_engine = source_engine
        self.target_engine = target_engine or source_engine
        self.logger = logging.getLogger("data_migrator")

    async def migrate_track_analysis_data(self, batch_size: int = 1000):
        """Migrate track analysis data to new format."""

        self.logger.info("Starting track analysis data migration")

        # Get total count for progress tracking
        async with self.source_engine.begin() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM tracks WHERE analysis_data IS NOT NULL")
            total_tracks = result.scalar()

        self.logger.info(f"Found {total_tracks} tracks with analysis data to migrate")

        processed = 0
        errors = []

        # Process in batches
        for offset in range(0, total_tracks, batch_size):
            try:
                batch_result = await self._migrate_analysis_batch(offset, batch_size)
                processed += batch_result['processed']
                errors.extend(batch_result['errors'])

                # Progress logging
                progress = (processed / total_tracks) * 100
                self.logger.info(f"Migration progress: {progress:.1f}% ({processed}/{total_tracks})")

            except Exception as e:
                self.logger.error(f"Batch migration failed at offset {offset}: {e}")
                errors.append({'offset': offset, 'error': str(e)})

        # Summary
        self.logger.info(f"Migration completed: {processed} tracks processed, {len(errors)} errors")

        if errors:
            self.logger.error(f"Migration errors: {json.dumps(errors, indent=2)}")

        return {'processed': processed, 'errors': errors}

    async def _migrate_analysis_batch(self, offset: int, batch_size: int) -> Dict[str, Any]:
        """Migrate a batch of track analysis data."""

        batch_processed = 0
        batch_errors = []

        async with self.source_engine.begin() as conn:
            # Fetch batch of tracks with old analysis format
            query = """
            SELECT id, file_path, analysis_data, created_at
            FROM tracks
            WHERE analysis_data IS NOT NULL
            ORDER BY id
            LIMIT :limit OFFSET :offset
            """

            result = await conn.execute(query, {'limit': batch_size, 'offset': offset})
            tracks = result.fetchall()

            for track in tracks:
                try:
                    # Convert old analysis format to new format
                    old_analysis = json.loads(track['analysis_data'])
                    new_analysis = await self._convert_analysis_format(old_analysis)

                    # Update track with new fields
                    update_query = """
                    UPDATE tracks SET
                        energy_level = :energy_level,
                        danceability = :danceability,
                        valence = :valence,
                        acousticness = :acousticness,
                        instrumentalness = :instrumentalness,
                        analysis_version = :analysis_version,
                        analysis_metadata = :analysis_metadata
                    WHERE id = :track_id
                    """

                    await conn.execute(update_query, {
                        'track_id': track['id'],
                        'energy_level': new_analysis['energy_level'],
                        'danceability': new_analysis['danceability'],
                        'valence': new_analysis['valence'],
                        'acousticness': new_analysis['acousticness'],
                        'instrumentalness': new_analysis['instrumentalness'],
                        'analysis_version': new_analysis['analysis_version'],
                        'analysis_metadata': json.dumps(new_analysis['metadata'])
                    })

                    # Insert into analysis history
                    history_query = """
                    INSERT INTO analysis_history
                    (track_id, analysis_type, analysis_version, results, confidence_score, created_at)
                    VALUES (:track_id, :analysis_type, :analysis_version, :results, :confidence_score, :created_at)
                    """

                    await conn.execute(history_query, {
                        'track_id': track['id'],
                        'analysis_type': 'full_analysis',
                        'analysis_version': new_analysis['analysis_version'],
                        'results': json.dumps(new_analysis),
                        'confidence_score': new_analysis.get('confidence_score', 0.8),
                        'created_at': track['created_at']
                    })

                    batch_processed += 1

                except Exception as e:
                    batch_errors.append({
                        'track_id': track['id'],
                        'file_path': track['file_path'],
                        'error': str(e)
                    })

        return {'processed': batch_processed, 'errors': batch_errors}

    async def _convert_analysis_format(self, old_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old analysis format to new format."""

        # Map old fields to new fields with defaults
        new_analysis = {
            'energy_level': old_analysis.get('energy', 0.5),
            'danceability': old_analysis.get('danceability', 0.5),
            'valence': old_analysis.get('mood_positivity', 0.5),
            'acousticness': old_analysis.get('acoustic_probability', 0.5),
            'instrumentalness': old_analysis.get('instrumental_probability', 0.1),
            'analysis_version': '2024.1',
            'confidence_score': old_analysis.get('overall_confidence', 0.8),
            'metadata': {
                'migration_date': datetime.now().isoformat(),
                'original_version': old_analysis.get('version', 'legacy'),
                'migration_notes': 'Migrated from legacy format'
            }
        }

        # Preserve additional legacy data
        if 'tempo_confidence' in old_analysis:
            new_analysis['metadata']['tempo_confidence'] = old_analysis['tempo_confidence']

        if 'key_confidence' in old_analysis:
            new_analysis['metadata']['key_confidence'] = old_analysis['key_confidence']

        # Validate ranges
        for field in ['energy_level', 'danceability', 'valence', 'acousticness', 'instrumentalness']:
            value = new_analysis[field]
            new_analysis[field] = max(0.0, min(1.0, float(value)))

        return new_analysis

# Migration utility functions
async def run_migration_safely(migration_func, description: str):
    """Run migration function with error handling and logging."""

    logger = logging.getLogger("migration")
    logger.info(f"Starting migration: {description}")

    start_time = datetime.now()

    try:
        result = await migration_func()

        duration = datetime.now() - start_time
        logger.info(f"Migration '{description}' completed successfully in {duration.total_seconds():.2f}s")

        return result

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"Migration '{description}' failed after {duration.total_seconds():.2f}s: {e}")
        raise

async def verify_migration_integrity():
    """Verify migration completed successfully."""

    checks = []

    # Check new columns exist and have data
    async with engine.begin() as conn:
        # Check for tracks with new analysis fields
        result = await conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(energy_level) as with_energy,
                   COUNT(analysis_version) as with_version
            FROM tracks
        """)

        stats = result.fetchone()
        checks.append({
            'check': 'new_fields_populated',
            'total_tracks': stats['total'],
            'with_energy': stats['with_energy'],
            'with_version': stats['with_version'],
            'success': stats['with_energy'] > 0 and stats['with_version'] > 0
        })

        # Check analysis history has entries
        result = await conn.execute("SELECT COUNT(*) FROM analysis_history")
        history_count = result.scalar()

        checks.append({
            'check': 'analysis_history_populated',
            'count': history_count,
            'success': history_count > 0
        })

        # Check data integrity
        result = await conn.execute("""
            SELECT COUNT(*) FROM tracks
            WHERE energy_level < 0 OR energy_level > 1
               OR danceability < 0 OR danceability > 1
        """)

        invalid_data = result.scalar()
        checks.append({
            'check': 'data_range_validation',
            'invalid_records': invalid_data,
            'success': invalid_data == 0
        })

    # Summary
    all_passed = all(check['success'] for check in checks)

    logger = logging.getLogger("migration_verify")
    logger.info(f"Migration integrity check: {'PASSED' if all_passed else 'FAILED'}")

    for check in checks:
        status = 'PASS' if check['success'] else 'FAIL'
        logger.info(f"  {check['check']}: {status} - {check}")

    return all_passed
```

## Configuration Migration

### 1. Configuration Version Management

```python
import yaml
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ConfigVersion(Enum):
    """Configuration versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    V2_1 = "2.1"

@dataclass
class MigrationStep:
    """Configuration migration step."""
    from_version: ConfigVersion
    to_version: ConfigVersion
    migration_func: callable
    description: str
    required: bool = True

class ConfigurationMigrator:
    """Migrate configuration files between versions."""

    def __init__(self):
        self.migration_steps = self._define_migration_steps()
        self.current_version = ConfigVersion.V2_1

    def _define_migration_steps(self) -> List[MigrationStep]:
        """Define all migration steps."""
        return [
            MigrationStep(
                ConfigVersion.V1_0,
                ConfigVersion.V1_1,
                self._migrate_v1_0_to_v1_1,
                "Add authentication settings"
            ),
            MigrationStep(
                ConfigVersion.V1_1,
                ConfigVersion.V2_0,
                self._migrate_v1_1_to_v2_0,
                "Restructure service configuration"
            ),
            MigrationStep(
                ConfigVersion.V2_0,
                ConfigVersion.V2_1,
                self._migrate_v2_0_to_v2_1,
                "Add security enhancements"
            ),
        ]

    def migrate_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Migrate configuration file to latest version."""

        # Load current configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Detect current version
        current_version = self._detect_config_version(config)

        if current_version == self.current_version:
            print(f"Configuration is already at latest version {current_version.value}")
            return config

        print(f"Migrating configuration from {current_version.value} to {self.current_version.value}")

        # Create backup
        backup_path = config_path.with_suffix(f'.{current_version.value}.backup')
        with open(backup_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)

        print(f"Backup created: {backup_path}")

        # Apply migration steps
        migrated_config = config.copy()

        for step in self.migration_steps:
            if step.from_version.value >= current_version.value:
                print(f"Applying migration: {step.description}")
                try:
                    migrated_config = step.migration_func(migrated_config)
                    migrated_config['version'] = step.to_version.value
                except Exception as e:
                    print(f"Migration step failed: {e}")
                    if step.required:
                        raise

        # Save migrated configuration
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(migrated_config, f, default_flow_style=False)
            else:
                json.dump(migrated_config, f, indent=2)

        print(f"Configuration migrated successfully to {self.current_version.value}")
        return migrated_config

    def _detect_config_version(self, config: Dict[str, Any]) -> ConfigVersion:
        """Detect configuration version."""

        if 'version' in config:
            try:
                return ConfigVersion(config['version'])
            except ValueError:
                pass

        # Detect based on structure
        if 'auth' in config and 'jwt_settings' in config['auth']:
            return ConfigVersion.V2_1
        elif 'services' in config and isinstance(config['services'], dict):
            return ConfigVersion.V2_0
        elif 'authentication' in config:
            return ConfigVersion.V1_1
        else:
            return ConfigVersion.V1_0

    def _migrate_v1_0_to_v1_1(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1.0 to v1.1 - Add authentication settings."""

        new_config = config.copy()

        # Add authentication section
        new_config['authentication'] = {
            'enabled': True,
            'jwt_secret': '${JWT_SECRET_KEY}',
            'jwt_expiration': 3600,
            'password_min_length': 8,
            'max_login_attempts': 5
        }

        # Move database settings to new structure
        if 'database_url' in config:
            new_config['database'] = {
                'url': config['database_url'],
                'pool_size': 10,
                'max_overflow': 20
            }
            del new_config['database_url']

        return new_config

    def _migrate_v1_1_to_v2_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1.1 to v2.0 - Restructure service configuration."""

        new_config = {
            'version': '2.0',
            'services': {
                'analysis_service': {
                    'enabled': True,
                    'port': config.get('analysis_port', 8001),
                    'workers': config.get('analysis_workers', 4)
                },
                'tracklist_service': {
                    'enabled': True,
                    'port': config.get('tracklist_port', 8002),
                    'workers': config.get('tracklist_workers', 2)
                },
                'file_watcher': {
                    'enabled': True,
                    'watch_paths': config.get('watch_directories', ['/music'])
                }
            },
            'database': config.get('database', {}),
            'authentication': config.get('authentication', {}),
            'logging': {
                'level': config.get('log_level', 'INFO'),
                'format': 'structured',
                'outputs': ['console', 'file']
            }
        }

        # Preserve any custom settings
        for key, value in config.items():
            if key not in new_config and not key.endswith('_port') and not key.endswith('_workers'):
                new_config[key] = value

        return new_config

    def _migrate_v2_0_to_v2_1(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v2.0 to v2.1 - Add security enhancements."""

        new_config = config.copy()

        # Enhanced authentication settings
        if 'authentication' in new_config:
            auth_config = new_config['authentication']

            # Add MFA support
            auth_config['mfa'] = {
                'enabled': False,
                'issuer_name': 'Tracktion',
                'backup_codes_count': 10
            }

            # Add password policy
            auth_config['password_policy'] = {
                'min_length': auth_config.get('password_min_length', 8),
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special': True
            }

            # Add rate limiting
            auth_config['rate_limiting'] = {
                'login_attempts': auth_config.get('max_login_attempts', 5),
                'lockout_duration': 900,  # 15 minutes
                'requests_per_minute': 60
            }

        # Add security headers configuration
        new_config['security'] = {
            'cors': {
                'enabled': True,
                'allowed_origins': ['https://app.tracktion.local'],
                'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
                'allow_credentials': True
            },
            'headers': {
                'csp_policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
                'hsts_max_age': 31536000,
                'frame_options': 'DENY'
            },
            'encryption': {
                'field_encryption_enabled': True,
                'master_key': '${ENCRYPTION_MASTER_KEY}'
            }
        }

        # Add monitoring configuration
        new_config['monitoring'] = {
            'metrics': {
                'enabled': True,
                'endpoint': '/metrics',
                'include_system_metrics': True
            },
            'health_checks': {
                'enabled': True,
                'endpoint': '/health',
                'detailed': False
            },
            'audit_logging': {
                'enabled': True,
                'include_requests': True,
                'include_responses': False
            }
        }

        return new_config

# Usage example
def migrate_all_configs():
    """Migrate all configuration files."""

    migrator = ConfigurationMigrator()
    config_files = [
        Path('config/development.yaml'),
        Path('config/production.yaml'),
        Path('config/testing.yaml'),
        Path('services/analysis_service/config.yaml'),
        Path('services/tracklist_service/config.yaml'),
    ]

    results = []

    for config_file in config_files:
        if config_file.exists():
            try:
                print(f"\nMigrating {config_file}")
                result = migrator.migrate_config_file(config_file)
                results.append({'file': str(config_file), 'success': True})
                print(f"Successfully migrated {config_file}")
            except Exception as e:
                print(f"Failed to migrate {config_file}: {e}")
                results.append({'file': str(config_file), 'success': False, 'error': str(e)})
        else:
            print(f"Configuration file not found: {config_file}")
            results.append({'file': str(config_file), 'success': False, 'error': 'File not found'})

    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nMigration Summary: {successful}/{total} files migrated successfully")

    return results
```

### 2. Environment Variable Migration

```python
import os
from typing import Dict, Set, List, Tuple
from pathlib import Path

class EnvironmentMigrator:
    """Migrate environment variables between versions."""

    def __init__(self):
        self.migrations = {
            'v1_to_v2': self._migrate_v1_to_v2,
            'v2_to_v2_1': self._migrate_v2_to_v2_1,
        }

    def migrate_env_file(self, env_file_path: Path, target_version: str = 'v2_1'):
        """Migrate .env file to target version."""

        # Load current environment variables
        env_vars = self._load_env_file(env_file_path)

        # Detect current version
        current_version = self._detect_env_version(env_vars)

        print(f"Current version: {current_version}, Target: {target_version}")

        # Create backup
        backup_path = env_file_path.with_suffix(f'.{current_version}.backup')
        with open(backup_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        # Apply migrations
        migrated_vars = env_vars.copy()

        if current_version == 'v1' and target_version in ['v2', 'v2_1']:
            migrated_vars = self.migrations['v1_to_v2'](migrated_vars)

        if target_version == 'v2_1':
            migrated_vars = self.migrations['v2_to_v2_1'](migrated_vars)

        # Write migrated file
        with open(env_file_path, 'w') as f:
            # Add version comment
            f.write(f"# Tracktion Environment Configuration - Version {target_version}\n")
            f.write(f"# Migrated on {datetime.now().isoformat()}\n\n")

            # Group variables by category
            categories = {
                'Database': ['DATABASE_URL', 'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'],
                'Redis': ['REDIS_URL', 'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD'],
                'RabbitMQ': ['RABBITMQ_URL', 'RABBITMQ_HOST', 'RABBITMQ_PORT', 'RABBITMQ_USER', 'RABBITMQ_PASSWORD'],
                'Authentication': ['JWT_SECRET_KEY', 'JWT_EXPIRATION', 'ENCRYPTION_MASTER_KEY'],
                'Services': ['ANALYSIS_SERVICE_URL', 'TRACKLIST_SERVICE_URL', 'FILE_WATCHER_ENABLED'],
                'Security': ['CORS_ORIGINS', 'RATE_LIMIT_ENABLED', 'MFA_ENABLED'],
                'Monitoring': ['METRICS_ENABLED', 'AUDIT_LOG_ENABLED', 'LOG_LEVEL'],
            }

            written_vars = set()

            for category, var_names in categories.items():
                category_vars = [(k, v) for k, v in migrated_vars.items() if k in var_names]

                if category_vars:
                    f.write(f"# {category}\n")
                    for key, value in sorted(category_vars):
                        f.write(f"{key}={value}\n")
                        written_vars.add(key)
                    f.write("\n")

            # Write remaining variables
            remaining_vars = [(k, v) for k, v in migrated_vars.items() if k not in written_vars]
            if remaining_vars:
                f.write("# Other Configuration\n")
                for key, value in sorted(remaining_vars):
                    f.write(f"{key}={value}\n")

        print(f"Environment file migrated successfully to {target_version}")
        return migrated_vars

    def _load_env_file(self, env_file_path: Path) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}

        if not env_file_path.exists():
            return env_vars

        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        return env_vars

    def _detect_env_version(self, env_vars: Dict[str, str]) -> str:
        """Detect environment configuration version."""

        # v2.1 indicators
        if 'ENCRYPTION_MASTER_KEY' in env_vars or 'MFA_ENABLED' in env_vars:
            return 'v2_1'

        # v2 indicators
        if 'ANALYSIS_SERVICE_URL' in env_vars or 'TRACKLIST_SERVICE_URL' in env_vars:
            return 'v2'

        # Default to v1
        return 'v1'

    def _migrate_v1_to_v2(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Migrate v1 to v2 environment variables."""

        migrated = env_vars.copy()

        # Service URLs (new in v2)
        migrated['ANALYSIS_SERVICE_URL'] = env_vars.get('ANALYSIS_SERVICE_URL', 'http://localhost:8001')
        migrated['TRACKLIST_SERVICE_URL'] = env_vars.get('TRACKLIST_SERVICE_URL', 'http://localhost:8002')

        # File watcher (new in v2)
        migrated['FILE_WATCHER_ENABLED'] = env_vars.get('FILE_WATCHER_ENABLED', 'true')
        migrated['WATCH_DIRECTORIES'] = env_vars.get('WATCH_DIRECTORIES', '/music,/audio')

        # Enhanced database settings
        if 'DATABASE_URL' not in env_vars:
            # Build from individual components
            db_host = env_vars.get('DB_HOST', 'localhost')
            db_port = env_vars.get('DB_PORT', '5432')
            db_name = env_vars.get('DB_NAME', 'tracktion')
            db_user = env_vars.get('DB_USER', 'tracktion')
            db_password = env_vars.get('DB_PASSWORD', 'password')

            migrated['DATABASE_URL'] = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

        # Redis settings
        if 'REDIS_URL' not in env_vars:
            redis_host = env_vars.get('REDIS_HOST', 'localhost')
            redis_port = env_vars.get('REDIS_PORT', '6379')
            redis_password = env_vars.get('REDIS_PASSWORD', '')

            if redis_password:
                migrated['REDIS_URL'] = f'redis://:{redis_password}@{redis_host}:{redis_port}/0'
            else:
                migrated['REDIS_URL'] = f'redis://{redis_host}:{redis_port}/0'

        # RabbitMQ settings
        if 'RABBITMQ_URL' not in env_vars:
            rabbit_host = env_vars.get('RABBITMQ_HOST', 'localhost')
            rabbit_port = env_vars.get('RABBITMQ_PORT', '5672')
            rabbit_user = env_vars.get('RABBITMQ_USER', 'guest')
            rabbit_password = env_vars.get('RABBITMQ_PASSWORD', 'guest')

            migrated['RABBITMQ_URL'] = f'amqp://{rabbit_user}:{rabbit_password}@{rabbit_host}:{rabbit_port}/'

        return migrated

    def _migrate_v2_to_v2_1(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Migrate v2 to v2.1 environment variables."""

        migrated = env_vars.copy()

        # Security enhancements
        migrated['ENCRYPTION_MASTER_KEY'] = env_vars.get('ENCRYPTION_MASTER_KEY', '${GENERATE_RANDOM_KEY}')
        migrated['MFA_ENABLED'] = env_vars.get('MFA_ENABLED', 'false')
        migrated['RATE_LIMIT_ENABLED'] = env_vars.get('RATE_LIMIT_ENABLED', 'true')

        # CORS configuration
        migrated['CORS_ORIGINS'] = env_vars.get('CORS_ORIGINS', 'https://app.tracktion.local')
        migrated['CORS_CREDENTIALS'] = env_vars.get('CORS_CREDENTIALS', 'true')

        # Enhanced JWT settings
        migrated['JWT_EXPIRATION'] = env_vars.get('JWT_EXPIRATION', '3600')
        migrated['JWT_REFRESH_EXPIRATION'] = env_vars.get('JWT_REFRESH_EXPIRATION', '604800')

        # Monitoring and logging
        migrated['METRICS_ENABLED'] = env_vars.get('METRICS_ENABLED', 'true')
        migrated['AUDIT_LOG_ENABLED'] = env_vars.get('AUDIT_LOG_ENABLED', 'true')
        migrated['STRUCTURED_LOGGING'] = env_vars.get('STRUCTURED_LOGGING', 'true')

        # Performance settings
        migrated['DB_POOL_SIZE'] = env_vars.get('DB_POOL_SIZE', '20')
        migrated['DB_MAX_OVERFLOW'] = env_vars.get('DB_MAX_OVERFLOW', '30')
        migrated['REDIS_POOL_SIZE'] = env_vars.get('REDIS_POOL_SIZE', '10')

        return migrated

# Usage
def migrate_environment_configs():
    """Migrate all environment configuration files."""

    migrator = EnvironmentMigrator()

    env_files = [
        Path('.env'),
        Path('.env.development'),
        Path('.env.production'),
        Path('.env.testing'),
    ]

    for env_file in env_files:
        if env_file.exists():
            print(f"\nMigrating {env_file}")
            try:
                migrator.migrate_env_file(env_file, target_version='v2_1')
                print(f"Successfully migrated {env_file}")
            except Exception as e:
                print(f"Failed to migrate {env_file}: {e}")
        else:
            print(f"Environment file not found: {env_file}")
```

This migration guide provides comprehensive tools and strategies for safely upgrading Tracktion services, including database schema changes, configuration updates, and data format migrations. Each migration includes safety checks, backups, and rollback capabilities to ensure system stability during upgrades.
