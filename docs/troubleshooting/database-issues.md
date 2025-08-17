# Database Troubleshooting Guide

Common database issues and their solutions for the Tracktion system.

## Connection Issues

### PostgreSQL Connection Refused

**Error**: `psycopg2.OperationalError: could not connect to server: Connection refused`

**Causes & Solutions**:

1. **Container not running**:
```bash
# Check if postgres is running
docker ps | grep postgres

# Start if not running
docker-compose -f infrastructure/docker-compose.yaml up -d postgres

# Check logs
docker-compose -f infrastructure/docker-compose.yaml logs postgres
```

2. **Wrong connection string**:
```bash
# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Should be:
DATABASE_URL=postgresql://tracktion:tracktion@localhost:5432/tracktion
```

3. **Port conflict**:
```bash
# Check if port 5432 is in use
lsof -i :5432

# Change port in docker-compose.yaml and .env if needed
```

### Neo4j Connection Failed

**Error**: `neo4j.exceptions.ServiceUnavailable: Unable to connect to localhost:7687`

**Solutions**:

1. **Container not running**:
```bash
# Check status
docker ps | grep neo4j

# Start Neo4j
docker-compose -f infrastructure/docker-compose.yaml up -d neo4j

# Wait for initialization (can take 30-60 seconds)
docker logs infrastructure_neo4j_1 --follow
```

2. **Authentication failed**:
```bash
# Check credentials match .env
cat .env | grep NEO4J

# Test connection
uv run python -c "
from neo4j import GraphDatabase
import os
driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))
driver.verify_connectivity()
print('Connected successfully!')
driver.close()
"
```

## Migration Issues

### Alembic: Can't Find Migration Head

**Error**: `alembic.util.exc.CommandError: Can't locate revision identified by 'head'`

**Solution**:
```bash
# Initialize alembic if not done
uv run alembic init alembic

# Create initial migration
uv run alembic revision --autogenerate -m "Initial schema"

# Apply migrations
uv run alembic upgrade head
```

### Duplicate Key/Constraint Violations

**Error**: `sqlalchemy.exc.IntegrityError: duplicate key value violates unique constraint`

**Solutions**:

1. **Check for duplicate data**:
```sql
-- Connect to database
psql postgresql://tracktion:tracktion@localhost:5432/tracktion

-- Find duplicates (example for recordings)
SELECT file_path, COUNT(*)
FROM recordings
GROUP BY file_path
HAVING COUNT(*) > 1;
```

2. **Reset sequence if needed**:
```sql
-- Reset UUID generation (rarely needed)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### Migration Stuck or Hanging

**Symptoms**: Migration command doesn't complete

**Solutions**:

1. **Check for locks**:
```sql
-- View active locks
SELECT pid, usename, application_name, client_addr, query
FROM pg_stat_activity
WHERE datname = 'tracktion';

-- Kill blocking query if needed (use carefully)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE pid = <blocking_pid>;
```

2. **Increase statement timeout**:
```python
# In database.py, already configured:
connect_args={"options": "-c statement_timeout=30000"}  # 30 seconds
```

## Performance Issues

### Slow Queries

**Symptoms**: Database operations taking too long

**Diagnostics**:

1. **Check missing indexes**:
```sql
-- View table indexes
\d+ recordings
\d+ metadata
\d+ tracklists

-- Check query plan
EXPLAIN ANALYZE SELECT * FROM metadata WHERE recording_id = '...';
```

2. **Add missing indexes**:
```sql
-- If metadata queries are slow
CREATE INDEX IF NOT EXISTS idx_metadata_recording_key
ON metadata(recording_id, key);
```

3. **Update statistics**:
```sql
ANALYZE recordings;
ANALYZE metadata;
ANALYZE tracklists;
```

### Connection Pool Exhausted

**Error**: `QueuePool limit of size X overflow Y reached`

**Solutions**:

1. **Increase pool size** in `database.py`:
```python
self.pg_engine = create_engine(
    database_url,
    pool_size=20,  # Increase from 10
    max_overflow=40,  # Increase from 20
)
```

2. **Check for connection leaks**:
```python
# Always use context managers
with db_manager.get_db_session() as session:
    # operations
    pass  # Session automatically closed
```

## Data Integrity Issues

### UUID Generation Problems

**Error**: `null value in column "id" violates not-null constraint`

**Solution**:
```sql
-- Ensure UUID extension is installed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify it's working
SELECT uuid_generate_v4();
```

### JSONB Validation Failures

**Error**: `Invalid tracks structure`

**Validation**:
```python
# Check tracks structure
tracks = [
    {
        "title": "Track Name",
        "artist": "Artist Name",
        "start_time": "00:00:00"
    }
]

# Validate before saving
from shared.core_types.src.models import Tracklist
tracklist = Tracklist(tracks=tracks)
if not tracklist.validate_tracks():
    print("Invalid structure!")
```

## Docker-Related Issues

### Container Keeps Restarting

**Diagnosis**:
```bash
# Check container status
docker ps -a | grep -E "postgres|neo4j"

# View logs
docker logs infrastructure_postgres_1 --tail 50
docker logs infrastructure_neo4j_1 --tail 50
```

**Common causes**:

1. **Insufficient memory**:
```bash
# Check Docker memory allocation
docker system info | grep Memory

# Increase in Docker Desktop settings
```

2. **Volume permissions**:
```bash
# Fix permissions
sudo chown -R $(whoami) infrastructure/postgres_data
sudo chown -R $(whoami) infrastructure/neo4j_data
```

### Data Not Persisting

**Issue**: Data lost after container restart

**Solution**:
```bash
# Verify volumes are mounted
docker inspect infrastructure_postgres_1 | grep -A 5 Mounts
docker inspect infrastructure_neo4j_1 | grep -A 5 Mounts

# Check docker-compose.yaml has volumes defined
grep -A 2 volumes infrastructure/docker-compose.yaml
```

## Recovery Procedures

### Backup PostgreSQL

```bash
# Create backup
docker exec infrastructure_postgres_1 \
  pg_dump -U tracktion tracktion > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i infrastructure_postgres_1 \
  psql -U tracktion tracktion < backup_20240101.sql
```

### Backup Neo4j

```bash
# Stop Neo4j first
docker-compose -f infrastructure/docker-compose.yaml stop neo4j

# Backup data directory
tar -czf neo4j_backup_$(date +%Y%m%d).tar.gz infrastructure/neo4j_data

# Restore
tar -xzf neo4j_backup_20240101.tar.gz
```

### Complete Reset (Development Only)

```bash
# Stop all services
docker-compose -f infrastructure/docker-compose.yaml down

# Remove all data volumes
docker volume rm infrastructure_postgres_data infrastructure_neo4j_data

# Restart fresh
docker-compose -f infrastructure/docker-compose.yaml up -d

# Re-run migrations
uv run alembic upgrade head

# Initialize Neo4j
uv run python scripts/initialize_neo4j.py
```

## Monitoring

### Health Checks

```bash
# PostgreSQL health
docker exec infrastructure_postgres_1 pg_isready

# Neo4j health
curl http://localhost:7474/db/neo4j/cluster/available

# Via Python
uv run python scripts/check_database_health.py
```

### Resource Usage

```bash
# Container stats
docker stats infrastructure_postgres_1 infrastructure_neo4j_1

# Database sizes
docker exec infrastructure_postgres_1 \
  psql -U tracktion -c "SELECT pg_database_size('tracktion');"
```

## Getting Help

If issues persist:

1. Check logs thoroughly:
```bash
docker-compose -f infrastructure/docker-compose.yaml logs --tail=100
```

2. Verify environment:
```bash
uv run python -c "import os; print(os.getenv('DATABASE_URL'))"
```

3. Test connections individually:
```bash
# PostgreSQL
pg_isready -h localhost -p 5432

# Neo4j
nc -zv localhost 7687
```

4. Review configuration files:
- `.env` - Environment variables
- `alembic.ini` - Migration configuration
- `infrastructure/docker-compose.yaml` - Container setup
