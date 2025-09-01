-- Initialize test database with required extensions and permissions

-- Create UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database user if not exists (for completeness)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tracktion_user') THEN
        CREATE USER tracktion_user WITH PASSWORD 'changeme';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE test_feedback TO tracktion_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO tracktion_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tracktion_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tracktion_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO tracktion_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO tracktion_user;
