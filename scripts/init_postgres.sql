-- ==============================================================================
-- Enterprise MLOps Platform — PostgreSQL Initialization Script
-- ==============================================================================

-- Create separate databases if needed for isolation
SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

-- Connect to primary mlops_db and ensure schemas
\c mlops_db;

CREATE SCHEMA IF NOT EXISTS serving;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Grant appropriate permissions
GRANT ALL PRIVILEGES ON SCHEMA serving TO CURRENT_USER;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO CURRENT_USER;
