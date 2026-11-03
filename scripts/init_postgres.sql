-- ==============================================================================
-- Enterprise MLOps Platform — PostgreSQL Initialization Script
-- ==============================================================================

-- Connect to primary mlops_db and ensure schemas
\c mlops_db;

CREATE SCHEMA IF NOT EXISTS serving;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Grant appropriate permissions to application user
GRANT ALL PRIVILEGES ON SCHEMA public TO CURRENT_USER;
GRANT ALL PRIVILEGES ON SCHEMA serving TO CURRENT_USER;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO CURRENT_USER;
