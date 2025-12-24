-- ABI Framework Database Initialization
-- This script sets up the necessary database schema for ABI metadata storage

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schema for ABI
CREATE SCHEMA IF NOT EXISTS abi;

-- Sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS abi.sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS abi.api_usage (
    id BIGSERIAL PRIMARY KEY,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(255),
    session_id UUID REFERENCES abi.sessions(session_id),
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,
    status_code INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    client_ip INET,
    user_agent TEXT
);

-- Model metadata storage
CREATE TABLE IF NOT EXISTS abi.models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100), -- 'transformer', 'cnn', 'rnn', etc.
    architecture JSONB, -- Model architecture details
    parameters_count BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(name, version)
);

-- Training runs tracking
CREATE TABLE IF NOT EXISTS abi.training_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES abi.models(model_id),
    status VARCHAR(50) DEFAULT 'running', -- 'running', 'completed', 'failed'
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    epochs_completed INTEGER DEFAULT 0,
    best_loss DECIMAL(10,6),
    best_accuracy DECIMAL(5,4),
    config JSONB, -- Training configuration
    metrics JSONB DEFAULT '{}', -- Training metrics over time
    created_by VARCHAR(255)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON abi.api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_user ON abi.api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON abi.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON abi.sessions(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_models_name ON abi.models(name);
CREATE INDEX IF NOT EXISTS idx_training_runs_model ON abi.training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON abi.training_runs(status);

-- Insert default data
INSERT INTO abi.models (name, version, model_type, architecture, parameters_count, metadata)
VALUES (
    'default-embedding',
    '1.0.0',
    'transformer',
    '{"layers": 12, "heads": 12, "d_model": 768, "d_ff": 3072}',
    110000000,
    '{"description": "Default embedding model for ABI framework", "vocabulary_size": 30000}'
)
ON CONFLICT (name, version) DO NOTHING;