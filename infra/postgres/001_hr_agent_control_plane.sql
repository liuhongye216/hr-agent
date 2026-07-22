CREATE TABLE IF NOT EXISTS governed_jobs (
    job_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    status TEXT NOT NULL,
    current_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    published_at TEXT,
    closed_at TEXT,
    deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_governed_jobs_company_status
    ON governed_jobs(company_id, status);

CREATE TABLE IF NOT EXISTS job_profile_versions (
    job_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    review_json TEXT,
    change_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    PRIMARY KEY (job_id, version),
    FOREIGN KEY (job_id) REFERENCES governed_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    occurred_at TEXT NOT NULL,
    request_id TEXT,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT,
    company_id TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    error_type TEXT
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
