SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Live chat exports reduced to 5-second activity buckets (backend/src/chat_signals.py).
-- A task points at one with source_metadata.chat_log_id (+ chat_offset_seconds, where the
-- video starts in the chat); its chat spikes then steer clip selection. Messages are not kept.
CREATE TABLE IF NOT EXISTS chat_logs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    message_count INTEGER NOT NULL DEFAULT 0,
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    buckets JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_logs_user ON chat_logs (user_id, created_at DESC);
