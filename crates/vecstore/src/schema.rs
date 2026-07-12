//! SQLite schema with sqlite-vec virtual table + FTS5 external-content table.
//!
//! Tables (verbatim from task spec):
//! - `files(path TEXT PK, mtime INTEGER, content_hash TEXT)`
//! - `chunks(id INTEGER PK, file_path TEXT, start_line INT, end_line INT, text TEXT, token_count INT)`
//! - `chunks_vec` = vec0 virtual table (embedding FLOAT[768]) keyed by chunk rowid
//! - `chunks_fts` = FTS5 external-content table (text, content='chunks', content_rowid='id')
//!
//! Migrations are idempotent (`user_version` pragma).

use rusqlite::Connection;

use agent_types::{AgentError, Result};

const CURRENT_VERSION: i32 = 1;

pub fn run_migrations(conn: &Connection) -> Result<()> {
    let version: i32 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|e| AgentError::Storage(e.to_string()))?;

    if version >= CURRENT_VERSION {
        return Ok(());
    }

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            mtime INTEGER NOT NULL,
            content_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
        ",
    )
    .map_err(|e| AgentError::Storage(format!("schema: {e}")))?;

    // sqlite-vec virtual table.
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[768]);"
    )
    .map_err(|e| AgentError::Storage(format!("vec0: {e}")))?;

    // FTS5 external-content table.
    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        );

        -- Triggers to keep FTS5 in sync with chunks table.
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        ",
    )
    .map_err(|e| AgentError::Storage(format!("fts5: {e}")))?;

    conn.pragma_update(None, "user_version", CURRENT_VERSION)
        .map_err(|e| AgentError::Storage(e.to_string()))?;

    Ok(())
}
