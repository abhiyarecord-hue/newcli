//! SQLite schema with sqlite-vec vector storage and FTS5 keyword storage.
//!
//! Schema changes are driven by `PRAGMA user_version`. Version 2 records the
//! provenance and validity of each chunk embedding so semantic search never
//! treats placeholders or vectors from another model as compatible.

use rusqlite::Connection;

use agent_types::{AgentError, Result};

pub const CURRENT_VERSION: i32 = 2;

/// Deterministic version-2 migration boundaries used by rollback tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationStep {
    MetadataAdded,
    LegacyVectorsRemoved,
    VersionUpdated,
}

pub fn run_migrations(conn: &Connection) -> Result<()> {
    run_migrations_inner(conn, None)
}

/// Run migrations with an injected failure before the version-2 transaction
/// commits. Public only for migration rollback tests.
#[doc(hidden)]
pub fn run_migrations_with_failure(conn: &Connection, fail_after: MigrationStep) -> Result<()> {
    run_migrations_inner(conn, Some(fail_after))
}

fn run_migrations_inner(conn: &Connection, fail_after: Option<MigrationStep>) -> Result<()> {
    let mut version = schema_version(conn)?;
    if version < 1 {
        create_version_one(conn)?;
        version = 1;
    }
    if version < CURRENT_VERSION {
        migrate_to_version_two(conn, fail_after)?;
    }
    Ok(())
}

fn schema_version(conn: &Connection) -> Result<i32> {
    conn.pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| AgentError::Storage(error.to_string()))
}

fn create_version_one(conn: &Connection) -> Result<()> {
    let tx = conn
        .unchecked_transaction()
        .map_err(|error| AgentError::Storage(format!("begin schema migration: {error}")))?;
    tx.execute_batch(
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

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[768]);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        );

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
        PRAGMA user_version = 1;
        ",
    )
    .map_err(|error| AgentError::Storage(format!("create schema version 1: {error}")))?;
    tx.commit()
        .map_err(|error| AgentError::Storage(format!("commit schema version 1: {error}")))
}

fn migrate_to_version_two(conn: &Connection, fail_after: Option<MigrationStep>) -> Result<()> {
    let tx = conn
        .unchecked_transaction()
        .map_err(|error| AgentError::Storage(format!("begin embedding migration: {error}")))?;

    tx.execute_batch(
        "
        ALTER TABLE chunks ADD COLUMN embedding_valid INTEGER NOT NULL DEFAULT 0
            CHECK (embedding_valid IN (0, 1));
        ALTER TABLE chunks ADD COLUMN embedding_provider TEXT;
        ALTER TABLE chunks ADD COLUMN embedding_model TEXT;
        ALTER TABLE chunks ADD COLUMN embedding_dimension INTEGER;
        CREATE INDEX idx_chunks_embedding_compatibility
            ON chunks(embedding_valid, embedding_provider, embedding_model, embedding_dimension);
        ",
    )
    .map_err(|error| AgentError::Storage(format!("add embedding metadata: {error}")))?;
    inject_failure(fail_after, MigrationStep::MetadataAdded)?;

    // Every version-1 vector lacks trustworthy provenance, including any
    // orphaned row. Remove only vector rows; chunks, files, and FTS remain.
    tx.execute("DELETE FROM chunks_vec", [])
        .map_err(|error| AgentError::Storage(format!("remove unknown legacy vectors: {error}")))?;
    inject_failure(fail_after, MigrationStep::LegacyVectorsRemoved)?;

    tx.pragma_update(None, "user_version", CURRENT_VERSION)
        .map_err(|error| AgentError::Storage(format!("set schema version: {error}")))?;
    inject_failure(fail_after, MigrationStep::VersionUpdated)?;

    tx.commit()
        .map_err(|error| AgentError::Storage(format!("commit embedding migration: {error}")))
}

fn inject_failure(fail_after: Option<MigrationStep>, step: MigrationStep) -> Result<()> {
    if fail_after == Some(step) {
        return Err(AgentError::Storage(format!(
            "injected embedding migration failure after {step:?}"
        )));
    }
    Ok(())
}
