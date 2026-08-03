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

/// Width the `chunks_vec` virtual table is currently built for.
///
/// Read back from the stored DDL rather than from a setting, so it cannot drift
/// out of sync with the table that actually exists.
pub fn vector_dimension(conn: &Connection) -> Result<Option<usize>> {
    let sql: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'",
            [],
            |row| row.get(0),
        )
        .or_else(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(other),
        })
        .map_err(|error| AgentError::Storage(format!("inspect chunks_vec: {error}")))?;

    Ok(sql.as_deref().and_then(parse_vector_dimension))
}

/// Extract `N` from a `float[N]` column declaration.
fn parse_vector_dimension(sql: &str) -> Option<usize> {
    let after = sql.split("float[").nth(1)?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Make the vector table match `dimension`, rebuilding it if the width changed.
///
/// Embedding models disagree on width: 768, 1024, 1536, and 3072 are all in use.
/// A `vec0` table is created for one fixed width, so switching models means the
/// table has to be rebuilt. Returns `true` when a rebuild happened.
///
/// Rebuilding **drops every stored vector**, which is unavoidable: vectors of the
/// old width cannot be reinterpreted at the new one, and keeping them would be
/// worse than losing them because comparisons would silently return nonsense.
/// Chunk text, file records, and the keyword index are all left intact, and every
/// chunk is marked `embedding_valid = 0`. That is what the existing compatibility
/// check already looks for, so search degrades to keyword-only and reports the
/// re-index guidance instead of returning wrong neighbours.
pub fn ensure_vector_dimension(conn: &Connection, dimension: usize) -> Result<bool> {
    if dimension == 0 {
        return Err(AgentError::Storage(
            "embedding dimension must be greater than zero".into(),
        ));
    }

    if vector_dimension(conn)? == Some(dimension) {
        return Ok(false);
    }

    let tx = conn
        .unchecked_transaction()
        .map_err(|error| AgentError::Storage(format!("begin vector rebuild: {error}")))?;
    tx.execute_batch(&format!(
        "
        DROP TABLE IF EXISTS chunks_vec;
        CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dimension}]);
        UPDATE chunks SET embedding_valid = 0;
        "
    ))
    .map_err(|error| {
        AgentError::Storage(format!(
            "rebuild chunks_vec at dimension {dimension}: {error}"
        ))
    })?;
    tx.commit()
        .map_err(|error| AgentError::Storage(format!("commit vector rebuild: {error}")))?;

    Ok(true)
}

#[cfg(test)]
mod dimension_tests {
    use super::*;

    /// Uses the production open path so the extension and migrations match what
    /// a real database gets.
    fn open() -> crate::VecStore {
        crate::VecStore::open_memory().unwrap()
    }

    #[test]
    fn the_declared_width_is_read_back_from_the_table() {
        let store = open();
        assert_eq!(vector_dimension(store.conn()).unwrap(), Some(768));
    }

    #[test]
    fn parsing_handles_the_declaration_and_rejects_nonsense() {
        assert_eq!(
            parse_vector_dimension(
                "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[1536])"
            ),
            Some(1536)
        );
        assert_eq!(parse_vector_dimension("CREATE TABLE t (x INTEGER)"), None);
        assert_eq!(parse_vector_dimension("float[]"), None);
        assert_eq!(parse_vector_dimension("float[abc]"), None);
    }

    #[test]
    fn matching_the_current_width_is_a_no_op() {
        let store = open();
        assert!(!ensure_vector_dimension(store.conn(), 768).unwrap());
        assert_eq!(vector_dimension(store.conn()).unwrap(), Some(768));
    }

    #[test]
    fn changing_width_rebuilds_and_invalidates_without_losing_chunk_text() {
        let store = open();
        let conn = store.conn();
        conn.execute(
            "INSERT INTO files (path, mtime, content_hash) VALUES ('a.rs', 1, 'h')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, text, token_count,
                embedding_valid, embedding_provider, embedding_model, embedding_dimension)
             VALUES ('a.rs', 1, 2, 'fn main() {}', 4, 1, 'gemini', 'text-embedding-004', 768)",
            [],
        )
        .unwrap();

        for width in [1536, 3072, 1024] {
            assert!(
                ensure_vector_dimension(conn, width).unwrap(),
                "switching to {width} must rebuild"
            );
            assert_eq!(vector_dimension(conn).unwrap(), Some(width));

            // The chunk survives, but its vector is no longer claimed as usable.
            let (text, valid): (String, i64) = conn
                .query_row(
                    "SELECT text, embedding_valid FROM chunks WHERE file_path = 'a.rs'",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(text, "fn main() {}");
            assert_eq!(
                valid, 0,
                "vectors from the old width must not stay marked valid"
            );

            // Keyword search must keep working while semantic data is missing.
            let hits: i64 = conn
                .query_row(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'main'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(hits, 1, "the keyword index must survive a vector rebuild");
        }
    }

    #[test]
    fn a_zero_width_is_refused() {
        let store = open();
        assert!(ensure_vector_dimension(store.conn(), 0).is_err());
        assert_eq!(vector_dimension(store.conn()).unwrap(), Some(768));
    }
}
