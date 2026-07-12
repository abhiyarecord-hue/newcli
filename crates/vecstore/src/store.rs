//! `VecStore` — the public API over the SQLite schema.
//!
//! `open(path)` → runs migrations → ready.
//! All writes run inside transactions so partial inserts never corrupt the
//! three linked tables (chunks, chunks_vec, chunks_fts).
//!
//! The `Connection` is `!Sync`; callers wrap behind a Mutex or run in
//! `spawn_blocking` (TASK-3.1 context guard).

use std::path::Path;

use rusqlite::{params, Connection};
use zerocopy::AsBytes;

use agent_types::{AgentError, Result};

use crate::schema::run_migrations;

pub struct VecStore {
    conn: Connection,
}

/// A chunk row to insert (text + embedding).
pub struct ChunkInsert {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    pub token_count: u32,
    pub embedding: Vec<f32>,
}

impl VecStore {
    /// Open (or create) the database at `path`, loading sqlite-vec and running
    /// migrations.
    pub fn open(path: &Path) -> Result<Self> {
        // Register sqlite-vec extension before opening.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(path)
            .map_err(|e| AgentError::Storage(format!("open db: {e}")))?;

        // Enable WAL mode + foreign keys.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        run_migrations(&conn)?;

        Ok(Self { conn })
    }

    /// Open an in-memory database (for tests).
    pub fn open_memory() -> Result<Self> {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open_in_memory()
            .map_err(|e| AgentError::Storage(format!("open :memory:: {e}")))?;

        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        run_migrations(&conn)?;

        Ok(Self { conn })
    }

    /// Upsert a file record (path + mtime + content_hash).
    pub fn upsert_file(&self, path: &str, mtime: i64, content_hash: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO files(path, mtime, content_hash) VALUES(?1, ?2, ?3)
                 ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, content_hash=excluded.content_hash",
                params![path, mtime, content_hash],
            )
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Insert chunks with their embeddings in a transaction.
    pub fn insert_chunks(&self, chunks: &[ChunkInsert]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        for c in chunks {
            tx.execute(
                "INSERT INTO chunks(file_path, start_line, end_line, text, token_count) VALUES(?1,?2,?3,?4,?5)",
                params![c.file_path, c.start_line, c.end_line, c.text, c.token_count],
            )
            .map_err(|e| AgentError::Storage(format!("insert chunk: {e}")))?;

            let rowid = tx.last_insert_rowid();

            // Insert embedding as little-endian f32 bytes.
            tx.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES(?1, ?2)",
                params![rowid, c.embedding.as_bytes()],
            )
            .map_err(|e| AgentError::Storage(format!("insert vec: {e}")))?;
        }

        tx.commit()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Delete a file and cascade to all its chunks (+ vec + fts via triggers).
    pub fn delete_file(&self, path: &str) -> Result<()> {
        // Manually delete from chunks_vec since vec0 virtual table doesn't
        // participate in ON DELETE CASCADE.
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        // Get chunk ids for this file.
        let ids: Vec<i64> = {
            let mut stmt = tx
                .prepare("SELECT id FROM chunks WHERE file_path = ?1")
                .map_err(|e| AgentError::Storage(e.to_string()))?;
            let rows: Vec<i64> = stmt
                .query_map(params![path], |row| row.get(0))
                .map_err(|e| AgentError::Storage(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect();
            rows
        };

        for id in &ids {
            tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![id])
                .map_err(|e| AgentError::Storage(format!("del vec: {e}")))?;
        }

        // Delete chunks (triggers handle FTS5).
        tx.execute("DELETE FROM chunks WHERE file_path = ?1", params![path])
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        // Delete the file record.
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        tx.commit()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Count rows across the three chunk tables (for consistency checks).
    pub fn chunk_counts(&self) -> Result<(i64, i64, i64)> {
        let chunks: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        let vecs: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |r| r.get(0))
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        let fts: i64 = self
            .conn
            .query_row(
                "SELECT count(*) FROM chunks_fts",
                [],
                |r| r.get(0),
            )
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok((chunks, vecs, fts))
    }

    /// Get raw connection (for advanced queries in hybrid search).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunks() -> Vec<ChunkInsert> {
        (0..3)
            .map(|i| ChunkInsert {
                file_path: "src/main.rs".to_string(),
                start_line: i * 10 + 1,
                end_line: (i + 1) * 10,
                text: format!("chunk {i} content here"),
                token_count: 5,
                embedding: vec![0.1 * (i as f32 + 1.0); 768],
            })
            .collect()
    }

    #[test]
    fn insert_and_count_agrees_across_tables() {
        let store = VecStore::open_memory().unwrap();
        store
            .upsert_file("src/main.rs", 12345, "abc123")
            .unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let (chunks, vecs, fts) = store.chunk_counts().unwrap();
        assert_eq!(chunks, 3);
        assert_eq!(vecs, 3);
        assert_eq!(fts, 3);
    }

    #[test]
    fn delete_file_cascades_to_all_tables() {
        let store = VecStore::open_memory().unwrap();
        store
            .upsert_file("src/main.rs", 12345, "abc123")
            .unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();
        store.delete_file("src/main.rs").unwrap();

        let (chunks, vecs, fts) = store.chunk_counts().unwrap();
        assert_eq!(chunks, 0);
        assert_eq!(vecs, 0);
        assert_eq!(fts, 0);
    }

    #[test]
    fn upsert_file_updates_existing() {
        let store = VecStore::open_memory().unwrap();
        store.upsert_file("a.rs", 1, "hash1").unwrap();
        store.upsert_file("a.rs", 2, "hash2").unwrap();

        let (mtime, hash): (i64, String) = store
            .conn()
            .query_row(
                "SELECT mtime, content_hash FROM files WHERE path = ?1",
                params!["a.rs"],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(mtime, 2);
        assert_eq!(hash, "hash2");
    }

    #[test]
    fn vec_knn_query_returns_results() {
        let store = VecStore::open_memory().unwrap();
        store
            .upsert_file("src/main.rs", 1, "abc")
            .unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let query_vec: Vec<f32> = vec![0.2; 768];
        let results: Vec<(i64, f64)> = store
            .conn()
            .prepare(
                "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT 2",
            )
            .unwrap()
            .query_map(params![query_vec.as_bytes()], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn fts5_search_returns_matching_chunks() {
        let store = VecStore::open_memory().unwrap();
        store
            .upsert_file("src/main.rs", 1, "abc")
            .unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let count: i64 = store
            .conn()
            .query_row(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '\"chunk\"'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 3);
    }
}
