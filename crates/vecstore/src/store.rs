//! `VecStore` — the public API over the SQLite schema.
//!
//! `open(path)` → runs migrations → ready.
//! All writes run inside transactions so partial inserts never corrupt the
//! three linked tables (chunks, chunks_vec, chunks_fts).
//!
//! The `Connection` is `!Sync`; callers wrap behind a Mutex or run in
//! `spawn_blocking` (TASK-3.1 context guard).

use std::path::Path;

use rusqlite::{params, Connection, Transaction};
use zerocopy::AsBytes;

use agent_types::{AgentError, Result};

use crate::schema::run_migrations;

pub const DEFAULT_EMBEDDING_PROVIDER: &str = "gemini";
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-004";
pub const EMBEDDING_DIMENSION: usize = 768;

/// Identity required for stored and query embeddings to be semantically
/// compatible. Provider, model, and dimension must all match exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingProfile {
    pub provider: String,
    pub model: String,
    pub dimension: usize,
}

impl EmbeddingProfile {
    pub fn new(provider: impl Into<String>, model: impl Into<String>, dimension: usize) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            dimension,
        }
    }

    pub fn default_gemini() -> Self {
        Self::new(
            DEFAULT_EMBEDDING_PROVIDER,
            DEFAULT_EMBEDDING_MODEL,
            EMBEDDING_DIMENSION,
        )
    }
}

impl Default for EmbeddingProfile {
    fn default() -> Self {
        Self::default_gemini()
    }
}

pub struct VecStore {
    conn: Connection,
}

/// Durable metadata associated with one indexed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRecord {
    pub path: String,
    pub mtime: i64,
    pub content_hash: String,
}

/// A chunk row to insert. An empty embedding means keyword-only indexing.
pub struct ChunkInsert {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    pub token_count: u32,
    pub embedding: Vec<f32>,
}

/// Deterministic replacement boundaries used by fault-injection tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplacementStep {
    StaleVectorsDeleted,
    StaleChunksDeleted,
    FileMetadataUpserted,
    ReplacementChunksInserted,
    ValidVectorsInserted,
    Committed,
}

struct PreparedChunk {
    start_line: u32,
    end_line: u32,
    text: String,
    token_count: u32,
    embedding: Option<Vec<u8>>,
}

struct ReplacementPlan {
    file: FileRecord,
    chunks: Vec<PreparedChunk>,
    profile: EmbeddingProfile,
}

impl ReplacementPlan {
    /// Validate and materialize all caller-owned data before SQLite opens the
    /// replacement transaction.
    fn build(file: FileRecord, chunks: &[ChunkInsert], profile: &EmbeddingProfile) -> Result<Self> {
        validate_storage_profile(profile)?;
        let mut prepared = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            if chunk.file_path != file.path {
                return Err(AgentError::Storage(format!(
                    "replacement chunk path {:?} does not match file {:?}",
                    chunk.file_path, file.path
                )));
            }
            prepared.push(PreparedChunk {
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                text: chunk.text.clone(),
                token_count: chunk.token_count,
                embedding: prepare_embedding(&chunk.embedding, profile)?,
            });
        }
        Ok(Self {
            file,
            chunks: prepared,
            profile: profile.clone(),
        })
    }
}

fn validate_storage_profile(profile: &EmbeddingProfile) -> Result<()> {
    if profile.provider.trim().is_empty() || profile.model.trim().is_empty() {
        return Err(AgentError::Storage(
            "valid embeddings require non-empty provider and model metadata".into(),
        ));
    }
    if profile.dimension == 0 {
        return Err(AgentError::Storage(
            "valid embeddings require a dimension greater than zero".into(),
        ));
    }
    Ok(())
}

/// Reject a profile whose width does not match the vector table it would be
/// written into.
///
/// This replaces a comparison against the compile-time `EMBEDDING_DIMENSION`
/// constant. That constant is only the default for a freshly created database,
/// not the width of the table in front of us, so checking against it rejected
/// every model that is not 768-wide even after the table had been rebuilt for
/// the correct width. Found by running a real 1536-wide embedding model.
fn validate_profile_against_table(conn: &Connection, profile: &EmbeddingProfile) -> Result<()> {
    let Some(table_dimension) = crate::schema::vector_dimension(conn)? else {
        return Err(AgentError::Storage(
            "vector table is missing; the database needs migration".into(),
        ));
    };
    if profile.dimension != table_dimension {
        return Err(AgentError::Storage(format!(
            "embedding profile dimension {} does not match the vector table dimension \
             {table_dimension}; re-index so the table is rebuilt for this model",
            profile.dimension
        )));
    }
    Ok(())
}

fn prepare_embedding(embedding: &[f32], profile: &EmbeddingProfile) -> Result<Option<Vec<u8>>> {
    match embedding.len() {
        0 => Ok(None),
        actual if actual != profile.dimension => Err(AgentError::Storage(format!(
            "embedding dimension {actual} does not match profile dimension {}",
            profile.dimension
        ))),
        _ if embedding.iter().any(|value| !value.is_finite()) => Err(AgentError::Storage(
            "embedding contains a non-finite value".into(),
        )),
        // All-zero vectors are the historical keyword-only placeholder. They
        // are retained as text but never written to semantic storage.
        _ if embedding.iter().all(|value| *value == 0.0) => Ok(None),
        _ => Ok(Some(embedding.as_bytes().to_vec())),
    }
}

/// Signature sqlite expects for an auto-loaded extension entry point.
type SqliteExtensionInit = unsafe extern "C" fn(
    *mut rusqlite::ffi::sqlite3,
    *mut *const std::os::raw::c_char,
    *const rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int;

/// Register the sqlite-vec extension so every new connection loads it.
fn enable_vec_extension() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            SqliteExtensionInit,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    }
}

impl VecStore {
    /// Open (or create) the database at `path`, loading sqlite-vec and running
    /// migrations.
    pub fn open(path: &Path) -> Result<Self> {
        enable_vec_extension();

        let conn =
            Connection::open(path).map_err(|e| AgentError::Storage(format!("open db: {e}")))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        run_migrations(&conn)?;
        Ok(Self { conn })
    }

    /// Open an in-memory database (for tests).
    pub fn open_memory() -> Result<Self> {
        enable_vec_extension();

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

    /// Insert chunks using the default Gemini embedding identity.
    ///
    /// This legacy append API remains for callers that are not replacing a
    /// file. File replacement must use [`Self::replace_file`].
    pub fn insert_chunks(&self, chunks: &[ChunkInsert]) -> Result<()> {
        self.insert_chunks_with_profile(chunks, &EmbeddingProfile::default())
    }

    /// Insert chunks and persist the exact embedding identity atomically.
    pub fn insert_chunks_with_profile(
        &self,
        chunks: &[ChunkInsert],
        profile: &EmbeddingProfile,
    ) -> Result<()> {
        validate_storage_profile(profile)?;
        let prepared = chunks
            .iter()
            .map(|chunk| prepare_embedding(&chunk.embedding, profile))
            .collect::<Result<Vec<_>>>()?;
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        for (chunk, embedding) in chunks.iter().zip(prepared) {
            let embedding_valid = i64::from(embedding.is_some());
            let provider = embedding.as_ref().map(|_| profile.provider.as_str());
            let model = embedding.as_ref().map(|_| profile.model.as_str());
            let dimension = embedding.as_ref().map(|_| profile.dimension as i64);
            tx.execute(
                "INSERT INTO chunks(
                    file_path, start_line, end_line, text, token_count,
                    embedding_valid, embedding_provider, embedding_model, embedding_dimension
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.token_count,
                    embedding_valid,
                    provider,
                    model,
                    dimension
                ],
            )
            .map_err(|e| AgentError::Storage(format!("insert chunk: {e}")))?;
            if let Some(embedding) = embedding {
                let rowid = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES(?1, ?2)",
                    params![rowid, embedding],
                )
                .map_err(|e| AgentError::Storage(format!("insert vec: {e}")))?;
            }
        }

        tx.commit()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Atomically replace every searchable row for one file using the default
    /// Gemini embedding identity.
    pub fn replace_file(&self, file: FileRecord, chunks: &[ChunkInsert]) -> Result<()> {
        self.replace_file_with_profile(file, chunks, &EmbeddingProfile::default())
    }

    /// Atomically replace a file while recording the exact identity of every
    /// real embedding. Keyword-only chunks receive null metadata and no vector.
    pub fn replace_file_with_profile(
        &self,
        file: FileRecord,
        chunks: &[ChunkInsert],
        profile: &EmbeddingProfile,
    ) -> Result<()> {
        // Checked here rather than inside the plan builder because only this
        // layer has the connection, and the width that matters is the one the
        // vector table actually declares.
        validate_profile_against_table(&self.conn, profile)?;
        self.replace_file_inner(ReplacementPlan::build(file, chunks, profile)?, None)
    }

    /// Execute replacement with a deterministic error immediately after the
    /// selected boundary. This is public solely for cross-crate durability
    /// and restart tests.
    #[doc(hidden)]
    pub fn replace_file_with_failure(
        &self,
        file: FileRecord,
        chunks: &[ChunkInsert],
        fail_after: ReplacementStep,
    ) -> Result<()> {
        self.replace_file_inner(
            ReplacementPlan::build(file, chunks, &EmbeddingProfile::default())?,
            Some(fail_after),
        )
    }

    fn replace_file_inner(
        &self,
        plan: ReplacementPlan,
        fail_after: Option<ReplacementStep>,
    ) -> Result<()> {
        let ReplacementPlan {
            file,
            chunks,
            profile,
        } = plan;
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        let old_ids = chunk_ids(&tx, &file.path)?;
        for id in old_ids {
            tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![id])
                .map_err(|e| AgentError::Storage(format!("delete stale vector: {e}")))?;
        }
        inject_failure(fail_after, ReplacementStep::StaleVectorsDeleted)?;

        tx.execute(
            "DELETE FROM chunks WHERE file_path = ?1",
            params![file.path],
        )
        .map_err(|e| AgentError::Storage(format!("delete stale chunks: {e}")))?;
        inject_failure(fail_after, ReplacementStep::StaleChunksDeleted)?;

        tx.execute(
            "INSERT INTO files(path, mtime, content_hash) VALUES(?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, content_hash=excluded.content_hash",
            params![file.path, file.mtime, file.content_hash],
        )
        .map_err(|e| AgentError::Storage(format!("replace file metadata: {e}")))?;
        inject_failure(fail_after, ReplacementStep::FileMetadataUpserted)?;

        let mut vector_rows = Vec::new();
        for chunk in chunks {
            let embedding_valid = i64::from(chunk.embedding.is_some());
            let provider = chunk.embedding.as_ref().map(|_| profile.provider.as_str());
            let model = chunk.embedding.as_ref().map(|_| profile.model.as_str());
            let dimension = chunk.embedding.as_ref().map(|_| profile.dimension as i64);
            tx.execute(
                "INSERT INTO chunks(
                    file_path, start_line, end_line, text, token_count,
                    embedding_valid, embedding_provider, embedding_model, embedding_dimension
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    file.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.token_count,
                    embedding_valid,
                    provider,
                    model,
                    dimension
                ],
            )
            .map_err(|e| AgentError::Storage(format!("insert replacement chunk: {e}")))?;
            if let Some(embedding) = chunk.embedding {
                vector_rows.push((tx.last_insert_rowid(), embedding));
            }
        }
        inject_failure(fail_after, ReplacementStep::ReplacementChunksInserted)?;

        for (rowid, embedding) in vector_rows {
            tx.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES(?1, ?2)",
                params![rowid, embedding],
            )
            .map_err(|e| AgentError::Storage(format!("insert replacement vector: {e}")))?;
        }
        inject_failure(fail_after, ReplacementStep::ValidVectorsInserted)?;

        tx.commit()
            .map_err(|e| AgentError::Storage(format!("commit replacement: {e}")))?;
        inject_failure(fail_after, ReplacementStep::Committed)
    }

    /// Delete one file and all searchable rows in an independent transaction.
    pub fn remove_file(&self, path: &str) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        for id in chunk_ids(&tx, path)? {
            tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![id])
                .map_err(|e| AgentError::Storage(format!("delete vector: {e}")))?;
        }
        tx.execute("DELETE FROM chunks WHERE file_path = ?1", params![path])
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        tx.commit()
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Backward-compatible alias for standalone atomic deletion.
    pub fn delete_file(&self, path: &str) -> Result<()> {
        self.remove_file(path)
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
            .query_row("SELECT count(*) FROM chunks_fts", [], |r| r.get(0))
            .map_err(|e| AgentError::Storage(e.to_string()))?;
        Ok((chunks, vecs, fts))
    }

    /// Get raw connection (for advanced queries in hybrid search).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

fn chunk_ids(tx: &Transaction<'_>, path: &str) -> Result<Vec<i64>> {
    let mut statement = tx
        .prepare("SELECT id FROM chunks WHERE file_path = ?1")
        .map_err(|e| AgentError::Storage(e.to_string()))?;
    let rows = statement
        .query_map(params![path], |row| row.get(0))
        .map_err(|e| AgentError::Storage(e.to_string()))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| AgentError::Storage(e.to_string()))
}

fn inject_failure(fail_after: Option<ReplacementStep>, step: ReplacementStep) -> Result<()> {
    if fail_after == Some(step) {
        return Err(AgentError::Storage(format!(
            "injected replacement failure after {step:?}"
        )));
    }
    Ok(())
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
        store.upsert_file("src/main.rs", 12345, "abc123").unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let (chunks, vecs, fts) = store.chunk_counts().unwrap();
        assert_eq!(chunks, 3);
        assert_eq!(vecs, 3);
        assert_eq!(fts, 3);
    }

    #[test]
    fn delete_file_cascades_to_all_tables() {
        let store = VecStore::open_memory().unwrap();
        store.upsert_file("src/main.rs", 12345, "abc123").unwrap();
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
        store.upsert_file("src/main.rs", 1, "abc").unwrap();
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
        store.upsert_file("src/main.rs", 1, "abc").unwrap();
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
