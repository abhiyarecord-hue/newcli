use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::{params, Connection};
use vecstore::schema::{run_migrations_with_failure, MigrationStep, CURRENT_VERSION};
use vecstore::{
    search, search_with_profile, ChunkInsert, EmbeddingProfile, FileRecord, SearchMode, VecStore,
};
use zerocopy::AsBytes;

const FILE: &str = "src/legacy.rs";
static NEXT_DB: AtomicU64 = AtomicU64::new(0);

fn database_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vecstore-embedding-{label}-{}-{}.db",
        std::process::id(),
        NEXT_DB.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Signature sqlite expects for an auto-loaded extension entry point.
type SqliteExtensionInit = unsafe extern "C" fn(
    *mut rusqlite::ffi::sqlite3,
    *mut *const std::os::raw::c_char,
    *const rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int;

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

fn create_legacy_database(path: &Path) -> Connection {
    enable_vec_extension();
    let connection = Connection::open(path).expect("open legacy database");
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE files (
                path TEXT PRIMARY KEY, mtime INTEGER NOT NULL, content_hash TEXT NOT NULL
             );
             CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, token_count INTEGER NOT NULL
             );
             CREATE INDEX idx_chunks_file ON chunks(file_path);
             CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[768]);
             CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
             );
             CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
             END;
             CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES('delete', old.id, old.text);
             END;
             PRAGMA user_version=1;",
        )
        .expect("create version-one schema");
    connection
}
fn seed_legacy_rows(connection: &Connection, token: &str) {
    connection
        .execute(
            "INSERT INTO files(path, mtime, content_hash) VALUES(?1, 1, 'legacy-hash')",
            [FILE],
        )
        .expect("insert legacy file");
    connection
        .execute(
            "INSERT INTO chunks(file_path, start_line, end_line, text, token_count)
             VALUES(?1, 1, 1, ?2, 1)",
            params![FILE, token],
        )
        .expect("insert legacy chunk");
    let rowid = connection.last_insert_rowid();
    connection
        .execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES(?1, ?2)",
            params![rowid, vec![0.0_f32; 768].as_bytes()],
        )
        .expect("insert unknown legacy vector");
}

fn remove_database(path: &Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("db-wal"));
    let _ = std::fs::remove_file(path.with_extension("db-shm"));
}

fn replacement(token: &str, embedding: Vec<f32>) -> ChunkInsert {
    ChunkInsert {
        file_path: FILE.into(),
        start_line: 1,
        end_line: 1,
        text: token.into(),
        token_count: 1,
        embedding,
    }
}

#[test]
fn version_two_upgrade_removes_only_unknown_vectors_and_preserves_text_state() {
    // **Validates: Requirements 2.12, 3.2**
    let path = database_path("upgrade");
    let legacy = create_legacy_database(&path);
    seed_legacy_rows(&legacy, "legacy_keyword");
    drop(legacy);

    let store = VecStore::open(&path).expect("migrate legacy database");
    let version: i32 = store
        .conn()
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .unwrap();
    let metadata: (i64, Option<String>, Option<String>, Option<i64>) = store
        .conn()
        .query_row(
            "SELECT embedding_valid, embedding_provider, embedding_model, embedding_dimension
             FROM chunks",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();

    assert_eq!(version, CURRENT_VERSION);
    assert_eq!(metadata, (0, None, None, None));
    assert_eq!(store.chunk_counts().unwrap(), (1, 0, 1));
    assert_eq!(
        search(&store, "legacy_keyword", None, &[], SearchMode::Keyword, 5)
            .unwrap()
            .len(),
        1
    );
    let file_count: i64 = store
        .conn()
        .query_row("SELECT count(*) FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(file_count, 1);

    drop(store);
    remove_database(&path);
}

#[test]
fn embedding_metadata_migration_rolls_back_every_change_on_failure() {
    // **Validates: Requirements 2.12**
    for step in [
        MigrationStep::MetadataAdded,
        MigrationStep::LegacyVectorsRemoved,
        MigrationStep::VersionUpdated,
    ] {
        let path = database_path(&format!("rollback-{step:?}"));
        let connection = create_legacy_database(&path);
        seed_legacy_rows(&connection, "rollback_keyword");

        let result = run_migrations_with_failure(&connection, step);
        assert!(result.is_err(), "missing injected failure after {step:?}");

        let version: i32 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        let columns: Vec<String> = connection
            .prepare("PRAGMA table_info(chunks)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let vector_count: i64 = connection
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))
            .unwrap();
        let fts_count: i64 = connection
            .query_row("SELECT count(*) FROM chunks_fts", [], |row| row.get(0))
            .unwrap();

        assert_eq!(version, 1, "version changed after {step:?}");
        assert!(
            !columns.iter().any(|column| column == "embedding_valid"),
            "metadata survived rollback after {step:?}"
        );
        assert_eq!(vector_count, 1, "legacy vector lost after {step:?}");
        assert_eq!(fts_count, 1, "legacy FTS row lost after {step:?}");

        drop(connection);
        remove_database(&path);
    }
}

#[test]
fn reindex_repopulates_valid_embedding_metadata_and_vectors() {
    // **Validates: Requirements 2.12, 3.2**
    let path = database_path("reindex");
    let legacy = create_legacy_database(&path);
    seed_legacy_rows(&legacy, "legacy_keyword");
    drop(legacy);

    let store = VecStore::open(&path).expect("migrate legacy database");
    let profile = EmbeddingProfile::new("gemini", "replacement-model", 768);
    store
        .replace_file_with_profile(
            FileRecord {
                path: FILE.into(),
                mtime: 2,
                content_hash: "replacement-hash".into(),
            },
            &[replacement("replacement_keyword", vec![0.75; 768])],
            &profile,
        )
        .expect("re-index migrated file");

    let metadata: (i64, String, String, i64) = store
        .conn()
        .query_row(
            "SELECT embedding_valid, embedding_provider, embedding_model, embedding_dimension
             FROM chunks",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    let report = search_with_profile(
        &store,
        "",
        Some(&vec![0.75; 768]),
        &[],
        SearchMode::Vector,
        5,
        &profile,
    )
    .expect("compatible vector search");

    assert_eq!(
        metadata,
        (1, "gemini".into(), "replacement-model".into(), 768)
    );
    assert_eq!(store.chunk_counts().unwrap(), (1, 1, 1));
    assert_eq!(report.hits.len(), 1);
    assert!(report.bm25_only_reason.is_none());
    assert!(
        search(&store, "legacy_keyword", None, &[], SearchMode::Keyword, 5)
            .unwrap()
            .is_empty()
    );

    drop(store);
    remove_database(&path);
}

#[test]
fn model_and_dimension_mismatches_are_excluded_with_actionable_reasons() {
    // **Validates: Requirements 2.12**
    let store = VecStore::open_memory().expect("open database");
    let indexed_profile = EmbeddingProfile::new("gemini", "indexed-model", 768);
    store.upsert_file(FILE, 1, "hash").unwrap();
    store
        .insert_chunks_with_profile(
            &[replacement("compatible_keyword", vec![0.4; 768])],
            &indexed_profile,
        )
        .unwrap();

    let model_report = search_with_profile(
        &store,
        "",
        Some(&vec![0.4; 768]),
        &[],
        SearchMode::Vector,
        5,
        &EmbeddingProfile::new("gemini", "other-model", 768),
    )
    .expect("model mismatch report");
    assert!(model_report.hits.is_empty());
    let model_reason = model_report.bm25_only_reason.unwrap();
    assert!(model_reason.contains("other-model"));
    assert!(model_reason.contains("srijandev index"));

    let dimension_report = search_with_profile(
        &store,
        "",
        Some(&vec![0.4; 384]),
        &[],
        SearchMode::Vector,
        5,
        &EmbeddingProfile::new("gemini", "indexed-model", 384),
    )
    .expect("dimension mismatch report");
    assert!(dimension_report.hits.is_empty());
    let dimension_reason = dimension_report.bm25_only_reason.unwrap();
    assert!(dimension_reason.contains("dimension 384"));
    assert!(dimension_reason.contains("srijandev index"));
}

#[test]
fn hybrid_search_falls_back_to_keyword_for_invalid_or_incompatible_embeddings() {
    // **Validates: Requirements 2.12, 3.2**
    let store = VecStore::open_memory().expect("open database");
    store.upsert_file(FILE, 1, "keyword-hash").unwrap();
    store
        .insert_chunks(&[replacement("keyword_fallback_token", vec![0.0; 768])])
        .expect("keyword-only chunk");

    let metadata: (i64, Option<String>, Option<String>, Option<i64>) = store
        .conn()
        .query_row(
            "SELECT embedding_valid, embedding_provider, embedding_model, embedding_dimension
             FROM chunks",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    let report = search_with_profile(
        &store,
        "keyword_fallback_token",
        Some(&vec![0.2; 768]),
        &[],
        SearchMode::Hybrid,
        5,
        &EmbeddingProfile::default(),
    )
    .expect("hybrid keyword fallback");

    assert_eq!(metadata, (0, None, None, None));
    assert_eq!(store.chunk_counts().unwrap(), (1, 0, 1));
    assert_eq!(report.hits.len(), 1);
    assert!(report.hits[0].text.contains("keyword_fallback_token"));
    let reason = report.bm25_only_reason.unwrap();
    assert!(reason.starts_with("BM25-only:"));
    assert!(reason.contains("srijandev index"));
}

/// A width other than the historical 768 must be storable once the table is
/// rebuilt for it.
///
/// This is a regression test for a real failure found by running a live
/// 1536-wide model: the storage guard compared the profile against the
/// compile-time `EMBEDDING_DIMENSION` constant instead of the width the vector
/// table actually declares, so `cli index` failed with
/// "embedding profile dimension 1536 does not match VecStore dimension 768"
/// even though the table had already been rebuilt correctly. Unit tests missed
/// it because they only ever exercised 768.
#[test]
fn a_rebuilt_table_accepts_its_own_width() {
    let store = VecStore::open_memory().expect("open database");

    for dimension in [1536, 3072, 1024, 768] {
        vecstore::ensure_vector_dimension(store.conn(), dimension).expect("rebuild vector table");
        let profile = EmbeddingProfile::new("azure", "text-embedding-3-small", dimension);

        store
            .replace_file_with_profile(
                FileRecord {
                    path: "wide.rs".into(),
                    mtime: 1,
                    content_hash: "h".into(),
                },
                &[ChunkInsert {
                    file_path: "wide.rs".into(),
                    start_line: 1,
                    end_line: 2,
                    text: "fn wide() {}".into(),
                    token_count: 3,
                    embedding: vec![0.01; dimension],
                }],
                &profile,
            )
            .unwrap_or_else(|error| panic!("storing a {dimension}-wide vector failed: {error}"));

        let (stored_dimension, valid): (i64, i64) = store
            .conn()
            .query_row(
                "SELECT embedding_dimension, embedding_valid FROM chunks WHERE file_path = 'wide.rs'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(stored_dimension, dimension as i64);
        assert_eq!(
            valid, 1,
            "a {dimension}-wide vector must be stored as valid"
        );
    }
}

/// Writing a width the table was not built for must still be refused, and the
/// message must say what to do about it.
#[test]
fn a_width_the_table_was_not_built_for_is_refused() {
    let store = VecStore::open_memory().expect("open database");
    vecstore::ensure_vector_dimension(store.conn(), 1536).expect("rebuild vector table");

    let error = store
        .replace_file_with_profile(
            FileRecord {
                path: "mismatch.rs".into(),
                mtime: 1,
                content_hash: "h".into(),
            },
            &[ChunkInsert {
                file_path: "mismatch.rs".into(),
                start_line: 1,
                end_line: 2,
                text: "fn mismatch() {}".into(),
                token_count: 3,
                embedding: vec![0.01; 768],
            }],
            &EmbeddingProfile::new("gemini", "text-embedding-004", 768),
        )
        .expect_err("a 768-wide profile must not be written into a 1536-wide table")
        .to_string();

    assert!(
        error.contains("768") && error.contains("1536"),
        "the error must name both widths: {error}"
    );
    assert!(
        error.contains("re-index"),
        "the error must tell the operator to re-index: {error}"
    );
}
