use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vecstore::{search, ChunkInsert, FileRecord, ReplacementStep, SearchMode, VecStore};

const FILE: &str = "src/a.rs";
static NEXT_DB: AtomicU64 = AtomicU64::new(0);

fn database_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vecstore-{label}-{}-{}.db",
        std::process::id(),
        NEXT_DB.fetch_add(1, Ordering::Relaxed)
    ))
}

fn record(hash: &str, mtime: i64) -> FileRecord {
    FileRecord {
        path: FILE.into(),
        mtime,
        content_hash: hash.into(),
    }
}

fn chunk(token: &str, embedding: Vec<f32>) -> ChunkInsert {
    ChunkInsert {
        file_path: FILE.into(),
        start_line: 1,
        end_line: 1,
        text: token.into(),
        token_count: 1,
        embedding,
    }
}

fn seed_old(store: &VecStore) {
    store
        .replace_file(
            record("old-hash", 1),
            &[chunk("old_token", vec![0.25; 768])],
        )
        .expect("seed old searchable state");
}
fn snapshot(store: &VecStore) -> (String, bool, bool, (i64, i64, i64)) {
    let hash = store
        .conn()
        .query_row(
            "SELECT content_hash FROM files WHERE path = ?1",
            [FILE],
            |row| row.get(0),
        )
        .expect("file metadata");
    let old_found = !search(store, "old_token", None, &[], SearchMode::Keyword, 5)
        .expect("search old token")
        .is_empty();
    let new_found = !search(store, "new_token", None, &[], SearchMode::Keyword, 5)
        .expect("search new token")
        .is_empty();
    (hash, old_found, new_found, store.chunk_counts().unwrap())
}

fn remove_database(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("db-wal"));
    let _ = std::fs::remove_file(path.with_extension("db-shm"));
}

#[test]
fn every_replacement_failure_exposes_a_complete_state_after_restart() {
    // **Validates: Requirements 2.1**
    let steps = [
        ReplacementStep::StaleVectorsDeleted,
        ReplacementStep::StaleChunksDeleted,
        ReplacementStep::FileMetadataUpserted,
        ReplacementStep::ReplacementChunksInserted,
        ReplacementStep::ValidVectorsInserted,
        ReplacementStep::Committed,
    ];

    for step in steps {
        let path = database_path(&format!("{step:?}"));
        let store = VecStore::open(&path).expect("open database");
        seed_old(&store);
        let result = store.replace_file_with_failure(
            record("new-hash", 2),
            &[chunk("new_token", vec![0.75; 768])],
            step,
        );
        assert!(result.is_err(), "missing injected failure at {step:?}");
        drop(store);

        let reopened = VecStore::open(&path).expect("restart database");
        let actual = snapshot(&reopened);
        let expected = if step == ReplacementStep::Committed {
            ("new-hash".into(), false, true, (1, 1, 1))
        } else {
            ("old-hash".into(), true, false, (1, 1, 1))
        };
        assert_eq!(actual, expected, "partial state after restart at {step:?}");
        drop(reopened);
        remove_database(&path);
    }
}
#[test]
fn keyword_only_replacement_removes_stale_vectors_and_remains_searchable() {
    // **Validates: Requirements 3.2**
    let store = VecStore::open_memory().expect("open database");
    seed_old(&store);

    store
        .replace_file(record("keyword-hash", 3), &[chunk("new_token", Vec::new())])
        .expect("keyword-only replacement");

    assert_eq!(
        snapshot(&store),
        ("keyword-hash".into(), false, true, (1, 0, 1))
    );
    assert!(search(
        &store,
        "",
        Some(&vec![0.0; 768]),
        &[],
        SearchMode::Vector,
        5,
    )
    .expect("vector search")
    .is_empty());
}

#[test]
fn malformed_replacement_plan_is_rejected_before_state_changes() {
    // **Validates: Requirements 2.1**
    let store = VecStore::open_memory().expect("open database");
    seed_old(&store);

    let result = store.replace_file(
        record("invalid-hash", 4),
        &[chunk("new_token", vec![0.5; 12])],
    );

    assert!(result.is_err());
    assert_eq!(
        snapshot(&store),
        ("old-hash".into(), true, false, (1, 1, 1))
    );
}
