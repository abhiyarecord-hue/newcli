//! Hybrid 3-mode retrieval: Vector KNN, FTS5 BM25, Graph CTE.
//! Fusion via Reciprocal Rank Fusion (RRF) `score = sum(1/(60 + rank_i))`.

use rusqlite::params;
use zerocopy::AsBytes;

use agent_types::{AgentError, Result};

use crate::store::VecStore;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub chunk_id: i64,
    pub text: String,
    pub score: f64,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    Vector,
    Keyword,
    Graph,
    Hybrid,
}

/// Main search entry point.
pub fn search(
    store: &VecStore,
    query: &str,
    query_embedding: Option<&[f32]>,
    seed_entity_names: &[String],
    mode: SearchMode,
    k: usize,
) -> Result<Vec<SearchHit>> {
    match mode {
        SearchMode::Vector => {
            let emb = query_embedding
                .ok_or_else(|| AgentError::Index("vector search requires embedding".into()))?;
            vector_search(store, emb, k)
        }
        SearchMode::Keyword => keyword_search(store, query, k),
        SearchMode::Graph => graph_search(store, seed_entity_names, k),
        SearchMode::Hybrid => {
            let mut ranked: Vec<(i64, f64)> = Vec::new();

            // Vector results (if embedding provided).
            if let Some(emb) = query_embedding {
                let vec_hits = vector_search(store, emb, k * 2)?;
                for (rank, hit) in vec_hits.iter().enumerate() {
                    add_rrf(&mut ranked, hit.chunk_id, rank);
                }
            }

            // Keyword results.
            if !query.trim().is_empty() {
                let kw_hits = keyword_search(store, query, k * 2)?;
                for (rank, hit) in kw_hits.iter().enumerate() {
                    add_rrf(&mut ranked, hit.chunk_id, rank);
                }
            }

            // Graph results.
            if !seed_entity_names.is_empty() {
                let graph_hits = graph_search(store, seed_entity_names, k * 2)?;
                for (rank, hit) in graph_hits.iter().enumerate() {
                    add_rrf(&mut ranked, hit.chunk_id, rank);
                }
            }

            // Sort by RRF score descending, dedupe (already unique by chunk_id).
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(k);

            // Hydrate results.
            let mut results = Vec::new();
            for (chunk_id, score) in ranked {
                if let Ok(hit) = hydrate(store, chunk_id, score) {
                    results.push(hit);
                }
            }
            Ok(results)
        }
    }
}

fn add_rrf(ranked: &mut Vec<(i64, f64)>, chunk_id: i64, rank: usize) {
    let rrf_score = 1.0 / (60.0 + rank as f64);
    if let Some(entry) = ranked.iter_mut().find(|(id, _)| *id == chunk_id) {
        entry.1 += rrf_score;
    } else {
        ranked.push((chunk_id, rrf_score));
    }
}

fn vector_search(store: &VecStore, embedding: &[f32], k: usize) -> Result<Vec<SearchHit>> {
    let conn = store.conn();
    let mut stmt = conn
        .prepare(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
        )
        .map_err(|e| AgentError::Storage(e.to_string()))?;

    let rows: Vec<(i64, f64)> = stmt
        .query_map(params![embedding.as_bytes(), k as i64], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .map_err(|e| AgentError::Storage(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();

    let mut hits = Vec::new();
    for (rowid, distance) in rows {
        // Convert distance to a similarity score (lower distance = higher score).
        let score = 1.0 / (1.0 + distance);
        if let Ok(hit) = hydrate(store, rowid, score) {
            hits.push(hit);
        }
    }
    Ok(hits)
}

fn keyword_search(store: &VecStore, query: &str, k: usize) -> Result<Vec<SearchHit>> {
    let conn = store.conn();

    // Sanitize FTS5 query: quote each term to avoid syntax errors from special chars.
    let sanitized = sanitize_fts_query(query);
    if sanitized.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn
        .prepare(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY rank LIMIT ?2",
        )
        .map_err(|e| AgentError::Storage(e.to_string()))?;

    let rows: Vec<(i64, f64)> = stmt
        .query_map(params![sanitized, k as i64], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .map_err(|e| AgentError::Storage(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();

    let mut hits = Vec::new();
    for (rowid, bm25_rank) in rows {
        // BM25 in SQLite returns negative-is-better; normalize to positive score.
        let score = -bm25_rank;
        if let Ok(hit) = hydrate(store, rowid, score) {
            hits.push(hit);
        }
    }
    Ok(hits)
}

fn graph_search(store: &VecStore, seed_names: &[String], k: usize) -> Result<Vec<SearchHit>> {
    let conn = store.conn();

    // Ensure entity_edges table exists (created lazily for graph mode).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entity_edges (
            src_id INTEGER NOT NULL,
            dst_id INTEGER NOT NULL,
            kind TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(src_id);",
    )
    .map_err(|e| AgentError::Storage(e.to_string()))?;

    // Also need entity_chunks mapping.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entity_chunks_eid ON entity_chunks(entity_id);",
    )
    .map_err(|e| AgentError::Storage(e.to_string()))?;

    // Find seed entity ids by name. We need an entities table for this.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            qualified_name TEXT NOT NULL,
            kind TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(qualified_name);",
    )
    .map_err(|e| AgentError::Storage(e.to_string()))?;

    if seed_names.is_empty() {
        return Ok(Vec::new());
    }

    // Build placeholders for seed names.
    let placeholders: Vec<String> = seed_names.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect();
    let in_clause = placeholders.join(",");

    let query_str = format!(
        "WITH RECURSIVE reachable(id, depth) AS (
            SELECT id, 0 FROM entities WHERE qualified_name IN ({in_clause})
            UNION
            SELECT e.dst_id, r.depth + 1
            FROM entity_edges e
            JOIN reachable r ON e.src_id = r.id
            WHERE r.depth < 2
            LIMIT 500
        )
        SELECT DISTINCT ec.chunk_id
        FROM reachable r
        JOIN entity_chunks ec ON ec.entity_id = r.id
        LIMIT ?{}", seed_names.len() + 1
    );

    let mut stmt = conn
        .prepare(&query_str)
        .map_err(|e| AgentError::Storage(e.to_string()))?;

    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for name in seed_names {
        param_values.push(Box::new(name.clone()));
    }
    param_values.push(Box::new(k as i64));

    let params_ref: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|b| b.as_ref()).collect();

    let chunk_ids: Vec<i64> = stmt
        .query_map(params_ref.as_slice(), |r| r.get(0))
        .map_err(|e| AgentError::Storage(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();

    let mut hits = Vec::new();
    for (rank, chunk_id) in chunk_ids.iter().enumerate() {
        let score = 1.0 / (1.0 + rank as f64);
        if let Ok(hit) = hydrate(store, *chunk_id, score) {
            hits.push(hit);
        }
    }
    Ok(hits)
}

/// Hydrate a chunk id into a full SearchHit.
fn hydrate(store: &VecStore, chunk_id: i64, score: f64) -> Result<SearchHit> {
    let conn = store.conn();
    conn.query_row(
        "SELECT id, file_path, start_line, end_line, text FROM chunks WHERE id = ?1",
        params![chunk_id],
        |r| {
            Ok(SearchHit {
                chunk_id: r.get(0)?,
                file_path: r.get(1)?,
                start_line: r.get::<_, u32>(2)?,
                end_line: r.get::<_, u32>(3)?,
                text: r.get(4)?,
                score,
            })
        },
    )
    .map_err(|e| AgentError::Storage(e.to_string()))
}

/// Sanitize an FTS5 query by quoting each whitespace-separated term.
/// Strips characters that break FTS5 syntax (`"`, `*`, `(`, `)`, `:`).
fn sanitize_fts_query(query: &str) -> String {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|t| {
            let clean: String = t
                .chars()
                .filter(|c| !matches!(c, '"' | '*' | '(' | ')' | ':'))
                .collect();
            clean
        })
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\""))
        .collect();
    terms.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ChunkInsert;

    fn setup_store() -> VecStore {
        let store = VecStore::open_memory().unwrap();
        store.upsert_file("a.rs", 1, "h1").unwrap();
        let chunks: Vec<ChunkInsert> = vec![
            ChunkInsert {
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 10,
                text: "fn hello_world() { println!(\"hi\"); }".into(),
                token_count: 8,
                embedding: vec![0.1; 768],
            },
            ChunkInsert {
                file_path: "a.rs".into(),
                start_line: 11,
                end_line: 20,
                text: "fn search_algo(query: &str) { /* bm25 ranking */ }".into(),
                token_count: 10,
                embedding: vec![0.9; 768],
            },
            ChunkInsert {
                file_path: "a.rs".into(),
                start_line: 21,
                end_line: 30,
                text: "struct Config { max_results: usize }".into(),
                token_count: 6,
                embedding: vec![0.5; 768],
            },
        ];
        store.insert_chunks(&chunks).unwrap();
        store
    }

    #[test]
    fn vector_search_returns_nearest() {
        let store = setup_store();
        let query_emb: Vec<f32> = vec![0.9; 768]; // closest to chunk 2
        let hits = search(&store, "", Some(&query_emb), &[], SearchMode::Vector, 2).unwrap();
        assert_eq!(hits.len(), 2);
        // First hit should be chunk 2 (search_algo) since its embedding is [0.9; 768].
        assert!(hits[0].text.contains("search_algo"));
    }

    #[test]
    fn keyword_search_finds_matching_text() {
        let store = setup_store();
        let hits = search(&store, "search_algo", None, &[], SearchMode::Keyword, 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].text.contains("search_algo"));
    }

    #[test]
    fn hybrid_returns_union_not_intersection() {
        let store = setup_store();
        // Query embedding close to chunk 1 (hello_world), keyword matches chunk 2.
        let query_emb: Vec<f32> = vec![0.1; 768];
        let hits = search(
            &store,
            "search_algo",
            Some(&query_emb),
            &[],
            SearchMode::Hybrid,
            5,
        )
        .unwrap();
        // Should contain both hello_world (from vector) and search_algo (from keyword).
        let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("hello_world")),
            "vector hit missing from hybrid"
        );
        assert!(
            texts.iter().any(|t| t.contains("search_algo")),
            "keyword hit missing from hybrid"
        );
    }

    #[test]
    fn sanitize_fts_strips_special_chars() {
        assert_eq!(sanitize_fts_query("hello \"world*"), "\"hello\" \"world\"");
        assert_eq!(sanitize_fts_query(""), "");
    }

    #[test]
    fn keyword_search_with_empty_query_is_empty() {
        let store = setup_store();
        let hits = search(&store, "   ", None, &[], SearchMode::Keyword, 5).unwrap();
        assert!(hits.is_empty());
    }
}
