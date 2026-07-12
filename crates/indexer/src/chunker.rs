//! Greedy recursive AST chunker.
//!
//! Walks top-level entities in byte order; an entity that fits the token
//! budget becomes one intact unit (a function is never cut mid-body when it
//! fits). An entity over budget is decomposed into its children via an
//! EXPLICIT stack of entity ids (`Vec<u64>`) — same stack-overflow hazard as
//! the parser, so no recursion. A childless leaf over budget is split at line
//! boundaries as a last resort.
//!
//! Overlap: the last `overlap_nodes` units of the previous chunk are prepended
//! to the next chunk's *text* but do NOT count against the budget — otherwise
//! chunks could shrink to overlap-only and the loop would never advance. Unit
//! iteration is linear, so the byte cursor makes forward progress every step.
//!
//! The token heuristic mirrors `compaction::estimate_tokens` exactly
//! (`(chars/4).max(words)`); it is re-stated locally to keep the crate
//! dependency matrix (agent-types + runtime-core only) intact.

use std::collections::HashMap;

use crate::entity::{Chunk, CodeEntity};

/// Deterministic token estimate — identical formula to
/// `compaction::estimate_tokens`.
fn estimate_tokens(s: &str) -> u32 {
    let chars = s.chars().count() as u32;
    let words = s.split_whitespace().count() as u32;
    (chars / 4).max(words)
}

struct Unit {
    start_line: u32,
    end_line: u32,
    text: String,
    tokens: u32,
    entity_ids: Vec<u64>,
}

/// Chunk `entities` (of a single file) into token-bounded windows.
pub fn chunk(
    entities: &[CodeEntity],
    source: &str,
    budget: u32,
    overlap_nodes: usize,
) -> Vec<Chunk> {
    if entities.is_empty() {
        return Vec::new();
    }
    let file = entities[0].path.clone();

    let by_id: HashMap<u64, &CodeEntity> = entities.iter().map(|e| (e.id, e)).collect();
    let mut children: HashMap<Option<u64>, Vec<u64>> = HashMap::new();
    for e in entities {
        children.entry(e.parent_id).or_default().push(e.id);
    }
    for v in children.values_mut() {
        v.sort_by_key(|id| by_id[id].byte_range.0);
    }

    // 1) Flatten into token-bounded units via an explicit stack.
    let mut units: Vec<Unit> = Vec::new();
    let mut stack: Vec<u64> = Vec::new();
    if let Some(tops) = children.get(&None) {
        for id in tops.iter().rev() {
            stack.push(*id);
        }
    }
    while let Some(id) = stack.pop() {
        let e = by_id[&id];
        let text = source
            .get(e.byte_range.0..e.byte_range.1)
            .unwrap_or_default();
        let tokens = estimate_tokens(text);

        if tokens <= budget {
            units.push(Unit {
                start_line: e.line_range.0,
                end_line: e.line_range.1,
                text: text.to_string(),
                tokens,
                entity_ids: vec![e.id],
            });
            continue;
        }

        match children.get(&Some(id)) {
            Some(kids) if !kids.is_empty() => {
                for k in kids.iter().rev() {
                    stack.push(*k);
                }
            }
            _ => split_lines(e, text, budget, &mut units),
        }
    }

    // 2) Greedily pack units into budget-bounded chunks with overlap.
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut window: Vec<usize> = Vec::new();
    let mut window_tokens: u32 = 0;
    let mut prev_overlap: Vec<usize> = Vec::new();

    for idx in 0..units.len() {
        let unit_tokens = units[idx].tokens;
        if !window.is_empty() && window_tokens + unit_tokens > budget {
            chunks.push(make_chunk(&units, &window, &prev_overlap, &file));
            let keep_from = window.len().saturating_sub(overlap_nodes);
            prev_overlap = window[keep_from..].to_vec();
            window.clear();
            window_tokens = 0;
        }
        window.push(idx);
        window_tokens += unit_tokens;
    }
    if !window.is_empty() {
        chunks.push(make_chunk(&units, &window, &prev_overlap, &file));
    }

    chunks
}

/// Split an over-budget, childless leaf at line boundaries. A single line
/// longer than the budget is emitted as its own (oversized) unit.
fn split_lines(e: &CodeEntity, text: &str, budget: u32, out: &mut Vec<Unit>) {
    let base_line = e.line_range.0;
    let mut cur = String::new();
    let mut cur_tokens = 0u32;
    let mut cur_start = base_line;
    let mut line_no = base_line;

    let flush = |cur: &mut String, cur_tokens: &mut u32, start: u32, end: u32, out: &mut Vec<Unit>, id: u64| {
        if cur.is_empty() {
            return;
        }
        out.push(Unit {
            start_line: start,
            end_line: end,
            text: std::mem::take(cur),
            tokens: *cur_tokens,
            entity_ids: vec![id],
        });
        *cur_tokens = 0;
    };

    for line in text.split('\n') {
        let lt = estimate_tokens(line);
        if !cur.is_empty() && cur_tokens + lt > budget {
            flush(&mut cur, &mut cur_tokens, cur_start, line_no.saturating_sub(1), out, e.id);
            cur_start = line_no;
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        cur_tokens += lt;
        line_no += 1;
    }
    flush(&mut cur, &mut cur_tokens, cur_start, line_no.saturating_sub(1), out, e.id);
}

fn make_chunk(units: &[Unit], window: &[usize], overlap: &[usize], file: &std::path::Path) -> Chunk {
    let overlap_text: String = overlap
        .iter()
        .map(|&i| units[i].text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let body_text: String = window
        .iter()
        .map(|&i| units[i].text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let text = if overlap_text.is_empty() {
        body_text
    } else {
        format!("{overlap_text}\n{body_text}")
    };

    let token_count: u32 = window.iter().map(|&i| units[i].tokens).sum();
    let start_line = window.first().map(|&i| units[i].start_line).unwrap_or(0);
    let end_line = window.last().map(|&i| units[i].end_line).unwrap_or(0);

    let mut entity_ids: Vec<u64> = Vec::new();
    for &i in window {
        for id in &units[i].entity_ids {
            if !entity_ids.contains(id) {
                entity_ids.push(*id);
            }
        }
    }

    Chunk {
        file: file.to_path_buf(),
        start_line,
        end_line,
        text,
        token_count,
        entity_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use std::path::PathBuf;

    fn rust_src() -> String {
        // Three small free functions with distinctive bodies.
        "fn alpha() {\n    let a = 1;\n    let b = 2;\n}\n\
         fn beta() {\n    let c = 3;\n    let d = 4;\n}\n\
         fn gamma() {\n    let e = 5;\n    let f = 6;\n}\n"
            .to_string()
    }

    #[test]
    fn every_chunk_within_budget_when_units_fit() {
        let src = rust_src();
        let entities = parse(&PathBuf::from("m.rs"), &src).unwrap();
        let budget = 30;
        let chunks = chunk(&entities, &src, budget, 1);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(
                c.token_count <= budget,
                "chunk over budget: {} > {budget}",
                c.token_count
            );
        }
    }

    #[test]
    fn fitting_function_is_never_cut_mid_body() {
        let src = rust_src();
        let entities = parse(&PathBuf::from("m.rs"), &src).unwrap();
        // Budget large enough for a whole function but small enough to force
        // multiple chunks across the three functions.
        let chunks = chunk(&entities, &src, 12, 0);
        // The full body of `beta` must appear intact in exactly one chunk.
        let beta_body = "fn beta() {\n    let c = 3;\n    let d = 4;\n}";
        assert!(
            chunks.iter().any(|c| c.text.contains(beta_body)),
            "beta was split across chunks"
        );
    }

    #[test]
    fn overlap_prepends_previous_units() {
        let src = rust_src();
        let entities = parse(&PathBuf::from("m.rs"), &src).unwrap();
        // Force one function per chunk.
        let chunks = chunk(&entities, &src, 12, 1);
        assert!(chunks.len() >= 2, "expected multiple chunks");
        // The second chunk's text must begin with the last unit of the first.
        let first_last_line = "fn alpha()";
        assert!(
            chunks[1].text.contains(first_last_line),
            "overlap from previous chunk missing"
        );
    }

    #[test]
    fn empty_entities_yields_no_chunks() {
        assert!(chunk(&[], "", 100, 2).is_empty());
    }
}
