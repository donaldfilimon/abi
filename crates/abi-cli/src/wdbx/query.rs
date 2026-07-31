//! WDBX `query` subcommand: hybrid semantic x temporal x causal x persona retrieval and stats.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use crate::wdbx::paths_from_cli_base;
use abi_ai::text_embedding;
use abi_wdbx::{
    DurableStore, HybridScorer, TemporalCausalGraph, hybrid_search_scoped,
    hybrid_search_with_persona,
};
use std::fmt::Write;

pub(crate) const QUERY_HELP: &str = "usage: abi wdbx query <path> [text] [persona] [--limit N] [--json] [--text T] [--persona P]\n\nPrint store stats (no text) or run hybrid semantic retrieval (semantic × temporal × causal × persona).\nPersona isolates results to that persona's memories. --limit defaults to 10. --json emits a machine-\nreadable result list (ranking=hybrid) with borrowed vector dims (zero-copy getVector view).\n";

#[derive(Debug, PartialEq, Eq)]
struct QueryOptions<'a> {
    path: &'a str,
    text: Option<&'a str>,
    persona: Option<&'a str>,
    limit: usize,
    json: bool,
}

fn parse_query(args: &[String]) -> Result<QueryOptions<'_>, ()> {
    let path = args.first().ok_or(())?.as_str();
    let mut text = None;
    let mut persona = None;
    let mut positionals = Vec::with_capacity(2);
    let mut limit = 10;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--limit" => {
                index += 1;
                limit = args.get(index).ok_or(())?.parse().map_err(|_| ())?;
                if limit == 0 {
                    return Err(());
                }
            }
            "--text" => {
                index += 1;
                text = Some(args.get(index).ok_or(())?.as_str());
            }
            "--persona" => {
                index += 1;
                persona = Some(args.get(index).ok_or(())?.as_str());
            }
            token if token.starts_with("--") => return Err(()),
            token => {
                if positionals.len() == 2 {
                    return Err(());
                }
                positionals.push(token);
            }
        }
        index += 1;
    }
    if text.is_none() {
        text = positionals.first().copied();
    }
    if persona.is_none() {
        persona = positionals.get(1).copied();
    }
    Ok(QueryOptions {
        path,
        text,
        persona,
        limit,
        json,
    })
}

pub(crate) fn backend_label() -> &'static str {
    abi_gpu::detect_backend().backend.name()
}

fn store_manifest(store: &DurableStore) -> String {
    let stats = store.stats();
    let dimensions = store
        .snapshot()
        .vector_dimensions()
        .map_or_else(|| "null".to_owned(), |value| value.to_string());
    format!(
        "{{\"kv_entries\":{},\"vectors\":{},\"blocks\":{},\"spatial_records\":{},\"temporal_nodes\":{},\"temporal_edges\":{},\"vector_dimensions\":{dimensions},\"next_vector_id\":{},\"backend\":\"{}\",\"mode\":\"cpu_fallback\"}}",
        stats.kv_entries,
        stats.vectors,
        stats.blocks,
        stats.spatial_records,
        stats.temporal_nodes,
        stats.temporal_edges,
        store.next_vector_id(),
        backend_label()
    )
}

pub(crate) fn query(args: &[String]) -> Outcome {
    let Ok(options) = parse_query(args) else {
        return super::usage();
    };
    let paths = match paths_from_cli_base(options.path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("query failed", detail),
    };
    let store = match DurableStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return super::error(&format!("error: {}", options.path), detail),
    };
    let manifest = store_manifest(&store);
    let Some(text) = options.text else {
        if options.json {
            let path = serde_json::to_string(options.path).expect("a string always serializes");
            return Outcome::stderr(
                format!(
                    "{{\"path\":{path},\"mode\":\"stats\",\"ranking\":null,\"manifest\":{manifest}}}\n"
                ),
                0,
            );
        }
        return Outcome::stderr(format!("{manifest}\n"), 0);
    };

    if store.stats().vectors == 0 {
        if options.json {
            let path = serde_json::to_string(options.path).expect("a string always serializes");
            let query = serde_json::to_string(text).expect("a string always serializes");
            let persona = options.persona.unwrap_or("all");
            return Outcome::stderr(
                format!(
                    "{{\"path\":{path},\"query\":{query},\"persona\":{},\"ranking\":\"hybrid\",\"limit\":{},\"vectors\":0,\"results\":[]}}\n",
                    serde_json::to_string(persona).expect("string"),
                    options.limit
                ),
                0,
            );
        }
        return Outcome::stderr(
            format!(
                "no vectors in {}; nothing to rank (populate with `abi complete`)\n",
                options.path
            ),
            0,
        );
    }

    query_with_text(
        &store,
        options.path,
        text,
        options.persona,
        options.limit,
        options.json,
    )
}

/// Hybrid semantic×temporal×causal×persona ranking over a recovered store.
fn query_with_text(
    store: &DurableStore,
    path: &str,
    text: &str,
    persona: Option<&str>,
    limit: usize,
    json: bool,
) -> Outcome {
    let query_vec = text_embedding(text);
    let graph = TemporalCausalGraph::from_records(&store.snapshot().temporal);
    let now_ms = abi_foundation::time::unix_ms();
    let scorer = HybridScorer {
        now_ms,
        ..HybridScorer::new(now_ms)
    };
    let focus_id = store.snapshot().vectors.keys().next().copied().unwrap_or(1);

    let ranked = if let Some(filter) = persona {
        let filter_owned = filter.to_ascii_lowercase();
        hybrid_search_scoped(store, &query_vec, limit, &graph, scorer, focus_id, |id| {
            vector_persona_matches(store, id, &filter_owned)
        })
    } else {
        hybrid_search_with_persona(store, &query_vec, limit, &graph, scorer, focus_id, |_id| {
            0.5
        })
    };

    let ranked = match ranked {
        Ok(rows) => rows,
        Err(detail) => return super::error("query failed", detail),
    };

    let persona_label = persona.unwrap_or("all");
    if json {
        let path_json = serde_json::to_string(path).expect("path serializes");
        let query_json = serde_json::to_string(text).expect("query serializes");
        let persona_json = serde_json::to_string(persona_label).expect("persona serializes");
        let mut results = String::from("[");
        for (i, row) in ranked.iter().enumerate() {
            if i > 0 {
                results.push(',');
            }
            let dims = row.vector.len();
            let _ = write!(
                results,
                "{{\"id\":{},\"score\":{:.6},\"semantic\":{:.6},\"temporal\":{:.6},\"causal\":{:.6},\"persona\":{:.6},\"dims\":{dims}}}",
                row.id,
                row.score,
                row.components.semantic,
                row.components.temporal,
                row.components.causal,
                row.components.persona,
            );
        }
        results.push(']');
        return Outcome::stderr(
            format!(
                "{{\"path\":{path_json},\"query\":{query_json},\"persona\":{persona_json},\"ranking\":\"hybrid\",\"limit\":{limit},\"vectors\":{},\"results\":{results}}}\n",
                store.stats().vectors
            ),
            0,
        );
    }

    if ranked.is_empty() {
        return Outcome::stderr(
            format!("no hybrid matches in {path} for query (persona={persona_label})\n"),
            0,
        );
    }

    let mut out =
        format!("query results (hybrid ranking, limit={limit}, persona={persona_label}):\n");
    for (i, row) in ranked.iter().enumerate() {
        let _ = writeln!(
            out,
            "  #{} id={} score={:.4} semantic={:.4} temporal={:.4} causal={:.4} persona={:.4} dims={}",
            i + 1,
            row.id,
            row.score,
            row.components.semantic,
            row.components.temporal,
            row.components.causal,
            row.components.persona,
            row.vector.len(),
        );
    }
    Outcome::stderr(out, 0)
}

fn vector_persona_matches(store: &DurableStore, id: u64, filter_lower: &str) -> bool {
    if let Some(label) = store.get(&format!("wdbx:profile:{id}")) {
        return label.eq_ignore_ascii_case(filter_lower);
    }
    // Fall back to block profile tags that reference this vector as query or response.
    for block in &store.snapshot().blocks {
        if (block.query_id == id || block.response_id == id)
            && block.profile.eq_ignore_ascii_case(filter_lower)
        {
            return true;
        }
    }
    false
}
