//! Evidence recall and prompt augmentation for SEA.
//!
//! Ported from `src/features/sea/evidence.zig`.

use abi_ai::{PROFILE_LABELS, text_embedding};
use abi_wdbx::DurableStore;

use crate::query_plan::{QueryPlan, infer};
use crate::types::Authority;

/// When a plan requests `exact_recall`, blend semantic score with lexical
/// overlap at this weight (even mix).
const EXACT_RECALL_KEYWORD_WEIGHT: f32 = 0.5;

/// Upper bound on the augmented-prompt preamble.
pub const MAX_PROMPT_BYTES: usize = 4096;

const UNKNOWN_PROFILE: &str = "unknown";

/// One recalled record. `snippet` is owned; `profile_label` is a static borrow.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceItem {
    /// Vector id of the hit.
    pub vector_id: u64,
    /// Persona label (`abbey`/`aviva`/`abi`/`unknown`).
    pub profile_label: &'static str,
    /// Forced-to-inferred authority for generic-store records.
    pub authority: Authority,
    /// Owned metadata snippet.
    pub snippet: String,
    /// Final relevance score.
    pub score: f32,
}

/// Owned collection of recalled evidence.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvidenceContext {
    /// Recalled items, highest score first.
    pub items: Vec<EvidenceItem>,
}

impl EvidenceContext {
    /// Whether the context holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

struct ParsedStoredMetadata {
    profile_label: &'static str,
    // Authority is always forced to Inferred for generic store records; the
    // field is parsed only to document that a claim was observed.
}

/// Parse only exact top-level JSON fields. Generic-store authority cannot
/// self-promote above [`Authority::Inferred`].
fn parse_stored_metadata(metadata: &str) -> ParsedStoredMetadata {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(metadata) else {
        return ParsedStoredMetadata {
            profile_label: UNKNOWN_PROFILE,
        };
    };
    let Some(object) = value.as_object() else {
        return ParsedStoredMetadata {
            profile_label: UNKNOWN_PROFILE,
        };
    };

    let mut profile_label = UNKNOWN_PROFILE;
    if let Some(profile) = object.get("profile").and_then(|v| v.as_str()) {
        for label in PROFILE_LABELS {
            if profile == label {
                profile_label = label;
                break;
            }
        }
    }

    // Observe but do not accept a self-asserted authority claim.
    if let Some(auth) = object.get("authority").and_then(|v| v.as_str()) {
        let _ = Authority::parse(auth);
    }

    ParsedStoredMetadata { profile_label }
}

/// Gather evidence, inferring a plan from `input`.
#[must_use]
pub fn gather_evidence(store: &DurableStore, input: &str, limit: usize) -> EvidenceContext {
    gather_evidence_with_plan(store, input, limit, &infer(input))
}

/// Gather evidence under an explicit plan.
#[must_use]
pub fn gather_evidence_with_plan(
    store: &DurableStore,
    input: &str,
    limit: usize,
    plan: &QueryPlan,
) -> EvidenceContext {
    if input.is_empty() || limit == 0 || store.stats().vectors == 0 {
        return EvidenceContext::default();
    }

    let embedding = text_embedding(input);
    let Ok(hits) = store.search(&embedding, limit) else {
        return EvidenceContext::default();
    };

    let mut items = Vec::new();
    for hit in hits {
        let key = format!("completion:{}", hit.id);
        let Some(metadata) = store.get(&key) else {
            continue;
        };
        let parsed = parse_stored_metadata(metadata);
        let authority = Authority::Inferred;
        let relevance = if plan.exact_recall {
            (1.0 - EXACT_RECALL_KEYWORD_WEIGHT) * hit.score
                + EXACT_RECALL_KEYWORD_WEIGHT * keyword_overlap(input, metadata)
        } else {
            hit.score
        };
        let final_score = relevance * authority.score();
        items.push(EvidenceItem {
            vector_id: hit.id,
            profile_label: parsed.profile_label,
            authority,
            snippet: metadata.to_string(),
            score: final_score,
        });
    }

    items.sort_by(|a, b| {
        let a_nan = a.score.is_nan();
        let b_nan = b.score.is_nan();
        if a_nan != b_nan {
            return if a_nan {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            };
        }
        match b.score.partial_cmp(&a.score) {
            Some(std::cmp::Ordering::Equal) | None => a.vector_id.cmp(&b.vector_id),
            Some(order) => order,
        }
    });

    EvidenceContext { items }
}

/// Fraction of significant (>=3 char) query tokens that appear in `text`.
#[allow(clippy::cast_precision_loss)] // evidence counts stay well under f32 mantissa range
fn keyword_overlap(query: &str, text: &str) -> f32 {
    let separators = [
        ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']', '{',
        '}', '<', '>', '/', '\\',
    ];
    let mut total = 0_usize;
    let mut matched = 0_usize;
    for tok in query.split(|c| separators.contains(&c)) {
        if tok.len() < 3 {
            continue;
        }
        total += 1;
        if contains_ignore_case(text, tok) {
            matched += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    matched as f32 / total as f32
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    let hay = haystack.as_bytes();
    let ned = needle.as_bytes();
    let mut i = 0;
    while i + ned.len() <= hay.len() {
        if hay[i..i + ned.len()].eq_ignore_ascii_case(ned) {
            return true;
        }
        i += 1;
    }
    false
}

/// Prepend recalled snippets as a preamble, capped at [`MAX_PROMPT_BYTES`].
///
/// With no evidence, returns `input` unchanged. The original input is always
/// present after the `[query]` marker.
#[must_use]
pub fn augment_prompt(input: &str, ctx: &EvidenceContext) -> String {
    if ctx.is_empty() {
        return input.to_string();
    }

    let mut out = String::from("[SEA evidence]\n");
    for item in &ctx.items {
        if out.len() >= MAX_PROMPT_BYTES {
            break;
        }
        let line = format!(
            "- (vec {}, {}, authority={}): {}\n",
            item.vector_id,
            item.profile_label,
            item.authority.text(),
            item.snippet
        );
        out.push_str(&line);
        if out.len() > MAX_PROMPT_BYTES {
            out.truncate(MAX_PROMPT_BYTES);
            break;
        }
    }
    out.push_str("[query]\n");
    out.push_str(input);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_wdbx::{DurableStore, StorePaths};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("abi_sea_evidence_{}_{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn empty_store_yields_empty_context_and_passthrough_prompt() {
        let dir = Scratch::new();
        let store = DurableStore::open(StorePaths::new(&dir.0)).unwrap();
        let ctx = gather_evidence(&store, "hello", 5);
        assert!(ctx.is_empty());
        assert_eq!(augment_prompt("hello", &ctx), "hello");
    }

    #[test]
    fn recalls_a_stored_completion_with_resolved_persona() {
        let dir = Scratch::new();
        let mut store = DurableStore::open(StorePaths::new(&dir.0)).unwrap();
        let embedding = text_embedding("aviva said hello");
        let id = store.put_vector(&embedding).unwrap();
        store
            .put(
                &format!("completion:{id}"),
                r#"{"profile":"aviva","authority":"user_stated","text":"hello there"}"#,
            )
            .unwrap();

        let ctx = gather_evidence(&store, "hello", 5);
        assert_eq!(ctx.items.len(), 1);
        assert_eq!(ctx.items[0].profile_label, "aviva");
        assert_eq!(ctx.items[0].authority, Authority::Inferred);
        assert!(ctx.items[0].snippet.contains("hello there"));
    }

    #[test]
    fn unrecognized_profile_maps_to_unknown() {
        let dir = Scratch::new();
        let mut store = DurableStore::open(StorePaths::new(&dir.0)).unwrap();
        let embedding = text_embedding("a mystery turn");
        let id = store.put_vector(&embedding).unwrap();
        store
            .put(&format!("completion:{id}"), r#"{"profile":"nobody"}"#)
            .unwrap();
        let ctx = gather_evidence(&store, "mystery", 5);
        assert_eq!(ctx.items.len(), 1);
        assert_eq!(ctx.items[0].profile_label, "unknown");
    }

    #[test]
    fn augment_prompt_caps_preamble() {
        let items: Vec<_> = (0..8)
            .map(|i| EvidenceItem {
                vector_id: i,
                profile_label: "abbey",
                authority: Authority::Inferred,
                snippet: "x".repeat(1024),
                score: 1.0,
            })
            .collect();
        let ctx = EvidenceContext { items };
        let prompt = augment_prompt("the-query", &ctx);
        assert!(prompt.len() <= MAX_PROMPT_BYTES + "[query]\n".len() + "the-query".len());
        assert!(prompt.ends_with("the-query"));
    }
}
