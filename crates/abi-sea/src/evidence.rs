//! Evidence recall and prompt augmentation for SEA.
//!
//! Ported from `src/features/sea/evidence.zig`.

use abi_ai::{PROFILE_LABELS, text_embedding};
use abi_wdbx::{RecordId, VersionedStore};
use serde_json::{Map, Value};

use crate::query_plan::{QueryPlan, infer};
use crate::scorer::{
    DEFAULT_SEA_WEIGHTS, SeaCandidate, SeaOptions, SeaSignals, adjust_weights_for_task, sea_score,
    select_sea_candidates,
};
use crate::types::{Authority, MemoryKind};

/// Upper bound on the augmented-prompt preamble.
pub const MAX_PROMPT_BYTES: usize = 4096;

/// Hard upper bound on both recalled evidence and the WDBX search pool.
///
/// This is applied before embedding search, metadata cloning, and scoring so
/// an untrusted public limit cannot turn a request into unbounded recall work.
pub const MAX_EVIDENCE_LIMIT: usize = 100;

const MAX_SUMMARY_BYTES: usize = 512;
const RECENCY_SOFT_HALF_LIFE_MS: i64 = 30 * 24 * 60 * 60 * 1_000;
const EXACT_RECALL_KEYWORD_WEIGHT: f32 = 0.5;
const EVIDENCE_HEADER: &str = "[SEA evidence]\n";
const GRAPH_SUPERSEDES: u8 = 1 << 0;
const GRAPH_NOT_SUPERSEDED: u8 = 1 << 1;
const GRAPH_SOURCE: u8 = 1 << 2;
const GRAPH_TAGS: u8 = 1 << 3;

const UNKNOWN_PROFILE: &str = "unknown";

/// One recalled record. `snippet` is owned; `profile_label` is a static borrow.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceItem {
    /// Vector id of the hit.
    pub vector_id: RecordId,
    /// Persona label (`abbey`/`aviva`/`abi`/`unknown`).
    pub profile_label: &'static str,
    /// Forced-to-inferred authority for generic-store records.
    pub authority: Authority,
    /// Parsed memory kind, defaulting to `note` for generic completions.
    pub kind: MemoryKind,
    /// Bounded owned summary of the opaque metadata.
    pub snippet: String,
    /// All eight sub-scores used for selection.
    pub signals: SeaSignals,
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
    kind: MemoryKind,
    project: Option<String>,
    importance: f32,
    updated_ms: Option<i64>,
    graph_flags: u8,
    valid_json: bool,
}

/// Parse only exact top-level JSON fields. Generic-store authority cannot
/// self-promote above [`Authority::Inferred`].
#[allow(clippy::cast_possible_truncation)] // JSON importance is deliberately clamped to [0,1]
fn parse_stored_metadata(metadata: &str) -> ParsedStoredMetadata {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(metadata) else {
        return ParsedStoredMetadata::default();
    };
    let Some(object) = value.as_object() else {
        return ParsedStoredMetadata::default();
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

    // Observe but do not accept a self-asserted authority claim. Generic-store
    // records are always forced to Inferred below.
    if let Some(auth) = object.get("authority").and_then(|v| v.as_str()) {
        let _ = Authority::parse(auth);
    }

    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .and_then(MemoryKind::parse)
        .unwrap_or(MemoryKind::Note);
    let project = object
        .get("project")
        .and_then(Value::as_str)
        .filter(|project| !project.is_empty())
        .map(str::to_owned);
    let importance = object
        .get("importance")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32;
    let updated_ms = timestamp_ms(object);
    let mut graph_flags = 0;
    if nonempty_value(object.get("supersedes")) {
        graph_flags |= GRAPH_SUPERSEDES;
    }
    if !nonempty_value(object.get("superseded_by")) {
        graph_flags |= GRAPH_NOT_SUPERSEDED;
    }
    if nonempty_value(object.get("source_uri")) {
        graph_flags |= GRAPH_SOURCE;
    }
    if nonempty_value(object.get("tags")) {
        graph_flags |= GRAPH_TAGS;
    }

    ParsedStoredMetadata {
        profile_label,
        kind,
        project,
        importance,
        updated_ms,
        graph_flags,
        valid_json: true,
    }
}

impl Default for ParsedStoredMetadata {
    fn default() -> Self {
        Self {
            profile_label: UNKNOWN_PROFILE,
            kind: MemoryKind::Note,
            project: None,
            importance: 0.0,
            updated_ms: None,
            graph_flags: 0,
            valid_json: false,
        }
    }
}

/// Gather evidence, inferring a plan from `input`.
#[must_use]
pub fn gather_evidence(store: &VersionedStore, input: &str, limit: usize) -> EvidenceContext {
    gather_evidence_with_plan(store, input, limit, &infer(input))
}

/// Gather evidence under an explicit plan.
#[must_use]
pub fn gather_evidence_with_plan(
    store: &VersionedStore,
    input: &str,
    limit: usize,
    plan: &QueryPlan,
) -> EvidenceContext {
    if input.is_empty() {
        return EvidenceContext::default();
    }

    let (limit, candidate_limit) = recall_limits(store.stats().vectors, limit);
    if limit == 0 || candidate_limit == 0 {
        return EvidenceContext::default();
    }

    let embedding = text_embedding(input);
    // Over-fetch bounded candidates so cluster/token selection can remain
    // sparse without returning fewer records merely because an early hit was
    // missing metadata or duplicated by an upstream index implementation. The
    // shared hard cap still bounds the WDBX search and every later clone.
    let Ok(hits) = store.search(&embedding, candidate_limit) else {
        return EvidenceContext::default();
    };

    let snapshot = store.snapshot();
    let mut block_timestamp_by_query = std::collections::BTreeMap::new();
    let mut latest_timestamp_ms = None;
    for block in snapshot.audit_blocks() {
        latest_timestamp_ms = latest_timestamp_ms.max(Some(block.timestamp_ms));
        block_timestamp_by_query
            .entry(block.query_id)
            .and_modify(|timestamp: &mut i64| *timestamp = (*timestamp).max(block.timestamp_ms))
            .or_insert(block.timestamp_ms);
    }
    let mut recalled = Vec::new();
    let mut seen_ids = std::collections::BTreeSet::new();

    for hit in hits {
        if !seen_ids.insert(hit.id) {
            continue;
        }
        let key = format!("completion:{}", hit.id);
        let Some(metadata) = store.get(&key) else {
            continue;
        };
        let parsed = parse_stored_metadata(&metadata);
        let block_timestamp_ms = block_timestamp_by_query.get(&hit.id).copied();
        latest_timestamp_ms = latest_timestamp_ms.max(parsed.updated_ms.or(block_timestamp_ms));
        recalled.push((hit.id, hit.score, metadata, parsed, block_timestamp_ms));
    }

    let weights = adjust_weights_for_task(DEFAULT_SEA_WEIGHTS, plan.task);
    let mut candidates = Vec::new();
    let mut evidence_by_id = std::collections::BTreeMap::new();
    for (id, hit_score, metadata, parsed, block_timestamp_ms) in recalled {
        let authority = Authority::Inferred;
        let signals = SeaSignals {
            semantic: semantic_score(hit_score),
            keyword: keyword_overlap(input, &metadata),
            metadata: metadata_score(input, &parsed),
            recency: recency_score(
                parsed.updated_ms.or(block_timestamp_ms),
                latest_timestamp_ms,
            ),
            authority: authority.score(),
            graph: graph_score(&parsed),
            contradiction: contradiction_score(parsed.kind),
            task_fit: task_fit_score(plan.task, parsed.kind),
        };
        let final_score = score_for_plan(signals, weights, plan);
        let snippet = summarize_metadata(&metadata, MAX_SUMMARY_BYTES);
        let estimated_tokens = snippet.len().div_ceil(4).max(1);
        candidates.push(SeaCandidate {
            record_id: id,
            cluster_id: kind_cluster(parsed.kind),
            estimated_tokens,
            signals,
            final_score,
        });
        evidence_by_id.insert(
            id,
            EvidenceItem {
                vector_id: id,
                profile_label: parsed.profile_label,
                authority,
                kind: parsed.kind,
                snippet,
                signals,
                score: final_score,
            },
        );
    }

    let selection = select_sea_candidates(
        candidates,
        SeaOptions {
            max_tokens: MAX_PROMPT_BYTES.div_ceil(4),
            max_records: limit,
            ..SeaOptions::default()
        },
    );
    let items = selection
        .selected_ids
        .into_iter()
        .filter_map(|id| evidence_by_id.remove(&id))
        .collect();

    EvidenceContext { items }
}

fn recall_limits(store_vectors: usize, requested_limit: usize) -> (usize, usize) {
    let evidence_limit = requested_limit.min(MAX_EVIDENCE_LIMIT);
    let candidate_limit = store_vectors
        .min(evidence_limit.saturating_mul(4))
        .min(MAX_EVIDENCE_LIMIT);
    (evidence_limit, candidate_limit)
}

fn score_for_plan(
    signals: SeaSignals,
    weights: crate::scorer::SeaWeights,
    plan: &QueryPlan,
) -> f32 {
    let base_score = sea_score(signals, weights);
    if !plan.exact_recall {
        return base_score;
    }
    let lexical_score = (1.0 - EXACT_RECALL_KEYWORD_WEIGHT) * signals.semantic
        + EXACT_RECALL_KEYWORD_WEIGHT * signals.keyword;
    f32::midpoint(base_score, lexical_score).clamp(0.0, 1.0)
}

fn nonempty_value(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(values)) => !values.is_empty(),
        Some(Value::Object(values)) => !values.is_empty(),
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(_)) => true,
    }
}

fn semantic_score(cosine: f32) -> f32 {
    if !cosine.is_finite() {
        return 0.0;
    }
    f32::midpoint(cosine.clamp(-1.0, 1.0), 1.0).clamp(0.0, 1.0)
}

fn timestamp_ms(object: &Map<String, Value>) -> Option<i64> {
    for key in ["updated_ms", "timestamp_ms", "created_ms"] {
        if let Some(value) = object.get(key).and_then(Value::as_i64) {
            return Some(value);
        }
    }
    for key in ["updated_ns", "created_ns"] {
        if let Some(value) = object.get(key).and_then(Value::as_i64) {
            return Some(value / 1_000_000);
        }
    }
    None
}

fn metadata_score(input: &str, metadata: &ParsedStoredMetadata) -> f32 {
    if !metadata.valid_json {
        return 0.0;
    }
    let mut score = 0.25;
    if metadata
        .project
        .as_deref()
        .is_some_and(|project| contains_ignore_case(input, project))
    {
        score += 0.40;
    }
    if matches!(
        metadata.kind,
        MemoryKind::Constraint | MemoryKind::ProjectDecision
    ) {
        score += 0.20;
    }
    (score + metadata.importance * 0.15).clamp(0.0, 1.0)
}

#[allow(clippy::cast_precision_loss)]
fn recency_score(timestamp_ms: Option<i64>, latest_timestamp_ms: Option<i64>) -> f32 {
    let (Some(timestamp_ms), Some(latest_timestamp_ms)) = (timestamp_ms, latest_timestamp_ms)
    else {
        return 0.0;
    };
    let age_ms = latest_timestamp_ms.saturating_sub(timestamp_ms).max(0);
    let age_periods = age_ms as f32 / RECENCY_SOFT_HALF_LIFE_MS as f32;
    1.0 / (1.0 + age_periods)
}

fn graph_score(metadata: &ParsedStoredMetadata) -> f32 {
    if !metadata.valid_json {
        return 0.0;
    }
    let flag_count = u8::try_from(metadata.graph_flags.count_ones()).unwrap_or(0);
    f32::from(flag_count) * 0.25
}

const fn contradiction_score(kind: MemoryKind) -> f32 {
    if matches!(kind, MemoryKind::Contradiction) {
        1.0
    } else {
        0.0
    }
}

fn task_fit_score(task: crate::query_plan::TaskType, kind: MemoryKind) -> f32 {
    use crate::query_plan::TaskType;
    match task {
        TaskType::General => 0.50,
        TaskType::ImplementationDesign => match kind {
            MemoryKind::ProjectDecision | MemoryKind::CodeFact | MemoryKind::Constraint => 1.0,
            MemoryKind::Summary => 0.70,
            _ => 0.25,
        },
        TaskType::CodeRepair => match kind {
            MemoryKind::CodeFact | MemoryKind::ToolOutput | MemoryKind::Constraint => 1.0,
            MemoryKind::Contradiction => 0.80,
            MemoryKind::Summary => 0.70,
            _ => 0.20,
        },
        TaskType::LegalReview => match kind {
            MemoryKind::Constraint | MemoryKind::Contradiction => 1.0,
            MemoryKind::Summary | MemoryKind::ProjectDecision => 0.80,
            _ => 0.20,
        },
        TaskType::ResearchSynthesis => match kind {
            MemoryKind::ToolOutput | MemoryKind::Summary => 1.0,
            MemoryKind::Benchmark => 0.80,
            MemoryKind::CodeFact => 0.60,
            _ => 0.30,
        },
        TaskType::ProjectRecall => match kind {
            MemoryKind::ProjectDecision | MemoryKind::UserPreference | MemoryKind::Summary => 1.0,
            MemoryKind::Constraint => 0.80,
            _ => 0.40,
        },
        TaskType::BenchmarkReview => match kind {
            MemoryKind::Benchmark => 1.0,
            MemoryKind::Contradiction | MemoryKind::Summary => 0.80,
            MemoryKind::ToolOutput => 0.70,
            _ => 0.20,
        },
    }
}

const fn kind_cluster(kind: MemoryKind) -> u8 {
    match kind {
        MemoryKind::Note => 0,
        MemoryKind::UserPreference => 1,
        MemoryKind::ProjectDecision => 2,
        MemoryKind::CodeFact => 3,
        MemoryKind::ToolOutput => 4,
        MemoryKind::Benchmark => 5,
        MemoryKind::Constraint => 6,
        MemoryKind::Contradiction => 7,
        MemoryKind::Summary => 8,
    }
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

fn summarize_metadata(metadata: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    let summary = serde_json::from_str::<Value>(metadata)
        .ok()
        .and_then(|value| {
            let object = value.as_object()?;
            let mut fields = Vec::new();
            for key in ["summary", "text", "input", "output"] {
                if let Some(text) = object
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    fields.push(format!("{key}={text}"));
                }
            }
            if fields.is_empty() {
                None
            } else {
                Some(fields.join("; "))
            }
        })
        .unwrap_or_else(|| metadata.to_owned());
    truncate_summary(&summary, max_bytes)
}

fn truncate_summary(summary: &str, max_bytes: usize) -> String {
    if summary.len() <= max_bytes {
        return summary.to_owned();
    }
    if max_bytes <= 3 {
        return ".".repeat(max_bytes);
    }
    let mut end = max_bytes - 3;
    while end > 0 && !summary.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = summary[..end].to_owned();
    out.push_str("...");
    out
}

/// Prepend recalled snippets as a preamble, capped at [`MAX_PROMPT_BYTES`].
///
/// With no evidence, returns `input` unchanged. The original input is always
/// present after the `[query]` marker.
#[must_use]
pub fn augment_prompt(input: &str, ctx: &EvidenceContext) -> String {
    augment_prompt_with_limit(input, ctx, MAX_PROMPT_BYTES)
}

/// Prepend evidence using an explicit byte budget for the entire preamble.
/// The raw query is appended after the bounded preamble and is never truncated.
#[must_use]
pub fn augment_prompt_with_limit(
    input: &str,
    ctx: &EvidenceContext,
    max_prompt_bytes: usize,
) -> String {
    if ctx.is_empty() {
        return input.to_string();
    }

    let budget = max_prompt_bytes.min(MAX_PROMPT_BYTES);
    if budget < EVIDENCE_HEADER.len() {
        return input.to_string();
    }

    let mut out = String::from(EVIDENCE_HEADER);
    let mut appended = false;
    for item in &ctx.items {
        let line = format!(
            "- (vec {}, profile={}, kind={}, authority={}, score={:.3}): {}\n",
            item.vector_id,
            item.profile_label,
            item.kind.text(),
            item.authority.text(),
            item.score,
            item.snippet
        );
        if out.len().saturating_add(line.len()) > budget {
            break;
        }
        out.push_str(&line);
        appended = true;
    }
    if !appended {
        return input.to_string();
    }
    out.push_str("[query]\n");
    out.push_str(input);
    out
}

#[cfg(test)]
mod tests;
