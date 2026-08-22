use super::*;
use abi_wdbx::{RecordId, StorePaths, VersionedStore};
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
    let store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let ctx = gather_evidence(&store, "hello", 5);
    assert_eq!(ctx.items.len(), 0);
    assert_eq!(augment_prompt("hello", &ctx), "hello");
}

#[test]
fn huge_limit_is_bounded_before_wdbx_search_and_candidate_work() {
    assert_eq!(recall_limits(usize::MAX, usize::MAX), (100, 100));
    assert_eq!(recall_limits(250, 5), (5, 20));
    assert_eq!(recall_limits(3, usize::MAX), (100, 3));
    assert_eq!(recall_limits(usize::MAX, 0), (0, 0));
}

#[test]
fn exact_recall_is_reserved_for_project_recall() {
    let signals = SeaSignals {
        semantic: 0.10,
        keyword: 1.0,
        metadata: 0.20,
        recency: 0.30,
        authority: 0.40,
        graph: 0.20,
        contradiction: 0.0,
        task_fit: 0.50,
    };

    let general = crate::query_plan::infer("neutral request");
    assert!(!general.exact_recall);
    let general_weights = adjust_weights_for_task(DEFAULT_SEA_WEIGHTS, general.task);
    assert_eq!(
        score_for_plan(signals, general_weights, &general),
        sea_score(signals, general_weights)
    );

    let recall_exact = crate::query_plan::infer("remember the prior decision");
    assert_eq!(
        recall_exact.task,
        crate::query_plan::TaskType::ProjectRecall
    );
    assert!(recall_exact.exact_recall);
    let mut recall_fuzzy = recall_exact.clone();
    recall_fuzzy.exact_recall = false;
    let recall_weights = adjust_weights_for_task(DEFAULT_SEA_WEIGHTS, recall_exact.task);
    assert!(
        score_for_plan(signals, recall_weights, &recall_exact)
            > score_for_plan(signals, recall_weights, &recall_fuzzy)
    );
}

#[test]
fn recalls_a_stored_completion_with_resolved_persona() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let input = "fix the abi compiler bug";
    let embedding = text_embedding(input);
    let id = store.put_vector(&embedding).unwrap();
    let metadata = r#"{"profile":"aviva","authority":"system_pinned","kind":"code_fact","project":"abi","importance":0.8,"supersedes":4,"source_uri":"file:///repo","tags":["compiler"],"text":"the compiler fix is deterministic"}"#;
    store.put(&format!("completion:{id}"), metadata).unwrap();
    store
        .add_block("aviva", id, RecordId::Legacy(0), metadata, 2_000)
        .unwrap();

    let ctx = gather_evidence(&store, input, 5);
    assert_eq!(ctx.items.len(), 1);
    assert_eq!(ctx.items[0].profile_label, "aviva");
    assert_eq!(ctx.items[0].authority, Authority::Inferred);
    assert_eq!(ctx.items[0].kind, MemoryKind::CodeFact);
    assert!(ctx.items[0].snippet.contains("compiler fix"));
    let signals = ctx.items[0].signals;
    assert!(signals.semantic > 0.99);
    assert!(signals.keyword > 0.5);
    assert!((signals.metadata - 0.77).abs() < 1e-5);
    assert!((signals.recency - 1.0).abs() < 1e-5);
    assert!((signals.authority - Authority::Inferred.score()).abs() < 1e-5);
    assert!((signals.graph - 1.0).abs() < 1e-5);
    assert!(signals.contradiction.abs() < 1e-6);
    assert!((signals.task_fit - 1.0).abs() < 1e-6);
}

#[test]
fn unrecognized_profile_maps_to_unknown() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
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
fn missing_metadata_is_not_admitted_as_evidence() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    store.put_vector(&text_embedding("orphan vector")).unwrap();
    assert_eq!(gather_evidence(&store, "orphan vector", 5).items.len(), 0);
}

#[test]
fn malformed_metadata_is_bounded_and_low_trust() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let id = store
        .put_vector(&text_embedding("malformed evidence"))
        .unwrap();
    store
        .put(&format!("completion:{id}"), "not-json evidence")
        .unwrap();
    let ctx = gather_evidence(&store, "malformed evidence", 5);
    assert_eq!(ctx.items.len(), 1);
    let item = &ctx.items[0];
    assert_eq!(item.profile_label, "unknown");
    assert_eq!(item.kind, MemoryKind::Note);
    assert!(item.signals.metadata.abs() < 1e-6);
    assert!(item.signals.graph.abs() < 1e-6);
    assert!(item.snippet.len() <= MAX_SUMMARY_BYTES);
}

#[test]
fn contradiction_records_surface_the_explicit_signal() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let id = store
        .put_vector(&text_embedding("benchmark conflict"))
        .unwrap();
    store
        .put(
            &format!("completion:{id}"),
            r#"{"kind":"contradiction","summary":"conflicting benchmark result"}"#,
        )
        .unwrap();
    let ctx = gather_evidence(&store, "benchmark conflict", 5);
    assert_eq!(ctx.items[0].kind, MemoryKind::Contradiction);
    assert!((ctx.items[0].signals.contradiction - 1.0).abs() < 1e-6);
    assert!((ctx.items[0].signals.task_fit - 0.8).abs() < 1e-6);
}

#[test]
fn metadata_timestamps_define_relative_recency_without_audit_blocks() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let embedding = text_embedding("metadata timestamp evidence");
    let older = store.put_vector(&embedding).unwrap();
    let newer = store.put_vector(&embedding).unwrap();
    store
        .put(
            &format!("completion:{older}"),
            r#"{"kind":"note","updated_ms":1000,"text":"metadata timestamp evidence"}"#,
        )
        .unwrap();
    store
        .put(
            &format!("completion:{newer}"),
            r#"{"kind":"note","updated_ms":2592001000,"text":"metadata timestamp evidence"}"#,
        )
        .unwrap();

    let ctx = gather_evidence(&store, "metadata timestamp evidence", 5);
    let older_item = ctx
        .items
        .iter()
        .find(|item| item.vector_id == older)
        .unwrap();
    let newer_item = ctx
        .items
        .iter()
        .find(|item| item.vector_id == newer)
        .unwrap();
    assert!((older_item.signals.recency - 0.5).abs() < 1e-5);
    assert!((newer_item.signals.recency - 1.0).abs() < 1e-5);
}

#[test]
fn equal_scores_are_ordered_by_stable_vector_id() {
    let dir = Scratch::new();
    let mut store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
    let embedding = text_embedding("same deterministic evidence");
    let first = store.put_vector(&embedding).unwrap();
    let second = store.put_vector(&embedding).unwrap();
    let metadata = r#"{"kind":"note","text":"same deterministic evidence"}"#;
    store.put(&format!("completion:{first}"), metadata).unwrap();
    store
        .put(&format!("completion:{second}"), metadata)
        .unwrap();
    let ctx = gather_evidence(&store, "same deterministic evidence", 5);
    let mut expected = vec![first, second];
    expected.sort();
    assert_eq!(
        ctx.items
            .iter()
            .map(|item| item.vector_id)
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn augment_prompt_caps_preamble() {
    let items: Vec<_> = (0..8)
        .map(|i| EvidenceItem {
            vector_id: RecordId::Legacy(i),
            profile_label: "abbey",
            authority: Authority::Inferred,
            kind: MemoryKind::Note,
            snippet: "x".repeat(1024),
            signals: SeaSignals::default(),
            score: 1.0,
        })
        .collect();
    let ctx = EvidenceContext { items };
    let prompt = augment_prompt("the-query", &ctx);
    assert!(prompt.len() <= MAX_PROMPT_BYTES + "[query]\n".len() + "the-query".len());
    assert!(prompt.ends_with("the-query"));
}

#[test]
fn summarization_is_utf8_safe_and_prompt_retains_raw_query() {
    let metadata = format!(r#"{{"text":"{}"}}"#, "é".repeat(600));
    let summary = summarize_metadata(&metadata, 63);
    assert!(summary.len() <= 63);
    assert!(summary.is_char_boundary(summary.len()));

    let ctx = EvidenceContext {
        items: vec![EvidenceItem {
            vector_id: RecordId::Legacy(7),
            profile_label: "abi",
            authority: Authority::Inferred,
            kind: MemoryKind::Summary,
            snippet: summary,
            signals: SeaSignals::default(),
            score: 0.75,
        }],
    };
    let raw_query = "Aviva, keep this raw query exactly: é";
    let prompt = augment_prompt_with_limit(raw_query, &ctx, 180);
    let (preamble, query) = prompt.split_once("[query]\n").unwrap();
    assert!(preamble.len() <= 180);
    assert_eq!(query, raw_query);
    assert!(preamble.contains("vec 7"));
    assert!(preamble.contains("authority=inferred"));
}
