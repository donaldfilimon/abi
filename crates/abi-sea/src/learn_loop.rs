//! SEA self-learning loop: evidence → augment → adaptive complete → save weights.
//!
//! Ported from `src/features/sea/learn_loop.zig`.

use abi_ai::{
    AdaptiveModulator, CompletionResult, EMBED_DIM, EmptyInputError, MODULATOR_STORE_KEY,
    analyze_sentiment, complete_adaptive, completion, text_embedding,
};
use abi_wdbx::{RecordId, VersionedError, VersionedStore};

use crate::evidence::{self, MAX_PROMPT_BYTES};
use crate::query_plan::{self, TaskType};

/// Configuration for one self-learning pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LearnLoopConfig {
    /// Number of prior records to recall as evidence.
    pub evidence_limit: usize,
    /// Persist the completion (vectors + metadata + block) into the store.
    pub persist: bool,
    /// Update + save the adaptive persona-router weights from this turn.
    pub adapt_router: bool,
    /// Hard cap on the augmented prompt preamble (mirrors Zig's surface).
    pub max_prompt_bytes: usize,
}

impl Default for LearnLoopConfig {
    fn default() -> Self {
        Self {
            evidence_limit: 5,
            persist: true,
            adapt_router: true,
            max_prompt_bytes: MAX_PROMPT_BYTES,
        }
    }
}

/// Result of [`run_learn_loop`].
#[derive(Debug, Clone, PartialEq)]
pub struct LearnLoopResult {
    /// The underlying completion (persona response + audit).
    pub completion: CompletionResult,
    /// How many prior records were recalled.
    pub evidence_count: usize,
    /// Whether router weights were updated and saved.
    pub adapted: bool,
    /// Task intent inferred from the input.
    pub query_task: TaskType,
    /// Query / response vector ids and block hash, when `persist` succeeded.
    pub persisted: Option<PersistedIds>,
    /// Why no completion record was reported, or that one committed.
    pub persistence_status: LearnPersistenceStatus,
}

/// Completion-evidence persistence outcome for one learning pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearnPersistenceStatus {
    /// Completion evidence was committed and its ids are in `persisted`.
    Committed,
    /// Persistence was disabled by the caller.
    Disabled,
    /// A non-passing or vetoed audit intentionally prevented the write.
    AuditRejected,
    /// WDBX returned an error; publication may or may not have occurred.
    OutcomeUnknown,
}

/// Ids produced when a completion is persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedIds {
    /// Stored query vector id.
    pub query_vector_id: RecordId,
    /// Stored response vector id.
    pub response_vector_id: RecordId,
    /// Audit-chain block hash as lowercase hex.
    pub block_hash_hex: String,
}

/// Run one SEA self-learning pass against `store`.
///
/// # Errors
///
/// Returns [`EmptyInputError`] when the generation input is empty.
pub fn run_learn_loop(
    store: &mut VersionedStore,
    input: &str,
    model: &str,
    config: LearnLoopConfig,
    now_ms: i64,
) -> Result<LearnLoopResult, EmptyInputError> {
    let plan = query_plan::infer(input);
    let ctx = evidence::gather_evidence_with_plan(store, input, config.evidence_limit, &plan);
    let evidence_count = ctx.items.len();
    let augmented = evidence::augment_prompt_with_limit(input, &ctx, config.max_prompt_bytes);

    // Generation sees the evidence-augmented prompt; routing uses the raw user
    // text so an explicit "Aviva, ..." address is not masked by the preamble.
    let persisted_weights = store.get(MODULATOR_STORE_KEY);
    let completion = complete_adaptive(&augmented, model, input, persisted_weights.as_deref())?;
    // A refusal may remain observable to the caller, but it must not train
    // routing weights or enter the completion evidence pool.
    let audit_allows_learning = completion.audit.passed && !completion.audit.vetoed;

    let mut adapted = false;
    if config.adapt_router && audit_allows_learning {
        // Zig reloads weights from the store for the save path, independently
        // of the routing-time update inside complete_adaptive.
        let mut modulator = match store.get(MODULATOR_STORE_KEY) {
            Some(data) => AdaptiveModulator::deserialize(&data),
            None => AdaptiveModulator::new(),
        };
        modulator.update(analyze_sentiment(input));
        let serialized = modulator.serialize();
        if store.put(MODULATOR_STORE_KEY, &serialized).is_ok() {
            adapted = true;
        }
    }

    // Zig's completeWithStoreAdaptive embeds request.input (the augmented
    // prompt) as the query vector — not the raw user text.
    let (persisted, persistence_status) = if !config.persist {
        (None, LearnPersistenceStatus::Disabled)
    } else if !audit_allows_learning {
        (None, LearnPersistenceStatus::AuditRejected)
    } else {
        match persist_completion(store, &augmented, &completion, now_ms) {
            Ok(ids) => (Some(ids), LearnPersistenceStatus::Committed),
            Err(_) => (None, LearnPersistenceStatus::OutcomeUnknown),
        }
    };

    Ok(LearnLoopResult {
        completion,
        evidence_count,
        adapted,
        query_task: plan.task,
        persisted,
        persistence_status,
    })
}

fn persist_completion(
    store: &mut VersionedStore,
    generation_input: &str,
    result: &CompletionResult,
    now_ms: i64,
) -> Result<PersistedIds, VersionedError> {
    let query_vec: [f32; EMBED_DIM] = text_embedding(generation_input);
    let response_vec: [f32; EMBED_DIM] = text_embedding(&result.output);

    let recorded = store.record_vector_pair(
        &query_vec,
        &response_vec,
        result.selected_profile.label(),
        now_ms,
        |query_id, response_id| {
            (
                completion::metadata_key(query_id),
                completion::metadata_json(generation_input, result, query_id, response_id),
            )
        },
    )?;

    Ok(PersistedIds {
        query_vector_id: recorded.query_id,
        response_vector_id: recorded.response_id,
        block_hash_hex: recorded.block_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_wdbx::StorePaths;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("abi_sea_learn_{}_{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn open() -> (Scratch, VersionedStore) {
        let dir = Scratch::new();
        let store = VersionedStore::open(StorePaths::new(&dir.0)).unwrap();
        (dir, store)
    }

    #[test]
    fn surfaces_inferred_task_intent() {
        let (_dir, mut store) = open();
        let result = run_learn_loop(
            &mut store,
            "remember the prior decision we made",
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert_eq!(result.query_task, TaskType::ProjectRecall);
        assert!(!result.adapted);
        assert!(result.persisted.is_none());
        assert_eq!(result.persistence_status, LearnPersistenceStatus::Disabled);
    }

    #[test]
    fn adapts_and_persists_weights() {
        let (_dir, mut store) = open();
        let result = run_learn_loop(
            &mut store,
            "hello world",
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: true,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert!(result.adapted);
        assert!(store.get(MODULATOR_STORE_KEY).is_some());
    }

    #[test]
    fn completion_evidence_uses_one_wdbx_transaction() {
        let (_dir, mut store) = open();
        let result = run_learn_loop(
            &mut store,
            "ordinary completion evidence",
            "abi-local",
            LearnLoopConfig {
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert!(result.persisted.is_some());
        assert_eq!(result.persistence_status, LearnPersistenceStatus::Committed);
        assert_eq!(store.snapshot().committed_transactions(), 1);
        assert_eq!(store.stats().kv_entries, 1);
        assert_eq!(store.stats().vectors, 2);
        assert_eq!(store.stats().blocks, 1);
    }

    #[test]
    fn vetoed_completion_neither_adapts_nor_enters_the_evidence_store() {
        let (_dir, mut store) = open();
        let before = store.stats();
        let result = run_learn_loop(
            &mut store,
            "this will cause harm",
            "abi-local",
            LearnLoopConfig::default(),
            1_700_000_000_000,
        )
        .unwrap();
        assert!(result.completion.audit.vetoed);
        assert!(!result.completion.audit.passed);
        assert!(!result.adapted);
        assert!(result.persisted.is_none());
        assert_eq!(
            result.persistence_status,
            LearnPersistenceStatus::AuditRejected
        );
        assert_eq!(store.stats(), before);
        assert!(store.get(MODULATOR_STORE_KEY).is_none());
    }

    #[test]
    fn failed_completion_write_reports_an_unknown_outcome_without_partial_mutations() {
        let (_dir, mut store) = open();
        store.put_vector(&[1.0]).unwrap();
        let before = store.stats();
        let result = run_learn_loop(
            &mut store,
            "ordinary completion evidence",
            "abi-local",
            LearnLoopConfig {
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert_eq!(
            result.persistence_status,
            LearnPersistenceStatus::OutcomeUnknown
        );
        assert!(result.persisted.is_none());
        assert_eq!(store.stats(), before);
        assert_eq!(store.snapshot().committed_transactions(), 1);
    }

    #[test]
    fn uses_saved_adaptive_weights_for_later_completions() {
        let (_dir, mut store) = open();
        // Heavy aviva prior — same shape as Zig's learn_loop test.
        store
            .put(MODULATOR_STORE_KEY, "0.010000,0.980000,0.010000,8,0.050000")
            .unwrap();
        let result = run_learn_loop(
            &mut store,
            "analyze the logical structure",
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert_eq!(
            result.completion.selected_profile,
            abi_ai::AgentProfile::Aviva
        );
        assert!(result.completion.output.contains("Aviva direct expert"));
    }

    #[test]
    fn explicit_persona_survives_evidence_and_opposing_adaptive_weights() {
        let (_dir, mut store) = open();
        let seed_text = "architecture evidence for the release plan";
        let seed_id = store.put_vector(&text_embedding(seed_text)).unwrap();
        let metadata = r#"{"kind":"project_decision","profile":"abi","summary":"architecture evidence for the release plan"}"#;
        store
            .put(&format!("completion:{seed_id}"), metadata)
            .unwrap();
        store
            .add_block(
                "abi",
                seed_id,
                RecordId::Legacy(0),
                metadata,
                1_700_000_000_000,
            )
            .unwrap();
        // Persisted EMA overwhelmingly favors ABI, while the explicit leading
        // address must still select Aviva after SEA prepends ABI evidence.
        store
            .put(
                MODULATOR_STORE_KEY,
                "0.005000,0.005000,0.990000,42,0.300000",
            )
            .unwrap();

        let result = run_learn_loop(
            &mut store,
            "Aviva, summarize the architecture release plan.",
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_001,
        )
        .unwrap();

        assert!(result.evidence_count > 0);
        assert_eq!(
            result.completion.selected_profile,
            abi_ai::AgentProfile::Aviva
        );
        assert!(
            result
                .completion
                .output
                .starts_with("Aviva direct expert: ")
        );
        assert!(result.completion.output.contains("[SEA evidence]\n- (vec"));
        assert!(
            result
                .completion
                .output
                .contains("[query]\nAviva, summarize the architecture release plan.")
        );
    }

    #[test]
    fn first_turn_is_recalled_as_evidence_on_related_second_turn() {
        let (_dir, mut store) = open();
        let first = run_learn_loop(
            &mut store,
            "the capital of france is paris",
            "abi-local",
            LearnLoopConfig {
                persist: true,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_000,
        )
        .unwrap();
        assert!(first.persisted.is_some());

        let second = run_learn_loop(
            &mut store,
            "tell me about paris in france",
            "abi-local",
            LearnLoopConfig {
                persist: true,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1_700_000_000_001,
        )
        .unwrap();
        assert!(second.evidence_count > 0);
    }

    #[test]
    fn corrupted_weights_route_like_missing_weights() {
        let input = "analyze the logical structure";
        let (_d1, mut empty) = open();
        let baseline = run_learn_loop(
            &mut empty,
            input,
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1,
        )
        .unwrap();

        let (_d2, mut corrupt) = open();
        corrupt
            .put(MODULATOR_STORE_KEY, "nan,0.3,0.4,1,0.3")
            .unwrap();
        let corrupted = run_learn_loop(
            &mut corrupt,
            input,
            "abi-local",
            LearnLoopConfig {
                persist: false,
                adapt_router: false,
                ..LearnLoopConfig::default()
            },
            1,
        )
        .unwrap();

        assert_eq!(
            baseline.completion.selected_profile,
            corrupted.completion.selected_profile
        );
        assert_eq!(baseline.completion.output, corrupted.completion.output);
    }
}
