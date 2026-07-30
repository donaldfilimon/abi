//! SEA self-learning loop: evidence → augment → adaptive complete → save weights.
//!
//! Ported from `src/features/sea/learn_loop.zig`.

use abi_ai::{
    AdaptiveModulator, CompletionResult, EMBED_DIM, EmptyInputError, MODULATOR_STORE_KEY,
    analyze_sentiment, complete_adaptive, completion, text_embedding,
};
use abi_wdbx::DurableStore;

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
}

/// Ids produced when a completion is persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedIds {
    /// Stored query vector id.
    pub query_vector_id: u64,
    /// Stored response vector id.
    pub response_vector_id: u64,
    /// Audit-chain block hash as lowercase hex.
    pub block_hash_hex: String,
}

/// Run one SEA self-learning pass against `store`.
///
/// # Errors
///
/// Returns [`EmptyInputError`] when the generation input is empty.
pub fn run_learn_loop(
    store: &mut DurableStore,
    input: &str,
    model: &str,
    config: LearnLoopConfig,
    now_ms: i64,
) -> Result<LearnLoopResult, EmptyInputError> {
    let plan = query_plan::infer(input);
    let ctx = evidence::gather_evidence_with_plan(store, input, config.evidence_limit, &plan);
    let evidence_count = ctx.items.len();
    let augmented = evidence::augment_prompt(input, &ctx);
    let _ = config.max_prompt_bytes; // operative bound is evidence::MAX_PROMPT_BYTES

    // Generation sees the evidence-augmented prompt; routing uses the raw user
    // text so an explicit "Aviva, ..." address is not masked by the preamble.
    let persisted_weights = store.get(MODULATOR_STORE_KEY).map(str::to_string);
    let completion = complete_adaptive(&augmented, model, input, persisted_weights.as_deref())?;

    let mut adapted = false;
    if config.adapt_router {
        // Zig reloads weights from the store for the save path, independently
        // of the routing-time update inside complete_adaptive.
        let mut modulator = match store.get(MODULATOR_STORE_KEY) {
            Some(data) => AdaptiveModulator::deserialize(data),
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
    let persisted = if config.persist {
        persist_completion(store, &augmented, &completion, now_ms)
    } else {
        None
    };

    Ok(LearnLoopResult {
        completion,
        evidence_count,
        adapted,
        query_task: plan.task,
        persisted,
    })
}

fn persist_completion(
    store: &mut DurableStore,
    generation_input: &str,
    result: &CompletionResult,
    now_ms: i64,
) -> Option<PersistedIds> {
    let query_vec: [f32; EMBED_DIM] = text_embedding(generation_input);
    let response_vec: [f32; EMBED_DIM] = text_embedding(&result.output);

    let query_id = store.put_vector(&query_vec).ok()?;
    let response_id = store.put_vector(&response_vec).ok()?;

    let metadata = completion::metadata_json(generation_input, result, query_id, response_id);
    let key = completion::metadata_key(query_id);
    store.put(&key, &metadata).ok()?;

    let block = store
        .add_block(
            result.selected_profile.label(),
            query_id,
            response_id,
            &metadata,
            now_ms,
        )
        .ok()?;

    Some(PersistedIds {
        query_vector_id: query_id,
        response_vector_id: response_id,
        block_hash_hex: block.hash.to_hex(),
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

    fn open() -> (Scratch, DurableStore) {
        let dir = Scratch::new();
        let store = DurableStore::open(StorePaths::new(&dir.0)).unwrap();
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
