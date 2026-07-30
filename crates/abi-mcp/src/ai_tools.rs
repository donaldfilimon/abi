//! Backing implementation for the `ai_complete` and `ai_learn` MCP tools.
//!
//! Ported from `src/mcp/ai_tools.zig`'s `runLocalCompletion`, `runLearn`, and
//! `appendCompletionTail`.
//!
//! ## Design: the store is a parameter, not read from env inside this module
//!
//! [`report`] takes `Option<&mut DurableStore>` rather than resolving
//! `ABI_WDBX_PATH`/`ABI_WDBX_PERSIST` itself. That split — pure logic taking
//! explicit inputs, with env resolution only at [`run`], the MCP-facing
//! wrapper — is deliberate: it means every test in this module constructs a
//! [`DurableStore`] on a tempdir and never touches process environment. Env
//! vars are global mutable state shared by every test thread in the crate, so
//! a test that set one and a test that didn't would race — the same shape of
//! bug the scheduler-status telemetry lock in `abi-cli` exists to prevent, but
//! here the race would write into the real `~/.abi/`, not just fail an
//! assertion. Parameter-passing removes the race instead of serializing it.
//!
//! ## What is intentionally not reproduced from Zig
//!
//! Zig's `getWdbxStore()` always returns *some* store — an in-memory one when
//! no persistence path resolves, backed by the same `Store` type as the
//! on-disk case. No in-memory-backed [`DurableStore`] analog exists in
//! `abi-wdbx` yet, and building one is real additional scope. So when no store
//! path resolves, this reports `persisted=false` with an explicit
//! `wdbx_status` — reusing the branch Zig already has for its own
//! feature-disabled case, rather than inventing a new one.

use std::fmt::Write as _;

use abi_ai::{
    EMBED_DIM, TrainingConfig, complete, text_embedding, train_inspect, training_store_key,
    training_store_value, training_vectors,
};
use abi_sea::{LearnLoopConfig, run_learn_loop};
use abi_wdbx::DurableStore;

use crate::state::{WdbxStatsError, open_wdbx_store};

/// Why persistence did not happen, when `report` is called with `store = None`.
const NO_STORE_STATUS: &str = "no persistent WDBX path configured (ABI_WDBX_PERSIST=0 or no HOME); \
     in-memory persistence is not yet ported";

/// Run `ai_complete`: canonicalize `requested_model`, complete `input`,
/// persist it if a store is available, and render the MCP report line.
///
/// `requested_model` is resolved through [`abi_ai::models::canonical`] so a
/// short alias (`fable-5`) and its canonical id (`claude-fable-5`) record the
/// same value — matching Zig's `models.canonical` at this exact call site.
///
/// # Errors
///
/// Returns [`WdbxStatsError`] only if a store path is configured but fails to
/// open — never for "no path configured", which is itself a reportable,
/// non-error outcome.
pub fn run(input: &str, requested_model: &str) -> Result<String, WdbxStatsError> {
    let model = abi_ai::models::canonical(requested_model);
    let mut store = open_wdbx_store()?;
    Ok(report(
        store.as_mut(),
        input,
        model,
        abi_foundation::time::unix_ms(),
    ))
}

/// The pure core of `ai_complete`: everything but env resolution.
///
/// `now_ms` is a parameter rather than read internally so tests can pin the
/// audit-chain block's timestamp and get a reproducible `block_id` — the block
/// hash is a function of it, so an internally-read wall clock would make any
/// hash-bearing test flaky.
#[must_use]
pub fn report(store: Option<&mut DurableStore>, input: &str, model: &str, now_ms: i64) -> String {
    // Zig's `complete` returns `error.InvalidCompletionInput` for empty input,
    // which `runLocalCompletion` never catches — it propagates out of the MCP
    // handler as an internal error. Middleware validation upstream
    // (`ai_complete`'s required `input` field) makes an empty string here
    // unreachable in practice; this arm exists so `report` itself is total
    // rather than trusting that invariant silently.
    let Ok(result) = complete(input, model) else {
        return "error: completion input must not be empty".to_string();
    };

    let Some(store) = store else {
        return report_unpersisted(&result, NO_STORE_STATUS);
    };

    let before = store.stats();
    let persisted = persist(store, input, &result, now_ms);
    let after = store.stats();

    let mut out = format!(
        "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={} kv_entries={} vectors={} blocks={} total_kv_entries={} total_vectors={} total_blocks={}",
        result.model,
        result.selected_profile.label(),
        result.audit.passed,
        result.audit.escore,
        result.audit.vetoed,
        persisted.is_some(),
        stat_delta(after.kv_entries, before.kv_entries),
        stat_delta(after.vectors, before.vectors),
        stat_delta(after.blocks, before.blocks),
        after.kv_entries,
        after.vectors,
        after.blocks,
    );

    match persisted {
        Some(persisted) => {
            let _ = write!(
                out,
                " query_vector_id={} metadata_key=completion:{} response_vector_id={} block_id={}",
                persisted.query_vector_id,
                persisted.query_vector_id,
                persisted.response_vector_id,
                persisted.block_hash.to_hex(),
            );
        }
        None => {
            let _ = write!(out, " wdbx_status={}", persistence_failure_status());
        }
    }
    let _ = write!(out, ": {}", result.output);
    out
}

/// The report line for the no-store case: the same head format as the
/// persisted branch, with every counter at zero and `wdbx_status` in place of
/// the persisted-record ids.
fn report_unpersisted(result: &abi_ai::CompletionResult, status: &str) -> String {
    format!(
        "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted=false kv_entries=0 vectors=0 blocks=0 total_kv_entries=0 total_vectors=0 total_blocks=0 wdbx_status={status}: {}",
        result.model,
        result.selected_profile.label(),
        result.audit.passed,
        result.audit.escore,
        result.audit.vetoed,
        result.output,
    )
}

/// The ids and hash of a successfully persisted completion.
struct Persisted {
    query_vector_id: u64,
    response_vector_id: u64,
    block_hash: abi_wdbx::Hash,
}

/// Embed, store, and chain-append one completion. Returns `None` if nothing
/// backs persistence for this build (currently unreachable — WDBX is always
/// linked — kept as a `Result`-shaped seam for the day a build-time WDBX
/// toggle exists, matching Zig's `isFeatureDisabled` catch).
fn persist(
    store: &mut DurableStore,
    input: &str,
    result: &abi_ai::CompletionResult,
    now_ms: i64,
) -> Option<Persisted> {
    let query_vec: [f32; EMBED_DIM] = text_embedding(input);
    let response_vec: [f32; EMBED_DIM] = text_embedding(&result.output);

    let query_id = store.put_vector(&query_vec).ok()?;
    let response_id = store.put_vector(&response_vec).ok()?;

    let metadata = abi_ai::completion::metadata_json(input, result, query_id, response_id);
    let key = abi_ai::completion::metadata_key(query_id);
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

    Some(Persisted {
        query_vector_id: query_id,
        response_vector_id: response_id,
        block_hash: block.hash,
    })
}

/// Placeholder disclosure for the (currently unreachable) in-store persist
/// failure path — distinct from [`NO_STORE_STATUS`], which covers "no store at
/// all".
fn persistence_failure_status() -> &'static str {
    "wdbx write failed"
}

/// Saturating difference, matching `state.statDelta` in Zig.
fn stat_delta(after: usize, before: usize) -> usize {
    after.saturating_sub(before)
}

/// Run `ai_learn`: one SEA self-learning pass against the ambient store.
///
/// # Errors
///
/// Returns [`WdbxStatsError`] only if a store path is configured but fails to
/// open.
pub fn run_learn(
    input: &str,
    requested_model: &str,
    evidence_limit: usize,
) -> Result<String, WdbxStatsError> {
    let model = abi_ai::models::canonical(requested_model);
    let mut store = open_wdbx_store()?;
    Ok(report_learn(
        store.as_mut(),
        input,
        model,
        evidence_limit,
        abi_foundation::time::unix_ms(),
    ))
}

/// Pure core of `ai_learn`: everything but env resolution.
#[must_use]
pub fn report_learn(
    store: Option<&mut DurableStore>,
    input: &str,
    model: &str,
    evidence_limit: usize,
    now_ms: i64,
) -> String {
    let Some(store) = store else {
        // Without a store there is no evidence and no weight persistence.
        // Still run a plain completion so the tool returns a persona response.
        let Ok(result) = complete(input, model) else {
            return "error: completion input must not be empty".to_string();
        };
        return format!(
            "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted=false evidence_count=0 adapted=false kv_entries=0 vectors=0 blocks=0 total_kv_entries=0 total_vectors=0 total_blocks=0 wdbx_status={NO_STORE_STATUS}: {}",
            result.model,
            result.selected_profile.label(),
            result.audit.passed,
            result.audit.escore,
            result.audit.vetoed,
            result.output,
        );
    };

    let before = store.stats();
    let Ok(learned) = run_learn_loop(
        store,
        input,
        model,
        LearnLoopConfig {
            evidence_limit,
            persist: true,
            adapt_router: true,
            ..LearnLoopConfig::default()
        },
        now_ms,
    ) else {
        return "error: completion input must not be empty".to_string();
    };
    let after = store.stats();
    let result = &learned.completion;
    let persisted = learned.persisted.is_some();

    let mut out = format!(
        "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={} evidence_count={} adapted={} kv_entries={} vectors={} blocks={} total_kv_entries={} total_vectors={} total_blocks={}",
        result.model,
        result.selected_profile.label(),
        result.audit.passed,
        result.audit.escore,
        result.audit.vetoed,
        persisted,
        learned.evidence_count,
        learned.adapted,
        stat_delta(after.kv_entries, before.kv_entries),
        stat_delta(after.vectors, before.vectors),
        stat_delta(after.blocks, before.blocks),
        after.kv_entries,
        after.vectors,
        after.blocks,
    );

    if let Some(ids) = &learned.persisted {
        let _ = write!(
            out,
            " query_vector_id={} metadata_key=completion:{} response_vector_id={} block_id={}",
            ids.query_vector_id, ids.query_vector_id, ids.response_vector_id, ids.block_hash_hex,
        );
    } else {
        let _ = write!(out, " wdbx_status={}", persistence_failure_status());
    }
    let _ = write!(out, ": {}", result.output);
    out
}

/// Run `ai_train`: validate, inspect the dataset, persist profile vectors into
/// the store when available, and render the MCP report line.
///
/// `backend` is always `cpu` — no Rust GPU backend is linked (disclosed, not
/// Zig's `gpu-metal`).
///
/// # Errors
///
/// Returns [`WdbxStatsError`] only if a configured store path fails to open.
/// Validation failures become a reportable error string rather than a transport
/// error, matching the handler's Internal-vs-text split for empty completion
/// input — except training validation is always a tool-body error string.
pub fn run_train(config: &TrainingConfig) -> Result<String, WdbxStatsError> {
    let mut store = open_wdbx_store()?;
    Ok(report_train(
        store.as_mut(),
        config,
        abi_foundation::time::unix_ms(),
    ))
}

/// Pure core of `ai_train`.
#[must_use]
pub fn report_train(
    store: Option<&mut DurableStore>,
    config: &TrainingConfig,
    now_ms: i64,
) -> String {
    let (mut result, _summary) = match train_inspect(config) {
        Ok(pair) => pair,
        Err(err) => return format!("error: {err}"),
    };

    let Some(store) = store else {
        return format!(
            "training accepted profile={} dataset={} records={} backend=cpu wdbx_kv_entries=0 wdbx_vectors=0 wdbx_blocks=0 total_kv_entries=0 total_vectors=0 total_blocks=0 wdbx_status={NO_STORE_STATUS}: {}",
            result.profile, result.dataset_path, result.records_stored, result.message,
        );
    };

    let before = store.stats();
    let profile = match abi_ai::parse_agent_profile(&config.profile) {
        Ok(p) => p,
        Err(err) => return format!("error: {err}"),
    };
    let (query_vec, response_vec) = training_vectors(profile);
    let Ok(query_id) = store.put_vector(&query_vec) else {
        return format!(
            "training accepted profile={} dataset={} records=0 backend=cpu wdbx_kv_entries=0 wdbx_vectors=0 wdbx_blocks=0 total_kv_entries={} total_vectors={} total_blocks={}: training accepted; wdbx write failed",
            result.profile, result.dataset_path, before.kv_entries, before.vectors, before.blocks,
        );
    };
    let Ok(response_id) = store.put_vector(&response_vec) else {
        return format!(
            "training accepted profile={} dataset={} records=0 backend=cpu wdbx_kv_entries=0 wdbx_vectors=0 wdbx_blocks=0 total_kv_entries={} total_vectors={} total_blocks={}: training accepted; wdbx write failed",
            result.profile,
            result.dataset_path,
            store.stats().kv_entries,
            store.stats().vectors,
            store.stats().blocks,
        );
    };

    let value = training_store_value(config, query_id, response_id, "cpu");
    let key = training_store_key(&config.profile);
    let _ = store.put(&key, &value);
    let _ = store.add_block(&config.profile, query_id, response_id, &value, now_ms);

    result = abi_ai::apply_store_persistence(result, query_id, response_id);
    let after = store.stats();

    format!(
        "training accepted profile={} dataset={} records={} backend=cpu wdbx_kv_entries={} wdbx_vectors={} wdbx_blocks={} total_kv_entries={} total_vectors={} total_blocks={}: {}",
        result.profile,
        result.dataset_path,
        result.records_stored,
        stat_delta(after.kv_entries, before.kv_entries),
        stat_delta(after.vectors, before.vectors),
        stat_delta(after.blocks, before.blocks),
        after.kv_entries,
        after.vectors,
        after.blocks,
        result.message,
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use abi_wdbx::StorePaths;

    use super::*;

    /// A fixed timestamp, so every test's `block_id` is reproducible rather
    /// than depending on wall-clock time — the block hash is a function of it.
    const FIXED_NOW_MS: i64 = 1_700_000_000_000;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

    /// A private scratch directory, cleaned up on drop.
    ///
    /// Matches the `Fixture` pattern already used in `abi-wdbx`'s own tests
    /// (`retrieval.rs`) rather than adding a `tempfile` dependency for one
    /// module: a PID+sequence-numbered subdirectory of the OS temp dir, unique
    /// per test even when the test binary runs many in parallel threads.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new() -> Self {
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "abi_ai_tools_test_{}_{sequence}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create the scratch directory");
            Self(path)
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch_store() -> (ScratchDir, DurableStore) {
        let dir = ScratchDir::new();
        let store = DurableStore::open(StorePaths::new(&dir.0)).expect("open a fresh store");
        (dir, store)
    }

    #[test]
    fn no_store_reports_the_disclosed_unpersisted_status() {
        let output = report(None, "hello world", "claude-fable-5", FIXED_NOW_MS);
        assert!(output.starts_with(
            "model=claude-fable-5 profile=abbey audit_passed=true audit_escore=1.000 audit_vetoed=false persisted=false"
        ));
        assert!(output.contains("kv_entries=0 vectors=0 blocks=0"));
        assert!(output.contains(&format!("wdbx_status={NO_STORE_STATUS}")));
        assert!(output.ends_with(": Abbey: hello world\n\nI\u{2019}ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit."));
    }

    #[test]
    fn a_fresh_store_persists_and_reports_the_new_ids() {
        let (_dir, mut store) = scratch_store();
        let output = report(
            Some(&mut store),
            "hello world",
            "claude-fable-5",
            FIXED_NOW_MS,
        );

        assert!(output.contains("persisted=true"));
        // First completion in an empty store: `next_vector_id` starts at 1, so
        // the query gets id 1 and the response gets id 2.
        assert!(
            output.contains("query_vector_id=1 metadata_key=completion:1 response_vector_id=2")
        );
        assert!(output.contains("kv_entries=1 vectors=2 blocks=1"));
        assert!(output.contains("total_kv_entries=1 total_vectors=2 total_blocks=1"));
        assert!(output.contains("block_id="));
        // The hex block_id is 64 lowercase hex characters (32-byte SHA-256).
        let hex = output
            .split("block_id=")
            .nth(1)
            .and_then(|rest| rest.split(':').next())
            .unwrap()
            .trim();
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn deltas_are_reported_relative_to_the_store_already_holding_records() {
        let (_dir, mut store) = scratch_store();
        let _ = report(Some(&mut store), "first", "m", FIXED_NOW_MS);
        let output = report(Some(&mut store), "second", "m", FIXED_NOW_MS + 1);

        // The second call's own delta is still +1 kv, +2 vectors, +1 block...
        assert!(output.contains("kv_entries=1 vectors=2 blocks=1"));
        // ...but the totals reflect both calls.
        assert!(output.contains("total_kv_entries=2 total_vectors=4 total_blocks=2"));
        // The second completion's query is id 3 (1, 2 already taken by the first).
        assert!(output.contains("query_vector_id=3"));
    }

    #[test]
    fn a_safety_veto_is_still_persisted_with_the_refusal_text() {
        let (_dir, mut store) = scratch_store();
        let output = report(Some(&mut store), "this will cause harm", "m", FIXED_NOW_MS);
        assert!(output.contains("audit_vetoed=true"));
        assert!(output.contains("persisted=true"));
        assert!(output.ends_with(
            ": I cannot provide that response because it violates the safety constitution (hard veto)."
        ));
    }

    #[test]
    fn repeated_completions_produce_distinct_block_hashes() {
        let (_dir, mut store) = scratch_store();
        let first = report(Some(&mut store), "alpha", "m", FIXED_NOW_MS);
        let second = report(Some(&mut store), "beta", "m", FIXED_NOW_MS + 1);
        let extract = |line: &str| {
            line.split("block_id=")
                .nth(1)
                .unwrap()
                .split(':')
                .next()
                .unwrap()
                .trim()
                .to_string()
        };
        assert_ne!(extract(&first), extract(&second));
    }

    #[test]
    fn learn_on_a_fresh_store_adapts_and_persists() {
        let (_dir, mut store) = scratch_store();
        let output = report_learn(
            Some(&mut store),
            "hello world",
            "claude-fable-5",
            5,
            FIXED_NOW_MS,
        );
        assert!(output.contains("persisted=true"));
        assert!(output.contains("adapted=true"));
        assert!(output.contains("evidence_count=0"));
        assert!(output.contains("query_vector_id=1"));
        assert!(output.contains("profile=abbey"));
    }

    #[test]
    fn learn_recalls_evidence_on_a_related_second_turn() {
        let (_dir, mut store) = scratch_store();
        let _ = report_learn(
            Some(&mut store),
            "the capital of france is paris",
            "m",
            5,
            FIXED_NOW_MS,
        );
        let second = report_learn(
            Some(&mut store),
            "tell me about paris in france",
            "m",
            5,
            FIXED_NOW_MS + 1,
        );
        assert!(second.contains("evidence_count="));
        // extract evidence_count value
        let count: usize = second
            .split("evidence_count=")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(count > 0, "second turn should recall first: {second}");
    }

    #[test]
    fn learn_without_store_reports_disclosed_status() {
        let output = report_learn(None, "hello world", "claude-fable-5", 5, FIXED_NOW_MS);
        assert!(output.contains("persisted=false"));
        assert!(output.contains("adapted=false"));
        assert!(output.contains("evidence_count=0"));
        assert!(output.contains(&format!("wdbx_status={NO_STORE_STATUS}")));
    }

    #[test]
    fn train_on_a_fresh_store_persists_profile_vectors() {
        let (_dir, mut store) = scratch_store();
        let config = TrainingConfig {
            profile: "abi".into(),
            dataset: abi_ai::DatasetSpec {
                path: "/no/such/dataset".into(),
                format: abi_ai::DatasetFormat::Text,
            },
            artifact_dir: "out".into(),
        };
        let output = report_train(Some(&mut store), &config, FIXED_NOW_MS);
        assert!(output.starts_with("training accepted profile=abi dataset=/no/such/dataset"));
        assert!(output.contains("backend=cpu"));
        assert!(output.contains("records=1"));
        assert!(output.contains("wdbx_vectors=2"));
        assert!(output.contains("wdbx_blocks=1"));
        assert!(output.contains("training metadata recorded in wdbx"));
    }

    #[test]
    fn train_without_store_discloses_status() {
        let config = TrainingConfig {
            profile: "abbey".into(),
            dataset: abi_ai::DatasetSpec {
                path: "demo".into(),
                format: abi_ai::DatasetFormat::Text,
            },
            artifact_dir: "out".into(),
        };
        let output = report_train(None, &config, FIXED_NOW_MS);
        assert!(output.contains("backend=cpu"));
        assert!(output.contains(&format!("wdbx_status={NO_STORE_STATUS}")));
    }
}
