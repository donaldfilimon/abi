//! ABI framework AI: persona identity, routing, generation, and governance.
//!
//! The Rust successor to the store-independent core of `src/features/ai/` in the
//! Zig tree. This crate is **pure**: it has no WDBX dependency, does no I/O, and
//! every function in it is deterministic. That is what makes [`run`] — the
//! backing implementation of the MCP `ai_run` tool — byte-reproducible and
//! golden-testable.
//!
//! ## What this covers, and what it does not
//!
//! Ported here: identity contracts, the keyword router, incremental persona
//! generation, and constitutional governance. Together they are exactly the
//! `ai_run` path.
//!
//! Ported here for `ai_complete` as well: pure completion, the model catalog,
//! signed-feature text embeddings (via Zig-compatible Wyhash), and the
//! completion metadata JSON/key helpers. The MCP crate owns the store-facing
//! persistence tail (`put_vector` / `put` / `add_block`) so this crate stays
//! pure and free of a WDBX dependency.
//!
//! Deliberately **not** here, because each depends on a later plan step:
//!
//! - `AdaptiveModulator` — only on the SEA path (`completeAdaptive`); lives with
//!   `abi-sea` / plan step 5c.
//! - The SEA learn loop — `ai_learn`, plan step 5c.
//! - Training — `ai_train`, whose Zig output reports `backend=gpu-metal`; no Rust
//!   GPU backend is linked, so that field will need the same explicit disclosure
//!   `wdbx_stats`'s `backend` already carries (plan step 5d).
//! - `point_neural_net`, `file_context`, `soul_layout`, `multimodal_fusion`,
//!   `streaming`, and `orchestration` — none is on the path to the four MCP AI
//!   tools.
//!
//! ## On the golden fixtures
//!
//! Only `ai_run`'s captured line is a byte-equality target. The
//! `ai_complete`/`ai_learn`/`ai_train` lines in
//! `tests/golden/mcp-tool-calls-args.jsonl` embed live store counters
//! (`total_vectors`, `query_vector_id`, a SHA-256 `block_id`) captured mid-run, so
//! they are shape and field-order references, not values to reproduce. The
//! persona substring inside them *is* contract, and it is asserted.

pub mod completion;
pub mod constitution;
pub mod embedding;
pub mod identity;
pub mod incremental;
pub mod keywords;
pub mod models;
pub mod router;

pub use completion::{CompletionResult, EmptyInputError, complete};
pub use constitution::{AuditResult, Principle, validate};
pub use embedding::{EMBED_DIM, count_non_empty_lines, response_embedding, text_embedding};
pub use identity::{AgentProfile, ProfileContract, profile_contract};
pub use incremental::{StreamChunk, StreamMode, generate_profile_incremental};
pub use router::{ProfileWeights, analyze_sentiment, route_profile, select_best_profile};

/// Generate `profile`'s response to `input`.
///
/// The one-shot path shares [`generate_profile_incremental`] with the streaming
/// path so the two can never drift.
#[must_use]
pub fn route_to_profile(profile: AgentProfile, input: &str) -> String {
    generate_profile_incremental(profile, input, None)
}

/// Route `input` to a persona and generate that persona's response.
#[must_use]
pub fn route_input(input: &str) -> String {
    route_to_profile(route_profile(input), input)
}

/// Run AI inference with internal profile routing — the MCP `ai_run` tool.
///
/// Routes, generates, then audits. The audit's outcome does **not** change the
/// response: Zig logged a warning and returned the text unchanged, and altering
/// that would change a frozen tool's output. The audit is returned alongside so
/// callers that do gate on it (`ai_complete`) can, and so the warning is the
/// caller's choice rather than a hidden write to stderr from a library.
#[must_use]
pub fn run(input: &str) -> (String, AuditResult) {
    let response = route_input(input);
    let audit = validate(&response);
    (response, audit)
}

/// Run AI inference, discarding the audit — the exact `ai_run` response bytes.
#[must_use]
pub fn run_text(input: &str) -> String {
    run(input).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `ai_run` response captured from the Zig server for `"hello world"`.
    const GOLDEN_AI_RUN: &str = "Abbey: hello world\n\nI\u{2019}ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit.";

    #[test]
    fn run_matches_the_golden_ai_run_output_byte_for_byte() {
        assert_eq!(run_text("hello world"), GOLDEN_AI_RUN);
    }

    #[test]
    fn run_is_deterministic_across_repeated_calls() {
        // The property that makes ai_run golden-testable at all: unlike
        // ai_complete, nothing here reads or writes persisted state.
        let first = run_text("hello world");
        for _ in 0..16 {
            assert_eq!(run_text("hello world"), first);
        }
    }

    #[test]
    fn run_audits_a_clean_persona_response_as_passing() {
        let (_, audit) = run("hello world");
        assert!(audit.passed);
        assert!(!audit.vetoed);
    }

    #[test]
    fn a_vetoing_input_still_returns_the_response_unchanged() {
        // "harm" trips the safety veto, but ai_run's contract is to return the
        // generated text regardless. Suppressing it here would change a frozen
        // tool's output.
        let (response, audit) = run("this may cause harm");
        assert!(audit.vetoed);
        assert!(response.starts_with("Abbey: this may cause harm"));
    }

    #[test]
    fn explicit_addresses_route_through_run() {
        assert!(run_text("Aviva, be direct.").starts_with("Aviva direct expert: "));
        assert!(run_text("ABI, orchestrate this.").starts_with("ABI orchestration review: "));
        assert!(run_text("hello").starts_with("Abbey: "));
    }

    #[test]
    fn one_shot_and_streaming_paths_agree() {
        let mut deltas = Vec::new();
        let streamed = {
            let mut sink = |chunk: StreamChunk<'_>| {
                if !chunk.done {
                    deltas.push(chunk.delta.to_string());
                }
            };
            generate_profile_incremental(AgentProfile::Abbey, "hello world", Some(&mut sink))
        };
        assert_eq!(streamed, run_text("hello world"));
        assert_eq!(deltas.concat(), streamed);
    }

    #[test]
    fn empty_input_produces_the_bare_template() {
        // Zig's ai.run had no empty-input guard — only `complete` rejected it with
        // InvalidCompletionInput — so this returns prefix+suffix rather than failing.
        let output = run_text("");
        assert!(output.starts_with("Abbey: \n\n"));
    }
}
