//! Local model completion, and the metadata record persisted alongside it.
//!
//! Ported from `src/features/ai/completion.zig`'s pure completion path and its
//! `completionMetadataJson`/`completionMetadataKey` helpers.
//!
//! ## What `ai_complete` actually calls
//!
//! Zig's `runLocalCompletion` calls `completeWithScheduler` →
//! `completeWithStore` → [`complete`], which routes with the plain
//! [`crate::analyze_sentiment`]/[`crate::select_best_profile`] pair — **not**
//! `completeAdaptive`/`AdaptiveModulator`. The modulator's persisted
//! `modulator:weights` KV entry is only reached from the SEA learn loop. So
//! persona selection for `ai_complete` is deterministic and store-independent;
//! only the counters, the metadata record, and the audit-chain block below are
//! store-derived. (`RUST-REWRITE-PLAN.md` step 5b corrects an earlier note that
//! assumed otherwise.)
//!
//! Zig's scheduler indirection (`submitCompletionTask` → `runAll`) exists so the
//! completion runs as a scheduled task. `abi-core`'s scheduler is one-shot and
//! synchronous, so calling [`complete`] directly is the same observable behavior
//! without the indirection — nothing genuinely concurrent was happening in Zig
//! either.

use std::fmt::Write as _;

use crate::{AgentProfile, AuditResult, route_to_profile, select_best_profile, validate};

/// The frozen public-API contract for the persisted metadata key: "stores JSON
/// completion metadata under `completion:<query_vector_id>`".
const COMPLETION_KEY_PREFIX: &str = "completion:";

/// The text substituted for a hard safety veto.
///
/// The response is replaced, not withheld — a caller sees this string, plus
/// `audit.vetoed == true`.
const VETO_REFUSAL: &str =
    "I cannot provide that response because it violates the safety constitution (hard veto).";

/// A completed turn, before persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionResult {
    /// The model identifier echoed back (not currently resolved against a
    /// catalog — see the module docs on scope).
    pub model: String,
    /// The persona that generated `output`.
    pub selected_profile: AgentProfile,
    /// The generated text, or the hard safety-veto refusal string if the audit
    /// vetoed it.
    pub output: String,
    /// The constitutional audit of the original (pre-veto-substitution) text.
    pub audit: AuditResult,
}

/// Complete `input` with `model`, routing deterministically by keyword.
///
/// # Errors
///
/// Returns an error if `input` is empty, matching Zig's
/// `error.InvalidCompletionInput`.
pub fn complete(input: &str, model: &str) -> Result<CompletionResult, EmptyInputError> {
    if input.is_empty() {
        return Err(EmptyInputError);
    }
    let selected = select_best_profile(crate::analyze_sentiment(input));
    Ok(complete_with_profile(input, model, selected))
}

/// `input` was empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyInputError;

impl std::fmt::Display for EmptyInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("completion input must not be empty")
    }
}

impl std::error::Error for EmptyInputError {}

fn complete_with_profile(input: &str, model: &str, selected: AgentProfile) -> CompletionResult {
    let response = route_to_profile(selected, input);
    let audit = validate(&response);

    // Hard gate: a safety-class veto replaces the body so a caller never sees
    // vetoed text, even though `audit` (computed over the original response)
    // still reports `vetoed = true`.
    let output = if audit.vetoed {
        VETO_REFUSAL.to_string()
    } else {
        response
    };

    CompletionResult {
        model: model.to_string(),
        selected_profile: selected,
        output,
        audit,
    }
}

/// The persisted metadata record for one completion, and the ids/hash needed to
/// render it and the report line around it.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedCompletion {
    /// The completion this record describes.
    pub result: CompletionResult,
    /// The stored query vector's id.
    pub query_vector_id: u64,
    /// The stored response vector's id.
    pub response_vector_id: u64,
    /// The metadata JSON, ready to store under [`metadata_key`].
    pub metadata_json: String,
}

/// The persisted-record key for a query vector id: `completion:<id>`.
///
/// Per the committed contract (`docs/contracts/public-api.mdx`): "stores JSON
/// completion metadata under `completion:<query_vector_id>`".
#[must_use]
pub fn metadata_key(query_vector_id: u64) -> String {
    format!("{COMPLETION_KEY_PREFIX}{query_vector_id}")
}

/// Build the metadata JSON for one completion.
///
/// `input_bytes`/`output_bytes` are the **caller's** input and the completion's
/// generated output length — not the persisted record's own size, and not
/// affected by the veto substitution (Zig reads `result.output.len` after
/// substitution, which this mirrors: the vetoed refusal's length is what gets
/// recorded).
#[must_use]
pub fn metadata_json(
    input: &str,
    result: &CompletionResult,
    query_vector_id: u64,
    response_vector_id: u64,
) -> String {
    let mut out = String::from(
        "{\"kind\":\"completion\",\"persistence\":\"explicit_store_result\",\"authority\":\"inferred\",\"epistemic_status\":\"generated_output\",\"model\":",
    );
    push_json_string(&mut out, &result.model);
    out.push_str(",\"profile\":");
    push_json_string(&mut out, result.selected_profile.label());
    // Manual append keeps field order fixed and avoids an intermediate String.
    let _ = write!(
        out,
        ",\"audit_passed\":{},\"audit_vetoed\":{},\"escore\":{:.3},\"input_bytes\":{},\"output_bytes\":{},\"query_vector_id\":{},\"response_vector_id\":{}}}",
        result.audit.passed,
        result.audit.vetoed,
        result.audit.escore,
        input.len(),
        result.output.len(),
        query_vector_id,
        response_vector_id,
    );
    out
}

/// Append `value` to `out` as a JSON string literal, escaping the same six
/// classes Zig's `appendMetadataJsonString` did: quote, backslash, `\n`, `\r`,
/// `\t`, and every other C0 control byte as `\u00XX` (uppercase hex).
///
/// Iterates by `char`, not by byte. Zig iterated raw bytes, which was correct
/// there because copying a valid-UTF-8 byte unchanged reproduces that byte.
/// Casting a byte to `char` in Rust is a different operation — it reinterprets
/// the byte as a Unicode scalar value, so a continuation byte of a multi-byte
/// sequence (e.g. `0xC3` of `é`) would be re-encoded as a *different*, two-byte
/// character, corrupting every non-ASCII input. Iterating by `char` and pushing
/// each `char` reproduces the identical UTF-8 bytes for anything not in the six
/// escaped classes, which are all single-byte ASCII — so this is a bug fix, not
/// a behavior change, for input Zig's byte loop already handled correctly.
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u00{:02X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_rejects_empty_input() {
        assert_eq!(complete("", "m"), Err(EmptyInputError));
    }

    #[test]
    fn complete_routes_and_audits() {
        let result = complete("hello world", "claude-fable-5").unwrap();
        assert_eq!(result.selected_profile, AgentProfile::Abbey);
        assert!(result.output.starts_with("Abbey: hello world"));
        assert!(result.audit.passed);
        assert!(!result.audit.vetoed);
    }

    #[test]
    fn a_safety_veto_replaces_the_output_but_not_the_audit_flag() {
        let result = complete("this will cause harm", "m").unwrap();
        assert!(result.audit.vetoed);
        assert_eq!(result.output, VETO_REFUSAL);
    }

    #[test]
    fn metadata_key_matches_the_public_api_contract() {
        let key = metadata_key(42);
        assert!(key.starts_with("completion:"));
        assert!(key.contains("42"));
        assert_eq!(key, "completion:42");
    }

    #[test]
    fn metadata_json_matches_the_expected_shape() {
        let result = complete("hello world", "claude-fable-5").unwrap();
        let json = metadata_json("hello world", &result, 655, 656);
        let expected = format!(
            "{{\"kind\":\"completion\",\"persistence\":\"explicit_store_result\",\"authority\":\"inferred\",\"epistemic_status\":\"generated_output\",\"model\":\"claude-fable-5\",\"profile\":\"abbey\",\"audit_passed\":true,\"audit_vetoed\":false,\"escore\":1.000,\"input_bytes\":11,\"output_bytes\":{},\"query_vector_id\":655,\"response_vector_id\":656}}",
            result.output.len()
        );
        assert_eq!(json, expected);

        // Must be valid JSON, not merely shaped like it.
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["query_vector_id"], 655);
        assert_eq!(parsed["escore"], 1.0);
    }

    #[test]
    fn metadata_json_escapes_every_reserved_byte_class() {
        let result = CompletionResult {
            model: "quote\"back\\slash\nline\rreturn\ttab\x01ctrl".to_string(),
            selected_profile: AgentProfile::Abbey,
            output: "x".to_string(),
            audit: AuditResult::new(),
        };
        let json = metadata_json("in", &result, 1, 2);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(
            parsed["model"],
            "quote\"back\\slash\nline\rreturn\ttab\x01ctrl"
        );
    }

    #[test]
    fn multibyte_characters_survive_json_escaping_unmangled() {
        // Regression test for a real bug caught during review: naive
        // byte-by-byte `as char` casting reinterprets UTF-8 continuation bytes
        // as separate Latin-1 code points, corrupting the string.
        let result = CompletionResult {
            model: "café 日本語 emoji 🎉".to_string(),
            selected_profile: AgentProfile::Abbey,
            output: "x".to_string(),
            audit: AuditResult::new(),
        };
        let json = metadata_json("in", &result, 1, 2);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["model"], "café 日本語 emoji 🎉");
    }

    #[test]
    fn escore_renders_to_exactly_three_decimals() {
        // audit_escore=1.000, not "1" or "1.0" — the MCP report line depends on
        // this exact width.
        let result = complete("hello world", "m").unwrap();
        let json = metadata_json("hello world", &result, 1, 2);
        assert!(json.contains("\"escore\":1.000"));
    }
}
