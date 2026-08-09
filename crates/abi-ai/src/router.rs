//! Keyword-based persona routing.
//!
//! Ported from `src/features/ai/router_weights.zig`,
//! `router_sentiment.zig`, and the store-independent half of
//! `router_route.zig`.
//!
//! ## Arithmetic fidelity
//!
//! Routing is decided by comparing `f32` sums, so the accumulation has to happen
//! in the same order and the same precision as Zig's. Both sides multiply an
//! `f32` score by an `f32` `0.1` and add into an `f32` accumulator, so the two
//! agree bit for bit. Widening to `f64` here would be a silent behavior change
//! at near-ties, not an improvement.
//!
//! Soul-network blending lives in [`crate::route_with_soul`] (CLI `complete
//! --soul`). The pure [`crate::AdaptiveModulator`] lives in
//! [`crate::modulator`]; store I/O for its weights is the SEA learn loop's
//! concern.

use crate::identity::{
    AgentProfile, DEFAULT_ABBEY_WEIGHT, DEFAULT_ABI_WEIGHT, DEFAULT_AVIVA_WEIGHT,
};
use crate::keywords::SENTIMENT_KEYWORDS;

/// A routing distribution over the three personas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileWeights {
    /// Weight for Abbey.
    pub w_abbey: f32,
    /// Weight for Aviva.
    pub w_aviva: f32,
    /// Weight for ABI.
    pub w_abi: f32,
}

impl ProfileWeights {
    /// Scale the weights to sum to 1.0.
    ///
    /// A non-positive total is left untouched, matching Zig's `if (total > 0)`.
    pub fn normalize(&mut self) {
        let total = self.w_abbey + self.w_aviva + self.w_abi;
        if total > 0.0 {
            self.w_abbey /= total;
            self.w_aviva /= total;
            self.w_abi /= total;
        }
    }

    /// The neutral prior, favoring Abbey.
    #[must_use]
    pub const fn prior() -> Self {
        Self {
            w_abbey: DEFAULT_ABBEY_WEIGHT,
            w_aviva: DEFAULT_AVIVA_WEIGHT,
            w_abi: DEFAULT_ABI_WEIGHT,
        }
    }

    /// A distribution putting all weight on `profile`.
    #[must_use]
    pub const fn only(profile: AgentProfile) -> Self {
        match profile {
            AgentProfile::Abbey => Self {
                w_abbey: 1.0,
                w_aviva: 0.0,
                w_abi: 0.0,
            },
            AgentProfile::Aviva => Self {
                w_abbey: 0.0,
                w_aviva: 1.0,
                w_abi: 0.0,
            },
            AgentProfile::Abi => Self {
                w_abbey: 0.0,
                w_aviva: 0.0,
                w_abi: 1.0,
            },
        }
    }
}

/// Blend `a` and `b`, where `alpha` of `0.0` yields `a` and `1.0` yields `b`.
#[must_use]
pub fn blend_weights(a: ProfileWeights, b: ProfileWeights, alpha: f32) -> ProfileWeights {
    let a_alpha = 1.0 - alpha;
    ProfileWeights {
        w_abbey: a.w_abbey * a_alpha + b.w_abbey * alpha,
        w_aviva: a.w_aviva * a_alpha + b.w_aviva * alpha,
        w_abi: a.w_abi * a_alpha + b.w_abi * alpha,
    }
}

/// Recognize an explicit leading persona address such as `Aviva, be direct.` or
/// `ABI: orchestrate this.`.
///
/// Only an exact ASCII name token at the very start counts. A name inside a
/// larger word (`avivacious`) or mentioned later in prose (`Please ask Aviva
/// to…`) is not a selector, so ordinary text about the personas still routes
/// heuristically.
#[must_use]
pub fn explicit_profile_selector(input: &str) -> Option<AgentProfile> {
    let mut remaining = input.trim_matches([' ', '\t', '\r', '\n']);
    remaining = remaining.strip_prefix('@').unwrap_or(remaining);

    // Byte indexing is safe: the scan stops at the first non-ASCII-alphabetic
    // byte, and every byte of a multi-byte UTF-8 sequence is outside that class.
    let bytes = remaining.as_bytes();
    let name_end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_alphabetic())
        .unwrap_or(bytes.len());
    if name_end == 0 {
        return None;
    }
    if name_end < bytes.len() {
        let separator = bytes[name_end];
        if !is_zig_ascii_whitespace(separator) && !matches!(separator, b',' | b':') {
            return None;
        }
    }

    let name = &remaining[..name_end];
    if name.eq_ignore_ascii_case("abbey") {
        return Some(AgentProfile::Abbey);
    }
    if name.eq_ignore_ascii_case("aviva") {
        return Some(AgentProfile::Aviva);
    }
    if name.eq_ignore_ascii_case("abi") {
        return Some(AgentProfile::Abi);
    }
    None
}

/// Score `input` into a normalized routing distribution.
///
/// An explicit leading persona address short-circuits to that persona. Otherwise
/// the Abbey-favoring prior is adjusted by every keyword whose stem **prefixes**
/// a whitespace-split token.
///
/// Prefix-only matching is deliberate: it keeps intended stems (`quickly` →
/// `quick`, `running` → `run`) while rejecting the suffix false positives that
/// used to shift routing on unrelated words (`overrun` → `run`, `unsafe` →
/// `safe`, `prefix` → `fix`).
#[must_use]
pub fn analyze_sentiment(input: &str) -> ProfileWeights {
    if let Some(profile) = explicit_profile_selector(input) {
        return ProfileWeights::only(profile);
    }

    let mut weights = ProfileWeights::prior();

    // Equivalent ASCII whitespace must not change routing. This also keeps the
    // parser bounded to the same byte-oriented domain as the keyword table.
    for word in input.split_ascii_whitespace() {
        let trimmed = word.trim_end_matches(['.', ',', '!', '?', ':', ';', '"', '\'']);
        for keyword in &SENTIMENT_KEYWORDS {
            if starts_with_ignore_case(trimmed, keyword.word) {
                weights.w_abbey += keyword.abbey_score * 0.1;
                weights.w_aviva += keyword.aviva_score * 0.1;
                weights.w_abi += keyword.abi_score * 0.1;
            }
        }
    }

    weights.normalize();
    weights
}

/// Pick the highest-weighted persona.
///
/// Ties break Abbey, then Aviva, then ABI — the same `>=` chain Zig used, which
/// makes an all-equal distribution route to Abbey.
#[must_use]
pub fn select_best_profile(weights: ProfileWeights) -> AgentProfile {
    if weights.w_abbey >= weights.w_aviva && weights.w_abbey >= weights.w_abi {
        AgentProfile::Abbey
    } else if weights.w_aviva >= weights.w_abi {
        AgentProfile::Aviva
    } else {
        AgentProfile::Abi
    }
}

/// Whether `byte` is whitespace by Zig's definition.
///
/// Zig's `std.ascii.whitespace` is `{' ', '\t', '\n', '\r', 0x0B, 0x0C}`, but
/// Rust's `u8::is_ascii_whitespace` deliberately omits vertical tab (0x0B). Using
/// the Rust predicate would reject `"Abbey\x0bhello"` as a persona address where
/// Zig accepts it, so the Zig set is spelled out.
const fn is_zig_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Whether `haystack` begins with `needle`, ignoring ASCII case.
fn starts_with_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack.len() >= needle.len()
        && haystack.as_bytes()[..needle.len()].eq_ignore_ascii_case(needle.as_bytes())
}

/// The persona `input` routes to.
#[must_use]
pub fn route_profile(input: &str) -> AgentProfile {
    select_best_profile(analyze_sentiment(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_input_defaults_to_abbey() {
        assert_eq!(route_profile("hello there"), AgentProfile::Abbey);
        // The golden ai_run input.
        assert_eq!(route_profile("hello world"), AgentProfile::Abbey);
    }

    #[test]
    fn analyze_sentiment_returns_normalized_weights() {
        let weights = analyze_sentiment("analyze the logical structure of this system");
        assert!(weights.w_abbey > weights.w_aviva);
        assert!(weights.w_abbey > weights.w_abi);
        let total = weights.w_abbey + weights.w_aviva + weights.w_abi;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn action_input_favors_aviva() {
        let weights = analyze_sentiment("execute deploy run the build quickly");
        assert!(weights.w_aviva > weights.w_abbey);
        assert!(weights.w_aviva > weights.w_abi);
    }

    #[test]
    fn orchestration_input_favors_abi() {
        let weights = analyze_sentiment("orchestrate routing governance policy profile");
        assert!(weights.w_abi > weights.w_abbey);
        assert!(weights.w_abi > weights.w_aviva);
    }

    #[test]
    fn suffix_false_positives_do_not_shift_routing() {
        let neutral = analyze_sentiment("zzzqqq");
        for word in ["overrun", "unsafe", "prefix", "redesign"] {
            let weights = analyze_sentiment(word);
            assert!(
                (neutral.w_abbey - weights.w_abbey).abs() < 0.0001,
                "{word} must not match by suffix"
            );
            assert!((neutral.w_aviva - weights.w_aviva).abs() < 0.0001, "{word}");
            assert!((neutral.w_abi - weights.w_abi).abs() < 0.0001, "{word}");
        }
    }

    #[test]
    fn prefix_stems_still_match() {
        let neutral = analyze_sentiment("zzzqqq");
        assert!(analyze_sentiment("quickly").w_aviva > neutral.w_aviva);
        assert!(analyze_sentiment("running").w_aviva > neutral.w_aviva);
    }

    #[test]
    fn analyze_sentiment_is_case_insensitive() {
        let weights = analyze_sentiment("ANALYZE the LOGICAL structure");
        assert!(weights.w_abbey > weights.w_aviva);
        assert!(weights.w_abbey > weights.w_abi);
    }

    #[test]
    fn explicit_leading_addresses_override_the_heuristic() {
        assert_eq!(
            explicit_profile_selector("Aviva, be direct."),
            Some(AgentProfile::Aviva)
        );
        assert_eq!(
            explicit_profile_selector("ABI, orchestrate this."),
            Some(AgentProfile::Abi)
        );
        assert_eq!(
            explicit_profile_selector("  @Abbey: explain this warmly."),
            Some(AgentProfile::Abbey)
        );
        // An explicit address wins even when the prose points elsewhere.
        assert_eq!(
            route_profile("Aviva, explain this creatively."),
            AgentProfile::Aviva
        );
        assert_eq!(route_profile("ABI, help me brainstorm."), AgentProfile::Abi);
    }

    #[test]
    fn explicit_selector_rejects_embedded_and_incidental_names() {
        for input in [
            "habitual orchestration",
            "stability review",
            "avivacious idea",
            "Please ask Aviva to review this",
            "ABI2, orchestrate this",
            "ABI-compatible design",
            "Abbey-road directions",
            "Aviva-like response",
        ] {
            assert_eq!(explicit_profile_selector(input), None, "{input}");
        }
    }

    #[test]
    fn equivalent_ascii_whitespace_routes_identically() {
        let words = "execute deploy run the build quickly";
        let expected = analyze_sentiment(words);
        for input in [
            "execute\tdeploy\trun\tthe\tbuild\tquickly",
            "execute\ndeploy\nrun\nthe\nbuild\nquickly",
            "execute\r\ndeploy\r\nrun\r\nthe\r\nbuild\r\nquickly",
        ] {
            assert_eq!(analyze_sentiment(input), expected, "{input:?}");
        }
    }

    #[test]
    fn explicit_selector_handles_non_ascii_without_panicking() {
        // The byte scan must not split a multi-byte character.
        assert_eq!(explicit_profile_selector("héllo"), None);
        assert_eq!(explicit_profile_selector("日本語"), None);
        assert_eq!(explicit_profile_selector(""), None);
        assert_eq!(explicit_profile_selector("   "), None);
        // A persona name followed by a multi-byte character is not an address.
        assert_eq!(explicit_profile_selector("Abbeyé"), None);
    }

    #[test]
    fn select_best_profile_tie_breaks_abbey_then_aviva() {
        let three_way = ProfileWeights {
            w_abbey: 0.33,
            w_aviva: 0.33,
            w_abi: 0.33,
        };
        assert_eq!(select_best_profile(three_way), AgentProfile::Abbey);

        let aviva_abi = ProfileWeights {
            w_abbey: 0.2,
            w_aviva: 0.4,
            w_abi: 0.4,
        };
        assert_eq!(select_best_profile(aviva_abi), AgentProfile::Aviva);

        let abi_only = ProfileWeights {
            w_abbey: 0.2,
            w_aviva: 0.2,
            w_abi: 0.6,
        };
        assert_eq!(select_best_profile(abi_only), AgentProfile::Abi);
    }

    #[test]
    fn normalize_leaves_a_zero_distribution_alone() {
        let mut zero = ProfileWeights {
            w_abbey: 0.0,
            w_aviva: 0.0,
            w_abi: 0.0,
        };
        zero.normalize();
        // Exact: a zero total must leave the field bit-identical, not merely
        // near zero — dividing by zero would have produced NaN.
        assert_eq!(zero.w_abbey.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn blend_weights_interpolates_the_endpoints() {
        let a = ProfileWeights::only(AgentProfile::Abbey);
        let b = ProfileWeights::only(AgentProfile::Abi);
        assert_eq!(blend_weights(a, b, 0.0), a);
        assert_eq!(blend_weights(a, b, 1.0), b);
        let mid = blend_weights(a, b, 0.5);
        assert!((mid.w_abbey - 0.5).abs() < f32::EPSILON);
        assert!((mid.w_abi - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn tokens_split_on_ascii_whitespace() {
        let tabbed = analyze_sentiment("x\tanalyze");
        let neutral = analyze_sentiment("zzzqqq");
        assert!(tabbed.w_abbey > neutral.w_abbey);
    }

    #[test]
    fn trailing_punctuation_is_trimmed_before_matching() {
        let neutral = analyze_sentiment("zzzqqq");
        // "analyze." trims to "analyze" and matches.
        assert!(analyze_sentiment("analyze.").w_abbey > neutral.w_abbey);
        assert!(analyze_sentiment("deploy!").w_aviva > neutral.w_aviva);
    }
}
