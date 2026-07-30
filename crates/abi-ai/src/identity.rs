//! Persona identity contracts: the three profiles and their response templates.
//!
//! Ported from `src/features/ai/identity.zig`.
//!
//! The prefix/suffix pairs below are the single source of truth for every
//! generated response, and they are frozen contract:
//! `tests/golden/mcp-tool-calls-args.jsonl` pins Abbey's pair byte-for-byte
//! through `ai_run`. Note the suffix opens with a **U+2019 right single
//! quotation mark** in "I’ll", not an ASCII apostrophe.

/// The three personas ABI routes between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentProfile {
    /// The empathetic polymath, and the default prior.
    Abbey,
    /// The concise direct-expert mode.
    Aviva,
    /// The orchestration and governance layer.
    Abi,
}

impl AgentProfile {
    /// The canonical lowercase label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Abbey => "abbey",
            Self::Aviva => "aviva",
            Self::Abi => "abi",
        }
    }
}

/// Profile labels in canonical order, matching Zig's `PROFILE_LABELS`.
pub const PROFILE_LABELS: [&str; 3] = ["abbey", "aviva", "abi"];

/// Every profile in canonical order, matching Zig's `known_profiles`.
pub const KNOWN_PROFILES: [AgentProfile; 3] =
    [AgentProfile::Abbey, AgentProfile::Aviva, AgentProfile::Abi];

/// A profile's response template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileContract {
    /// The profile this contract describes.
    pub id: AgentProfile,
    /// Text prepended to the user's input.
    pub response_prefix: &'static str,
    /// Text appended after the user's input.
    pub response_suffix: &'static str,
}

/// The response template for `id`.
#[must_use]
pub const fn profile_contract(id: AgentProfile) -> ProfileContract {
    match id {
        AgentProfile::Abbey => ProfileContract {
            id,
            response_prefix: "Abbey: ",
            response_suffix: "\n\nI\u{2019}ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit.",
        },
        AgentProfile::Aviva => ProfileContract {
            id,
            response_prefix: "Aviva direct expert: ",
            response_suffix: "\n\nLeading with the concrete answer, assumptions, and next action.",
        },
        AgentProfile::Abi => ProfileContract {
            id,
            response_prefix: "ABI orchestration review: ",
            response_suffix: "\n\nEvaluating intent, risk, context, and the appropriate response mode.",
        },
    }
}

/// Abbey's share of the neutral routing prior.
///
/// Abbey is the primary personality, so neutral input routes to her. The three
/// defaults sum to 1.0.
pub const DEFAULT_ABBEY_WEIGHT: f32 = 0.40;
/// Aviva's share of the neutral routing prior.
pub const DEFAULT_AVIVA_WEIGHT: f32 = 0.30;
/// ABI's share of the neutral routing prior.
pub const DEFAULT_ABI_WEIGHT: f32 = 0.30;

#[cfg(test)]
mod tests {
    use super::*;

    // These hold over constants, so they are checked at compile time; a `#[test]`
    // over constants is what clippy::assertions_on_constants objects to.
    const _: () = {
        assert!(DEFAULT_ABBEY_WEIGHT > DEFAULT_AVIVA_WEIGHT);
        assert!(DEFAULT_ABBEY_WEIGHT > DEFAULT_ABI_WEIGHT);
    };

    #[test]
    fn priors_sum_to_one() {
        let total = DEFAULT_ABBEY_WEIGHT + DEFAULT_AVIVA_WEIGHT + DEFAULT_ABI_WEIGHT;
        assert!((total - 1.0).abs() < 0.0001, "got {total}");
    }

    #[test]
    fn abbey_suffix_uses_a_typographic_apostrophe() {
        // An ASCII `'` here would silently break the golden ai_run byte match.
        let suffix = profile_contract(AgentProfile::Abbey).response_suffix;
        assert!(suffix.starts_with("\n\nI\u{2019}ll approach this"));
        assert!(!suffix.contains('\''));
    }

    #[test]
    fn labels_match_the_canonical_order() {
        let labels: Vec<&str> = KNOWN_PROFILES.iter().map(|p| p.label()).collect();
        assert_eq!(labels, PROFILE_LABELS);
    }
}
