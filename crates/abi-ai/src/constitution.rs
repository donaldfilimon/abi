//! Constitutional governance: per-principle scoring, the weighted E-score, and
//! the hard safety veto.
//!
//! Ported from `src/features/ai/constitution.zig`.
//!
//! ## Scope, stated plainly
//!
//! This is **case-insensitive substring matching** against a fixed table of
//! thirteen patterns. It is a deterministic, auditable governance gate, not a
//! semantic safety classifier, and it cannot detect a harmful response that
//! avoids the listed phrases. The value is that the score is reproducible and the
//! veto is explicit — `ai_complete` reports `audit_escore` to three decimals, so
//! the arithmetic below is contract.

/// The six governed principles.
///
/// The discriminant order is contract: it indexes [`PRINCIPLE_WEIGHTS`] and the
/// `scores` array, and Zig used `@backingInt` on this enum for exactly that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Principle {
    /// Honesty about what is and is not known.
    Truthfulness = 0,
    /// Avoiding harm.
    Safety = 1,
    /// Being useful.
    Helpfulness = 2,
    /// Even-handedness.
    Fairness = 3,
    /// Protecting personal information.
    Privacy = 4,
    /// Disclosing reasoning and AI nature.
    Transparency = 5,
}

impl Principle {
    /// The principle's own name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Truthfulness => "truthfulness",
            Self::Safety => "safety",
            Self::Helpfulness => "helpfulness",
            Self::Fairness => "fairness",
            Self::Privacy => "privacy",
            Self::Transparency => "transparency",
        }
    }

    /// The name used in the WDBX governance spec, which differs for two of six.
    #[must_use]
    pub const fn spec_alias(self) -> &'static str {
        match self {
            Self::Truthfulness => "honesty",
            Self::Helpfulness => "autonomy",
            other => other.label(),
        }
    }

    /// This principle's index into the score array.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Every principle in discriminant order.
pub const PRINCIPLES: [Principle; 6] = [
    Principle::Truthfulness,
    Principle::Safety,
    Principle::Helpfulness,
    Principle::Fairness,
    Principle::Privacy,
    Principle::Transparency,
];

/// Per-principle weights for the constitutional E-score.
///
/// Adapts the WDBX four-pillar formula
/// (`E = α·Autonomy + β·Non-Maleficence + γ·Beneficence + δ·Justice`) onto ABI's
/// six principles, splitting each pillar across the principles that realize it:
///
/// | Pillar | Weight | Principles |
/// |---|---|---|
/// | Beneficence | 0.20 | truthfulness 0.20 |
/// | Non-Maleficence | 0.35 | safety 0.175 + privacy 0.175 |
/// | Autonomy | 0.25 | helpfulness 0.125 + transparency 0.125 |
/// | Justice | 0.20 | fairness 0.20 |
///
/// Non-Maleficence is deliberately over-weighted so a safety regression
/// dominates. The six weights sum to 1.0, keeping the E-score in `[0, 1]`.
pub const PRINCIPLE_WEIGHTS: [f32; 6] = [
    0.20,  // truthfulness (Beneficence)
    0.175, // safety       (Non-Maleficence, safety class)
    0.125, // helpfulness  (Autonomy)
    0.20,  // fairness     (Justice)
    0.175, // privacy      (Non-Maleficence, safety class)
    0.125, // transparency (Autonomy)
];

/// Principles in the safety class — the Non-Maleficence pillar.
///
/// A score below [`SAFETY_VETO_THRESHOLD`] on either one hard-fails the turn
/// regardless of the weighted E-score.
pub const SAFETY_CLASS: [Principle; 2] = [Principle::Safety, Principle::Privacy];

/// Severity floor for the safety class.
pub const SAFETY_VETO_THRESHOLD: f32 = 0.5;

/// One pattern check against a response.
struct PrincipleCheck {
    principle: Principle,
    pattern: &'static str,
    /// Kept for parity with Zig's table and for diagnostics; the scoring logic
    /// only reads `principle`, `pattern`, and `is_negative`.
    #[allow(dead_code)]
    description: &'static str,
    /// A negative check zeroes the principle and fails the audit; a positive one
    /// adds credit.
    is_negative: bool,
}

/// Build one check.
macro_rules! check {
    ($principle:ident, $pattern:literal, $description:literal, $negative:literal) => {
        PrincipleCheck {
            principle: Principle::$principle,
            pattern: $pattern,
            description: $description,
            is_negative: $negative,
        }
    };
}

/// The thirteen checks, in Zig's declaration order.
///
/// Order matters for positive checks: each adds `0.2` and saturates at `1.0`, so
/// two positives on one principle compose.
const PRINCIPLE_CHECKS: [PrincipleCheck; 13] = [
    check!(
        Truthfulness,
        "i don't know",
        "admits uncertainty honestly",
        false
    ),
    check!(
        Truthfulness,
        "definitely false",
        "contains known falsehood",
        true
    ),
    check!(Safety, "harm", "references potential harm", true),
    check!(Safety, "unsafe", "describes unsafe behavior", true),
    check!(
        Safety,
        "safe alternative",
        "suggests safe alternative",
        false
    ),
    check!(
        Helpfulness,
        "here is how",
        "provides actionable guidance",
        false
    ),
    check!(
        Helpfulness,
        "cannot help",
        "refuses to help without reason",
        true
    ),
    check!(
        Fairness,
        "discriminate",
        "contains discriminatory language",
        true
    ),
    check!(Fairness, "everyone", "uses inclusive language", false),
    check!(Privacy, "password", "references credentials", true),
    check!(
        Privacy,
        "personal data",
        "handles personal data references",
        true
    ),
    check!(Transparency, "as an ai", "discloses AI nature", false),
    check!(
        Transparency,
        "i think",
        "expresses reasoning transparently",
        false
    ),
];

/// The outcome of auditing one response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditResult {
    /// Whether the response cleared governance.
    pub passed: bool,
    /// Bit `i` set means principle `i` was violated.
    pub violations: u8,
    /// Per-principle scores in `[0, 1]`, indexed by [`Principle::index`].
    pub scores: [f32; 6],
    /// The weighted E-score over all six principles, in `[0, 1]`.
    pub escore: f32,
    /// Whether the hard safety veto tripped.
    pub vetoed: bool,
}

impl Default for AuditResult {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditResult {
    /// A clean audit: everything passing at `1.0`.
    ///
    /// Zig's `init` also stamped `timestamp: time.unixMs()`. That field is
    /// dropped: no ported caller reads it, and a wall-clock value inside an
    /// otherwise pure, comparable result makes the type non-deterministic for no
    /// benefit. Callers that need a time can stamp one themselves.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            passed: true,
            violations: 0,
            scores: [1.0; 6],
            escore: 1.0,
            vetoed: false,
        }
    }

    /// Whether `principle` was violated.
    #[must_use]
    pub const fn violated(&self, principle: Principle) -> bool {
        self.violations & (1 << principle.index()) != 0
    }

    fn set_violation(&mut self, principle: Principle) {
        self.violations |= 1 << principle.index();
    }

    /// Compute the weighted E-score and apply the hard safety veto.
    ///
    /// Idempotent; call once after every score is assigned. A tripped veto folds
    /// into `passed`.
    pub fn finalize(&mut self) {
        let mut e = 0.0;
        for (score, weight) in self.scores.iter().zip(PRINCIPLE_WEIGHTS) {
            e += score * weight;
        }
        self.escore = e.clamp(0.0, 1.0);

        self.vetoed = SAFETY_CLASS
            .iter()
            .any(|principle| self.scores[principle.index()] < SAFETY_VETO_THRESHOLD);
        if self.vetoed {
            self.passed = false;
        }
    }
}

/// Audit `response` against the full check table.
///
/// An empty response fails on truthfulness, matching Zig.
#[must_use]
pub fn validate(response: &str) -> AuditResult {
    let mut result = AuditResult::new();

    if response.is_empty() {
        result.passed = false;
        result.set_violation(Principle::Truthfulness);
        result.scores[Principle::Truthfulness.index()] = 0.0;
        result.finalize();
        return result;
    }

    for check in &PRINCIPLE_CHECKS {
        if !contains_ignore_case(response, check.pattern) {
            continue;
        }
        let index = check.principle.index();
        if check.is_negative {
            result.set_violation(check.principle);
            result.scores[index] = 0.0;
            result.passed = false;
        } else {
            result.scores[index] = (result.scores[index] + 0.2).min(1.0);
        }
    }

    result.finalize();
    result
}

/// Case-insensitive substring search, matching `foundation/utils.containsIgnoreCase`.
///
/// ASCII-only case folding, which is what the Zig helper did: the patterns are
/// all ASCII, so folding non-ASCII would change matching without making any
/// pattern reachable.
#[must_use]
pub fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let (haystack, needle) = (haystack.as_bytes(), needle.as_bytes());
    if needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_so_the_escore_stays_bounded() {
        let total: f32 = PRINCIPLE_WEIGHTS.iter().sum();
        assert!((total - 1.0).abs() < 0.0001, "got {total}");
    }

    #[test]
    fn principle_indices_match_declaration_order() {
        for (expected, principle) in PRINCIPLES.iter().enumerate() {
            assert_eq!(principle.index(), expected, "{}", principle.label());
        }
    }

    #[test]
    fn a_clean_response_scores_exactly_one() {
        // This is the golden case: ai_complete reports audit_escore=1.000 for
        // Abbey's response to "hello world".
        let audit = validate(
            "Abbey: hello world\n\nI\u{2019}ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit.",
        );
        assert!(audit.passed);
        assert!(!audit.vetoed);
        assert!(
            (audit.escore - 1.0).abs() < 0.0005,
            "escore renders as {:.3}",
            audit.escore
        );
    }

    #[test]
    fn an_empty_response_fails_on_truthfulness() {
        let audit = validate("");
        assert!(!audit.passed);
        assert!(audit.violated(Principle::Truthfulness));
        // Exact: validate assigns the literal 0.0, so bit equality is the
        // precise claim (and clippy::float_cmp rightly rejects `==` on f32).
        assert_eq!(
            audit.scores[Principle::Truthfulness.index()].to_bits(),
            0.0_f32.to_bits()
        );
        // Truthfulness carries 0.20, so the rest of the score survives.
        assert!((audit.escore - 0.80).abs() < 0.0005, "got {}", audit.escore);
    }

    #[test]
    fn a_safety_hit_trips_the_hard_veto() {
        let audit = validate("this will cause harm");
        assert!(!audit.passed);
        assert!(audit.vetoed, "safety below the floor must veto");
        assert!(audit.violated(Principle::Safety));
    }

    #[test]
    fn a_privacy_hit_also_vetoes() {
        let audit = validate("your password is stored here");
        assert!(audit.vetoed);
        assert!(audit.violated(Principle::Privacy));
    }

    #[test]
    fn a_non_safety_violation_fails_without_vetoing() {
        let audit = validate("we discriminate between cases");
        assert!(!audit.passed, "fairness violation fails the audit");
        assert!(!audit.vetoed, "fairness is outside the safety class");
        assert!(audit.violated(Principle::Fairness));
    }

    #[test]
    fn positive_checks_add_credit_and_saturate() {
        // Both transparency positives hit; the score is already 1.0 and must not
        // exceed it.
        let audit = validate("As an AI, I think this is fine");
        // Exact: the second positive check saturates at the literal 1.0.
        assert_eq!(
            audit.scores[Principle::Transparency.index()].to_bits(),
            1.0_f32.to_bits()
        );
        assert!(audit.escore <= 1.0);
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert!(validate("HARM").vetoed);
        assert!(validate("Harm").vetoed);
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut audit = validate("this will cause harm");
        let once = audit;
        audit.finalize();
        assert_eq!(audit, once);
    }

    #[test]
    fn contains_ignore_case_edge_cases() {
        assert!(contains_ignore_case("anything", ""));
        assert!(!contains_ignore_case("ab", "abc"));
        assert!(contains_ignore_case("xxABCxx", "abc"));
        assert!(!contains_ignore_case("", "a"));
        // Non-ASCII in the haystack must not panic or match spuriously.
        assert!(!contains_ignore_case("日本語", "abc"));
        assert!(contains_ignore_case("日本語 HARM here", "harm"));
    }

    #[test]
    fn spec_aliases_differ_for_exactly_two_principles() {
        let renamed: Vec<&str> = PRINCIPLES
            .iter()
            .filter(|p| p.label() != p.spec_alias())
            .map(|p| p.spec_alias())
            .collect();
        assert_eq!(renamed, vec!["honesty", "autonomy"]);
    }
}
