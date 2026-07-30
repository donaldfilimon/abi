//! Memory taxonomy and provenance ladder for SEA.
//!
//! Ported from `src/features/sea/types.zig`.

/// Typed memory taxonomy for durable SEA records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Free-form note.
    Note,
    /// Stated user preference.
    UserPreference,
    /// Project decision.
    ProjectDecision,
    /// Code fact.
    CodeFact,
    /// Tool output.
    ToolOutput,
    /// Benchmark result.
    Benchmark,
    /// Constraint.
    Constraint,
    /// Contradiction.
    Contradiction,
    /// Summary.
    Summary,
}

impl MemoryKind {
    /// Parse a `snake_case` tag name.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "note" => Self::Note,
            "user_preference" => Self::UserPreference,
            "project_decision" => Self::ProjectDecision,
            "code_fact" => Self::CodeFact,
            "tool_output" => Self::ToolOutput,
            "benchmark" => Self::Benchmark,
            "constraint" => Self::Constraint,
            "contradiction" => Self::Contradiction,
            "summary" => Self::Summary,
            _ => return None,
        })
    }

    /// `snake_case` tag name.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::UserPreference => "user_preference",
            Self::ProjectDecision => "project_decision",
            Self::CodeFact => "code_fact",
            Self::ToolOutput => "tool_output",
            Self::Benchmark => "benchmark",
            Self::Constraint => "constraint",
            Self::Contradiction => "contradiction",
            Self::Summary => "summary",
        }
    }
}

/// Provenance/trust ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Authority {
    /// Least trusted: inferred by the system.
    #[default]
    Inferred,
    /// Stated by the user.
    UserStated,
    /// Verified by a tool.
    ToolVerified,
    /// Verified against a file.
    FileVerified,
    /// System-pinned invariant.
    SystemPinned,
}

impl Authority {
    /// Parse a `snake_case` tag name.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "inferred" => Self::Inferred,
            "user_stated" => Self::UserStated,
            "tool_verified" => Self::ToolVerified,
            "file_verified" => Self::FileVerified,
            "system_pinned" => Self::SystemPinned,
            _ => return None,
        })
    }

    /// `snake_case` tag name.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::Inferred => "inferred",
            Self::UserStated => "user_stated",
            Self::ToolVerified => "tool_verified",
            Self::FileVerified => "file_verified",
            Self::SystemPinned => "system_pinned",
        }
    }

    /// Deterministic scalar trust weight for this rung.
    #[must_use]
    pub const fn score(self) -> f32 {
        match self {
            Self::Inferred => 0.30,
            Self::UserStated => 0.78,
            Self::ToolVerified => 0.86,
            Self::FileVerified => 0.90,
            Self::SystemPinned => 1.00,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_kind_round_trips() {
        for kind in [
            MemoryKind::Note,
            MemoryKind::UserPreference,
            MemoryKind::ProjectDecision,
            MemoryKind::CodeFact,
            MemoryKind::ToolOutput,
            MemoryKind::Benchmark,
            MemoryKind::Constraint,
            MemoryKind::Contradiction,
            MemoryKind::Summary,
        ] {
            assert_eq!(MemoryKind::parse(kind.text()), Some(kind));
        }
        assert_eq!(MemoryKind::parse("bogus"), None);
    }

    #[test]
    fn authority_scores_match_the_ladder() {
        assert!((Authority::Inferred.score() - 0.30).abs() < 1e-5);
        assert!((Authority::UserStated.score() - 0.78).abs() < 1e-5);
        assert!((Authority::ToolVerified.score() - 0.86).abs() < 1e-5);
        assert!((Authority::FileVerified.score() - 0.90).abs() < 1e-5);
        assert!((Authority::SystemPinned.score() - 1.00).abs() < 1e-5);
    }
}
