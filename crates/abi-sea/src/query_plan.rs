//! `QueryPlan` / `TaskType` inference for SEA retrieval.
//!
//! Ported from `src/features/sea/query_plan.zig`. Deterministic keyword
//! heuristic — no model call.

/// Recognized retrieval intents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TaskType {
    /// No more-specific intent matched.
    #[default]
    General,
    /// Implementation / design work.
    ImplementationDesign,
    /// Bug fix / compile repair.
    CodeRepair,
    /// Legal / compliance review.
    LegalReview,
    /// Research synthesis.
    ResearchSynthesis,
    /// Project / decision recall (`project_recall`).
    ProjectRecall,
    /// Benchmark review.
    BenchmarkReview,
}

impl TaskType {
    /// `snake_case` tag name (matches Zig `@tagName`).
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::ImplementationDesign => "implementation_design",
            Self::CodeRepair => "code_repair",
            Self::LegalReview => "legal_review",
            Self::ResearchSynthesis => "research_synthesis",
            Self::ProjectRecall => "project_recall",
            Self::BenchmarkReview => "benchmark_review",
        }
    }

    /// Stable ordinal used by the scorer's task-weight adjustments.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::General => 0,
            Self::ImplementationDesign => 1,
            Self::CodeRepair => 2,
            Self::LegalReview => 3,
            Self::ResearchSynthesis => 4,
            Self::ProjectRecall => 5,
            Self::BenchmarkReview => 6,
        }
    }
}

/// Inferred structured intent for a query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    /// Inferred task type.
    pub task: TaskType,
    /// Borrowed query text (never owned by the plan).
    pub query: String,
    /// Answers should be backed by selected evidence.
    pub require_grounding: bool,
    /// Prefer exact wording over fuzzy similarity.
    pub exact_recall: bool,
    /// Relative preference for fresher records, `[0,1]`.
    pub recency_bias: f32,
    /// Risk posture of the task, `[0,1]`.
    pub risk: f32,
}

/// Keyword groups in ascending precedence — later groups overwrite earlier
/// matches (last-match-wins).
const KEYWORD_GROUPS: &[(TaskType, &[&str])] = &[
    (
        TaskType::ImplementationDesign,
        &["zig", "code", "build.zig", "implement", "design"],
    ),
    (
        TaskType::CodeRepair,
        &["bug", "patch", "compile", "fix", "error"],
    ),
    (
        TaskType::LegalReview,
        &["contract", "legal", "license", "compliance"],
    ),
    (
        TaskType::ResearchSynthesis,
        &["paper", "research", "eval", "study"],
    ),
    (
        TaskType::BenchmarkReview,
        &["benchmark", "throughput", "latency", "perf"],
    ),
    (
        TaskType::ProjectRecall,
        &["remember", "prior", "decision", "recall", "earlier"],
    ),
];

/// Deterministically infer a [`QueryPlan`] from `query`.
#[must_use]
pub fn infer(query: &str) -> QueryPlan {
    let mut plan = QueryPlan {
        task: TaskType::General,
        query: query.to_string(),
        require_grounding: true,
        exact_recall: false,
        recency_bias: 0.40,
        risk: 0.50,
    };

    for &(task, keywords) in KEYWORD_GROUPS {
        for kw in keywords {
            if contains_ignore_case(query, kw) {
                plan.task = task;
                break;
            }
        }
    }

    match plan.task {
        TaskType::ProjectRecall => {
            plan.exact_recall = true;
            plan.recency_bias = 0.20;
        }
        TaskType::CodeRepair => {
            plan.recency_bias = 0.60;
        }
        _ => {}
    }

    plan
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    let hay = haystack.as_bytes();
    let ned = needle.as_bytes();
    let mut i = 0;
    while i + ned.len() <= hay.len() {
        if eq_ignore_case_ascii(&hay[i..i + ned.len()], ned) {
            return true;
        }
        i += 1;
    }
    false
}

fn eq_ignore_case_ascii(a: &[u8], b: &[u8]) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_general() {
        let plan = infer("what is the weather like today");
        assert_eq!(plan.task, TaskType::General);
        assert!(!plan.exact_recall);
    }

    #[test]
    fn detects_each_task_class() {
        assert_eq!(
            infer("how do I write this in zig").task,
            TaskType::ImplementationDesign
        );
        assert_eq!(
            infer("help me fix this compile bug").task,
            TaskType::CodeRepair
        );
        assert!(infer("help me fix this compile bug").recency_bias > 0.5);
        assert_eq!(
            infer("review this contract for compliance").task,
            TaskType::LegalReview
        );
        assert_eq!(
            infer("summarize the research paper").task,
            TaskType::ResearchSynthesis
        );
        assert_eq!(
            infer("what is the latency benchmark").task,
            TaskType::BenchmarkReview
        );
        let recall = infer("remember the prior decision we made");
        assert_eq!(recall.task, TaskType::ProjectRecall);
        assert!(recall.exact_recall);
        assert!(recall.recency_bias < 0.40);
    }

    #[test]
    fn case_insensitive_and_last_match_wins() {
        assert_eq!(
            infer("Rewrite the CODE").task,
            TaskType::ImplementationDesign
        );
        // both "code" and "remember"; project_recall is later and wins
        assert_eq!(
            infer("remember the code decision").task,
            TaskType::ProjectRecall
        );
    }
}
