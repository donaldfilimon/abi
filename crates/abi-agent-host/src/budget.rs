//! Host-wide limits spanning every provider continuation and tool call.

use std::time::Duration;

/// The host-wide limit that stopped admission of more work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostBudgetLimit {
    /// Too many model or host events.
    Events,
    /// One provider event exceeded its byte ceiling.
    EventBytes,
    /// Too many provider-reported output tokens across continuations.
    OutputTokens,
    /// Too many assistant-output bytes across continuations.
    OutputBytes,
    /// Too many tool calls across continuations.
    ToolCalls,
    /// Too many provider turns containing tool calls.
    ToolRounds,
    /// Too many total provider invocations.
    ProviderRuns,
    /// The total-run deadline elapsed.
    Deadline,
}

impl HostBudgetLimit {
    /// Stable lowercase label for logs and machine-readable reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Events => "events",
            Self::EventBytes => "event_bytes",
            Self::OutputTokens => "output_tokens",
            Self::OutputBytes => "output_bytes",
            Self::ToolCalls => "tool_calls",
            Self::ToolRounds => "tool_rounds",
            Self::ProviderRuns => "provider_runs",
            Self::Deadline => "deadline",
        }
    }
}

/// Mandatory host-wide ceilings for one complete agent run.
///
/// Defaults are deliberately finite. Builder methods accept zero so tests and
/// callers can express "admit none" and prove the corresponding boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    /// Maximum externally emitted non-terminal events.
    pub max_events: u64,
    /// Maximum bytes carried by one provider event.
    pub max_event_bytes: usize,
    /// Maximum provider-reported output tokens across all provider runs.
    pub max_output_tokens: u64,
    /// Maximum assistant text bytes across all provider runs.
    pub max_output_bytes: usize,
    /// Maximum tool calls across the complete run.
    pub max_tool_calls: u64,
    /// Maximum provider turns that request at least one tool.
    pub max_tool_rounds: u64,
    /// Maximum provider invocations, including the final continuation.
    pub max_provider_runs: u64,
    /// Maximum raw bytes accepted from one tool execution.
    pub max_tool_result_bytes: usize,
    /// Total elapsed-time deadline for provider and tool work.
    pub max_duration: Duration,
}

impl Default for HostBudget {
    fn default() -> Self {
        Self {
            max_events: 1_024,
            max_event_bytes: 65_536,
            max_output_tokens: 16_384,
            max_output_bytes: 1_048_576,
            max_tool_calls: 32,
            max_tool_rounds: 8,
            max_provider_runs: 9,
            max_tool_result_bytes: 65_536,
            max_duration: Duration::from_secs(300),
        }
    }
}

impl HostBudget {
    /// A finite, production-safe default budget.
    #[must_use]
    pub fn bounded() -> Self {
        Self::default()
    }

    /// Set the total event ceiling.
    #[must_use]
    pub fn with_max_events(mut self, value: u64) -> Self {
        self.max_events = value;
        self
    }

    /// Set the per-provider-event byte ceiling.
    #[must_use]
    pub fn with_max_event_bytes(mut self, value: usize) -> Self {
        self.max_event_bytes = value;
        self
    }

    /// Set the total output-token ceiling.
    #[must_use]
    pub fn with_max_output_tokens(mut self, value: u64) -> Self {
        self.max_output_tokens = value;
        self
    }

    /// Set the total assistant-output byte ceiling.
    #[must_use]
    pub fn with_max_output_bytes(mut self, value: usize) -> Self {
        self.max_output_bytes = value;
        self
    }

    /// Set the complete-run tool-call ceiling.
    #[must_use]
    pub fn with_max_tool_calls(mut self, value: u64) -> Self {
        self.max_tool_calls = value;
        self
    }

    /// Set the recursive tool-round ceiling.
    #[must_use]
    pub fn with_max_tool_rounds(mut self, value: u64) -> Self {
        self.max_tool_rounds = value;
        self
    }

    /// Set the total provider-invocation ceiling.
    #[must_use]
    pub fn with_max_provider_runs(mut self, value: u64) -> Self {
        self.max_provider_runs = value;
        self
    }

    /// Set the raw result ceiling applied before bounded capture.
    #[must_use]
    pub fn with_max_tool_result_bytes(mut self, value: usize) -> Self {
        self.max_tool_result_bytes = value;
        self
    }

    /// Set the complete-run elapsed-time deadline.
    #[must_use]
    pub fn with_max_duration(mut self, value: Duration) -> Self {
        self.max_duration = value;
        self
    }
}
