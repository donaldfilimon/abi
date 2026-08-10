//! Complete-run report types.

use crate::HostBudgetLimit;
use abi_agent_runtime::{CapturedText, ToolResult, Usage};

/// Why the host stopped the complete provider/tool loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostStopReason {
    /// A provider turn completed without another tool call.
    Completed,
    /// The shared cancellation token was observed.
    Cancelled,
    /// A host-wide ceiling was reached.
    BudgetExhausted(HostBudgetLimit),
}

/// Evidence from one complete host run across every continuation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRunReport {
    /// Complete-run terminal reason.
    pub stop: HostStopReason,
    /// Saturating merged provider-reported usage.
    pub usage: Usage,
    /// Externally emitted non-terminal event count.
    pub events: u64,
    /// Number of admitted tool calls.
    pub tool_calls: u64,
    /// Number of provider invocations.
    pub provider_runs: u64,
    /// Bounded assistant text from all provider turns.
    pub output: CapturedText,
    /// Bounded tool results in execution order.
    pub tool_results: Vec<ToolResult>,
}

impl HostRunReport {
    /// Whether the provider/tool loop reached a natural completion.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stop == HostStopReason::Completed
    }

    /// Retained assistant text across every continuation.
    #[must_use]
    pub fn text(&self) -> &str {
        self.output.text()
    }
}
