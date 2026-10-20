//! Redacted driver failures retain all already-admitted execution accounting.
use crate::{ApprovalPreview, HostBudgetLimit, HostError};
use abi_capability::lifecycle::{LifecycleError, PreparedExecution, TerminalOutcome};

/// Content-free classification; provider diagnostics and model identifiers are excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernedDriverFailureKind {
    /// Startup schema or registration failed.
    Registration,
    /// Provider failed its invocation.
    Provider,
    /// Untrusted event stream violated the continuation protocol.
    Protocol,
    /// A configured host bound was exceeded.
    Budget(HostBudgetLimit),
    /// Input or result admission failed.
    Admission,
}

/// A failed run is not a completed driver report. Earlier durable accounting survives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GovernedRunFailure {
    /// Closed error category without raw provider/model content.
    pub kind: GovernedDriverFailureKind,
    /// A terminal rejection audit failed; the driver failure remains independently visible.
    pub audit_error: Option<LifecycleError>,
    /// Outcomes observed before the driver failed.
    pub outcomes: Vec<TerminalOutcome>,
    /// Durable execution proofs, including any unfinished intent.
    pub executions: Vec<PreparedExecution>,
    /// Any pending exact preview already produced.
    pub pending_approval: Option<ApprovalPreview>,
}
impl GovernedRunFailure {
    pub(crate) fn from_driver(error: &HostError) -> Self {
        let kind = match error {
            HostError::DuplicateToolName { .. } | HostError::InvalidToolSchema { .. } => {
                GovernedDriverFailureKind::Registration
            }
            HostError::Provider(_) => GovernedDriverFailureKind::Provider,
            HostError::DuplicateCallId { .. }
            | HostError::ProviderToolResult { .. }
            | HostError::PostTerminalEvent { .. }
            | HostError::InvalidTerminalSequence => GovernedDriverFailureKind::Protocol,
            HostError::BudgetExceeded(limit) => GovernedDriverFailureKind::Budget(*limit),
            _ => GovernedDriverFailureKind::Admission,
        };
        Self {
            kind,
            audit_error: None,
            outcomes: Vec::new(),
            executions: Vec::new(),
            pending_approval: None,
        }
    }
}
impl std::fmt::Display for GovernedRunFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "governed driver failure: {:?}", self.kind)
    }
}
impl std::error::Error for GovernedRunFailure {}
