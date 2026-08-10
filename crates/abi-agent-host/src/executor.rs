//! Object-safe tool execution boundary.

use abi_agent_runtime::{CancellationToken, ToolCall, ToolSpec, ToolStatus};
use std::{fmt, time::Instant};

/// Cooperative execution context supplied by the host.
#[derive(Clone, Copy, Debug)]
pub struct ToolExecutionContext<'a> {
    /// Shared cancellation token for the complete run.
    pub cancellation: &'a CancellationToken,
    /// Absolute deadline for the complete run.
    pub deadline: Instant,
}

impl ToolExecutionContext<'_> {
    /// Whether cancellation or the deadline already refuses new work.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.cancellation.is_cancelled() || Instant::now() >= self.deadline
    }
}

/// An allowed tool's raw outcome, before the host applies its byte ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    status: ToolStatus,
    payload: String,
}

impl ToolOutput {
    /// Successful raw tool output.
    #[must_use]
    pub fn ok(payload: impl Into<String>) -> Self {
        Self {
            status: ToolStatus::Ok,
            payload: payload.into(),
        }
    }

    /// Failed raw tool output that may be returned to the model.
    #[must_use]
    pub fn error(payload: impl Into<String>) -> Self {
        Self {
            status: ToolStatus::Error,
            payload: payload.into(),
        }
    }

    /// Terminal result status.
    #[must_use]
    pub fn status(&self) -> ToolStatus {
        self.status
    }

    /// Raw payload before host admission.
    #[must_use]
    pub fn payload(&self) -> &str {
        &self.payload
    }
}

/// Executor-side failure represented as a bounded model-visible error result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecutionError {
    message: String,
}

impl ToolExecutionError {
    /// Build an executor failure without exposing an implementation error type.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Model-visible failure description.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ToolExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ToolExecutionError {}

/// Startup-owned implementation of registered tool behavior.
///
/// The model supplies only `call.name` and validated JSON in `call.input`.
/// Executable paths, argv prefixes, environment policy, credentials, and
/// workspace mapping stay encapsulated inside the implementation.
pub trait ToolExecutor: Send + Sync {
    /// Execute one already-validated and authorized call.
    fn execute(
        &self,
        call: &ToolCall,
        spec: &ToolSpec,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError>;
}

#[cfg(test)]
mod tests {
    use super::{ToolExecutionContext, ToolExecutionError, ToolExecutor, ToolOutput};
    use abi_agent_runtime::{CancellationToken, ToolCall, ToolSpec, ToolStatus};
    use std::time::{Duration, Instant};

    struct Fixture;

    impl ToolExecutor for Fixture {
        fn execute(
            &self,
            _call: &ToolCall,
            _spec: &ToolSpec,
            _context: ToolExecutionContext<'_>,
        ) -> Result<ToolOutput, ToolExecutionError> {
            Ok(ToolOutput::ok("done"))
        }
    }

    #[test]
    fn executor_is_object_safe_and_outputs_are_explicit() {
        let executor: Box<dyn ToolExecutor> = Box::new(Fixture);
        let cancel = CancellationToken::new();
        let output = executor
            .execute(
                &ToolCall::new("c1", "read", "{}"),
                &ToolSpec::new("read"),
                ToolExecutionContext {
                    cancellation: &cancel,
                    deadline: Instant::now() + Duration::from_secs(1),
                },
            )
            .expect("fixture executes");
        assert_eq!(output.status(), ToolStatus::Ok);
        assert_eq!(output.payload(), "done");
        assert_eq!(ToolOutput::error("bad").status(), ToolStatus::Error);
        assert_eq!(ToolExecutionError::new("no").message(), "no");
    }

    #[test]
    fn context_observes_cancellation_and_deadline() {
        let cancel = CancellationToken::new();
        let future = ToolExecutionContext {
            cancellation: &cancel,
            deadline: Instant::now() + Duration::from_secs(1),
        };
        assert!(!future.should_stop());
        cancel.cancel();
        assert!(future.should_stop());

        let expired = ToolExecutionContext {
            cancellation: &CancellationToken::new(),
            deadline: Instant::now(),
        };
        assert!(expired.should_stop());
    }
}
