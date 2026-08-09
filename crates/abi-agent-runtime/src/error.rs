//! Runtime error vocabulary.
//!
//! Errors are described, never swallowed. A provider that cannot produce a
//! reply must return a [`RuntimeError`] rather than emitting a fabricated
//! answer.

use std::fmt;

/// Failure raised while preparing or driving a model run.
///
/// This crate owns no transport, so there is deliberately no I/O, HTTP, or
/// authentication variant here. A provider adapter wraps its own transport
/// failure in [`RuntimeError::Provider`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeError {
    /// The request carried no messages, so there is nothing to run.
    EmptyRequest,
    /// A tool call named a tool that the registry does not describe.
    UnknownTool {
        /// Requested tool name.
        name: String,
    },
    /// A scripted provider ran out of scripted events before finishing.
    ScriptExhausted {
        /// Number of events the script contained.
        scripted: usize,
    },
    /// A provider adapter failed for a provider-specific reason.
    Provider {
        /// Provider identifier, as reported by [`ModelProvider::id`].
        ///
        /// [`ModelProvider::id`]: crate::ModelProvider::id
        provider: String,
        /// Human-readable description supplied by the adapter.
        message: String,
    },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRequest => f.write_str("request carried no messages"),
            Self::UnknownTool { name } => write!(f, "unknown tool: {name}"),
            Self::ScriptExhausted { scripted } => {
                write!(f, "scripted provider exhausted after {scripted} events")
            }
            Self::Provider { provider, message } => {
                write!(f, "provider {provider} failed: {message}")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::RuntimeError;

    #[test]
    fn messages_name_the_cause() {
        assert_eq!(
            RuntimeError::EmptyRequest.to_string(),
            "request carried no messages"
        );
        assert_eq!(
            RuntimeError::UnknownTool {
                name: "shell".to_owned()
            }
            .to_string(),
            "unknown tool: shell"
        );
        assert_eq!(
            RuntimeError::ScriptExhausted { scripted: 3 }.to_string(),
            "scripted provider exhausted after 3 events"
        );
        assert_eq!(
            RuntimeError::Provider {
                provider: "echo".to_owned(),
                message: "no model".to_owned(),
            }
            .to_string(),
            "provider echo failed: no model"
        );
    }
}
