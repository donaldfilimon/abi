//! Fail-closed host error vocabulary.

use crate::HostBudgetLimit;
use abi_agent_runtime::RuntimeError;
use std::fmt;

/// A failure that prevents a trustworthy agent continuation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostError {
    /// The registry contains the same tool name more than once.
    DuplicateToolName {
        /// Duplicated name.
        name: String,
    },
    /// A registered tool carries an invalid JSON Schema.
    InvalidToolSchema {
        /// Tool whose schema is invalid.
        tool: String,
        /// Bounded validator diagnostic.
        message: String,
    },
    /// A provider failed before producing a usable turn.
    Provider(RuntimeError),
    /// A model call identifier was already used in this run.
    DuplicateCallId {
        /// Repeated identifier.
        call_id: String,
    },
    /// A model named a tool absent from the startup registry.
    UnknownTool {
        /// Correlation identifier.
        call_id: String,
        /// Requested tool name.
        tool: String,
    },
    /// Tool input is not valid JSON.
    MalformedToolInput {
        /// Correlation identifier.
        call_id: String,
        /// Bounded parser diagnostic.
        message: String,
    },
    /// Tool input JSON does not satisfy the registered schema.
    ToolSchemaViolation {
        /// Correlation identifier.
        call_id: String,
        /// Bounded validator diagnostic.
        message: String,
    },
    /// A provider emitted a result that only the host may create.
    ProviderToolResult {
        /// Correlation identifier from the fabricated result.
        call_id: String,
    },
    /// A provider emitted an event after a terminal event.
    PostTerminalEvent {
        /// Kind of the event after the terminal marker.
        kind: String,
    },
    /// A provider turn did not end in exactly one terminal event.
    InvalidTerminalSequence,
    /// Raw tool output exceeded its admission ceiling.
    ToolResultTooLarge {
        /// Correlation identifier.
        call_id: String,
        /// Actual raw byte length.
        bytes: usize,
        /// Configured ceiling.
        limit: usize,
    },
    /// A host-wide budget refused more work.
    BudgetExceeded(HostBudgetLimit),
}

impl fmt::Display for HostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateToolName { name } => write!(formatter, "duplicate tool name: {name}"),
            Self::InvalidToolSchema { tool, message } => {
                write!(formatter, "invalid schema for tool {tool}: {message}")
            }
            Self::Provider(error) => write!(formatter, "provider failed: {error}"),
            Self::DuplicateCallId { call_id } => write!(formatter, "duplicate call id: {call_id}"),
            Self::UnknownTool { call_id, tool } => {
                write!(formatter, "tool call {call_id} names unknown tool: {tool}")
            }
            Self::MalformedToolInput { call_id, message } => {
                write!(
                    formatter,
                    "tool call {call_id} has malformed JSON: {message}"
                )
            }
            Self::ToolSchemaViolation { call_id, message } => {
                write!(
                    formatter,
                    "tool call {call_id} violates its schema: {message}"
                )
            }
            Self::ProviderToolResult { call_id } => {
                write!(
                    formatter,
                    "provider fabricated tool result for call: {call_id}"
                )
            }
            Self::PostTerminalEvent { kind } => {
                write!(formatter, "provider emitted {kind} after a terminal event")
            }
            Self::InvalidTerminalSequence => {
                formatter.write_str("provider turn did not end with one terminal event")
            }
            Self::ToolResultTooLarge {
                call_id,
                bytes,
                limit,
            } => write!(
                formatter,
                "tool call {call_id} returned {bytes} bytes, exceeding limit {limit}"
            ),
            Self::BudgetExceeded(limit) => {
                write!(formatter, "host budget exceeded: {}", limit.as_str())
            }
        }
    }
}

impl std::error::Error for HostError {}
