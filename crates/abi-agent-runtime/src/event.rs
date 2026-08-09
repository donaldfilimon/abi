//! The event stream a provider produces, and where it is delivered.

use crate::{
    tool::{ToolCall, ToolResult},
    usage::BudgetLimit,
};

/// Why a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The provider finished on its own.
    Completed,
    /// A [`CancellationToken`] was set and the run stopped at an event boundary.
    ///
    /// [`CancellationToken`]: crate::CancellationToken
    Cancelled,
    /// A [`RunBudget`] ceiling was reached.
    ///
    /// [`RunBudget`]: crate::RunBudget
    BudgetExhausted(BudgetLimit),
    /// The provider returned an error; see the error the run returned.
    Failed,
}

impl StopReason {
    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::BudgetExhausted(_) => "budget_exhausted",
            Self::Failed => "failed",
        }
    }

    /// Whether the provider ran to its own end.
    #[must_use]
    pub fn is_complete(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// One typed item in a provider's output stream.
///
/// The stream is the whole output contract: text arrives as ordered
/// [`ModelEvent::TextDelta`] chunks, and tool use arrives as structured
/// [`ModelEvent::ToolCall`] items rather than text a caller must parse. Exactly
/// one [`ModelEvent::Finished`] is delivered, always last.
///
/// The enum is `#[non_exhaustive]`: new event kinds may be added, so downstream
/// matches need a wildcard arm.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelEvent {
    /// The provider accepted the request and began work.
    Started {
        /// Opaque model identifier the provider actually used.
        model: String,
    },
    /// An incremental chunk of assistant text.
    TextDelta {
        /// Chunk text, carried verbatim.
        text: String,
    },
    /// The model asked for a tool invocation.
    ToolCall(ToolCall),
    /// The host reported the outcome of an invocation.
    ToolResult(ToolResult),
    /// The run ended. Always the final event.
    Finished {
        /// Why the run ended.
        stop: StopReason,
    },
}

impl ModelEvent {
    /// Build a [`ModelEvent::TextDelta`].
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::TextDelta { text: text.into() }
    }

    /// Build a [`ModelEvent::Started`].
    #[must_use]
    pub fn started(model: impl Into<String>) -> Self {
        Self::Started {
            model: model.into(),
        }
    }

    /// Stable lowercase label for the event kind.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Started { .. } => "started",
            Self::TextDelta { .. } => "text_delta",
            Self::ToolCall(_) => "tool_call",
            Self::ToolResult(_) => "tool_result",
            Self::Finished { .. } => "finished",
        }
    }

    /// Text carried by this event, when it carries any.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextDelta { text } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Whether this is the terminal event.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished { .. })
    }
}

/// Destination for a run's events.
///
/// The trait is object-safe and receives events by reference, so a sink may
/// forward, count, or ignore them without cloning. A sink must not block the
/// run: it is called synchronously between provider steps.
pub trait EventSink: Send {
    /// Receive one event.
    fn emit(&mut self, event: &ModelEvent);
}

/// An [`EventSink`] that discards every event.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&mut self, _event: &ModelEvent) {}
}

/// An [`EventSink`] that retains every event in arrival order.
///
/// Unlike streamed text, this buffer is unbounded by design: it exists for
/// tests and small in-process consumers. Use [`RunReport::output`] when a
/// bounded transcript is what is wanted.
///
/// [`RunReport::output`]: crate::RunReport::output
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CollectingSink {
    events: Vec<ModelEvent>,
}

impl CollectingSink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Events received so far, in order.
    #[must_use]
    pub fn events(&self) -> &[ModelEvent] {
        &self.events
    }

    /// Number of events received.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no event has been received.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Kind labels of the received events, in order.
    #[must_use]
    pub fn kinds(&self) -> Vec<&'static str> {
        self.events.iter().map(ModelEvent::kind).collect()
    }

    /// Concatenated text of every [`ModelEvent::TextDelta`] received.
    ///
    /// This is unbounded; it reflects what the sink saw, not what the run
    /// captured.
    #[must_use]
    pub fn text(&self) -> String {
        self.events
            .iter()
            .filter_map(ModelEvent::as_text)
            .collect::<Vec<_>>()
            .concat()
    }
}

impl EventSink for CollectingSink {
    fn emit(&mut self, event: &ModelEvent) {
        self.events.push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{CollectingSink, EventSink, ModelEvent, NullSink, StopReason};
    use crate::{tool::ToolCall, usage::BudgetLimit};

    #[test]
    fn collecting_sink_preserves_order_and_text() {
        let mut sink = CollectingSink::new();
        assert!(sink.is_empty());
        sink.emit(&ModelEvent::started("m"));
        sink.emit(&ModelEvent::text("he"));
        sink.emit(&ModelEvent::text("llo"));
        sink.emit(&ModelEvent::ToolCall(ToolCall::new("c1", "grep", "{}")));
        sink.emit(&ModelEvent::Finished {
            stop: StopReason::Completed,
        });

        assert_eq!(sink.len(), 5);
        assert_eq!(
            sink.kinds(),
            vec![
                "started",
                "text_delta",
                "text_delta",
                "tool_call",
                "finished"
            ]
        );
        assert_eq!(sink.text(), "hello");
        assert!(sink.events()[4].is_terminal());
        assert!(!sink.events()[0].is_terminal());
    }

    #[test]
    fn null_sink_discards() {
        let mut sink = NullSink;
        sink.emit(&ModelEvent::text("ignored"));
    }

    #[test]
    fn stop_reason_labels_are_stable() {
        assert_eq!(StopReason::Completed.as_str(), "completed");
        assert!(StopReason::Completed.is_complete());
        assert_eq!(StopReason::Cancelled.as_str(), "cancelled");
        assert!(!StopReason::Cancelled.is_complete());
        assert_eq!(
            StopReason::BudgetExhausted(BudgetLimit::Events).as_str(),
            "budget_exhausted"
        );
        assert_eq!(StopReason::Failed.as_str(), "failed");
    }

    #[test]
    fn only_text_events_expose_text() {
        assert_eq!(ModelEvent::text("x").as_text(), Some("x"));
        assert_eq!(ModelEvent::started("m").as_text(), None);
    }
}
