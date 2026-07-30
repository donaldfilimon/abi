//! Twilio `ConversationRelay` local conversation builder.
//!
//! Ported from `src/connectors/twilio_relay.zig` for the offline / MCP
//! `connector_test` path. Does not contact the live Twilio API. Live event
//! streaming over WebSocket remains a disclosed non-goal for this module.

/// `ConversationRelay` event kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// Call setup.
    Setup,
    /// Caller speech transcript.
    UserTranscript,
    /// Keypad input.
    Dtmf,
    /// Caller interrupt.
    Interrupt,
    /// Call hangup.
    Disconnect,
}

/// Why a transcript should escalate to a human queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationReason {
    /// Caller asked for a human.
    HumanRequested,
    /// Empty transcript.
    EmptyTranscript,
    /// Sensitive topic keywords.
    SensitiveTopic,
    /// Short / low-confidence transcript.
    LowConfidence,
    /// Intelligence signal recommended escalation.
    IntelligenceSignal,
}

impl EscalationReason {
    /// Stable reason code string.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::HumanRequested => "human_requested",
            Self::EmptyTranscript => "empty_transcript",
            Self::SensitiveTopic => "sensitive_topic",
            Self::LowConfidence => "low_confidence",
            Self::IntelligenceSignal => "intelligence_signal",
        }
    }
}

/// Optional memory recall attached to an event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversationMemory {
    /// Profile id.
    pub profile_id: String,
    /// Profile summary.
    pub profile_summary: String,
    /// Recall summary injected into local replies.
    pub recall_summary: String,
}

/// Optional intelligence signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntelligenceSignal {
    /// Sentiment label.
    pub sentiment: String,
    /// Compliance status.
    pub compliance_status: String,
    /// Whether escalation is recommended.
    pub escalation_recommended: bool,
}

impl Default for IntelligenceSignal {
    fn default() -> Self {
        Self {
            sentiment: "neutral".into(),
            compliance_status: "clear".into(),
            escalation_recommended: false,
        }
    }
}

/// One `ConversationRelay` event (local synthesis path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationRelayEvent {
    /// Event kind.
    pub kind: EventKind,
    /// Conversation / call id.
    pub conversation_id: String,
    /// Customer id (often E.164).
    pub customer_id: String,
    /// Transcript text for user events.
    pub transcript: String,
    /// DTMF digit when applicable.
    pub digit: String,
    /// Optional memory.
    pub memory: Option<ConversationMemory>,
    /// Optional intelligence signal.
    pub intelligence: Option<IntelligenceSignal>,
}

impl ConversationRelayEvent {
    /// Build a user-transcript event with the MCP fixture identifiers.
    #[must_use]
    pub fn user_transcript(transcript: impl Into<String>) -> Self {
        Self {
            kind: EventKind::UserTranscript,
            conversation_id: "CA0123456789abcdef0123456789abcdef".into(),
            customer_id: "+15551234567".into(),
            transcript: transcript.into(),
            digit: String::new(),
            memory: None,
            intelligence: None,
        }
    }
}

/// Escalation handoff payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EscalationPayload {
    /// Conversation id.
    pub conversation_id: String,
    /// Customer id.
    pub customer_id: String,
    /// Reason code.
    pub reason_code: String,
    /// Summary for the human queue.
    pub summary: String,
    /// Routing hints.
    pub routing_hints: String,
}

/// Local `ConversationRelay` response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationRelayResponse {
    /// Reply text returned to the voice path.
    pub text: String,
    /// Optional escalation.
    pub escalation: Option<EscalationPayload>,
}

/// Classify whether `event` should escalate.
#[must_use]
pub fn classify_escalation(event: &ConversationRelayEvent) -> Option<EscalationReason> {
    if event
        .intelligence
        .as_ref()
        .is_some_and(|s| s.escalation_recommended)
    {
        return Some(EscalationReason::IntelligenceSignal);
    }
    let transcript = event.transcript.trim();
    if transcript.is_empty() {
        return Some(EscalationReason::EmptyTranscript);
    }
    if contains_any_ignore_case(
        transcript,
        &["human", "representative", "real person", "support agent"],
    ) {
        return Some(EscalationReason::HumanRequested);
    }
    if contains_any_ignore_case(
        transcript,
        &[
            "credit card",
            "card number",
            "ssn",
            "social security",
            "medical",
            "diagnosis",
            "debt collection",
        ],
    ) {
        return Some(EscalationReason::SensitiveTopic);
    }
    if transcript.len() < 3
        || contains_any_ignore_case(transcript, &["not sure", "confused", "unknown error"])
    {
        return Some(EscalationReason::LowConfidence);
    }
    None
}

/// Build the local `ConversationRelay` response for `event`.
#[must_use]
pub fn build_local_conversation_response(
    event: &ConversationRelayEvent,
    agent_reply: &str,
) -> ConversationRelayResponse {
    match event.kind {
        EventKind::Setup => {
            return ConversationRelayResponse {
                text: "Hello, this is ABI support. How can I help today?".into(),
                escalation: None,
            };
        }
        EventKind::Disconnect => {
            return ConversationRelayResponse {
                text: "Thanks for contacting ABI support.".into(),
                escalation: None,
            };
        }
        EventKind::Dtmf => {
            let digit = if event.digit.is_empty() {
                "unknown"
            } else {
                event.digit.as_str()
            };
            return ConversationRelayResponse {
                text: format!("I received keypad input {digit}."),
                escalation: None,
            };
        }
        EventKind::Interrupt => {
            return ConversationRelayResponse {
                text: "I heard an interruption. Please continue and I will adjust.".into(),
                escalation: None,
            };
        }
        EventKind::UserTranscript => {}
    }

    if let Some(reason) = classify_escalation(event) {
        return ConversationRelayResponse {
            text: "I can connect you with a support specialist. Please hold while I pass along the context.".into(),
            escalation: Some(build_escalation_payload(event, reason)),
        };
    }

    let text = if let Some(memory) = &event.memory
        && !memory.recall_summary.is_empty()
    {
        format!(
            "{agent_reply} I also found this customer context: {}",
            memory.recall_summary
        )
    } else {
        agent_reply.to_string()
    };
    ConversationRelayResponse {
        text,
        escalation: None,
    }
}

/// Build an escalation payload for `reason`.
#[must_use]
pub fn build_escalation_payload(
    event: &ConversationRelayEvent,
    reason: EscalationReason,
) -> EscalationPayload {
    let reason_code = reason.code().to_string();
    let transcript = event.transcript.trim();
    let summary = if transcript.is_empty() {
        format!(
            "Voice support escalation for {}: no usable transcript captured",
            event.customer_id
        )
    } else {
        format!(
            "Voice support escalation for {}: {transcript}",
            event.customer_id
        )
    };
    let priority = if matches!(
        reason,
        EscalationReason::SensitiveTopic | EscalationReason::IntelligenceSignal
    ) {
        "high"
    } else {
        "normal"
    };
    EscalationPayload {
        conversation_id: event.conversation_id.clone(),
        customer_id: event.customer_id.clone(),
        reason_code: reason_code.clone(),
        summary,
        routing_hints: format!(
            "queue=support;priority={priority};channel=voice;reason={reason_code}"
        ),
    }
}

fn contains_any_ignore_case(haystack: &str, needles: &[&str]) -> bool {
    let lower = haystack.to_ascii_lowercase();
    needles
        .iter()
        .any(|n| lower.contains(&n.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_replies_for_setup_disconnect_interrupt_dtmf() {
        let setup = ConversationRelayEvent {
            kind: EventKind::Setup,
            conversation_id: "c".into(),
            customer_id: "u".into(),
            transcript: String::new(),
            digit: String::new(),
            memory: None,
            intelligence: None,
        };
        assert!(
            build_local_conversation_response(&setup, "unused")
                .text
                .contains("ABI support")
        );
        let mut dtmf = setup.clone();
        dtmf.kind = EventKind::Dtmf;
        dtmf.digit = "5".into();
        assert_eq!(
            build_local_conversation_response(&dtmf, "unused").text,
            "I received keypad input 5."
        );
    }

    #[test]
    fn user_transcript_without_escalation_returns_agent_reply() {
        let event = ConversationRelayEvent::user_transcript("what are your hours?");
        let response = build_local_conversation_response(&event, "We're open 9-5.");
        assert_eq!(response.text, "We're open 9-5.");
        assert!(response.escalation.is_none());
    }

    #[test]
    fn user_transcript_with_human_request_escalates() {
        let event = ConversationRelayEvent::user_transcript("let me talk to a human");
        let response = build_local_conversation_response(&event, "unused");
        assert!(response.text.contains("support specialist"));
        let payload = response.escalation.expect("escalation");
        assert_eq!(payload.reason_code, "human_requested");
        assert!(payload.routing_hints.contains("priority=normal"));
    }

    #[test]
    fn sensitive_topic_gets_high_priority() {
        let event = ConversationRelayEvent::user_transcript("my ssn is 123");
        let reason = classify_escalation(&event).expect("sensitive");
        assert_eq!(reason, EscalationReason::SensitiveTopic);
        let payload = build_escalation_payload(&event, reason);
        assert!(payload.routing_hints.contains("priority=high"));
    }

    #[test]
    fn memory_recall_is_appended() {
        let mut event = ConversationRelayEvent::user_transcript("hours?");
        event.memory = Some(ConversationMemory {
            recall_summary: "prefers mornings".into(),
            ..ConversationMemory::default()
        });
        let response = build_local_conversation_response(&event, "We're open 9-5.");
        assert!(response.text.contains("customer context: prefers mornings"));
    }
}
