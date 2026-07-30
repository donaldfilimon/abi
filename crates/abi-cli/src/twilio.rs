//! `abi twilio simulate` — local `ConversationRelay`-shaped simulation.
//!
//! Ported from `src/cli/handlers/twilio.zig`'s offline path. Does not contact
//! the live Twilio API. Uses the shared `abi-connectors` `ConversationRelay`
//! builder for escalation classification and reply text.

use std::fmt::Write as _;

use abi_ai::run_text;
use abi_connectors::{ConversationRelayEvent, build_local_conversation_response};

use crate::app::Outcome;

const USAGE: &str = "usage: abi twilio simulate <input>";

/// Dispatch `abi twilio …` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        return Outcome::stderr(
            include_str!("../../../tests/golden/help-twilio.txt").to_owned(),
            0,
        );
    }
    if args.len() == 2
        && args[0] == "simulate"
        && matches!(args[1].as_str(), "--help" | "-h" | "help")
    {
        return Outcome::stderr(
            "usage: abi twilio simulate <input>\n\nRun a local ConversationRelay simulation without contacting Twilio.\n".into(),
            0,
        );
    }
    if args.len() != 2 || args[0] != "simulate" {
        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
    }
    let input = &args[1];
    let agent_reply = run_text(input);
    let event = ConversationRelayEvent::user_transcript(input);
    let response = build_local_conversation_response(&event, &agent_reply);
    let escalated = response.escalation.is_some();
    let mut out = format!(
        "Twilio ConversationRelay simulation\nresponse: {}\nescalation: {escalated}\n",
        response.text
    );
    if let Some(payload) = &response.escalation {
        let _ = write!(
            out,
            "escalation_reason: {}\nrouting: {}\n",
            payload.reason_code, payload.routing_hints
        );
    }
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulate_prints_persona_reply() {
        let outcome = run(&["simulate".into(), "hello".into()]);
        assert_eq!(outcome.exit_code, 0);
        assert!(
            outcome
                .stdout
                .contains("Twilio ConversationRelay simulation")
        );
        assert!(outcome.stdout.contains("response: Abbey: hello"));
        assert!(outcome.stdout.contains("escalation: false"));
    }

    #[test]
    fn simulate_escalates_human_request() {
        let outcome = run(&["simulate".into(), "let me talk to a human".into()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("escalation: true"));
        assert!(
            outcome
                .stdout
                .contains("escalation_reason: human_requested")
        );
    }

    #[test]
    fn bad_grammar_is_usage() {
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&["notsimulate".into(), "hi".into()]).exit_code, 2);
    }
}
