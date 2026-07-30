//! `abi twilio simulate` — local ConversationRelay-shaped simulation.
//!
//! Ported from `src/cli/handlers/twilio.zig`'s offline path. Does not contact
//! the live Twilio API. Full relay event parsing remains in step 3b; this
//! handler exercises persona routing + the observed CLI report shape.

use abi_ai::run_text;

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
    let reply = run_text(input);
    // Local non-live path: user transcript → persona reply, no escalation.
    let out =
        format!("Twilio ConversationRelay simulation\nresponse: {reply}\nescalation: false\n");
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
    fn bad_grammar_is_usage() {
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&["notsimulate".into(), "hi".into()]).exit_code, 2);
    }
}
