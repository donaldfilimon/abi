//! MCP `connector_test` tool: exercise a connector through its deterministic
//! local path.
//!
//! Ported from `src/mcp/connector_tools.zig`, reusing the already-ported
//! `abi-connectors` clients (including Twilio `ConversationRelay` local synthesis).

use abi_connectors::{
    Client, ConnectorConfig, ConversationRelayEvent, build_local_conversation_response, payload,
};

/// Why [`run_connector_test`] could not produce a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorToolError {
    /// The connector's local synthesis failed unexpectedly.
    Failed,
}

/// Run the deterministic local path for one connector and return a one-line
/// status string, matching `connector_tools.runConnectorTest` in Zig.
///
/// `service` is assumed already validated against the tool's enum (see
/// `handlers::CONNECTOR_TEST_FIELDS`) — this only distinguishes ported vs.
/// unported services among the five valid ones.
pub fn run_connector_test(service: &str, input: &str) -> Result<String, ConnectorToolError> {
    match service {
        "openai" => {
            let messages = build_user_messages(input);
            let client = Client::openai(ConnectorConfig::new(
                "mcp-local-key",
                "https://api.openai.com",
            ));
            let response = client
                .complete("gpt-local", &messages)
                .map_err(|_| ConnectorToolError::Failed)?;
            Ok(format!(
                "connector=openai status={} body={}",
                response.status, response.body
            ))
        }
        "anthropic" => {
            // The MCP tool asks for max_tokens=256, not abi-connectors'
            // general-purpose default of 4096 — call the payload builders
            // directly rather than through `Client::complete`.
            const MAX_TOKENS: u32 = 256;
            let body = payload::build_anthropic_body("claude-local", input, MAX_TOKENS, false);
            let text =
                payload::anthropic_local_response("claude-local", input, MAX_TOKENS, body.len());
            Ok(format!("connector=anthropic status=200 body={text}"))
        }
        "discord" => {
            let body = payload::discord_local_ack("234567890123456789", input);
            Ok(format!("connector=discord status=200 body={body}"))
        }
        "grok" => {
            let messages = build_user_messages(input);
            let client = Client::grok(ConnectorConfig::new("mcp-local-key", "https://api.x.ai"));
            let response = client
                .complete("grok-local", &messages)
                .map_err(|_| ConnectorToolError::Failed)?;
            Ok(format!(
                "connector=grok status={} body={}",
                response.status, response.body
            ))
        }
        "twilio" => {
            // MCP fixture: fixed conversation/customer ids + local agent reply,
            // matching Zig `handleConversationRelayEvent(.user_transcript, …)`.
            let event = ConversationRelayEvent::user_transcript(input);
            let response =
                build_local_conversation_response(&event, "ABI local relay acknowledged.");
            Ok(format!(
                "connector=twilio status=200 body={}",
                response.text
            ))
        }
        _ => Err(ConnectorToolError::Failed),
    }
}

fn build_user_messages(input: &str) -> String {
    let mut out = String::from(r#"[{"role":"user","content":"#);
    out.push_str(&abi_foundation::json::escape_json_string(input));
    out.push_str("}]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_matches_the_golden_local_synthesis() {
        let text = run_connector_test("openai", "ping").expect("openai is ported");
        assert_eq!(
            text,
            "connector=openai status=200 body={\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"OpenAI-compatible local response model=gpt-local messages_bytes=34 request_bytes=67\"}}]}"
        );
    }

    #[test]
    fn twilio_local_relay_synthesizes_body() {
        // "hi" is < 3 chars and escalates as low_confidence (Zig parity).
        let text = run_connector_test("twilio", "what are your hours?").expect("twilio is ported");
        assert_eq!(
            text,
            "connector=twilio status=200 body=ABI local relay acknowledged."
        );
        let escalate =
            run_connector_test("twilio", "let me talk to a human").expect("twilio escalate");
        assert!(
            escalate.contains("support specialist"),
            "expected escalation reply: {escalate}"
        );
        let short = run_connector_test("twilio", "hi").expect("short escalates");
        assert!(
            short.contains("support specialist"),
            "short transcript should escalate: {short}"
        );
    }

    #[test]
    fn anthropic_and_discord_and_grok_synthesize_locally() {
        assert!(
            run_connector_test("anthropic", "hi")
                .expect("anthropic is ported")
                .starts_with("connector=anthropic status=200 body=")
        );
        assert!(
            run_connector_test("discord", "hi")
                .expect("discord is ported")
                .starts_with("connector=discord status=200 body=")
        );
        assert!(
            run_connector_test("grok", "hi")
                .expect("grok is ported")
                .starts_with("connector=grok status=200 body=")
        );
    }
}
