//! MCP `connector_test` tool: exercise a connector through its deterministic
//! local path.
//!
//! Ported from `src/mcp/connector_tools.zig`, reusing the already-ported
//! `abi-connectors` clients (plan step 3a). `twilio` is not yet portable: its
//! local response goes through `twilio_relay.zig`'s conversation builder,
//! which is plan step 3b and still open — see `RUST-REWRITE-PLAN.md`.

use abi_connectors::{Client, ConnectorConfig, payload};

/// Why [`run_connector_test`] could not produce a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorToolError {
    /// `service` is a valid, known service, but its local dispatch has no
    /// Rust port yet.
    NotYetPorted,
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
        "twilio" => Err(ConnectorToolError::NotYetPorted),
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
    fn twilio_is_honestly_not_yet_ported() {
        assert_eq!(
            run_connector_test("twilio", "hi"),
            Err(ConnectorToolError::NotYetPorted)
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
