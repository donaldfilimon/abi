//! Request-body construction and local-transport response synthesis.
//!
//! Ported from `src/connectors/json.zig`.
//!
//! The synthesized local responses are byte-exact reproductions. They are not
//! cosmetic: `abi complete` prints them, the connector tests assert on them, and
//! `connector_test` surfaces them through MCP. Every format string here matches
//! the Zig original literally.
//!
//! The Zig module carried its own copy of the JSON string escaper with a comment
//! explaining why — the connectors were also compiled as an isolated module root,
//! so importing `foundation/json.zig` would have violated that build boundary.
//! Cargo has no such constraint, so this uses
//! [`abi_foundation::json`] directly and the duplicate (and its
//! "keep synchronized" hazard) is gone.

use crate::connector::{ConnectorError, Result};
use abi_foundation::json::append_json_string;
use std::fmt::Write as _;

/// The JSON shape a payload is required to have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedJson {
    /// Must be a JSON array.
    Array,
    /// Must be a JSON object.
    Object,
    /// Any valid JSON value.
    Any,
}

/// Verify that `input` is valid JSON of the expected shape.
pub fn validate_json(input: &str, expected: ExpectedJson) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_str(input).map_err(|_| ConnectorError::InvalidResponse)?;
    let ok = match expected {
        ExpectedJson::Any => true,
        ExpectedJson::Array => value.is_array(),
        ExpectedJson::Object => value.is_object(),
    };
    if ok {
        Ok(())
    } else {
        Err(ConnectorError::InvalidResponse)
    }
}

/// Build an `OpenAI`-compatible chat-completion request body.
///
/// `messages` must already be a JSON array; it is validated and then spliced in
/// verbatim, matching Zig. Validating up front is what makes a malformed
/// `messages` fail *before* any request is built or synthesized.
pub fn build_openai_body(model: &str, messages: &str, stream: bool) -> Result<String> {
    validate_json(messages, ExpectedJson::Array)?;
    let mut out = String::from("{\"model\":");
    append_json_string(&mut out, model);
    out.push_str(",\"messages\":");
    out.push_str(messages);
    if stream {
        out.push_str(",\"stream\":true");
    }
    out.push('}');
    Ok(out)
}

/// Build an Anthropic-compatible message request body.
#[must_use]
pub fn build_anthropic_body(model: &str, prompt: &str, max_tokens: u32, stream: bool) -> String {
    let mut out = String::from("{\"model\":");
    append_json_string(&mut out, model);
    let _ = write!(out, ",\"max_tokens\":{max_tokens}");
    if stream {
        out.push_str(",\"stream\":true");
    }
    out.push_str(",\"messages\":[{\"role\":\"user\",\"content\":");
    append_json_string(&mut out, prompt);
    out.push_str("}]}");
    out
}

/// Build a Discord create-message request body.
#[must_use]
pub fn build_discord_message_body(content: &str) -> String {
    let mut out = String::from("{\"content\":");
    append_json_string(&mut out, content);
    out.push('}');
    out
}

/// Build a Discord gateway `IDENTIFY` (opcode 2) payload.
#[must_use]
pub fn build_discord_identify_body(token: &str, intents: u32) -> String {
    let mut out = String::from("{\"op\":2,\"d\":{\"token\":");
    append_json_string(&mut out, token);
    let _ = write!(
        out,
        ",\"intents\":{intents},\"properties\":{{\"os\":\"abi\",\"browser\":\"abi\",\"device\":\"abi\"}}}}}}"
    );
    out
}

/// Synthesize an `OpenAI`-shaped local chat response.
#[must_use]
pub fn openai_local_response(model: &str, messages: &str, body_len: usize) -> String {
    let text = format!(
        "OpenAI-compatible local response model={model} messages_bytes={} request_bytes={body_len}",
        messages.len()
    );
    let mut out = String::from("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":");
    append_json_string(&mut out, &text);
    out.push_str("}}]}");
    out
}

/// Synthesize an `OpenAI`-shaped local SSE stream.
#[must_use]
pub fn openai_local_stream(model: &str, messages: &str, body_len: usize) -> String {
    let content = format!(
        "OpenAI-compatible local stream model={model} messages_bytes={} request_bytes={body_len}",
        messages.len()
    );
    let mut out = String::from("data: ");
    append_openai_delta(&mut out, &content);
    out.push_str("\n\ndata: [DONE]\n\n");
    out
}

/// Append one `OpenAI` streaming delta frame.
pub fn append_openai_delta(out: &mut String, content: &str) {
    out.push_str("{\"choices\":[{\"delta\":{\"content\":");
    append_json_string(out, content);
    out.push_str("}}]}");
}

/// Synthesize an Anthropic-shaped local message response.
#[must_use]
pub fn anthropic_local_response(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    body_len: usize,
) -> String {
    let text = format!(
        "Anthropic-compatible local response model={model} prompt_bytes={} max_tokens={max_tokens} request_bytes={body_len}",
        prompt.len()
    );
    let mut out = String::from("{\"content\":[{\"type\":\"text\",\"text\":");
    append_json_string(&mut out, &text);
    out.push_str("}]}");
    out
}

/// Synthesize an Anthropic-shaped local SSE stream.
#[must_use]
pub fn anthropic_local_stream(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    body_len: usize,
) -> String {
    let content = format!(
        "Anthropic-compatible local stream model={model} prompt_bytes={} max_tokens={max_tokens} request_bytes={body_len}",
        prompt.len()
    );
    let mut out = String::from("event: content_block_delta\ndata: ");
    append_anthropic_delta(&mut out, &content);
    out.push_str("\n\nevent: message_stop\n\n");
    out
}

/// Append one Anthropic streaming delta frame.
pub fn append_anthropic_delta(out: &mut String, content: &str) {
    out.push_str("{\"delta\":{\"text\":");
    append_json_string(out, content);
    out.push_str("}}");
}

/// Synthesize a Discord local send acknowledgement.
#[must_use]
pub fn discord_local_ack(channel_id: &str, content: &str) -> String {
    let mut out = String::from("{\"status\":\"queued-local\",\"channel_id\":");
    append_json_string(&mut out, channel_id);
    let _ = write!(out, ",\"content_bytes\":{}}}", content.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_json_enforces_the_expected_shape() {
        assert!(validate_json("[]", ExpectedJson::Array).is_ok());
        assert!(validate_json("{}", ExpectedJson::Object).is_ok());
        assert!(validate_json("7", ExpectedJson::Any).is_ok());

        assert_eq!(
            validate_json("{}", ExpectedJson::Array).unwrap_err(),
            ConnectorError::InvalidResponse
        );
        assert_eq!(
            validate_json("[]", ExpectedJson::Object).unwrap_err(),
            ConnectorError::InvalidResponse
        );
        assert_eq!(
            validate_json("not json", ExpectedJson::Any).unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn openai_body_splices_the_messages_array_verbatim() {
        let body =
            build_openai_body("gpt-4", r#"[{"role":"user","content":"hi"}]"#, false).unwrap();
        assert_eq!(
            body,
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#
        );
    }

    #[test]
    fn openai_body_appends_the_stream_flag() {
        let body = build_openai_body("gpt-4", "[]", true).unwrap();
        assert_eq!(body, r#"{"model":"gpt-4","messages":[],"stream":true}"#);
    }

    #[test]
    fn openai_body_rejects_non_array_messages() {
        // Must fail before any request is built.
        assert_eq!(
            build_openai_body("gpt-4", r#"{"not":"array"}"#, false).unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn openai_body_escapes_the_model_name() {
        let body = build_openai_body("weird\"model", "[]", false).unwrap();
        assert!(body.starts_with(r#"{"model":"weird\"model""#));
        // Must remain parseable — an unescaped quote would corrupt the request.
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(parsed["model"], "weird\"model");
    }

    #[test]
    fn anthropic_body_has_the_documented_field_order() {
        let body = build_anthropic_body("claude-opus-5", "hello", 1024, false);
        assert_eq!(
            body,
            r#"{"model":"claude-opus-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}"#
        );
    }

    #[test]
    fn anthropic_body_puts_stream_before_messages() {
        let body = build_anthropic_body("m", "p", 8, true);
        assert_eq!(
            body,
            r#"{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"p"}]}"#
        );
    }

    #[test]
    fn discord_bodies_are_valid_json() {
        let message = build_discord_message_body("hello \"world\"");
        assert_eq!(message, r#"{"content":"hello \"world\""}"#);
        let parsed: serde_json::Value = serde_json::from_str(&message).expect("valid JSON");
        assert_eq!(parsed["content"], "hello \"world\"");
    }

    #[test]
    fn discord_identify_body_nests_correctly() {
        // Zig built this with a literal `}}}}` in a format string, which is easy
        // to get wrong by one brace — so check it parses and nests.
        let body = build_discord_identify_body("tok", 513);
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(parsed["op"], 2);
        assert_eq!(parsed["d"]["token"], "tok");
        assert_eq!(parsed["d"]["intents"], 513);
        assert_eq!(parsed["d"]["properties"]["os"], "abi");
        assert_eq!(parsed["d"]["properties"]["browser"], "abi");
        assert_eq!(parsed["d"]["properties"]["device"], "abi");
    }

    #[test]
    fn openai_local_response_is_byte_exact() {
        let body = openai_local_response("gpt-4", "[]", 42);
        assert_eq!(
            body,
            r#"{"choices":[{"message":{"role":"assistant","content":"OpenAI-compatible local response model=gpt-4 messages_bytes=2 request_bytes=42"}}]}"#
        );
        // Zig's own test asserted this substring.
        assert!(body.contains("model=gpt-4"));
    }

    #[test]
    fn openai_local_stream_has_sse_framing() {
        let body = openai_local_stream("gpt-4", "[]", 30);
        assert!(body.starts_with("data: "));
        assert!(body.ends_with("data: [DONE]\n\n"));
        assert_eq!(
            body,
            "data: {\"choices\":[{\"delta\":{\"content\":\"OpenAI-compatible local stream model=gpt-4 messages_bytes=2 request_bytes=30\"}}]}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn anthropic_local_response_is_byte_exact() {
        let body = anthropic_local_response("claude-opus-5", "hi", 512, 77);
        assert_eq!(
            body,
            r#"{"content":[{"type":"text","text":"Anthropic-compatible local response model=claude-opus-5 prompt_bytes=2 max_tokens=512 request_bytes=77"}]}"#
        );
    }

    #[test]
    fn anthropic_local_stream_uses_named_events() {
        let body = anthropic_local_stream("m", "p", 16, 20);
        assert!(body.starts_with("event: content_block_delta\ndata: "));
        assert!(body.ends_with("event: message_stop\n\n"));
    }

    #[test]
    fn discord_local_ack_reports_content_length_not_content() {
        let ack = discord_local_ack("123456", "hello");
        assert_eq!(
            ack,
            r#"{"status":"queued-local","channel_id":"123456","content_bytes":5}"#
        );
    }

    #[test]
    fn every_synthesized_body_is_parseable_json() {
        // The local transport feeds `abi complete`, which parses these.
        for body in [
            openai_local_response("m", "[]", 1),
            anthropic_local_response("m", "p", 1, 1),
            discord_local_ack("c", "x"),
            build_discord_message_body("x"),
            build_discord_identify_body("t", 1),
            build_anthropic_body("m", "p", 1, false),
        ] {
            serde_json::from_str::<serde_json::Value>(&body)
                .unwrap_or_else(|e| panic!("not valid JSON: {body} ({e})"));
        }
    }

    #[test]
    fn control_characters_in_a_prompt_do_not_break_the_body() {
        let body = build_anthropic_body("m", "line1\nline2\ttab\u{1}ctl", 8, false);
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(
            parsed["messages"][0]["content"],
            "line1\nline2\ttab\u{1}ctl"
        );
    }
}
