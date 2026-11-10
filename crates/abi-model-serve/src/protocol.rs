//! Pure `OpenAI` chat-completions request parsing and response rendering.

use abi_agent_runtime::{Message, ModelRequest, Role, StopReason, Usage};
use serde_json::{Value, json};

/// One accepted chat-completions request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatRequest {
    /// Provider request carrying the exact model the caller selected.
    pub request: ModelRequest,
    /// Whether the caller asked for server-sent events.
    pub stream: bool,
}

/// A refusal rendered as an `OpenAI`-shaped error object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolError {
    /// HTTP status code.
    pub status: u16,
    /// HTTP reason phrase.
    pub reason: &'static str,
    /// `OpenAI` error `type`.
    pub kind: &'static str,
    /// Stable machine-readable `code`.
    pub code: &'static str,
    /// Human-readable detail.
    pub message: String,
}

impl ProtocolError {
    /// A `400` for a request this server will not interpret.
    #[must_use]
    pub fn invalid(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: 400,
            reason: "Bad Request",
            kind: "invalid_request_error",
            code,
            message: message.into(),
        }
    }

    /// A non-`400` refusal.
    #[must_use]
    pub fn new(
        status: u16,
        reason: &'static str,
        kind: &'static str,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            reason,
            kind,
            code,
            message: message.into(),
        }
    }

    /// The JSON error document.
    #[must_use]
    pub fn body(&self) -> String {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": null,
                "code": self.code
            }
        })
        .to_string()
    }
}

/// Parse a chat-completions body. Anything the server cannot honor is refused
/// instead of being dropped.
pub fn parse_chat(body: &[u8]) -> Result<ChatRequest, ProtocolError> {
    let document: Value = serde_json::from_slice(body)
        .map_err(|error| ProtocolError::invalid("invalid_json", error.to_string()))?;
    let Some(object) = document.as_object() else {
        return Err(ProtocolError::invalid(
            "invalid_json",
            "request body must be a JSON object",
        ));
    };
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ProtocolError::invalid("model_required", "'model' must be a string"))?;
    if object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ProtocolError::invalid(
            "tools_unsupported",
            "this server does not support tool definitions",
        ));
    }
    if object
        .get("n")
        .is_some_and(|choices| choices.as_u64() != Some(1))
    {
        return Err(ProtocolError::invalid(
            "n_unsupported",
            "this server returns exactly one choice",
        ));
    }
    let stream = match object.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => {
            return Err(ProtocolError::invalid(
                "invalid_stream",
                "'stream' must be a boolean",
            ));
        }
    };
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or_else(|| {
            ProtocolError::invalid("messages_required", "'messages' must be a non-empty array")
        })?;

    let mut request = ModelRequest::new(model);
    for (index, message) in messages.iter().enumerate() {
        let role = match message.get("role").and_then(Value::as_str) {
            Some("system" | "developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            _ => {
                return Err(ProtocolError::invalid(
                    "invalid_role",
                    format!("messages[{index}].role is not a supported role"),
                ));
            }
        };
        let content = message_text(index, message.get("content"))?;
        request = if index == 0 && role == Role::System {
            request.with_system(content)
        } else {
            request.with_message(Message::new(role, content))
        };
    }
    for key in ["max_completion_tokens", "max_tokens"] {
        match object.get(key) {
            None | Some(Value::Null) => {}
            Some(value) => {
                let tokens = value.as_u64().filter(|tokens| *tokens > 0).ok_or_else(|| {
                    ProtocolError::invalid(
                        "invalid_max_tokens",
                        format!("'{key}' must be a positive integer"),
                    )
                })?;
                request = request.with_max_output_tokens(tokens);
                break;
            }
        }
    }
    Ok(ChatRequest { request, stream })
}

fn message_text(index: usize, content: Option<&Value>) -> Result<String, ProtocolError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                let piece = part
                    .get("type")
                    .and_then(Value::as_str)
                    .filter(|kind| *kind == "text")
                    .and_then(|_| part.get("text"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ProtocolError::invalid(
                            "content_unsupported",
                            format!("messages[{index}].content has a non-text part"),
                        )
                    })?;
                text.push_str(piece);
            }
            Ok(text)
        }
        _ => Err(ProtocolError::invalid(
            "content_required",
            format!("messages[{index}].content must be a string or text parts"),
        )),
    }
}

/// `OpenAI` `finish_reason` for a run that produced a reportable turn.
#[must_use]
pub fn finish_reason(stop: StopReason) -> &'static str {
    match stop {
        StopReason::BudgetExhausted(_) => "length",
        StopReason::Completed | StopReason::Cancelled | StopReason::Failed => "stop",
    }
}

fn usage_value(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens()
    })
}

/// The buffered `chat.completion` document.
pub(crate) fn completion_body(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    stop: StopReason,
    usage: &Usage,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish_reason(stop)
        }],
        "usage": usage_value(usage)
    })
    .to_string()
}

/// One `chat.completion.chunk` document.
pub(crate) fn chunk_body(
    id: &str,
    created: u64,
    model: &str,
    delta: &Value,
    finish: Option<(StopReason, &Usage)>,
) -> String {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish.map(|(stop, _)| finish_reason(stop))
        }]
    });
    if let Some((_, usage)) = finish {
        chunk["usage"] = usage_value(usage);
    }
    chunk.to_string()
}

/// The `GET /v1/models` listing for the one loaded model.
pub(crate) fn models_body(model: &str, created: u64) -> String {
    json!({
        "object": "list",
        "data": [{ "id": model, "object": "model", "created": created, "owned_by": "abi" }]
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{finish_reason, parse_chat};
    use abi_agent_runtime::{BudgetLimit, Role, StopReason};

    fn code(body: &str) -> &'static str {
        parse_chat(body.as_bytes())
            .expect_err("request is refused")
            .code
    }

    #[test]
    fn a_full_request_maps_roles_text_parts_and_limits() {
        let parsed = parse_chat(
            br#"{"model":"m","stream":true,"max_tokens":7,"messages":[
                {"role":"system","content":"be terse"},
                {"role":"user","content":[{"type":"text","text":"hel"},{"type":"text","text":"lo"}]},
                {"role":"assistant","content":"hi"},
                {"role":"developer","content":"later rule"},
                {"role":"user","content":"again"}]}"#,
        )
        .expect("valid request");
        assert!(parsed.stream);
        assert_eq!(parsed.request.model(), "m");
        assert_eq!(parsed.request.system(), Some("be terse"));
        assert_eq!(parsed.request.max_output_tokens(), Some(7));
        let messages = parsed.request.messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[2].role, Role::System);
        assert_eq!(messages[3].content, "again");
    }

    #[test]
    fn max_completion_tokens_wins_and_stream_defaults_off() {
        let parsed = parse_chat(
            br#"{"model":"m","max_tokens":9,"max_completion_tokens":3,"n":1,"tools":[],
                "messages":[{"role":"user","content":"x"}]}"#,
        )
        .expect("valid request");
        assert!(!parsed.stream);
        assert_eq!(parsed.request.max_output_tokens(), Some(3));
    }

    #[test]
    fn unsupported_and_malformed_requests_are_refused_by_code() {
        let user = r#""messages":[{"role":"user","content":"x"}]"#;
        assert_eq!(code("["), "invalid_json");
        assert_eq!(code("[]"), "invalid_json");
        assert_eq!(code(&format!("{{{user}}}")), "model_required");
        assert_eq!(code(r#"{"model":"m"}"#), "messages_required");
        assert_eq!(code(r#"{"model":"m","messages":[]}"#), "messages_required");
        assert_eq!(
            code(&format!(
                r#"{{"model":"m","tools":[{{"type":"function"}}],{user}}}"#
            )),
            "tools_unsupported"
        );
        assert_eq!(
            code(&format!(r#"{{"model":"m","n":2,{user}}}"#)),
            "n_unsupported"
        );
        assert_eq!(
            code(&format!(r#"{{"model":"m","stream":"yes",{user}}}"#)),
            "invalid_stream"
        );
        assert_eq!(
            code(&format!(r#"{{"model":"m","max_tokens":0,{user}}}"#)),
            "invalid_max_tokens"
        );
        assert_eq!(
            code(r#"{"model":"m","messages":[{"role":"robot","content":"x"}]}"#),
            "invalid_role"
        );
        assert_eq!(
            code(r#"{"model":"m","messages":[{"role":"user"}]}"#),
            "content_required"
        );
        assert_eq!(
            code(
                r#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#
            ),
            "content_unsupported"
        );
    }

    #[test]
    fn only_budget_exhaustion_reports_length() {
        assert_eq!(finish_reason(StopReason::Completed), "stop");
        assert_eq!(
            finish_reason(StopReason::BudgetExhausted(BudgetLimit::OutputTokens)),
            "length"
        );
    }
}
