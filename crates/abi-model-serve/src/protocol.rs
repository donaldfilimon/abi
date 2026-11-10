//! Pure `OpenAI` chat-completions request parsing and response rendering.

use abi_agent_runtime::{Message, ModelRequest, Role, StopReason, ToolCall, ToolSpec, Usage};
use serde_json::{Value, json};
use std::collections::HashMap;

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
/// instead of being dropped. `tools_supported` says whether the loaded model
/// takes tool definitions.
pub fn parse_chat(body: &[u8], tools_supported: bool) -> Result<ChatRequest, ProtocolError> {
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
    let tools = parse_tools(object.get("tools"), object.get("tool_choice"))?;
    if !tools.is_empty() && !tools_supported {
        return Err(ProtocolError::invalid(
            "tools_unsupported",
            "the loaded model does not support tool definitions",
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

    let mut request = conversation(model, messages)?;
    for tool in tools {
        request = request.with_tool(tool);
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

/// Build the provider request's system text and messages.
fn conversation(model: &str, messages: &[Value]) -> Result<ModelRequest, ProtocolError> {
    let mut request = ModelRequest::new(model);
    // Calls the assistant made earlier, by id, so a `tool` message can be
    // rendered with the name and arguments of the call it answers.
    let mut calls: HashMap<String, (String, String)> = HashMap::new();
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
        let made_calls = role == Role::Assistant && record_calls(index, message, &mut calls)?;
        let content = match message.get("content") {
            None | Some(Value::Null) if made_calls => String::new(),
            content => message_text(index, content)?,
        };
        request = match role {
            Role::System if index == 0 => request.with_system(content),
            // The calls travel with their results, so an assistant message
            // that only carried calls adds nothing by itself.
            Role::Assistant if made_calls && content.is_empty() => request,
            Role::Tool => {
                let id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let (name, arguments) = calls.get(id).ok_or_else(|| {
                    ProtocolError::invalid(
                        "unknown_tool_call",
                        format!("messages[{index}].tool_call_id matches no earlier tool call"),
                    )
                })?;
                request.with_message(Message::new(
                    Role::Tool,
                    json!({
                        "call_id": id,
                        "tool": name,
                        "status": "ok",
                        "input": arguments,
                        "payload": content
                    })
                    .to_string(),
                ))
            }
            _ => request.with_message(Message::new(role, content)),
        };
    }
    Ok(request)
}

/// Tool definitions, unless `tool_choice` turns them off.
fn parse_tools(
    tools: Option<&Value>,
    choice: Option<&Value>,
) -> Result<Vec<ToolSpec>, ProtocolError> {
    let entries = match tools {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            return Err(ProtocolError::invalid(
                "invalid_tools",
                "'tools' must be an array",
            ));
        }
    };
    match choice.map(|choice| choice.as_str()) {
        None | Some(Some("auto")) => {}
        Some(Some("none")) => return Ok(Vec::new()),
        Some(_) => {
            return Err(ProtocolError::invalid(
                "tool_choice_unsupported",
                "'tool_choice' must be 'auto' or 'none'",
            ));
        }
    }
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let function = entry
                .get("function")
                .filter(|_| entry.get("type").and_then(Value::as_str) == Some("function"));
            let name = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    ProtocolError::invalid(
                        "invalid_tools",
                        format!("tools[{index}] must be a named function"),
                    )
                })?;
            let mut spec = ToolSpec::new(name);
            if let Some(description) = function
                .and_then(|function| function.get("description"))
                .and_then(Value::as_str)
            {
                spec = spec.with_description(description);
            }
            if let Some(parameters) = function
                .and_then(|function| function.get("parameters"))
                .filter(|parameters| parameters.is_object())
            {
                spec = spec.with_input_schema(parameters.to_string());
            }
            Ok(spec)
        })
        .collect()
}

/// Record an assistant message's tool calls. Returns whether it made any.
fn record_calls(
    index: usize,
    message: &Value,
    calls: &mut HashMap<String, (String, String)>,
) -> Result<bool, ProtocolError> {
    let entries = match message.get("tool_calls") {
        None | Some(Value::Null) => return Ok(false),
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            return Err(ProtocolError::invalid(
                "invalid_tool_calls",
                format!("messages[{index}].tool_calls must be an array"),
            ));
        }
    };
    for entry in entries {
        let id = entry.get("id").and_then(Value::as_str);
        let function = entry.get("function");
        let name = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);
        let arguments = function
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let (Some(id), Some(name)) = (id, name) else {
            return Err(ProtocolError::invalid(
                "invalid_tool_calls",
                format!("messages[{index}].tool_calls entries need an id and a function name"),
            ));
        };
        calls.insert(id.to_owned(), (name.to_owned(), arguments.to_owned()));
    }
    Ok(!entries.is_empty())
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
pub fn finish_reason(stop: StopReason, made_tool_calls: bool) -> &'static str {
    match stop {
        StopReason::BudgetExhausted(_) => "length",
        _ if made_tool_calls => "tool_calls",
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

/// One `tool_calls` entry. `index` is present only in streamed deltas.
pub(crate) fn tool_call_value(call: &ToolCall, index: Option<usize>) -> Value {
    let mut entry = json!({
        "id": call.id,
        "type": "function",
        "function": { "name": call.name, "arguments": call.input }
    });
    if let Some(index) = index {
        entry["index"] = json!(index);
    }
    entry
}

/// The buffered `chat.completion` document.
pub(crate) fn completion_body(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    calls: &[&ToolCall],
    stop: StopReason,
    usage: &Usage,
) -> String {
    let mut message = json!({ "role": "assistant", "content": text });
    if !calls.is_empty() {
        if text.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = calls
            .iter()
            .map(|call| tool_call_value(call, None))
            .collect();
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason(stop, !calls.is_empty())
        }],
        "usage": usage_value(usage)
    })
    .to_string()
}

/// One `chat.completion.chunk` document. `finish` carries the finish reason
/// and usage on the last chunk.
pub(crate) fn chunk_body(
    id: &str,
    created: u64,
    model: &str,
    delta: &Value,
    finish: Option<(&str, &Usage)>,
) -> String {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish.map(|(reason, _)| reason)
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
        parse_chat(body.as_bytes(), true)
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
            false,
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
            false,
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
            "invalid_tools"
        );
        assert_eq!(
            code(&format!(r#"{{"model":"m","tools":7,{user}}}"#)),
            "invalid_tools"
        );
        assert_eq!(
            code(&format!(
                r#"{{"model":"m","tools":[{{"type":"function","function":{{"name":"t"}}}}],"tool_choice":"required",{user}}}"#
            )),
            "tool_choice_unsupported"
        );
        assert_eq!(
            code(r#"{"model":"m","messages":[{"role":"tool","tool_call_id":"c9","content":"x"}]}"#),
            "unknown_tool_call"
        );
        assert_eq!(
            code(r#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c1"}]}]}"#),
            "invalid_tool_calls"
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

    const TOOLED: &str = r#"{"model":"m","tools":[{"type":"function","function":{
        "name":"recall","description":"Recall facts.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}],
        "messages":[
            {"role":"user","content":"what do you know"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"recall","arguments":"{\"query\":\"tea\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"likes tea"}]}"#;

    #[test]
    fn tools_and_a_tool_round_trip_become_specs_and_host_envelopes() {
        let parsed = parse_chat(TOOLED.as_bytes(), true).expect("valid tooled request");
        let tools = parsed.request.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "recall");
        assert_eq!(tools[0].description, "Recall facts.");
        assert!(tools[0].input_schema.contains(r#""query""#));

        // The call-only assistant message is folded into the tool envelope.
        let messages = parsed.request.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, Role::Tool);
        let envelope: serde_json::Value =
            serde_json::from_str(&messages[1].content).expect("envelope JSON");
        assert_eq!(envelope["call_id"], "c1");
        assert_eq!(envelope["tool"], "recall");
        assert_eq!(envelope["status"], "ok");
        assert_eq!(envelope["input"], r#"{"query":"tea"}"#);
        assert_eq!(envelope["payload"], "likes tea");
    }

    #[test]
    fn tools_are_refused_only_when_the_model_cannot_take_them() {
        assert_eq!(
            parse_chat(TOOLED.as_bytes(), false)
                .expect_err("unsupported")
                .code,
            "tools_unsupported"
        );
        let silenced = TOOLED.replacen(r#""messages""#, r#""tool_choice":"none","messages""#, 1);
        let parsed = parse_chat(silenced.as_bytes(), false).expect("tool_choice none drops tools");
        assert_eq!(parsed.request.tools(), []);
    }

    #[test]
    fn only_budget_exhaustion_reports_length() {
        assert_eq!(finish_reason(StopReason::Completed, false), "stop");
        assert_eq!(finish_reason(StopReason::Completed, true), "tool_calls");
        assert_eq!(
            finish_reason(StopReason::BudgetExhausted(BudgetLimit::OutputTokens), true),
            "length"
        );
    }
}
