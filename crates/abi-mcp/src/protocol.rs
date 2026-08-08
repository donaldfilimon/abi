//! JSON-RPC 2.0 envelope types and the structural pre-check shared by every
//! transport.
//!
//! Ported from `src/mcp/protocol.zig`.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Same bound the HTTP transport enforces on a request body
/// (`abi_foundation::http::MAX_REQUEST_SIZE`) — both transports share one cap.
pub const MAX_REQUEST_SIZE: usize = abi_foundation::http::MAX_REQUEST_SIZE;

/// Maximum nesting depth of `{`/`[` containers accepted by JSON-RPC parse
/// paths. Bounds CPU/stack pressure from adversarial deeply-nested payloads
/// (TM-008).
pub const MAX_JSON_DEPTH: usize = 32;

/// A parsed JSON-RPC request line.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Must be `"2.0"`.
    pub jsonrpc: String,
    /// The MCP method name, e.g. `"tools/call"`.
    pub method: String,
    /// The caller's correlation id. Absent for notifications.
    ///
    /// The custom deserializer is load-bearing: Serde's ordinary
    /// `Option<Value>` handling maps both an absent field and an explicit JSON
    /// `null` to `None`. JSON-RPC assigns different meanings to those cases, so
    /// an explicit `null` must survive as `Some(Value::Null)`.
    #[serde(default, deserialize_with = "deserialize_present_id")]
    pub id: Option<Value>,
    /// Method-specific parameters.
    #[serde(default)]
    pub params: Option<Value>,
}

fn deserialize_present_id<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Value::deserialize(deserializer).map(Some)
}

/// The `error` member of a JSON-RPC response.
#[derive(Debug, Serialize)]
pub struct JsonRpcErrorObj {
    /// The JSON-RPC error code.
    pub code: i32,
    /// A stable, non-leaking client-facing message.
    pub message: String,
}

/// The MCP methods this server recognizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpMethod {
    /// The handshake that opens a session.
    Initialize,
    /// List the tools this server offers.
    ToolsList,
    /// Invoke one tool.
    ToolsCall,
    /// List resources. Always empty — this server offers none.
    ResourcesList,
    /// List prompts. Always empty — this server offers none.
    PromptsList,
    /// A liveness check.
    Ping,
    /// Request server shutdown.
    Shutdown,
    /// Client lifecycle notification sent after initialization completes.
    NotificationsInitialized,
    /// Any method name not recognized above.
    Unknown,
}

impl McpMethod {
    /// Classify a raw method name.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "initialize" => Self::Initialize,
            "tools/list" => Self::ToolsList,
            "tools/call" => Self::ToolsCall,
            "resources/list" => Self::ResourcesList,
            "prompts/list" => Self::PromptsList,
            "ping" => Self::Ping,
            "shutdown" => Self::Shutdown,
            "notifications/initialized" => Self::NotificationsInitialized,
            _ => Self::Unknown,
        }
    }
}

/// The structural failure modes [`validate_request`] and [`check_json_depth`]
/// can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestError {
    /// The request line was empty.
    Empty,
    /// The request line exceeded [`MAX_REQUEST_SIZE`].
    TooLarge,
    /// The request did not start with `{`, or an object/array was unbalanced.
    InvalidFormat,
    /// Object/array nesting exceeded [`MAX_JSON_DEPTH`].
    TooDeep,
}

/// Structural pre-check for JSON-RPC request lines (stdio + HTTP body).
///
/// Rejects empty/oversize input, non-object roots, and over-nested containers
/// before the JSON parse, so both transports share one bound.
pub fn validate_request(line: &str) -> Result<(), RequestError> {
    if line.is_empty() {
        return Err(RequestError::Empty);
    }
    if line.len() > MAX_REQUEST_SIZE {
        return Err(RequestError::TooLarge);
    }
    let trimmed = line.trim_start_matches([' ', '\t']);
    if !trimmed.starts_with('{') {
        return Err(RequestError::InvalidFormat);
    }
    check_json_depth(line, MAX_JSON_DEPTH)
}

/// Walk JSON text and reject when object/array nesting exceeds `max_depth`.
///
/// String contents are skipped so braces inside quoted values do not count.
pub fn check_json_depth(src: &str, max_depth: usize) -> Result<(), RequestError> {
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escape = false;
    for byte in src.bytes() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if byte == b'\\' {
                escape = true;
                continue;
            }
            if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max_depth {
                    return Err(RequestError::TooDeep);
                }
            }
            b'}' | b']' => {
                if depth == 0 {
                    return Err(RequestError::InvalidFormat);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_method_maps_known_and_unknown() {
        assert_eq!(McpMethod::parse("initialize"), McpMethod::Initialize);
        assert_eq!(McpMethod::parse("tools/call"), McpMethod::ToolsCall);
        assert_eq!(McpMethod::parse("tools/list"), McpMethod::ToolsList);
        assert_eq!(McpMethod::parse("ping"), McpMethod::Ping);
        assert_eq!(
            McpMethod::parse("notifications/initialized"),
            McpMethod::NotificationsInitialized
        );
        assert_eq!(McpMethod::parse("tools/run"), McpMethod::Unknown);
        assert_eq!(McpMethod::parse(""), McpMethod::Unknown);
    }

    #[test]
    fn request_id_preserves_absent_versus_explicit_null() {
        let notification: JsonRpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","method":"ping"}"#)
                .expect("notification parses");
        assert!(notification.id.is_none());

        let explicit_null: JsonRpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#)
                .expect("null id parses");
        assert_eq!(explicit_null.id, Some(Value::Null));
    }

    #[test]
    fn validate_request_accepts_an_object_line_and_rejects_malformed_input() {
        assert!(validate_request(r#"{"jsonrpc":"2.0"}"#).is_ok());
        assert!(validate_request("  \t{\"a\":1}").is_ok());

        assert_eq!(validate_request(""), Err(RequestError::Empty));
        assert_eq!(
            validate_request("not json"),
            Err(RequestError::InvalidFormat)
        );
        assert_eq!(validate_request("   "), Err(RequestError::InvalidFormat));

        let oversized = "{".repeat(MAX_REQUEST_SIZE + 1);
        assert_eq!(validate_request(&oversized), Err(RequestError::TooLarge));
    }

    #[test]
    fn validate_request_rejects_over_nested_json_containers() {
        let mut deep = String::new();
        for _ in 0..=MAX_JSON_DEPTH {
            deep.push_str("{\"a\":");
        }
        deep.push('1');
        for _ in 0..=MAX_JSON_DEPTH {
            deep.push('}');
        }
        assert_eq!(validate_request(&deep), Err(RequestError::TooDeep));

        // Braces inside strings must not count toward depth.
        assert!(validate_request(r#"{"a":"{{{{{"}"#).is_ok());
    }

    #[test]
    fn check_json_depth_allows_shallow_nests_and_rejects_deeper_ones() {
        assert!(check_json_depth(r#"{"a":[1,2,{"b":3}]}"#, 4).is_ok());
        assert_eq!(
            check_json_depth("[[[[[]]]]]", 3),
            Err(RequestError::TooDeep)
        );
    }
}
