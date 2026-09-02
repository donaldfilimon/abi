//! Top-level JSON-RPC dispatch, shared by every transport.
//!
//! Ported from `src/mcp/rpc.zig` (the transport-agnostic core) and
//! `src/mcp/stdio_transport.zig`'s per-request handling (the error-response
//! shaping, since the stdio transport is this port's only transport so far).

use serde_json::{Value, json};

use crate::handlers::{self, ToolError};
use crate::protocol::{McpMethod, RequestError, validate_request};
use crate::state::McpState;

/// A ready-to-serialize JSON-RPC response line.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcResponse(pub Value);

impl RpcResponse {
    /// Serialize to the exact bytes written as one stdio response line.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        self.0.to_string()
    }
}

fn ok(id: &Value, result: &Value) -> RpcResponse {
    RpcResponse(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

fn err(id: &Value, code: i32, message: &str) -> RpcResponse {
    RpcResponse(json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}))
}

/// Parse-failure shaping shared by every caller: id is always `null` because
/// the id could not be recovered from unparseable input.
fn parse_error(message: &str) -> RpcResponse {
    err(&Value::Null, -32700, message)
}

/// The standard parse-error response used when a transport cannot decode a
/// request frame as JSON text.
pub(crate) fn parse_error_response() -> RpcResponse {
    parse_error("Parse error")
}

fn request_error_response(error: RequestError) -> RpcResponse {
    let message = match error {
        RequestError::TooDeep => "Parse error: JSON nesting too deep",
        RequestError::TooLarge | RequestError::InvalidFormat | RequestError::Empty => {
            return parse_error_response();
        }
    };
    parse_error(message)
}

/// Process one JSON-RPC request line and return the response to write back.
///
/// `None` for an empty line or a valid notification. Notifications are still
/// dispatched so their side effects occur, but JSON-RPC forbids writing a
/// response to them.
#[must_use]
pub fn process(state: McpState, line: &str) -> Option<RpcResponse> {
    if line.is_empty() {
        return None;
    }

    if let Err(structural) = validate_request(line) {
        return Some(request_error_response(structural));
    }

    let request: crate::protocol::JsonRpcRequest = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(_) => return Some(parse_error_response()),
    };

    let is_notification = request.id.is_none();
    let id = request.id.clone().unwrap_or(Value::Null);

    if request.jsonrpc != "2.0" {
        return Some(err(&id, -32600, "Invalid Request"));
    }

    // MCP 2024-11-05 narrows JSON-RPC's request id to a string or integer.
    // Most importantly, an explicit null is an invalid request, not a
    // notification; field presence was preserved during deserialization so it
    // receives an error response instead of being silently suppressed.
    if !is_notification && !is_valid_mcp_request_id(&id) {
        return Some(err(&Value::Null, -32600, "Invalid Request"));
    }

    let method = McpMethod::parse(&request.method);
    let response = match method {
        McpMethod::Initialize => ok(&id, &handlers::handle_initialize()),
        McpMethod::ToolsList => ok(&id, &handlers::handle_tools_list()),
        McpMethod::ToolsCall => match handlers::handle_tools_call(state, request.params.as_ref()) {
            Ok(result) => ok(&id, &result),
            Err(tool_error) => err(&id, -32603, tool_error.message()),
        },
        McpMethod::Ping => ok(&id, &json!({})),
        McpMethod::Shutdown | McpMethod::NotificationsInitialized => ok(&id, &Value::Null),
        McpMethod::ResourcesList => ok(&id, &json!({"resources": []})),
        McpMethod::PromptsList => ok(&id, &json!({"prompts": []})),
        McpMethod::Unknown => err(&id, -32601, ToolError::UnknownTool.message()),
    };
    if is_notification {
        None
    } else {
        Some(response)
    }
}

fn is_valid_mcp_request_id(id: &Value) -> bool {
    match id {
        Value::String(_) => true,
        Value::Number(number) => number.as_i64().is_some() || number.as_u64().is_some(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(line: &str) -> Value {
        process(McpState::new(), line).expect("a response").0
    }

    #[test]
    fn initialize_matches_the_golden_fixture() {
        let golden: Value =
            serde_json::from_str(include_str!("../../../tests/golden/mcp-initialize.json"))
                .expect("golden fixture parses");
        let got = resp(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#);
        assert_eq!(got, golden);
    }

    #[test]
    fn tools_list_matches_the_golden_fixture() {
        let golden: Value =
            serde_json::from_str(include_str!("../../../tests/golden/mcp-tools-list.json"))
                .expect("golden fixture parses");
        let got = resp(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        assert_eq!(got, golden);
    }

    #[test]
    fn ping_and_id_passthrough() {
        let numeric = resp(r#"{"jsonrpc":"2.0","id":42,"method":"ping"}"#);
        assert_eq!(numeric["id"], json!(42));
        assert_eq!(numeric["result"], json!({}));

        let string_id = resp(r#"{"jsonrpc":"2.0","id":"abc","method":"ping"}"#);
        assert_eq!(string_id["id"], json!("abc"));
    }

    #[test]
    fn notifications_are_dispatched_without_a_response() {
        assert!(
            process(
                McpState::new(),
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#
            )
            .is_none()
        );
        assert!(process(McpState::new(), r#"{"jsonrpc":"2.0","method":"ping"}"#).is_none());
        assert!(
            process(
                McpState::new(),
                r#"{"jsonrpc":"2.0","method":"unknown-notification"}"#
            )
            .is_none()
        );
    }

    #[test]
    fn explicit_null_and_non_mcp_ids_are_invalid_requests_not_notifications() {
        for line in [
            r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#,
            r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#,
            r#"{"jsonrpc":"2.0","id":{},"method":"ping"}"#,
        ] {
            let response = process(McpState::new(), line).expect("invalid request responds");
            assert_eq!(response.0["id"], Value::Null, "{line}");
            assert_eq!(response.0["error"]["code"], json!(-32600), "{line}");
        }
    }

    #[test]
    fn rejects_invalid_requests() {
        assert_eq!(resp("not json")["error"]["message"], json!("Parse error"));
        assert_eq!(
            resp(r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#)["error"]["message"],
            json!("Invalid Request")
        );
        assert_eq!(
            resp(r#"{"jsonrpc":"2.0","id":1,"method":"unknown"}"#)["error"]["message"],
            json!("Method not found")
        );
    }

    #[test]
    fn ai_run_matches_the_golden_success_path() {
        // Proves the wiring, not just `abi_ai::run_text`: the captured Zig
        // response has to survive dispatch, validation, and JSON framing.
        // `ai_run` is the only one of the four AI tools whose captured line is a
        // byte-equality target — the other three embed live store counters.
        let fixture = include_str!("../../../tests/golden/mcp-tool-calls-args.jsonl");
        let expected: Value = fixture
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str::<Value>(line).expect("fixture line parses"))
            .find(|value| value["tool"] == "ai_run")
            .expect("the fixture contains an ai_run call");

        let arguments = &expected["arguments"];
        let call = format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"ai_run","arguments":{arguments}}}}}"#
        );
        assert_eq!(resp(&call), expected["response"]);
    }

    #[test]
    fn empty_arguments_matches_every_line_of_the_golden_fixture() {
        let fixture = include_str!("../../../tests/golden/mcp-tool-calls.jsonl");
        for line in fixture.lines().filter(|l| !l.is_empty()) {
            let expected: Value = serde_json::from_str(line).expect("fixture line parses");
            let tool = expected["tool"].as_str().expect("tool field");
            let call = format!(
                r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{tool}","arguments":{{}}}}}}"#
            );
            let got = resp(&call);
            let expected_response = &expected["response"];

            // wdbx_stats discloses backend=cpu for the store path; gpu_status
            // shape-check only (accelerated depends on Metal kernel init).
            if matches!(tool, "wdbx_stats" | "gpu_status") {
                let text = got["result"]["content"][0]["text"]
                    .as_str()
                    .unwrap_or_else(|| {
                        panic!("tool={tool} expected text content, got {got}");
                    });
                if tool == "gpu_status" {
                    assert!(text.starts_with("backend="), "{text}");
                    assert!(text.contains(" available="), "{text}");
                    assert!(
                        text.contains(" accelerated=false") || text.contains(" accelerated=true"),
                        "{text}"
                    );
                    assert!(text.contains(" capabilities=7 "), "{text}");
                    assert!(text.contains(" message="), "{text}");
                    if abi_gpu::metal_kernels::kernels_active() {
                        assert!(text.contains("accelerated=true"), "{text}");
                    } else {
                        assert!(text.contains("accelerated=false"), "{text}");
                    }
                } else {
                    assert!(text.contains("backend=cpu"), "{text}");
                    assert!(text.contains("source=mcp-store"), "{text}");
                }
                continue;
            }

            assert_eq!(got, *expected_response, "tool={tool}");
        }
    }
}
