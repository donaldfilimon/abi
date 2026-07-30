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

fn request_error_response(error: RequestError) -> RpcResponse {
    let message = match error {
        RequestError::TooDeep => "Parse error: JSON nesting too deep",
        RequestError::TooLarge | RequestError::InvalidFormat | RequestError::Empty => "Parse error",
    };
    parse_error(message)
}

/// Process one JSON-RPC request line and return the response to write back.
///
/// `None` for a line that is empty after trimming — the stdio transport skips
/// writing anything for those, matching Zig's `processRequest` early return.
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
        Err(_) => return Some(parse_error("Parse error")),
    };

    let id = request.id.clone().unwrap_or(Value::Null);

    if request.jsonrpc != "2.0" {
        return Some(err(&id, -32600, "Invalid Request"));
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
        McpMethod::Shutdown => ok(&id, &Value::Null),
        McpMethod::ResourcesList => ok(&id, &json!({"resources": []})),
        McpMethod::PromptsList => ok(&id, &json!({"prompts": []})),
        McpMethod::Unknown => err(&id, -32601, ToolError::UnknownTool.message()),
    };
    Some(response)
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
    fn rejects_invalid_requests() {
        assert_eq!(resp("not json")["error"]["message"], json!("Parse error"));
        assert_eq!(
            resp(r#"{"jsonrpc":"1.0","method":"ping"}"#)["error"]["message"],
            json!("Invalid Request")
        );
        assert_eq!(
            resp(r#"{"jsonrpc":"2.0","method":"unknown"}"#)["error"]["message"],
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

            // wdbx_stats depends on a GPU backend that is not linked, and
            // gpu_status on a feature not yet ported, so both diverge by
            // disclosure — see McpState::wdbx_stats_text.
            if matches!(tool, "wdbx_stats" | "gpu_status") {
                continue;
            }

            // plugin_list is fully ported and matches every byte except the
            // entry-point file name: Zig reported `mod.zig`, and the Rust
            // plugins are `mod.rs`. Rewriting just that token keeps the rest of
            // the 16-plugin listing — names, versions, targets, descriptions,
            // and their declaration order — under a byte-exact assertion.
            if tool == "plugin_list" {
                let retargeted = serde_json::from_str::<Value>(
                    &serde_json::to_string(&got)
                        .expect("response reserializes")
                        .replace("entry=mod.rs ", "entry=mod.zig "),
                )
                .expect("retargeted response parses");
                assert_eq!(retargeted, *expected_response, "tool={tool}");
                continue;
            }

            assert_eq!(got, *expected_response, "tool={tool}");
        }
    }
}
