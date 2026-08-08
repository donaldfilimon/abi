---
name: mcp
description: Plan abi MCP server work — the 12-tool JSON-RPC 2.0 stdio surface plus its custom loopback HTTP compatibility listener. Use for abi-mcp, tools, transports, or middleware.
---

# mcp

Entry point for the abi MCP server (`crates/abi-mcp/src/`). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Smoke-test abi-mcp + verify the 12-tool contract | `mcp-smoke` |
| Deep-dive the MCP superpower | `abi-superpower-mcp` |
| Transport / middleware / protocol limits detail | `abi-mcp-transport` |

## Frozen contract (do not change without a parity/contract-test update)
- 12 tools, in source order: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`,
  `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`,
  `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- `protocol.MAX_REQUEST_SIZE` = 64 KB; `MAX_JSON_DEPTH` = 32; per-field 16 KB
  cap in `crates/abi-mcp/src/middleware.rs` (declarative validation before dispatch).
- Frozen enums: `connector_test` tool arg `service` ∈ {openai, anthropic, discord,
  twilio, grok}; `ai_train` tool arg `format` ∈ {jsonl, csv, text}.

## Honest boundary
Stdio exits on stdin EOF (not a long-lived daemon). Startup also attempts the
custom loopback listener (`127.0.0.1:8080` by default, configured with
`ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN`); bind failure leaves stdio running.
`GET /sse` emits one discovery event and closes, while `POST /message` handles
one JSON-RPC message per connection. This is not conforming persistent MCP
HTTP+SSE, and non-loopback serving is not supported. Rust handlers return
bounded JSON-RPC errors without exposing internal error chains.
