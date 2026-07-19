---
name: mcp
description: Plan abi MCP server work — the 12-tool JSON-RPC 2.0 surface over stdio plus optional loopback HTTP/SSE. Use when asked about abi-mcp, the tool list, transports, or middleware. Routes to mcp-smoke, abi-superpower-mcp, and abi-mcp-transport. Loopback-only; non-loopback HTTP hardening is a disclosed gap.
---

# mcp

Entry point for the abi MCP server (`src/mcp/`). Routes to specialists:

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
  cap in `src/mcp/middleware.zig` (declarative validation before dispatch).
- Frozen enums: `connector_test` tool arg `service` ∈ {openai, anthropic, discord,
  twilio, grok}; `ai_train` tool arg `format` ∈ {jsonl, csv, text}.

## Honest boundary
Stdio exits on stdin EOF (not a long-lived daemon). Optional HTTP/SSE is
loopback-only (`127.0.0.1:8080`, `ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN`).
Non-loopback hardening (TLS/authz/rate-limit) is **not** done — deploy behind
a TLS-terminating proxy. `handlers.errorMessage` normalizes every `anyerror`
so `@errorName` never leaks on either transport.
