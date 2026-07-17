---
name: abi-mcp-transport
description: MCP transport layer superpower. JSON-RPC 2.0 stdio, loopback HTTP/SSE, bearer auth, request size limits, JSON depth bounds.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["serve-stdio", "serve-http", "tools", "call", "health", "auth"]
      description: "Transport action"
    - name: "port"
      type: "integer"
      description: "HTTP/SSE port (default 8080)"
    - name: "token"
      type: "string"
      description: "Bearer token for auth"
    - name: "tool"
      type: "string"
      description: "Tool name to call"
    - name: "args"
      type: "object"
      description: "Tool arguments as JSON"
---

# ABI Superpower: MCP Transport

Exposes the MCP server transport layer as a superpower. **Honest scope**: JSON-RPC 2.0 over stdio + optional loopback HTTP/SSE. Not a long-lived daemon (exits on stdin EOF). Loopback only — non-loopback requires TLS/authz/rate-limit review.

## Actions

### serve-stdio
Start MCP server on stdio (for MCP client integration):
```
/abi-mcp-transport serve-stdio
```
Exits on stdin EOF/read failure.

### serve-http
Start MCP server with loopback HTTP/SSE:
```
/abi-mcp-transport serve-http --port 8080 --token <optional-bearer>
```
Endpoints:
- `GET /sse` — Server-Sent Events stream
- `POST /message` — JSON-RPC request
- Loopback only (`127.0.0.1`)
- Optional bearer auth via `ABI_MCP_HTTP_TOKEN`

### tools
List all 12 frozen MCP tools:
```
/abi-mcp-transport tools
```

### call
Invoke a tool directly:
```
/abi-mcp-transport call --tool ai_complete --args '{"input": "hello"}'
```

### health
Check server health:
```
/abi-mcp-transport health
```

### auth
Configure bearer tokens:
```
/abi-mcp-transport auth --mcp-token <token> --wdbx-token <token>
```
Sets `ABI_MCP_HTTP_TOKEN` and `ABI_WDBX_REST_TOKEN` for current session.

## Transport Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client                                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌─────────────┐                 ┌─────────────┐
       │  Stdio      │                 │  HTTP/SSE   │
       │  Transport  │                 │  Transport  │
       │             │                 │  (loopback) │
       │ stdin/stdout│                 │  127.0.0.1: │
       │             │                 │  8080       │
       └──────┬──────┘                 └──────┬──────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  JSON-RPC 2.0       │
                   │  Protocol Engine    │
                   │  (protocol.zig)     │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Middleware Chain   │
                   │  (middleware.zig)   │
                   │  - Size limits      │
                   │  - JSON depth       │
                   │  - Arg validation   │
                   │  - Error normalize  │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Tool Dispatch      │
                   │  (handlers.zig)     │
                   └─────────────────────┘
```

## Protocol Limits (Enforced in `middleware.zig` + `protocol.zig`)

| Limit | Value | Source |
|-------|-------|--------|
| Max request size | 64 KB | `protocol.MAX_REQUEST_SIZE` |
| Max JSON depth | 32 | `protocol.MAX_JSON_DEPTH` |
| Max field size | 16 KB | `middleware.zig` per-field cap |
| Stdio exit | EOF/read fail | `stdio_transport.zig` |

## Authentication

| Transport | Auth | Env Var |
|-----------|------|---------|
| Stdio | None (local IPC) | — |
| HTTP/SSE | Optional Bearer | `ABI_MCP_HTTP_TOKEN` |
| WDBX REST | Optional Bearer | `ABI_WDBX_REST_TOKEN` |

**Not production non-loopback hardening** — deploy behind TLS-terminating proxy with authz/rate-limit.

## Frozen Tool Surface (12 tools)

1. `ai_run` — AI inference with profile routing
2. `ai_complete` — Completion with metadata
3. `ai_learn` — SEA self-learning completion
4. `ai_train` — Train agent profile (paths confined to cwd/`ABI_TRAIN_DATA_ROOT`)
5. `wdbx_query` — Vector store hybrid search
6. `scheduler_stats` — Scheduler task counts
7. `scheduler_info` — Compatibility alias
8. `connector_test` — Local connector validation
9. `gpu_status` — GPU backend report
10. `plugin_list` — Bundled plugin metadata
11. `wdbx_stats` — WDBX statistics
12. `plugin_run` — Execute registered plugin

## Shutdown Semantics

- `shutdown` RPC **only signals** — doesn't teardown scheduler/store
- Actual teardown deferred to `main` after HTTP thread joins
- Prevents use-after-free during in-flight calls

## Error Handling

- `handlers.errorMessage` normalizes **every** `anyerror` to stable non-leaking string
- Raw `@errorName` **never** leaks on either transport
- Feature-disabled tools return `FeatureDisabled` error (not fabricated success)

## Feature Gates

| Tool Group | Required Feature |
|------------|-----------------|
| `ai_*` | `feat-ai=true` |
| `wdbx_*` | `feat-wdbx=true` |
| `scheduler_*` | `feat-metrics=true` |
| `connector_test` | `feat-ai=true` + connector features |
| `gpu_status` | `feat-gpu=true` |
| `plugin_*` | `feat-tui=true` + plugin features |

When disabled: explicit degraded response preserving surface shape.

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx`:
- ✅ JSON-RPC 2.0 stdio + loopback HTTP/SSE
- ✅ 12 frozen tools with contract coverage
- ✅ 64 KB request cap + 32 JSON depth + 16 KB field cap
- ✅ Optional bearer tokens on loopback HTTP
- ❌ NOT long-lived daemon (exits on pipe close)
- ❌ NOT production non-loopback without TLS/authz/rate-limit review
- ❌ NOT WebSocket/gRPC streaming (proposed)