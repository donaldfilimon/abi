---
name: abi-mcp-transport
description: Validate ABI MCP JSON-RPC stdio and its custom loopback HTTP compatibility endpoint, including auth, limits, notifications, and shutdown.
---

# ABI MCP transport

Use this skill for the real `abi-mcp` transport surface. ABI provides
JSON-RPC 2.0 over stdio and attempts a loopback-only custom HTTP compatibility
listener at startup. A bind failure leaves stdio running. The HTTP path serves
persistent MCP 2024-11-05 HTTP+SSE sessions (not Streamable HTTP, 2025-03-26);
the process itself is not a long-lived daemon.

## Real launch paths

Build and start stdio:

```bash
./tools/cargo.sh build -p abi-mcp
./mcp/launcher.sh stdio
```

The server exits on stdin EOF. List or call the frozen tools by sending normal
JSON-RPC requests over stdio; there are no `/abi-mcp-transport` commands.

Configure the automatically attempted custom loopback listener:

```bash
ABI_MCP_HTTP_PORT=8080 ABI_MCP_HTTP_TOKEN=local-secret \
  ./target/debug/abi-mcp
```

- `GET /sse` opens a persistent session: `event: endpoint` names
  `/message?sessionId=<id>` (a random v4 UUID) and the stream stays open on its
  own thread (`crates/abi-mcp/src/sse.rs`).
- `POST /message?sessionId=<id>` answers `202` with an empty body and publishes
  the JSON-RPC response as `event: message`, data byte-identical to stdio.
  Notifications queue no event. An unknown or closed session gets `404` before
  dispatch.
- At most 16 sessions (`503` beyond); a session ends on client disconnect or
  server stop, with a `:` keepalive after 15 s idle.
- `POST /message` without a `sessionId` keeps the one-shot compatibility mode:
  one JSON-RPC message per connection, response directly over HTTP (`200`), and
  accepted notifications get `202` with an empty body.
- A present Origin must identify `localhost` or `127.0.0.1` over HTTP(S).
- The listener binds only to loopback.

`ABI_MCP_HTTP_PORT=0`, empty, malformed, or out-of-range environment values
fall back to 8080. Direct Rust tests may still use `HttpConfig { port: 0 }` for
an ephemeral listener.

## Enforced protocol boundaries

| Boundary | Behavior |
| --- | --- |
| Stdio physical line | 64 KiB maximum; one parse error, discard through newline, then recover |
| JSON nesting | 32-container maximum before JSON parse |
| String/field input | Middleware applies its bounded field checks |
| Notification | Omitted `id` dispatches without a JSON-RPC response |
| Explicit `id: null` | Invalid request; omitted and null remain distinct |
| HTTP auth | Optional exact Bearer token via `ABI_MCP_HTTP_TOKEN` |
| HTTP+SSE session | Unknown/closed session `404` before dispatch; more than 16 open `503` |
| Shutdown | EOF wakes and joins the actual bound HTTP listener |

## Frozen tool catalog

The twelve contract-tested tools are `ai_run`, `ai_complete`, `ai_learn`,
`ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`,
`connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, and `plugin_run`.
Do not change the catalog or `tests/golden/mcp-tools-list.json` as incidental
transport cleanup.

## Validation

```bash
ABI_WDBX_PATH=:memory: ABI_WDBX_PERSIST=0 ./tools/cargo.sh test -p abi-mcp
./tools/cargo.sh clippy -p abi-mcp --all-targets -- -D warnings
./tools/cargo.sh build -p abi-mcp
```

Tests and smokes must never open the user's live `~/.abi` store.

## Claim boundary

- JSON-RPC stdio and the frozen twelve-tool catalog are contract-tested.
- The loopback HTTP surface serves persistent MCP 2024-11-05 HTTP+SSE sessions,
  contract-tested against stdio bytes (`transport_contract.rs`), plus the
  one-shot direct-reply `POST /message` mode.
- It is not Streamable HTTP (2025-03-26), WebSocket/gRPC streaming, a
  production non-loopback service, or a claim of TLS/authz/rate-limit
  completeness.
