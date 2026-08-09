---
name: abi-mcp-transport
description: Validate ABI MCP JSON-RPC stdio and its custom loopback HTTP compatibility endpoint, including auth, limits, notifications, and shutdown.
---

# ABI MCP transport

Use this skill for the real `abi-mcp` transport surface. ABI provides
JSON-RPC 2.0 over stdio and attempts a loopback-only custom HTTP compatibility
listener at startup. A bind failure leaves stdio running. The HTTP path is not a conforming persistent MCP
HTTP+SSE transport and the process is not a long-lived daemon.

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

- `GET /sse` emits one endpoint-discovery event and closes.
- `POST /message` handles one JSON-RPC message per connection.
- Requests receive direct HTTP JSON responses.
- Accepted notifications receive HTTP 202 with an empty body.
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
- The loopback HTTP surface is a custom one-request compatibility path.
- It is not persistent MCP HTTP+SSE, WebSocket/gRPC streaming, a production
  non-loopback service, or a claim of TLS/authz/rate-limit completeness.
