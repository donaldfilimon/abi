---
name: abi-superpower-mcp
description: ABI MCP JSON-RPC server skill for the frozen 12-tool stdio surface and custom loopback HTTP compatibility listener.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["serve", "tools", "call", "auth", "health"]
      description: "MCP action"
    - name: "tool"
      type: "string"
      description: "Tool name (ai_complete, wdbx_query, etc.)"
    - name: "args"
      type: "object"
      description: "Tool arguments as JSON"
---

# ABI Superpower: MCP

Exposes the MCP server and tool surface as a superpower.

## Actions

### serve
Start MCP server on stdio:
```
./mcp/launcher.sh stdio
```

### tools
List all 12 frozen tools:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | ./target/debug/abi-mcp
```

### call
Invoke a tool with arguments:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ai_complete","arguments":{"input":"hello"}}}' | ./target/debug/abi-mcp
```

### auth
The automatically attempted custom loopback listener can require an exact
bearer token configured before launch:
```
ABI_MCP_HTTP_TOKEN=local-secret ./target/debug/abi-mcp
```

### health
There is no dedicated health command. Use a JSON-RPC `ping` or `initialize`
request over stdio; the HTTP listener is only a custom compatibility surface.

## Frozen Tool Surface (12 tools)

1. `ai_run` - AI inference with profile routing
2. `ai_complete` - Completion with metadata
3. `ai_learn` - SEA self-learning completion
4. `ai_train` - Train agent profile
5. `wdbx_query` - Vector store query
6. `scheduler_stats` - Scheduler task counts
7. `scheduler_info` - Compatibility alias
8. `connector_test` - Local connector validation
9. `gpu_status` - GPU backend report
10. `plugin_list` - Bundled plugins
11. `wdbx_stats` - WDBX statistics
12. `plugin_run` - Execute plugin

## Implementation

Maps to:
- `crates/abi-mcp/src/main.rs` - JSON-RPC 2.0 server
- `crates/abi-mcp/src/handlers.rs` - 12 tool implementations
- `crates/abi-mcp/src/http.rs` - Custom loopback HTTP compatibility path
- `crates/abi-mcp/src/middleware.rs` - Arg validation, size limits

## Build and protocol boundary

The twelve handlers are built into `abi-mcp`; the historical `feat-ai`,
`feat-wdbx`, `feat-metrics`, and `feat-tui` switches do not exist in this Rust
workspace. The loopback listener is attempted at process startup and bind
failure leaves stdio running. `GET /sse` emits one discovery event and closes;
this is not a persistent conforming MCP HTTP+SSE transport.
