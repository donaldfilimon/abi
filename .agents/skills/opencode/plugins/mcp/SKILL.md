---
name: mcp
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

# MCP Superpower Plugin

Core MCP capabilities for OpenCode within the ABI framework.

## Capabilities

- MCP subsystem integration
- Plugin framework registration
- Runtime lifecycle management
- Configuration and settings management
- Status monitoring and reporting

## Integration Points

- ABI's MCP subsystem integration
- OpenCode plugin framework integration
- Runtime lifecycle management
- Configuration and settings management

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
Configure the optional HTTP bearer token with `ABI_MCP_HTTP_TOKEN` before
launch. Stdio stays tokenless.

### health
There is no dedicated health command; send a JSON-RPC `ping` or `initialize`
request over stdio.

## Frozen Tool Surface (12 tools)

ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run

## Implementation

Maps to:
- `crates/abi-mcp/src/main.rs` - JSON-RPC 2.0 server
- `crates/abi-mcp/src/handlers.rs` - 12 tool implementations
- `crates/abi-mcp/src/http.rs` - Custom loopback HTTP compatibility path

## Build and protocol boundary

All twelve handlers are built into `abi-mcp`; the historical `feat-*` switches
do not exist in this Rust workspace. The loopback listener is attempted at
startup, bind failure leaves stdio running, and one-event `GET /sse` is not a
persistent conforming MCP HTTP+SSE transport.
