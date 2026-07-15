---
name: mcp
description: MCP server and client superpower. JSON-RPC tools, stdio transport, HTTP/SSE loopback, authentication.
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
/abi-superpower-mcp serve
```

### tools
List all 12 frozen tools:
```
/abi-superpower-mcp tools
```

### call
Invoke a tool with arguments:
```
/abi-superpower-mcp call --tool ai_complete --args '{"input": "hello"}'
```

### auth
Configure bearer tokens.

### health
Check server health.

## Frozen Tool Surface (12 tools)

ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run

## Implementation

Maps to:
- `src/mcp/main.zig` - JSON-RPC 2.0 server
- `src/mcp/handlers.zig` - 12 tool implementations
- `src/mcp/http_transport.zig` - Loopback HTTP/SSE

## Feature Gates

- `feat-ai`, `feat-wdbx`, `feat-metrics`, `feat-tui` as appropriate for each tool.
