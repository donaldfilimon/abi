---
name: abi-superpower-mcp
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

# ABI Superpower: MCP

Exposes the MCP server and tool surface as a superpower.

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
/abi-superpower-mcp call --tool wdbx_query --args '{"query": "vector search"}'
```

### auth
Configure bearer tokens:
```
/abi-superpower-mcp auth --mcp-token <token> --wdbx-token <token>
```

### health
Check server health:
```
/abi-superpower-mcp health
```

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
- `src/mcp/main.zig` - JSON-RPC 2.0 server
- `src/mcp/handlers.zig` - 12 tool implementations
- `src/mcp/http_transport.zig` - Loopback HTTP/SSE
- `src/mcp/middleware.zig` - Arg validation, size limits

## Feature Gates

- `feat-ai` for ai_* tools
- `feat-wdbx` for wdbx_* tools
- `feat-metrics` for scheduler_* tools
- `feat-tui` for plugin_* tools