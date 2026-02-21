---
title: "MCP Server"
description: "Model Context Protocol server with JSON-RPC 2.0"
section: "Services"
order: 2
---

# MCP Server

The MCP (Model Context Protocol) service provides a JSON-RPC 2.0 server over
stdio for exposing ABI framework tools to MCP-compatible AI clients such as
Claude Desktop, Cursor, and other LLM tool-use environments.

- **Namespace:** `abi.mcp` (via `src/services/mcp/`)
- **Source:** `src/services/mcp/`
- **Protocol:** JSON-RPC 2.0 over stdio (newline-delimited)
- **MCP Version:** `2024-11-05`

## Overview

The MCP server reads newline-delimited JSON-RPC messages from stdin, dispatches
them to registered tool handlers, and writes responses to stdout. It implements
the core MCP lifecycle:

1. **Initialize** -- Client sends `initialize`, server responds with capabilities and version
2. **Initialized** -- Client sends `notifications/initialized` notification
3. **Tool discovery** -- Client calls `tools/list` to discover available tools
4. **Tool execution** -- Client calls `tools/call` with tool name and arguments
5. **Ping** -- Keepalive via `ping` method

### Pre-registered WDBX Tools

The server ships with 5 database tools for the WDBX vector database:

| Tool | Description |
|------|-------------|
| `db_query` | Vector similarity search with cosine distance |
| `db_insert` | Insert vectors with optional metadata |
| `db_stats` | Database statistics (count, dimensions, memory) |
| `db_list` | List stored vectors |
| `db_delete` | Delete a vector by ID |

## Quick Start

### Starting the Server

```bash
# Start MCP server via CLI (listens on stdio)
zig build run -- mcp serve

# List registered tools
zig build run -- mcp tools
```

### Client Interaction

Send JSON-RPC messages to stdin:

```bash
# Initialize
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | zig build run -- mcp serve

# List tools
echo '{"jsonrpc":"2.0","method":"tools/list","id":2}' | zig build run -- mcp serve

# Call a tool
echo '{"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"db_stats","arguments":{}}}' | zig build run -- mcp serve
```

## API Reference

### Server Types

| Type | Description |
|------|-------------|
| `Server` | MCP server that manages tools and dispatches JSON-RPC messages |
| `RegisteredTool` | Tool definition paired with a handler function |
| `ToolDef` | Tool metadata: `name`, `description`, `input_schema` (JSON Schema string) |
| `ToolHandler` | Function signature: `fn(allocator, ?ObjectMap, *ArrayListUnmanaged(u8)) anyerror!void` |
| `RequestId` | JSON-RPC request ID (integer or string) |
| `ErrorCode` | Standard JSON-RPC error codes (`parse_error`, `invalid_request`, `method_not_found`, `invalid_params`, `internal_error`) |
| `ServerCapabilities` | Advertised capabilities (tools, resources) |
| `ToolResult` | Tool call result with content blocks |
| `ContentBlock` | Content block in a tool result (`type` + `text`) |

### Server Functions

| Function | Description |
|----------|-------------|
| `Server.init(allocator, name, version)` | Create a new MCP server |
| `Server.deinit()` | Release server resources |
| `Server.addTool(RegisteredTool)` | Register a tool with the server |
| `Server.run(io)` | Run the stdio server loop (requires I/O backend) |
| `Server.runInfo()` | Log readiness without starting I/O loop |
| `createWdbxServer(allocator, version)` | Create a server with all 5 WDBX tools pre-registered |

### JSON-RPC Methods

| Method | Direction | Description |
|--------|-----------|-------------|
| `initialize` | Client -> Server | Handshake; server returns protocol version and capabilities |
| `notifications/initialized` | Client -> Server | Notification that client is ready |
| `tools/list` | Client -> Server | Returns list of available tools with schemas |
| `tools/call` | Client -> Server | Execute a tool by name with arguments |
| `ping` | Client -> Server | Keepalive; server returns `{}` |

### Response Format

Successful responses:
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"abi-wdbx","version":"0.4.0"}}}
```

Tool results are wrapped in MCP content format:
```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"WDBX Database Statistics:\n  Vectors: 0\n  Dimensions: 0"}]}}
```

Tool errors include `isError: true`:
```json
{"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"Error: InvalidParams"}],"isError":true}}
```

## Registering Custom Tools

```zig
const mcp = @import("abi").mcp;

var server = mcp.Server.init(allocator, "my-server", "1.0.0");
defer server.deinit();

// Register a custom tool
try server.addTool(.{
    .def = .{
        .name = "greet",
        .description = "Generate a greeting message",
        .input_schema =
        \\{"type":"object","properties":{"name":{"type":"string","description":"Name to greet"}},"required":["name"]}
        ,
    },
    .handler = struct {
        fn handle(
            alloc: std.mem.Allocator,
            params: ?std.json.ObjectMap,
            out: *std.ArrayListUnmanaged(u8),
        ) !void {
            const p = params orelse return error.InvalidParams;
            const name_val = p.get("name") orelse return error.InvalidParams;
            if (name_val != .string) return error.InvalidParams;
            try out.appendSlice(alloc, "Hello, ");
            try out.appendSlice(alloc, name_val.string);
            try out.append(alloc, '!');
        }
    }.handle,
});
```

## WDBX Tool Schemas

### db_query

```json
{
  "type": "object",
  "properties": {
    "vector": { "type": "array", "items": { "type": "number" }, "description": "Query vector (float32 array)" },
    "top_k": { "type": "integer", "description": "Number of results (default: 5)", "default": 5 },
    "db_name": { "type": "string", "description": "Database name (default: default)", "default": "default" }
  },
  "required": ["vector"]
}
```

### db_insert

```json
{
  "type": "object",
  "properties": {
    "id": { "type": "integer", "description": "Unique vector ID" },
    "vector": { "type": "array", "items": { "type": "number" }, "description": "Vector data (float32 array)" },
    "metadata": { "type": "string", "description": "Optional metadata string" },
    "db_name": { "type": "string", "description": "Database name (default: default)", "default": "default" }
  },
  "required": ["id", "vector"]
}
```

### db_stats

```json
{
  "type": "object",
  "properties": {
    "db_name": { "type": "string", "description": "Database name (default: default)", "default": "default" }
  },
  "required": []
}
```

### db_list

```json
{
  "type": "object",
  "properties": {
    "limit": { "type": "integer", "description": "Max vectors to return (default: 10)", "default": 10 },
    "db_name": { "type": "string", "description": "Database name (default: default)", "default": "default" }
  },
  "required": []
}
```

### db_delete

```json
{
  "type": "object",
  "properties": {
    "id": { "type": "integer", "description": "Vector ID to delete" },
    "db_name": { "type": "string", "description": "Database name (default: default)", "default": "default" }
  },
  "required": ["id"]
}
```

## CLI Commands

```bash
zig build run -- mcp serve     # Start MCP server (stdio JSON-RPC)
zig build run -- mcp tools     # List registered MCP tools
```

## Source Files

| File | Description |
|------|-------------|
| `src/services/mcp/mod.zig` | Module root, WDBX tool registration and handlers |
| `src/services/mcp/server.zig` | JSON-RPC 2.0 server loop and message dispatch |
| `src/services/mcp/types.zig` | Protocol types, error codes, response serialization |

## Related

- [ACP Protocol](acp.html) -- HTTP-based agent communication (complementary to MCP)
- [Connectors](connectors.html) -- LLM providers that agents can use via MCP
- [Deployment](deployment.html) -- Running MCP server in production

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
