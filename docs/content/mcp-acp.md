---
title: MCP & ACP
description: Model Context Protocol and Agent Communication Protocol servers
section: Reference
order: 18
---

# MCP & ACP

ABI ships two protocol servers for AI tool integration: **MCP** (Model Context
Protocol) for exposing database tools over JSON-RPC, and **ACP** (Agent
Communication Protocol) for multi-agent coordination.

Both are always available -- they do not require a feature flag.

---

## MCP (Model Context Protocol)

**Source:** `src/services/mcp/`

MCP provides a JSON-RPC 2.0 server that runs over stdio. It exposes WDBX
database operations as tools that can be called by any MCP-compatible client
(Claude Desktop, IDEs, custom agents).

### CLI

```bash
# Start the MCP server (reads JSON-RPC from stdin, writes to stdout)
zig build run -- mcp serve

# List available MCP tools
zig build run -- mcp tools
```

### Protocol

The server reads newline-delimited JSON-RPC 2.0 messages from stdin, dispatches
them to the appropriate tool handler, and writes responses to stdout.

**Supported JSON-RPC methods:**

| Method | Description |
|--------|-------------|
| `initialize` | Handshake, returns server capabilities |
| `tools/list` | List available tools with schemas |
| `tools/call` | Execute a tool by name with arguments |

### WDBX Tools

MCP exposes five tools for interacting with the WDBX vector database:

| Tool | Description | Parameters |
|------|-------------|------------|
| `wdbx_query` | Search vectors by similarity | `query`, `top_k`, `collection` |
| `wdbx_insert` | Insert a new vector | `id`, `vector`, `metadata`, `collection` |
| `wdbx_stats` | Get database statistics | `collection` (optional) |
| `wdbx_list` | List collections or entries | `collection` (optional), `limit` |
| `wdbx_delete` | Delete a vector by ID | `id`, `collection` |

### Example Session

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
  "name": "wdbx_query",
  "arguments": {"query": "machine learning", "top_k": 5}
}}
```

### Architecture

The MCP server is structured as a request loop:

1. Read a line from stdin
2. Parse as JSON-RPC 2.0
3. Dispatch to the matching tool handler
4. Serialize the response as JSON-RPC 2.0
5. Write to stdout and flush

The server is single-threaded and stateless between requests (the underlying
WDBX database maintains its own state).

---

## ACP (Agent Communication Protocol)

**Source:** `src/services/acp/`

ACP implements an agent-to-agent communication protocol. Each agent publishes
an **Agent Card** describing its capabilities, and tasks flow through a defined
lifecycle.

### CLI

```bash
# Print the agent card as JSON
zig build run -- acp card
```

### Agent Card

The agent card is a JSON document that describes what an agent can do:

```json
{
  "name": "abi-agent",
  "version": "0.4.0",
  "description": "ABI Framework Agent",
  "capabilities": ["chat", "search", "compute", "storage"],
  "protocols": ["mcp", "acp"],
  "endpoints": {
    "chat": "/api/v1/chat",
    "search": "/api/v1/search",
    "health": "/health"
  }
}
```

### Task Lifecycle

ACP tasks follow a state machine:

```
created -> running -> completed
                  \-> failed
```

| State | Description |
|-------|-------------|
| `created` | Task submitted, awaiting execution |
| `running` | Task is being processed by an agent |
| `completed` | Task finished successfully with a result |
| `failed` | Task failed with an error message |

### Integration with MCP

MCP and ACP are complementary:

- **MCP** exposes tools that clients call via JSON-RPC (tool-centric).
- **ACP** coordinates agents that can discover and delegate to each other (agent-centric).

An agent running ACP can internally use MCP tools, or expose its own capabilities
through the ACP card for other agents to consume.
