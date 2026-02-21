---
title: "ACP Protocol"
description: "Agent Communication Protocol for multi-agent systems"
section: "Services"
order: 3
---

# ACP Protocol

The ACP (Agent Communication Protocol) service provides an HTTP server for
agent-to-agent communication. It exposes an agent card at
`/.well-known/agent.json` and task management endpoints for submitting,
tracking, and retrieving agent tasks.

- **Namespace:** `abi.acp` (via `src/services/acp/`)
- **Source:** `src/services/acp/mod.zig`
- **Transport:** HTTP with JSON payloads

## Overview

The ACP service implements a task-based interaction model:

1. **Agent Discovery** -- Clients fetch `/.well-known/agent.json` to discover agent capabilities, skills, and version information.
2. **Task Submission** -- Clients send a message to `/tasks/send` (POST), which creates a task and returns its ID.
3. **Task Tracking** -- Clients poll `/tasks/{id}` (GET) to check task status and retrieve messages.

Tasks progress through a lifecycle: `submitted` -> `working` -> `completed` (or `failed`, `canceled`, `input_required`).

### Built-in Skills

The agent card advertises three default skills:

| Skill ID | Name | Description |
|----------|------|-------------|
| `db_query` | Vector Search | Search the WDBX vector database |
| `db_insert` | Vector Insert | Insert vectors with metadata |
| `agent_chat` | Chat | Conversational interaction |

## Quick Start

### Starting the Server

```bash
# Start ACP HTTP server
zig build run -- acp serve --port 8080

# Print the agent card as JSON
zig build run -- acp card
```

### Client Interaction

```bash
# Discover agent capabilities
curl http://localhost:8080/.well-known/agent.json

# Submit a task
curl -X POST http://localhost:8080/tasks/send \
    -H "Content-Type: application/json" \
    -d '{"message": "Search for similar vectors"}'

# Check task status
curl http://localhost:8080/tasks/task-1
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `AgentCard` | Agent metadata: name, description, version, URL, capabilities, skills |
| `AgentCard.Capabilities` | Feature flags: `streaming`, `pushNotifications` |
| `TaskStatus` | Enum: `submitted`, `working`, `input_required`, `completed`, `failed`, `canceled` |
| `Task` | Task with ID, status, and message history |
| `Task.Message` | Single message with `role` and `content` |
| `Server` | ACP server managing tasks with sequential ID assignment |

### AgentCard

```zig
const card = abi.acp.AgentCard{
    .name = "my-agent",
    .description = "An AI assistant powered by ABI",
    .version = "0.4.0",
    .url = "http://localhost:8080",
    .capabilities = .{
        .streaming = false,
        .pushNotifications = false,
    },
};

// Serialize to JSON
const json = try card.toJson(allocator);
defer allocator.free(json);
```

The `toJson()` method escapes all string fields for JSON safety and includes
the skills array in the output.

### Server

```zig
var server = abi.acp.Server.init(allocator, card);
defer server.deinit();

// Create a task from a user message
const task_id = try server.createTask("Hello, agent!");

// Retrieve the task
if (server.getTask(task_id)) |task| {
    // task.status == .submitted
    // task.messages.items[0].role == "user"
    // task.messages.items[0].content == "Hello, agent!"
}

// Check task count
const count = server.taskCount();
```

### Task Lifecycle

Tasks are created with `submitted` status and a single user message. The
server assigns sequential IDs (`task-1`, `task-2`, etc.).

```zig
const acp = abi.acp;

// Task status values
acp.TaskStatus.submitted       // "submitted"
acp.TaskStatus.working         // "working"
acp.TaskStatus.input_required  // "input-required"
acp.TaskStatus.completed       // "completed"
acp.TaskStatus.failed          // "failed"
acp.TaskStatus.canceled        // "canceled"

// Serialize task to JSON
const json = try task.toJson(allocator);
defer allocator.free(json);
// {"id":"task-1","status":"submitted","messages":[{"role":"user","parts":[{"type":"text","text":"Hello"}]}]}
```

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/.well-known/agent.json` | Agent card (capabilities, skills, version) |
| POST | `/tasks/send` | Submit a new task; body: `{"message": "..."}` or plain text |
| GET | `/tasks/{id}` | Get task status and messages |

### Response Formats

**Agent Card** (`GET /.well-known/agent.json`):
```json
{
  "name": "abi-agent",
  "description": "ABI Framework Agent",
  "version": "0.4.0",
  "url": "http://localhost:8080",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "skills": [
    {"id": "db_query", "name": "Vector Search", "description": "Search the WDBX vector database"},
    {"id": "db_insert", "name": "Vector Insert", "description": "Insert vectors with metadata"},
    {"id": "agent_chat", "name": "Chat", "description": "Conversational interaction"}
  ]
}
```

**Task Creation** (`POST /tasks/send`):
```json
{"id": "task-1"}
```

**Task Status** (`GET /tasks/task-1`):
```json
{
  "id": "task-1",
  "status": "submitted",
  "messages": [
    {"role": "user", "parts": [{"type": "text", "text": "Search for similar vectors"}]}
  ]
}
```

### Error Responses

```json
{"error": "not found"}           // 404 - Unknown path or task ID
{"error": "method not allowed"}  // 405 - Wrong HTTP method
{"error": "payload too large"}   // 413 - Body exceeds 256 KB limit
{"error": "invalid body"}        // 400 - Unreadable request body
```

## Running the HTTP Server

The `serveHttp` function starts a blocking HTTP server:

```zig
const acp = abi.acp;

// Requires an I/O backend
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.init.environ,
});
defer io_backend.deinit();
const io = io_backend.io();

const card = acp.AgentCard{
    .name = "my-agent",
    .description = "My ABI Agent",
    .version = "0.4.0",
    .url = "http://localhost:8080",
    .capabilities = .{},
};

// This blocks and runs the server loop
try acp.serveHttp(allocator, io, "0.0.0.0:8080", card);
```

## CLI Commands

```bash
zig build run -- acp card              # Print agent card JSON to stdout
zig build run -- acp serve             # Start ACP HTTP server (default port)
zig build run -- acp serve --port 8080 # Start on specific port
```

## Source Files

| File | Description |
|------|-------------|
| `src/services/acp/mod.zig` | Agent card, task management, HTTP server, and all types |

## Related

- [MCP Server](mcp.html) -- Stdio-based tool protocol (complementary to ACP)
- [AI Core](ai-core.html) -- Agent framework with multi-agent coordination
- [Connectors](connectors.html) -- LLM providers for agent task execution
- [Deployment](deployment.html) -- Running ACP in production with health checks

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
