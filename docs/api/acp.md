# acp

> ACP (Agent Communication Protocol) for agent-to-agent communication.

**Source:** [`src/services/acp/mod.zig`](../../src/services/acp/mod.zig)

**Availability:** Always enabled

---

ACP (Agent Communication Protocol) Service

Provides an HTTP server implementing the Agent Communication Protocol
for agent-to-agent communication. Exposes an agent card at
`/.well-known/agent.json` and task management endpoints.

## Usage
```bash
abi acp serve --port 8080
curl http://localhost:8080/.well-known/agent.json
```

---

## API

### `pub const AgentCard`

<sup>**type**</sup>

ACP Agent Card — describes this agent's capabilities

### `pub fn toJson(self: AgentCard, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup>

Serialize to JSON (escapes all string fields for safety)

### `pub const TaskStatus`

<sup>**type**</sup>

Task status in the ACP lifecycle

### `pub const Task`

<sup>**type**</sup>

ACP Task

### `pub fn toJson(self: *const Task, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup>

Serialize task to JSON

### `pub const Server`

<sup>**type**</sup>

ACP Server that manages tasks

### `pub fn createTask(self: *Server, message: []const u8) ![]const u8`

<sup>**fn**</sup>

Create a new task from a message

### `pub fn getTask(self: *Server, id: []const u8) ?*Task`

<sup>**fn**</sup>

Get a task by ID

### `pub fn taskCount(self: *const Server) u32`

<sup>**fn**</sup>

Get the number of tasks

### `pub fn serveHttp(`

<sup>**fn**</sup>

Run the ACP HTTP server loop. Blocks until the process exits.
Caller must provide an I/O backend (e.g. from std.Io.Threaded).

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
