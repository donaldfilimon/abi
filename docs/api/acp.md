# acp

> ACP (Agent Communication Protocol) for agent-to-agent communication.

**Source:** [`src/services/acp/mod.zig`](../../src/services/acp/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-agentcard"></a>`pub const AgentCard`

<sup>**const**</sup> | [source](../../src/services/acp/mod.zig#L17)

ACP Agent Card — describes this agent's capabilities

### <a id="pub-fn-tojson-self-agentcard-allocator-std-mem-allocator-u8"></a>`pub fn toJson(self: AgentCard, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L30)

Serialize to JSON (escapes all string fields for safety)

### <a id="pub-const-taskstatus"></a>`pub const TaskStatus`

<sup>**const**</sup> | [source](../../src/services/acp/mod.zig#L53)

Task status in the ACP lifecycle

### <a id="pub-const-task"></a>`pub const Task`

<sup>**const**</sup> | [source](../../src/services/acp/mod.zig#L74)

ACP Task

### <a id="pub-fn-tojson-self-const-task-allocator-std-mem-allocator-u8"></a>`pub fn toJson(self: *const Task, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L94)

Serialize task to JSON

### <a id="pub-const-server"></a>`pub const Server`

<sup>**const**</sup> | [source](../../src/services/acp/mod.zig#L119)

ACP Server that manages tasks

### <a id="pub-fn-createtask-self-server-message-const-u8-const-u8"></a>`pub fn createTask(self: *Server, message: []const u8) ![]const u8`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L144)

Create a new task from a message

### <a id="pub-fn-gettask-self-server-id-const-u8-task"></a>`pub fn getTask(self: *Server, id: []const u8) ?*Task`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L175)

Get a task by ID

### <a id="pub-fn-taskcount-self-const-server-u32"></a>`pub fn taskCount(self: *const Server) u32`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L180)

Get the number of tasks

### <a id="pub-fn-servehttp-allocator-std-mem-allocator-io-std-io-address-const-u8-card-agentcard-httperror-void"></a>`pub fn serveHttp( allocator: std.mem.Allocator, io: std.Io, address: []const u8, card: AgentCard, ) HttpError!void`

<sup>**fn**</sup> | [source](../../src/services/acp/mod.zig#L221)

Run the ACP HTTP server loop. Blocks until the process exits.
Caller must provide an I/O backend (e.g. from std.Io.Threaded).

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
