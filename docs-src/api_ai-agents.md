# ai-agents API Reference

> Agent runtime and orchestration

**Source:** [`src/ai/agents/mod.zig`](../../src/ai/agents/mod.zig)

---

Agents Sub-module

AI agent runtime with tool support and conversation management.

---

## API

### `pub const Context`

<sup>**type**</sup>

Agents context for framework integration.

### `pub fn createAgent(self: *Context, name: []const u8) !*Agent`

<sup>**fn**</sup>

Create a new agent.

### `pub fn getAgent(self: *Context, name: []const u8) ?*Agent`

<sup>**fn**</sup>

Get an existing agent.

### `pub fn getToolRegistry(self: *Context) !*ToolRegistry`

<sup>**fn**</sup>

Get or create the tool registry.

### `pub fn registerTool(self: *Context, tool: Tool) !void`

<sup>**fn**</sup>

Register a tool.

---

*Generated automatically by `zig build gendocs`*
