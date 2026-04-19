# MCP Protocol Registry Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple MCP tool registration from the monolithic `factories.zig` by introducing an `Endpoint`/`Tool` registry pattern that allows modular registration of MCP tools.

**Architecture:** Each handler module (e.g., `ai.zig`, `database.zig`, `discord.zig`) will export a `pub const tools = [_]ToolDef{...}`. The server factory will use `comptime` discovery to aggregate these definitions into the MCP server instance.

**Tech Stack:** Zig 0.17.x/dev, `abi` foundations, `comptime` reflection.

---

### Task 1: Define MCP Tool Registry Interface

**Files:**
- Create: `src/protocols/mcp/registry.zig`
- Modify: `src/protocols/mcp/mod.zig`

- [ ] **Step 1: Create `src/protocols/mcp/registry.zig`**

```zig
const std = @import("std");
const Server = @import("server.zig").Server;

pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
    input_schema: []const u8,
    handler: anytype,
};

pub fn registerTools(server: *Server, modules: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "tools")) {
            inline for (mod.tools) |t| {
                try server.addTool(.{
                    .def = .{
                        .name = t.name,
                        .description = t.description,
                        .input_schema = t.input_schema,
                    },
                    .handler = t.handler,
                });
            }
        }
    }
}
```

- [ ] **Step 2: Update `mod.zig` to include the new registry**

```zig
// Add to src/protocols/mcp/mod.zig
pub const registry = @import("registry.zig");
```

- [ ] **Step 3: Commit**

```bash
git add src/protocols/mcp/registry.zig src/protocols/mcp/mod.zig
git commit -m "feat: implement MCP tool registry interface"
```

### Task 2: Refactor Handler Modules to Export Tools

**Files:**
- Modify: `src/protocols/mcp/handlers/status.zig`
- Modify: `src/protocols/mcp/handlers/database.zig`

- [ ] **Step 1: Export tools from `status.zig`**

```zig
// Update in src/protocols/mcp/handlers/status.zig
const registry = @import("../registry.zig");

pub const tools = [_]registry.ToolDef{
    .{ .name = "abi_status", .description = "...", .input_schema = "{}", .handler = handleAbiStatus },
    // ... add remaining tools
};
```

- [ ] **Step 2: Export tools from `database.zig`**

```zig
// Update in src/protocols/mcp/handlers/database.zig
const registry = @import("../registry.zig");

pub const tools = [_]registry.ToolDef{
    .{ .name = "db_query", .description = "...", .input_schema = "{}", .handler = handleDbQuery },
    // ... add remaining tools
};
```

- [ ] **Step 3: Commit**

```bash
git add src/protocols/mcp/handlers/status.zig src/protocols/mcp/handlers/database.zig
git commit -m "feat: export tool definitions from handler modules"
```

### Task 3: Migrate MCP Factories to use Registry

**Files:**
- Modify: `src/protocols/mcp/factories.zig`

- [ ] **Step 1: Update `createCombinedServer` in `factories.zig`**

```zig
// Use registry.registerTools(server, .{ status, database, ai, discord })
pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-full", version);
    
    // Aggregate tools using registry
    const status = @import("handlers/status.zig");
    const database = @import("handlers/database.zig");
    const ai = @import("handlers/ai.zig");
    const discord = @import("handlers/discord.zig");
    
    try mcp_registry.registerTools(&server, .{ status, database, ai, discord });
    
    return server;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/protocols/mcp/factories.zig
git commit -m "refactor: migrate MCP factories to registry discovery"
```
