# Streaming Server Refactor and Protocol Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the streaming server and protocol handlers by implementing a compile-time registry-based dispatch mechanism.

**Architecture:** Implement a static `Endpoint` registry pattern using `comptime` discovery, allowing modules to expose endpoints that are aggregated by the core router.

**Tech Stack:** Zig 0.17.x/dev, `abi` foundation primitives, `comptime` reflection.

---

### Task 1: Define Core Registry Interface

**Files:**
- Create: `src/features/ai/streaming/server/types.zig`
- Modify: `src/features/ai/streaming/server/mod.zig`

- [ ] **Step 1: Create `src/features/ai/streaming/server/types.zig`**

```zig
const std = @import("std");

pub const Endpoint = struct {
    path: []const u8,
    method: std.http.Method,
    handler: *const fn (anyopaque, *std.http.Request, *anyopaque) anyerror!void,
};
```

- [ ] **Step 2: Update `mod.zig` to include the new types**

```zig
// Add to src/features/ai/streaming/server/mod.zig
pub const types = @import("types.zig");
```

- [ ] **Step 3: Commit**

```bash
git add src/features/ai/streaming/server/types.zig src/features/ai/streaming/server/mod.zig
git commit -m "feat: define streaming server endpoint types"
```

### Task 2: Implement Registry Discovery

**Files:**
- Create: `src/features/ai/streaming/server/registry.zig`
- Modify: `src/features/ai/streaming/server/mod.zig`

- [ ] **Step 1: Implement `src/features/ai/streaming/server/registry.zig`**

```zig
const std = @import("std");
const types = @import("types.zig");

pub fn dispatch(modules: anytype, path: []const u8, server: anytype, request: *std.http.Request, conn: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "endpoints")) {
            for (mod.endpoints) |ep| {
                if (std.mem.eql(u8, path, ep.path)) {
                    return ep.handler(server, request, conn);
                }
            }
        }
    }
    return error.NotFound;
}
```

- [ ] **Step 2: Update `mod.zig` to include registry**

```zig
// Add to src/features/ai/streaming/server/mod.zig
pub const registry = @import("registry.zig");
```

- [ ] **Step 3: Commit**

```bash
git add src/features/ai/streaming/server/registry.zig src/features/ai/streaming/server/mod.zig
git commit -m "feat: implement registry-based endpoint discovery"
```

### Task 3: Migrate OpenAI Handler to Registry

**Files:**
- Modify: `src/features/ai/streaming/server/openai.zig`
- Modify: `src/features/ai/streaming/server/routing.zig`

- [ ] **Step 1: Add `endpoints` declaration to `openai.zig`**

```zig
// In src/features/ai/streaming/server/openai.zig
pub const endpoints = [_]types.Endpoint{
    .{
        .path = "/v1/chat/completions",
        .method = .POST,
        .handler = handleChatCompletions,
    },
};

fn handleChatCompletions(server: anyopaque, request: *std.http.Request, conn: anyopaque) anyerror!void {
    // ... logic ...
}
```

- [ ] **Step 2: Update `routing.zig` to use the registry**

```zig
// In src/features/ai/streaming/server/routing.zig
const server_mod = @import("mod.zig");
const registry = server_mod.registry;

pub fn dispatchRequest(server: *StreamingServer, request: *std.http.Request, conn: *ConnectionContext) !void {
    // Collect modules to register
    const modules = .{ @import("openai.zig"), @import("admin.zig") };
    registry.dispatch(modules, request.head.target, server, request, conn) catch |err| {
        // ... handle error ...
    };
}
```

- [ ] **Step 3: Commit**

```bash
git add src/features/ai/streaming/server/openai.zig src/features/ai/streaming/server/routing.zig
git commit -m "feat: migrate openai handler to registry dispatch"
```
