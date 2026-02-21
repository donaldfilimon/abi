# runtime

> Runtime infrastructure (thread pool, channels, scheduling).

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

**Availability:** Always enabled

---

Runtime Module - Always-on Core Infrastructure

This module provides the foundational runtime infrastructure that is always
available regardless of which features are enabled. It includes:

- Task scheduling and execution engine
- Concurrency primitives (futures, task groups, cancellation)
- Memory management utilities

## Module Organization

```
runtime/
├── mod.zig          # This file - unified entry point
├── engine/          # Task execution engine
├── scheduling/      # Futures, cancellation, task groups
├── concurrency/     # Lock-free data structures
└── memory/          # Memory pools and allocators
```

## Usage

```zig
const runtime = @import("runtime/mod.zig");

// Create runtime context
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Use task groups for parallel work
var group = try ctx.createTaskGroup(.{});
defer group.deinit();
```

---

## API

### `pub const Context`

<sup>**type**</sup>

Runtime context - the always-available infrastructure.
This is created automatically by the Framework and provides
access to scheduling, concurrency, and memory primitives.

### `pub fn init(allocator: std.mem.Allocator) Error!*Context`

<sup>**fn**</sup>

Initialize the runtime context.

### `pub fn deinit(self: *Context) void`

<sup>**fn**</sup>

Shutdown the runtime context.

### `pub fn getEngine(self: *Context) Error!*Engine`

<sup>**fn**</sup>

Get or create the compute engine.

### `pub fn createTaskGroup(self: *Context, config: TaskGroupConfig) !TaskGroup`

<sup>**fn**</sup>

Create a new task group.

### `pub fn createFuture(self: *Context, comptime T: type) !Future(T)`

<sup>**fn**</sup>

Create a new future.

### `pub fn createCancellationSource(self: *Context) !CancellationSource`

<sup>**fn**</sup>

Create a cancellation source.

### `pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup>

Create an engine with configuration (2-arg version for compatibility).

### `pub fn createDefaultEngine(allocator: std.mem.Allocator) !Engine`

<sup>**fn**</sup>

Create an engine with default configuration.

### `pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup>

Create an engine with custom configuration.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
