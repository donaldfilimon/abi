---
title: runtime API
purpose: Generated API reference for runtime
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# runtime

> Runtime Module - Always-on Core Infrastructure

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
const runtime = @import("runtime");

// Create runtime context
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Use task groups for parallel work
var group = try ctx.createTaskGroup(.{});
defer group.deinit();
```

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L175)

Runtime context - the always-available infrastructure.
This is created automatically by the Framework and provides
access to scheduling, concurrency, and memory primitives.

### <a id="pub-fn-init-allocator-std-mem-allocator-error-context"></a>`pub fn init(allocator: std.mem.Allocator) Error!*Context`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L187)

Initialize the runtime context.

### <a id="pub-fn-deinit-self-context-void"></a>`pub fn deinit(self: *Context) void`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L197)

Shutdown the runtime context.

### <a id="pub-fn-getengine-self-context-error-engine"></a>`pub fn getEngine(self: *Context) Error!*Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L207)

Get or create the compute engine.

### <a id="pub-fn-createtaskgroup-self-context-config-taskgroupconfig-taskgroup"></a>`pub fn createTaskGroup(self: *Context, config: TaskGroupConfig) !TaskGroup`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L220)

Create a new task group.

### <a id="pub-fn-createfuture-self-context-comptime-t-type-future-t"></a>`pub fn createFuture(self: *Context, comptime T: type) !Future(T)`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L225)

Create a new future.

### <a id="pub-fn-createcancellationsource-self-context-cancellationsource"></a>`pub fn createCancellationSource(self: *Context) !CancellationSource`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L230)

Create a cancellation source.

### <a id="pub-fn-createengine-allocator-std-mem-allocator-config-engineconfig-engine"></a>`pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L240)

Create an engine with configuration (2-arg version for compatibility).

### <a id="pub-fn-createdefaultengine-allocator-std-mem-allocator-engine"></a>`pub fn createDefaultEngine(allocator: std.mem.Allocator) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L245)

Create an engine with default configuration.

### <a id="pub-fn-createenginewithconfig-allocator-std-mem-allocator-config-engineconfig-engine"></a>`pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L250)

Create an engine with custom configuration.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
