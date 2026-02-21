# runtime

> Runtime infrastructure (thread pool, channels, scheduling).

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

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
