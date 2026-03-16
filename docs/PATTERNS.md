# ABI Codebase Patterns (Zig 0.16)

Working reference for contributors. All examples come from the actual codebase
(pinned to `0.16.0-dev.2905+5d71e3051`).

---

## mod/stub Contract

Every feature lives in `src/features/<name>/` and has at least two files:

| File       | Purpose                                                  |
|------------|----------------------------------------------------------|
| `mod.zig`  | Real implementation, compiled when the feature flag is on |
| `stub.zig` | No-op stand-in, compiled when the feature flag is off     |
| `types.zig`| Shared type definitions imported by both mod and stub     |

**Rule**: `stub.zig` must export the same public symbols (functions, types,
constants) as `mod.zig`. If a caller can compile against the real module, it
must also compile against the stub without changes.

Stubs return `error.FeatureDisabled` (or a domain-specific disabled error) for
fallible functions and safe zero-value defaults for infallible ones.

### Example: cache feature

`src/features/cache/mod.zig` (abbreviated):

```zig
const types = @import("types.zig");

pub const CacheConfig = types.CacheConfig;
pub const CacheError  = types.CacheError;
pub const CacheStats  = types.CacheStats;

pub fn init(allocator: std.mem.Allocator, config: CacheConfig) CacheError!void {
    if (state != null) return;
    state = CacheState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit() void {
    if (state) |s| { s.destroy(); state = null; }
}

pub fn isEnabled() bool { return true; }
pub fn isInitialized() bool { return state != null; }

pub fn get(key: []const u8) CacheError!?[]const u8 { ... }
pub fn put(key: []const u8, value: []const u8) CacheError!void { ... }
pub fn delete(key: []const u8) CacheError!bool { ... }
pub fn contains(key: []const u8) bool { ... }
pub fn clear() void { ... }
pub fn size() u32 { ... }
pub fn stats() CacheStats { ... }
```

`src/features/cache/stub.zig` (complete):

```zig
const std = @import("std");
const stub_context = @import("../../core/stub_context.zig");
const types = @import("types.zig");

pub const CacheConfig = types.CacheConfig;
pub const EvictionPolicy = types.EvictionPolicy;
pub const CacheError = types.CacheError;
pub const CacheEntry = types.CacheEntry;
pub const CacheStats = types.CacheStats;

const feature = stub_context.StubFeature(CacheConfig, CacheError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

pub fn get(_: []const u8) CacheError!?[]const u8 {
    return error.FeatureDisabled;
}
pub fn put(_: []const u8, _: []const u8) CacheError!void {
    return error.FeatureDisabled;
}
pub fn putWithTtl(_: []const u8, _: []const u8, _: u64) CacheError!void {
    return error.FeatureDisabled;
}
pub fn delete(_: []const u8) CacheError!bool {
    return error.FeatureDisabled;
}
pub fn contains(_: []const u8) bool { return false; }
pub fn clear() void {}
pub fn size() u32 { return 0; }
pub fn stats() CacheStats { return .{}; }
```

Key points:
- Every public function in `mod.zig` has a matching stub.
- Fallible functions return `error.FeatureDisabled`.
- Infallible functions return zero/empty/false defaults.
- Both files import shared types from `types.zig`.

---

## StubFeature Helpers

`src/core/stub_context.zig` provides comptime generics that eliminate
boilerplate for the four standard lifecycle functions (`init`, `deinit`,
`isEnabled`, `isInitialized`) plus a `Context` type.

### StubFeature (for features with a config parameter)

```zig
pub fn StubFeature(comptime ConfigType: type, comptime ErrorType: type) type {
    return struct {
        pub const Context = StubContextWithConfig(ConfigType);

        pub fn init(_: std.mem.Allocator, _: ConfigType) ErrorType!void {
            return error.FeatureDisabled;
        }
        pub fn deinit() void {}
        pub fn isEnabled() bool { return false; }
        pub fn isInitialized() bool { return false; }
    };
}
```

Usage in a stub (from `src/features/cache/stub.zig`):

```zig
const stub_context = @import("../../core/stub_context.zig");
const types = @import("types.zig");

const feature = stub_context.StubFeature(types.CacheConfig, types.CacheError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;
```

### StubFeatureNoConfig (for features whose init takes only an allocator)

```zig
pub fn StubFeatureNoConfig(comptime ErrorType: type) type {
    return struct {
        pub fn init(_: std.mem.Allocator) ErrorType!void {
            return error.FeatureDisabled;
        }
        pub fn deinit() void {}
        pub fn isEnabled() bool { return false; }
        pub fn isInitialized() bool { return false; }
    };
}
```

Usage (from `src/features/analytics/stub.zig`):

```zig
const lifecycle = stub_context.StubFeatureNoConfig(AnalyticsError);
pub const init = lifecycle.init;
pub const deinit = lifecycle.deinit;
pub const isEnabled = lifecycle.isEnabled;
pub const isInitialized = lifecycle.isInitialized;
```

### StubContext and StubContextWithConfig

For stubs that only need a `Context` type (without the lifecycle functions),
two lower-level helpers exist:

- `StubContext(ConfigType)` -- allocator-only context, ignores config.
- `StubContextWithConfig(ConfigType)` -- stores both allocator and config.

Both provide `init(allocator, config) -> !*@This()` and `deinit(self) -> void`.

---

## types.zig Pattern

Shared type definitions go in `types.zig` within the feature directory. Both
`mod.zig` and `stub.zig` import from it. This avoids duplicating struct/enum/error
definitions and guarantees type identity across the enabled and disabled paths.

### Example: analytics types

`src/features/analytics/types.zig` (abbreviated):

```zig
const std = @import("std");

pub const Event = struct {
    name: []const u8,
    timestamp_ms: u64,
    session_id: ?[]const u8 = null,
    properties: []const Property = &.{},

    pub const Property = struct {
        key: []const u8,
        value: Value,
    };

    pub const Value = union(enum) {
        string: []const u8,
        int: i64,
        float: f64,
        boolean: bool,
    };
};

pub const AnalyticsConfig = struct {
    buffer_capacity: u32 = 1024,
    enable_timestamps: bool = true,
    app_id: []const u8 = "abi-app",
    flush_interval_ms: u64 = 0,
};

pub const AnalyticsError = error{
    BufferFull,
    InvalidEvent,
    FlushFailed,
    AnalyticsDisabled,
    FeatureDisabled,
    OutOfMemory,
};
```

Both `mod.zig` and `stub.zig` then re-export:

```zig
const types = @import("types.zig");
pub const Event = types.Event;
pub const AnalyticsConfig = types.AnalyticsConfig;
pub const AnalyticsError = types.AnalyticsError;
```

This keeps the public API identical regardless of which file the build system
selects.

---

## ArrayListUnmanaged Initialization

In Zig 0.16, `ArrayListUnmanaged` and other unmanaged containers **must** be
initialized with `.empty`, not `.{}`.

**Correct:**

```zig
events: std.ArrayListUnmanaged(StoredEvent) = .empty,
```

**Wrong (compile error: "missing struct field: items"):**

```zig
events: std.ArrayListUnmanaged(StoredEvent) = .{},
```

The `.{}` struct literal fails because `ArrayListUnmanaged` has required fields
(`items` and `capacity`) that have no defaults. The `.empty` constant is
declared by the type itself and provides the correct zero-initialized state
(null pointer, zero capacity).

The same applies to `StringHashMapUnmanaged` and other `*Unmanaged` containers:

```zig
key_map: std.StringHashMapUnmanaged(NodeIndex) = .empty,
metadata: std.StringHashMapUnmanaged([]const u8) = .empty,
```

---

## Feature Flag Gating

All 19 features are gated by compile-time booleans from `build_options`. The
root module (`src/root.zig`) selects between `mod.zig` and `stub.zig` at
comptime:

```zig
const build_options = @import("build_options");

pub const gpu       = if (build_options.feat_gpu)       @import("features/gpu/mod.zig")       else @import("features/gpu/stub.zig");
pub const ai        = if (build_options.feat_ai)        @import("features/ai/mod.zig")        else @import("features/ai/stub.zig");
pub const cache     = if (build_options.feat_cache)      @import("features/cache/mod.zig")      else @import("features/cache/stub.zig");
pub const analytics = if (build_options.feat_analytics) @import("features/analytics/mod.zig") else @import("features/analytics/stub.zig");
// ... 15 more features
```

Sub-module gating within a feature follows the same pattern. From
`src/features/ai/mod.zig`:

```zig
pub const llm      = if (build_options.feat_llm)       @import("llm/mod.zig")       else @import("llm/stub.zig");
pub const training = if (build_options.feat_training)   @import("training/mod.zig")   else @import("training/stub.zig");
pub const vision   = if (build_options.feat_vision)     @import("vision/mod.zig")     else @import("vision/stub.zig");
pub const reasoning = if (build_options.feat_reasoning) @import("reasoning/mod.zig") else @import("reasoning/stub.zig");
```

All flags are enabled by default. Disable at build time:

```bash
zig build -Dfeat-cache=false -Dfeat-analytics=false
```

25 flags are defined in `build/options.zig`, with 42 validated combinations in
`build/flags.zig`.

---

## Import Rules

### Framework API

Code anywhere in the project accesses the framework through the `abi` named module:

```zig
const abi = @import("abi");

// Use features:
const result = try abi.cache.get("key");
const engine = abi.analytics.Engine.init(allocator, .{});
```

### Relative imports within a feature

Files inside `src/features/<name>/` use relative paths to reach siblings:

```zig
const types = @import("types.zig");
const core_facade = @import("facades/core.zig");
```

### Explicit `.zig` extensions required

Zig 0.16 dev.2905+ requires explicit file extensions on all path imports:

```zig
// Correct:
const config = @import("../../core/config/mod.zig");

// Wrong (compile error):
const config = @import("../../core/config/mod");
```

### Single-module file ownership

Every `.zig` file belongs to exactly one named module. Cross-module relative-path
imports are illegal. Files under `src/` belong to the `abi` module. Files under
`src/services/shared/` belong to the `foundation` module and must be accessed
via `@import("foundation")` from other modules.

---

## Darwin 25+ Workarounds

On macOS Darwin 25+, `zig build` fails with undefined symbol linker errors
(`_malloc_size`, `_nanosleep`, etc.) because the build runner itself links
before `build.zig` runs. No `build.zig` setting can fix this.

### What works

| Command | Status |
|---------|--------|
| `./tools/scripts/run_build.sh <args>` | Use for all build operations |
| `zig fmt --check build.zig build/ src/ tools/` | Always works (no linking) |
| `zig test -fno-emit-bin <file>` | Works for single-file tests |

### What does not work

| Command | Problem |
|---------|---------|
| `zig build` | Linker errors on Darwin 25+ |
| `use_lld = true` | LLD has zero Mach-O support -- never use on macOS |

CI on Linux is authoritative for build correctness. Use format checks locally
on Darwin.

---

## Naming Conventions

### Files and functions

`lower_snake_case` for all file names and function names:

```
src/features/cache/mod.zig
src/core/stub_context.zig
build/test_discovery.zig
```

```zig
pub fn putWithTtl(key: []const u8, value: []const u8, ttl_ms: u64) CacheError!void { ... }
pub fn isInitialized() bool { ... }
pub fn getStepCounts(self: *const Funnel, buffer: []u64) []u64 { ... }
```

### Types

`PascalCase` for all type names:

```zig
pub const CacheConfig = struct { ... };
pub const EvictionPolicy = enum { lru, lfu, fifo, random };
pub const AnalyticsError = error{ BufferFull, InvalidEvent, ... };
pub const PersonaType = enum { assistant, coder, writer, ... };
```

### Commits

Conventional commits with atomic scope:

```
feat: add TTL support to cache module
fix: resolve analytics buffer overflow on flush
refactor: migrate stubs to StubFeature helpers
docs: add codebase patterns reference
```

---

## Error Handling

### Explicit error sets

Define named error sets rather than using `anyerror`:

```zig
pub const CacheError = error{
    FeatureDisabled,
    CacheFull,
    KeyNotFound,
    InvalidTTL,
    OutOfMemory,
};

pub const AnalyticsError = error{
    BufferFull,
    InvalidEvent,
    FlushFailed,
    AnalyticsDisabled,
    FeatureDisabled,
    OutOfMemory,
};
```

### Propagation with try

Use `try` to propagate errors up. Do not silently swallow them:

```zig
pub fn init(allocator: std.mem.Allocator, config: CacheConfig) CacheError!void {
    if (state != null) return;
    state = CacheState.create(allocator, config) catch return error.OutOfMemory;
}
```

### Stub error returns

Stubs return `error.FeatureDisabled` for fallible functions. For infallible
functions, return safe defaults:

```zig
// Fallible -- return error
pub fn get(_: []const u8) CacheError!?[]const u8 {
    return error.FeatureDisabled;
}

// Infallible -- return safe default
pub fn contains(_: []const u8) bool { return false; }
pub fn size() u32 { return 0; }
pub fn stats() CacheStats { return .{}; }
```

For stubs with inline sub-module types, use a domain-specific error:

```zig
pub fn autonomousSearch(_: *DeepResearcher, _: []const u8) error{AiDisabled}![]const u8 {
    return error.AiDisabled;
}
```
