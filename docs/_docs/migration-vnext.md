---
title: Migration (vNext)
description: Migration guide for upcoming breaking changes with one-release compatibility window
section: Core
order: 5
---

# Migration (vNext)

This guide covers upcoming breaking API changes in the next major ABI release. ABI
follows a **one-release compatibility window**: deprecated APIs are marked in the current
release (0.4.0) and removed in the next. This gives you one full release cycle to
migrate your code.

## Overview

The vNext release modernizes the public API surface to be more consistent and
composable. The key changes are:

1. **`abi.Framework`** is renamed to **`abi.vnext.App`** with a streamlined interface.
2. **`abi.Config`** is replaced by **`abi.vnext.AppConfig`** with flatter field layout.
3. **Builder methods** are renamed from `.withFeature()` to `.enable(feature, config)`.
4. **Feature getters** move from `.getGpu()` to `.feature(.gpu)` with unified return type.
5. **Init helpers** are consolidated: `initDefault` and `initMinimal` become static methods on `App`.

## Migration Table

| Old API (0.4.0) | New API (vNext) | Notes |
|------------------|----------------|-------|
| `abi.Framework` | `abi.vnext.App` | Primary type rename |
| `abi.Config` | `abi.vnext.AppConfig` | Flatter field layout |
| `abi.initDefault(alloc)` | `abi.vnext.App.start(alloc, .{})` | Empty config = all defaults |
| `abi.Framework.initDefault(alloc)` | `abi.vnext.App.start(alloc, .{})` | Same as above |
| `abi.Framework.initMinimal(alloc)` | `abi.vnext.App.start(alloc, .minimal)` | Sentinel config value |
| `abi.Framework.init(alloc, cfg)` | `abi.vnext.App.start(alloc, cfg)` | Config struct passed directly |
| `abi.Framework.builder(alloc)` | `abi.vnext.App.configure(alloc)` | Returns `AppBuilder` |
| `builder.withGpu(cfg)` | `builder.enable(.gpu, cfg)` | Unified method for all features |
| `builder.withGpuDefaults()` | `builder.enable(.gpu, .{})` | Empty config = defaults |
| `builder.withAi(cfg)` | `builder.enable(.ai, cfg)` | Same pattern for all features |
| `builder.withIo(io)` | `builder.io(io)` | Shorter method name |
| `builder.build()` | `builder.start()` | Returns `App` directly |
| `fw.getGpu()` | `app.feature(.gpu)` | Returns `?*GpuContext` (no error) |
| `fw.getAi()` | `app.feature(.ai)` | Returns `?*AiContext` (no error) |
| `fw.isEnabled(.gpu)` | `app.has(.gpu)` | Boolean check |
| `fw.isRunning()` | `app.state() == .running` | Explicit state query |
| `fw.getState()` | `app.state()` | Same enum values |
| `fw.getRuntime()` | `app.runtime()` | Always non-null |
| `fw.getRegistry()` | `app.registry()` | Same `Registry` type |
| `fw.deinit()` | `app.stop()` | Renamed for clarity |
| `fw.shutdownWithTimeout(ms)` | `app.stop(.{ .timeout_ms = ms })` | Options struct |
| `Framework.Error` | `App.Error` | Same error set |
| `Framework.State` | `App.State` | Same state enum |

## Code Examples

### Before (0.4.0)

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Default initialization
    var fw = try abi.initDefault(allocator);
    defer fw.deinit();

    // Check and use features
    if (fw.isEnabled(.gpu)) {
        const gpu_ctx = try fw.getGpu();
        _ = gpu_ctx;
    }

    std.debug.print("State: {t}\n", .{fw.getState()});
}
```

### After (vNext)

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Default initialization
    var app = try abi.vnext.App.start(allocator, .{});
    defer app.stop();

    // Check and use features
    if (app.feature(.gpu)) |gpu_ctx| {
        _ = gpu_ctx;
    }

    std.debug.print("State: {t}\n", .{app.state()});
}
```

### Builder Before (0.4.0)

```zig
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .metal })
    .withAiDefaults()
    .withDatabaseDefaults()
    .withCache(.{ .max_entries = 50_000, .eviction_policy = .lru })
    .build();
defer fw.deinit();
```

### Builder After (vNext)

```zig
var app = try abi.vnext.App.configure(allocator)
    .enable(.gpu, .{ .backend = .metal })
    .enable(.ai, .{})
    .enable(.database, .{})
    .enable(.cache, .{ .max_entries = 50_000, .eviction_policy = .lru })
    .start();
defer app.stop();
```

### Feature Access Before (0.4.0)

```zig
// Error-returning getter -- must handle error.FeatureDisabled
const db = try fw.getDatabase();
const results = try db.query("SELECT * FROM vectors");

// Or check first
if (fw.isEnabled(.database)) {
    const db2 = try fw.getDatabase();
    _ = db2;
}
```

### Feature Access After (vNext)

```zig
// Optional return -- null if disabled (no error to handle)
if (app.feature(.database)) |db| {
    const results = try db.query("SELECT * FROM vectors");
    _ = results;
}

// Boolean check
if (app.has(.database)) {
    // ...
}
```

## Deprecation Markers

In the current release (0.4.0), deprecated APIs continue to work but emit compile-time
deprecation warnings when the `abi_vnext_warnings` build option is enabled:

```bash
# Build with deprecation warnings enabled
zig build -Dabi-vnext-warnings=true
```

This lets you identify call sites that need migration without breaking your build.

## Validation Checklist

Before completing your migration, verify:

1. **Tests pass** -- Run `zig build test --summary all` (expect 1270 pass, 5 skip).
2. **Feature tests pass** -- Run `zig build feature-tests --summary all` (expect 1534 pass).
3. **No deprecated warnings** -- Build with `-Dabi-vnext-warnings=true` and confirm zero warnings.
4. **Builder pattern works** -- If you use the builder, verify `.configure()` ... `.start()` succeeds.
5. **Feature access compiles** -- Replace all `try fw.getXxx()` with `app.feature(.xxx)` optional checks.
6. **Shutdown is clean** -- Replace `fw.deinit()` with `app.stop()` and confirm tests still pass.
7. **Full validation** -- Run `zig build full-check` for the complete gate.

## Timeline

| Milestone | Version | Date | Status |
|-----------|---------|------|--------|
| Deprecation warnings added | 0.4.0 | Current | Available |
| vNext API available (dual support) | 0.4.x | Current | Available |
| Old API removed | 0.6.0 | Planned | -- |

Both the old (`Framework`) and new (`App`) APIs are available simultaneously in the
current release. The old API will emit deprecation warnings when
`-Dabi-vnext-warnings=true` is set. In 0.6.0, the old API will be removed entirely.

**Note:** Zig does not support function overloading, so `app.stop()` is split into
two methods: `stop()` (no arguments, for `defer app.stop()`) and
`stopWithOptions(.{ .timeout_ms = ms })` for controlled shutdown with timeout.

## Compatibility Notes

- **Build flags are unchanged.** All `-Denable-*` and `-Dgpu-backend=` flags work the
  same way in both old and new APIs.
- **Environment variables are unchanged.** All `ABI_*` and `DISCORD_*` variables
  continue to work.
- **Config types are unchanged.** `GpuConfig`, `AiConfig`, `DatabaseConfig`, etc. are
  the same structs -- only the top-level `Config` wrapper is renamed to `AppConfig`.
- **State enum values are unchanged.** `uninitialized`, `initializing`, `running`,
  `stopping`, `stopped`, `failed` remain the same.
- **Error set is unchanged.** `FrameworkError` is type-aliased as `App.Error`.
- **Registry is unchanged.** The `Registry` type and `RegistrationMode` enum are
  the same.

## Further Reading

- [Framework Lifecycle](framework.html) -- current initialization patterns (0.4.0)
- [Architecture](architecture.html) -- module hierarchy (unchanged in vNext)
- [Configuration](configuration.html) -- build flags and environment variables (unchanged)
