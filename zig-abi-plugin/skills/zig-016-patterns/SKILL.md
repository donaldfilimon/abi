---
name: zig-016-patterns
description: Use when writing or modifying Zig code in the ABI project. Provides Zig 0.16 API patterns, migration guidance from deprecated APIs, and platform-specific workarounds. Trigger when user writes Zig code, encounters Zig compilation errors, asks about Zig 0.16 APIs, or mentions "std.time", "std.posix", "std.Io", "LazyPath", "Alignment", or "HashMap".
---

# Zig 0.16 API Patterns for ABI

When writing Zig code in this project, follow these Zig 0.16 patterns exactly. Older patterns will not compile.

## Removed / Changed APIs

### Time
`std.time` has NO `timestamp()`. Use the project's time utility:
```zig
// Path is relative — adjust depth to match your file's location:
//   from src/features/database/: @import("../../services/shared/time.zig")
//   from src/features/ai/sub/:   @import("../../../services/shared/time.zig")
const time = @import("<depth>/services/shared/time.zig");
const now = time.unixSeconds(); // returns i64, 0 on WASM
```
For sub-second precision: `time.timestampNs()`.
For distributed components, prefer passing `now: i64` from the caller instead of importing time directly.

### Environment Variables
`std.posix.getenv()` is REMOVED. Use:
```zig
const value_ptr = std.c.getenv(name.ptr) orelse return null;
return std.mem.span(value_ptr);
```

### Enum Conversion
`std.meta.intToEnum` is REMOVED. Use:
```zig
const val = @enumFromInt(x); // direct builtin
// For validated parsing:
switch (x) { 0 => .a, 1 => .b, else => return error.Invalid }
```

### HashMap Iteration
`.values()` is NOT public API on `AutoHashMapUnmanaged`. Use:
```zig
var it = map.valueIterator();
while (it.next()) |v| { ... }
```

### Allocator Vtable
`alignment` parameter is `std.mem.Alignment`, not `u8`:
```zig
std.mem.Alignment.fromByteUnits(n)
```

### Build System
```zig
// Module creation (no root_source_file on compile step)
const mod = b.createModule(.{ .root_source_file = b.path("src/root.zig") });
const exe = b.addExecutable(.{ .name = "app", .root_module = mod });

// LazyPath — no .path field
b.path("relative/path")  // correct
// .cwd_relative for CWD-relative paths
```

### File I/O (std.Io.Dir)
```zig
// No makeDirAbsolute — use .cwd() base
std.Io.Dir.createDirPath(.cwd(), io, path);
// No deleteTreeAbsolute
std.Io.Dir.deleteTree(.cwd(), io, path);
// No File.writeAll
file.writeStreamingAll(io, data);
// File existence check
Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
```

### mem.readInt / writeInt
Takes `*const [N]u8` / `*[N]u8`. Use `std.builtin.Endian.little` / `.big`.

### usingnamespace
REMOVED in 0.16. Pass parent context as parameters to submodule init functions instead.

### Module System (dev.2905+)

**Explicit extensions required** — `@import("path/to/file")` must end in `.zig`:
```zig
// WRONG (dev.1503 style)
const config = @import("core/config");
// CORRECT (dev.2905+)
const config = @import("core/config/mod.zig");
```

**Single-module file ownership** — every `.zig` file belongs to exactly one named module. All `src/` files belong to the single `abi` module. No `shared_services` or `core` named modules exist.
```zig
// WRONG — cross-module relative path
const shared = @import("../../../../services/shared/simd/mod.zig");  // if shared_services is a named module
// CORRECT — relative path within same module (all src/ is one module)
const shared = @import("../../../../services/shared/simd/mod.zig");  // fine, same abi module
// ALSO CORRECT — use abi's exported namespace from tools/cli/ (separate module)
const shared = @import("abi").foundation;
```

**Named modules in build system**: `abi`, `build_options`, and `cli` exist. Use `@import("build_options")` for feature flags and `@import("abi")` from external modules (CLI, tests). There is no separate `foundation` named module — shared services live at `src/services/shared/mod.zig` as part of the single `abi` module, accessible via `@import("abi").foundation` (external) or relative imports (internal). `wireAbiImports(module, build_opts)` wires only `build_options`.

### Format Specifiers
Zig's `std.fmt` does NOT support `{t}`. Common valid specifiers:
- `{}` — default formatter (works for enums, ints, etc.)
- `{s}` — string (for `[]const u8`)
- `{d}` — decimal integer
- `{x}` — hex
- `{any}` — any type

**Comptime gating trap:** Invalid format specifiers in comptime-dead branches (e.g., `if (builtin.os.tag != .macos)` on macOS) won't error on the gated platform but WILL error on other targets.

## Platform Notes

### Darwin 26+ Linker
Direct `zig build` fails with undefined symbols (`__availability_version_check`, `_arc4random_buf`). The build runner itself can't link — no `build.zig` workaround helps.

**Supported path**: use a host-built or otherwise known-good Zig matching `.zigversion` for `zig build full-check` / `zig build check-docs`.

**Fallback evidence**:
1. `./tools/scripts/run_build.sh typecheck --summary all` — temporary fallback when stock prebuilt Zig is linker-blocked
2. `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` — direct format check (no linking)
3. `zig test <file> -fno-emit-bin` — compile-only check (no binary output, no linking)
4. Linux CI or another host with a working Zig linker for binary-emitting gates

Note: Options 3-4 are partial validation only — they verify syntax and type correctness but cannot run tests or produce binaries.

### WASM
`time.unixSeconds()` returns 0 on WASM (no timer). Guard time-dependent logic accordingly.
