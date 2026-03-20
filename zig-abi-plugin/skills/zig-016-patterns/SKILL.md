---
name: zig-016-patterns
description: This skill should be used when writing or modifying Zig code in the ABI project. Provides Zig 0.16 API patterns (pinned at dev.2934+), migration guidance from deprecated APIs, and platform-specific linking notes. Trigger when user writes Zig code, encounters Zig compilation errors, asks about Zig 0.16 APIs, or mentions "std.time", "std.posix", "std.Io", "LazyPath", "Alignment", "HashMap", "DebugAllocator", or "main signature".
---

# Zig 0.16 API Patterns for ABI

Pinned at `0.16.0-dev.2934+47d2e5de9`. Follow these patterns exactly — older patterns will not compile.

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

### Module System (dev.2934+)

**Explicit extensions required** — `@import("path/to/file")` must end in `.zig`:
```zig
// WRONG (dev.1503 style)
const config = @import("core/config");
// CORRECT (dev.2934+)
const config = @import("core/config/mod.zig");
```

**Single-module file ownership** — every `.zig` file belongs to exactly one named module. All `src/` files belong to the single `abi` module. No `shared_services` or `core` named modules exist.
```zig
// WRONG — named module import from within src/ (causes circular dependency)
const shared = @import("abi").foundation;
// CORRECT — relative path within same module (all src/ is one module)
const shared = @import("../../../../services/shared/simd/mod.zig");
// ALSO CORRECT — use abi's exported namespace from tools/cli/ (separate module)
const shared = @import("abi").foundation;
```

**Named modules in build system**: `abi`, `build_options`, and `cli` exist. Use `@import("build_options")` for feature flags and `@import("abi")` from external modules (CLI, tests). There is no separate `foundation` named module — shared services live at `src/services/shared/mod.zig` as part of the single `abi` module, accessible via `@import("abi").foundation` (external) or relative imports (internal). `wireAbiImports(module, build_opts)` wires only `build_options`.

### Main Function Signature
```zig
// WRONG (old Zig)
pub fn main() !void { ... }
// CORRECT (0.16)
pub fn main(init: std.process.Init) !void { ... }
```

### Allocator
```zig
// WRONG
std.heap.GeneralPurposeAllocator(.{}){}
// CORRECT (0.16)
std.heap.DebugAllocator(.{}){}
```

### Discard Rules
```zig
// WRONG — referencing then discarding triggers "pointless discard"
fn foo(param: u32) void { doSomething(param); _ = param; }
// CORRECT — use _: prefix in signature for unused params
fn foo(_: u32) void { ... }
```

### Defer / errdefer with Owned Memory
```zig
// WRONG — defer on toOwnedSlice causes use-after-free when returning
defer list.deinit(allocator);
const result = try list.toOwnedSlice(allocator);
return result;
// CORRECT — use errdefer, only free on error path
errdefer list.deinit(allocator);
const result = try list.toOwnedSlice(allocator);
return result;
```

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
Stock prebuilt Zig's internal LLD linker fails on Darwin 25+ with undefined symbols (`__availability_version_check`, `_arc4random_buf`, `_malloc_size`). Compilation succeeds — only linking is blocked. The build runner itself cannot link, so no `build.zig` workaround helps.

**Recommended**: use a host-built or known-good Zig matching `.zigversion` (`0.16.0-dev.2934+47d2e5de9`) for `zig build full-check` / `zig build check-docs`.

**Fallback options** (when stock Zig cannot link):
1. `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` — format check (no linking needed)
2. `zig test <file> -fno-emit-bin` — compile-only check (no binary output, no linking)
3. Linux CI or another host with a working Zig linker for binary-emitting gates

Note: Options 2-3 are partial validation only — they verify syntax and type correctness but cannot run tests or produce binaries.

### Platform-Specific Linking (build/link.zig)
The build system handles linking per-platform via `applyAllPlatformLinks()`:
- **macOS/iOS**: Accelerate, Foundation, Metal/CoreML/MPS (when Metal backend), AppKit/Cocoa (macOS), UIKit (iOS)
- **Linux**: libc, libm, CUDA (libcuda/cublas/cudart/cudnn), Vulkan, OpenGL
- **Windows**: CUDA, Vulkan
- **BSD**: Vulkan, OpenGL
- **Android**: log, android, EGL, GLESv2
- **illumos**: socket, nsl, OpenGL (Mesa)
- **Haiku**: OpenGL

Never set `use_lld = true` on macOS (zero Mach-O support).

### WASM
`time.unixSeconds()` returns 0 on WASM (no timer). Guard time-dependent logic accordingly.
