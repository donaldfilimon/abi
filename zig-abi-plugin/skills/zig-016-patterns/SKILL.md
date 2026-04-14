---
name: zig-016-patterns
description: This skill should be used when writing or modifying Zig code in the ABI project. Provides Zig 0.16 API patterns (pinned at dev.2984+), migration guidance from deprecated APIs, and platform-specific linking notes. Trigger when user writes Zig code, encounters Zig compilation errors, asks about Zig 0.16 APIs, or mentions "std.time", "std.posix", "std.Io", "LazyPath", "Alignment", "HashMap", "DebugAllocator", or "main signature".
---

# Zig 0.16 API Patterns for ABI

Pinned at `0.16.0-dev.3153+d6f43caad`. Follow these patterns exactly — older patterns will not compile.

## Removed / Changed APIs

### Time
`std.time` has NO `timestamp()` or `milliTimestamp()`. Use the foundation time utility:
```zig
const foundation = @import("../../foundation/mod.zig"); // adjust depth
const now = foundation.time.unixSeconds(); // returns i64, 0 on WASM
const now_ms = foundation.time.unixMs();   // milliseconds
```
For sub-second precision: `foundation.time.timestampNs()`.
For distributed components, prefer passing `now: i64` from the caller instead of importing time directly.

### BoundedArray (REMOVED)
`std.BoundedArray` no longer exists. Use manual buffer + length:
```zig
// WRONG
messages: std.BoundedArray(Msg, 64) = .{},
// CORRECT
buffer: [64]Msg = undefined,
len: usize = 0,
```

### Thread.Mutex
`std.Thread.Mutex` may not be available. Use the project's sync wrapper:
```zig
const foundation = @import("../../../foundation/mod.zig");
mutex: foundation.sync.Mutex = .{},
```

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

### Module System (dev.2962+)

**Explicit extensions required** — `@import("path/to/file")` must end in `.zig`:
```zig
// WRONG (dev.1503 style)
const config = @import("core/config");
// CORRECT (dev.2962+)
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

**Named modules in build system**: `abi` and `build_options` exist. Use `@import("build_options")` for feature flags and `@import("abi")` from test/ files. Foundation utilities live at `src/foundation/mod.zig` as part of the single `abi` module, accessible via `@import("abi").foundation` (from test/) or relative imports (from src/). The build.zig is self-contained — no external build/ modules.

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

### Writer API (std.Io.Writer)
`std.io.AnyWriter` and `std.io.fixedBufferStream` are REMOVED. Use `std.Io.Writer`:
```zig
// WRONG
var fbs = std.io.fixedBufferStream(&buf);
try doWrite(fbs.writer().any());
const written = fbs.getWritten();

// CORRECT
var writer = std.Io.Writer.fixed(&buf);
try doWrite(&writer);
const written = buf[0..writer.end];

// Function signature: pass by pointer
fn doWrite(w: *std.Io.Writer) !void {
    try w.print("{d}", .{42});
    try w.writeAll("hello");
}
```

### Process Args (CLI)
`std.process.args()` is REMOVED. Use `Init` parameter:
```zig
// WRONG
var args = std.process.args();

// CORRECT
pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip(); // skip argv[0]
    const cmd = args.next() orelse return;
    // Use init.gpa as allocator (pre-configured, leak-checked in debug)
}
```

### Terminal Detection
`std.posix.isatty` is REMOVED. Use C library:
```zig
if (std.c.isatty(fd) != 0) { /* is a terminal */ }
```

### Termios (POSIX terminal control)
Termios flag types are packed structs with named fields, not integer bitmasks:
```zig
var raw = try std.posix.tcgetattr(fd);
raw.iflag.BRKINT = false;
raw.iflag.ICRNL = false;
raw.lflag.ECHO = false;
raw.lflag.ICANON = false;
raw.cflag.CSIZE = .CS8;  // multi-bit field via enum
try std.posix.tcsetattr(fd, .FLUSH, raw);
```

### Comptime Stub Compatibility
When calling methods that only exist on the real module (not the stub), use `@hasDecl`:
```zig
// WRONG — stub.dashboard = struct {} has no "run" method
try root.tui.dashboard.run(allocator);

// CORRECT — comptime check
if (comptime @hasDecl(root.tui.dashboard, "run")) {
    return root.tui.dashboard.run(allocator);
}
```

## Platform Notes

### Darwin 26+ Linker
Stock prebuilt Zig's internal LLD linker fails on Darwin 25+ with undefined symbols (`__availability_version_check`, `_arc4random_buf`, `_malloc_size`). Compilation succeeds — only linking is blocked.

**Solution**: Use `./build.sh` which auto-relinks with Apple's native ld + compiler_rt:
```bash
./build.sh lib                    # Build library
./build.sh test --summary all     # Run tests
```

Ensure zig is installed: `zigly --status` (auto-downloads if missing).

**Fallback options** (when build.sh cannot be used):
1. `zig build lint` — format check (no linking needed)
2. `zig test <file> -fno-emit-bin` — compile-only check (no binary output)
3. Linux CI for binary-emitting gates

### Platform-Specific Linking (build.zig)
The build system handles linking per-platform directly in `build.zig`:
- **macOS**: System, c, objc, IOKit, Accelerate (feat_gpu), Metal/MPS/CoreGraphics (gpu_metal)
- **Linux**: Default Zig libc linking
- **Windows**: Default Zig linking
- **WASM**: No linking (freestanding)

Never set `use_lld = true` on macOS (zero Mach-O support).

### WASM
`time.unixSeconds()` returns 0 on WASM (no timer). Guard time-dependent logic accordingly.

## Additional Patterns

### SerializationCursor for Decode

Use `SerializationCursor` from `foundation/utils/binary.zig` for decode methods. The auto-advancing cursor eliminates manual byte-offset arithmetic and makes decode logic clearer.

```zig
const binary = @import("../../foundation/utils/binary.zig");

pub fn decode(allocator: std.mem.Allocator, data: []const u8) !MyStruct {
    var cursor = binary.SerializationCursor{ .data = data };
    const version = cursor.readByte() catch return error.BufferTooSmall;
    const id = cursor.readInt(u32, .little) catch return error.BufferTooSmall;
    // ...
}
```

Key points:
- Map `EndOfData` to `BufferTooSmall` at each call site
- Use `readByte()` for single-byte enum discriminants
- Leave encode methods using manual `writeInt` (fixed buffer, no allocation needed)

### errdefer Before toOwnedSlice

Always use `errdefer` (not `defer`) before `toOwnedSlice` — `toOwnedSlice` transfers ownership to the caller, so freeing on success is a use-after-free:
```zig
var list = std.ArrayListUnmanaged(u8).empty;
errdefer list.deinit(allocator);  // only free if we error
const result = try list.toOwnedSlice(allocator);
return result;  // caller owns the slice
```

### foundation.time.unixSeconds() Over Raw Syscalls

Never use `std.c.clock_gettime` directly. Use the foundation wrapper:
```zig
const now = foundation.time.unixSeconds(); // portable, WASM-safe (returns 0)
```

### Comptime Asserts for Struct Padding

Verify struct sizes at comptime to catch padding issues, especially for cache-line-aligned structures:
```zig
comptime {
    const fields_size = @sizeOf(DataField) + @sizeOf(MetaField);
    std.debug.assert(CACHE_LINE_SIZE >= fields_size);
}
```
