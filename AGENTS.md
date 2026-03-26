# AGENTS.md

Guidance for AI coding agents working in this repository.

## Quick Reference

- **Language**: Zig 0.16 (pinned in `.zigversion`)
- **Build**: `zig build` (Linux) or `./build.sh` (macOS 26.4+)
- **Test**: `zig build test --summary all` or `./build.sh test --summary all`
- **Lint**: `zig build lint` / **Fix**: `zig build fix`
- **Parity check**: `zig build check-parity`
- **Full gate**: `zig build check` (lint + test + parity)
- **CLI**: `zig build cli` produces `zig-out/bin/abi`
- **MCP server**: `zig build mcp` produces `zig-out/bin/abi-mcp`

## Critical Rules

1. Never use `@import("abi")` from within `src/` — causes circular import error.
2. Cross-feature imports must use comptime gates: `if (build_options.feat_X) mod else stub`.
3. Use `.empty` not `.{}` for `ArrayListUnmanaged` / `HashMapUnmanaged` init (Zig 0.16).
4. Both `mod.zig` and `stub.zig` must be updated together — run `zig build check-parity`.
5. Use `foundation.time.unixMs()` not `std.time.milliTimestamp` (removed in 0.16).
6. Use `foundation.sync.Mutex` not `std.Thread.Mutex` (may be unavailable).
7. Never run `zig fmt .` at root — use `zig build fix` (scoped to `src/`, `build.zig`, `build/`, and `test/`).
8. All path imports require explicit `.zig` extensions.
9. `var` vs `const`: compiler enforces const for never-mutated locals.
10. On macOS 26.4+, use `./build.sh` instead of `zig build` (LLD linker issue).

## Build Commands

### Full builds and gates
```bash
./build.sh                         # Build (macOS 26.4+ auto-relinks with Apple ld)
zig build test --summary all       # Unit + integration tests
zig build lint                     # Check formatting (read-only)
zig build fix                      # Auto-format in place
zig build check                   # Lint + test + parity (full gate)
zig build check-parity            # Verify mod/stub declaration parity
```

### Running single tests
```bash
zig build test --summary all -- --test-filter "test_name_pattern"
./build.sh test --summary all -- --test-filter "test_name_pattern"
```

### Focused test lanes
```bash
zig build feature-tests            # Feature integration + parity tests
zig build messaging-tests          # Messaging unit + integration
zig build agents-tests             # Agents unit + integration
zig build orchestration-tests     # Orchestration unit + integration
zig build gateway-tests            # Gateway unit + integration
zig build inference-tests          # Inference unit + integration
zig build secrets-tests            # Secrets unit + integration
zig build pitr-tests               # PITR unit + integration
zig build mcp-tests                # MCP integration tests
zig build cli-tests                # CLI tests
zig build tui-tests                # TUI tests
zig build typecheck               # Compile-only for current target
zig build cross-check              # Cross-compilation (linux, wasi, x86_64)
```

### Platform linking
- **macOS**: use `linkIfDarwin()` from `build/linking.zig` instead of inline `if (os.tag == .macos)` checks. 13 callsites consolidated.
- On macOS 26.4+: always use `./build.sh` (LLD cannot link system frameworks on Darwin 25.x).

## Architecture

- **Entrypoint**: `src/root.zig` re-exports all domains as `abi.<domain>`
- **Features**: 20 directories under `src/features/`, 35 features total (including AI sub-features)
- **Mod/Stub pattern**: each feature has `mod.zig` (real), `stub.zig` (no-op), `types.zig` (shared)
- **Comptime gating** in `root.zig`: `if (build_options.feat_gpu) mod else stub`
- **Core**: `src/core/` (config, errors, registry, feature catalog)
- **Foundation**: `src/foundation/` (logging, security, time, SIMD, sync)
- **Runtime**: `src/runtime/` (task scheduling, event loops)
- **Connectors**: `src/connectors/` (OpenAI, Anthropic, Discord, etc.)
- **Protocols**: `src/protocols/` (mcp/, lsp/, acp/, ha/) — all comptime-gated via `feat_mcp`, `feat_lsp`, `feat_acp`, `feat_ha`
- **Inference**: `src/inference/` (multi-backend ML engine)
- **Feature catalog**: `src/core/feature_catalog.zig` (canonical feature metadata)
- **Stub helpers**: `src/core/stub_helpers.zig` (reuse `StubFeature`, `StubContext` in stubs)

## Testing

- **Unit tests**: `src/root.zig` uses `refAllDecls` to walk all `test` blocks in `src/`
- **Integration tests**: `test/mod.zig` imports `@import("abi")` as external consumer
- Add new integration tests by importing them from `test/mod.zig`
- Both suites link macOS frameworks: System, IOKit, Accelerate, Metal, objc

## Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui`. Disable with `-Dfeat-<name>=false`. GPU backends: `-Dgpu-backend=metal` or `-Dgpu-backend=cuda,vulkan`.

## Import Rules

- **Within `src/`**: relative imports only (`@import("../../foundation/mod.zig")`)
- **From `test/`**: use `@import("abi")` and `@import("build_options")`
- **Cross-feature**: comptime gate, never import another feature's `mod.zig` directly

## Zig 0.16 Gotchas

- `ArrayListUnmanaged` init: use `.empty` not `.{}` (struct fields changed in 0.16)
- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- `std.mem.trimRight` renamed to `std.mem.trimEnd`
- `std.process.getEnvVarOwned` removed: use `b.graph.environ_map.get("KEY")` in build.zig
- `std.fs.cwd()` removed: use `std.Io.Threaded` + `std.Io.Dir.cwd()`
- Entry point signature: `pub fn main(init: std.process.Init) !void` (not `pub fn main() !void`)
- Function pointers: can call through `*const fn` directly without dereferencing
- `var` vs `const`: compiler enforces const for never-mutated locals

## Important Safety Notes

- **Database engine thread safety**: every public `Engine` method must acquire `db_lock` before reading `vectors_array`, `hnsw_index`, `ai_client`, or `cache`.
- **JSON utilities**: use `foundation/utils/json.zig` for escaping — never reimplement in protocol-specific files (ACP, MCP, etc.).
- **AI pipeline memory**: string literals in `ProfileResponse.content` crash on `deinit` — always `allocator.dupe()` heap copies before storing.
- **Abbey emotion files**: `emotion.zig` and `emotions.zig` both exist — `emotions.zig` is canonical; don't import `emotion.zig`.
- **Struct field renames**: grep for `.field_name` (with leading dot) to catch anonymous struct literals that won't match `StructName{` searches.
- **Platform-gated externs**: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`, not just OS. Otherwise symbols leak into feature-disabled builds.
- **Wall-clock vs monotonic**: `foundation.time.timestampSec()` is monotonic from process start (returns 0 in the first second). Use `std.posix.system.clock_gettime(.REALTIME, ...)` for wall-clock timestamps in persisted data.

## Code Style

- **Naming**: camelCase for functions/methods, PascalCase for types/structs/enums, SCREAMING_SNAKE_CASE for constants. Avoid abbreviations unless universally understood (e.g., `num` is fine, `cnt` is not).
- **Struct naming**: one main type per file named after the filename (e.g., `src/foo/bar.zig` defines `Bar`). Helper types can be nested or defined nearby.
- **Comments**: doc comments (`//!` or `///`) for public API; inline comments (`//`) for non-obvious implementation details. Avoid redundant comments that restate the code.
- **Error handling**: prefer error unions (`!`) for recoverable failures; use `error.FeatureDisabled` in stubs; `@panic` only in CLI entry points and tests, never in library code.
- **Memory**: always paired allocation/deallocation; use `defer` for cleanup; prefer arena allocators for temporary parsing work.
- **Optionals**: use `orelse`, `orelse_return`, and `if (x) |val|` patterns; avoid nested optional unwrapping.
- **Alignment**: 4-space indentation, no tabs. Max ~80 char line width for readability.
- **`test {}` blocks**: include `std.testing.refAllDecls(@This())` at the end of every public type file to ensure all declarations are exercised.
- **Defer pattern**: `defer x.deinit()` is preferred; `defer { ... multiple statements ... }` when cleanup spans multiple lines.
- **No comments unless requested**: the AGENTS.md rule "NEVER ADD ***ANY*** COMMENTS" applies to implementation code. Doc comments (`//!`, `///`) are still required for public API.
- **AI sub-feature stubs**: stubs under `src/features/ai/*/stub.zig` are domain-specific and intentionally don't use generic `stub_helpers.zig` helpers.

## Common Patterns

### Mod/Stub boilerplate
```zig
// mod.zig
pub const MyFeature = struct { ... };
pub fn isEnabled() bool { return true; }

// stub.zig
pub const MyFeature = struct {};
pub fn isEnabled() bool { return false; }

// types.zig — shared between both
pub const MyType = struct { ... };
```

### Comptime feature gate
```zig
pub const my_feature = if (build_options.feat_X)
    @import("features/my_feature/mod.zig")
else
    @import("features/my_feature/stub.zig");
```

### Using a shared type in stub (avoids duplication)
```zig
// stub.zig
const types = @import("types.zig");
pub const MyType = types.MyType;
pub const MyError = error{FeatureDisabled};
```

### Avoiding string literal ownership issues
```zig
// WRONG: string literal passed to struct with deinit()
const response = ProfileResponse{ .content = "Hello", ... }; // crashes on deinit

// CORRECT: heap-allocate the string
const safe_msg = "Hello";
const response = ProfileResponse{
    .content = try allocator.dupe(u8, safe_msg),
    ...
};
```

## GPU Patterns

### VTable Pattern for Backend-Agnostic Interfaces
```zig
// types.zig — shared types, no circular deps
pub const AiOpsError = error{ NotAvailable, OutOfMemory, ... };
pub const DeviceBuffer = struct { ptr: ?*anyopaque, size: usize, ... };
pub const Transpose = enum { no_trans, trans };

// interface.zig — VTable definition
pub const AiOps = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        sgemm: *const fn (ctx: *anyopaque, ...) AiOpsError!void,
        softmax: *const fn (ctx: *anyopaque, ...) AiOpsError!void,
        allocDevice: *const fn (ctx: *anyopaque, ...) AiOpsError!DeviceBuffer,
        deinit: *const fn (ctx: *anyopaque) void,
    };
    // Wrapper methods call through vtable
    pub fn sgemm(self: AiOps, ...) AiOpsError!void {
        return self.vtable.sgemm(self.ptr, ...);
    }
};

// cpu_fallback.zig — concrete implementation
pub const CpuFallbackAiOps = struct {
    pub fn isAvailable(self: *anyopaque) bool { return true; }
    pub fn sgemm(self: *anyopaque, ...) AiOpsError!void { ... }
};

// adapters.zig — wrap concrete impls
pub fn createAiOps(impl: anytype) AiOps { ... }
```

### Module Decomposition Best Practices
- When a file exceeds ~300 lines, split into sub-modules
- Keep parent file as thin re-export layer (see `src/features/gpu/ai_ops.zig`)
- Move tests to dedicated `tests.zig` sub-module
- Define shared types in `types.zig` to avoid circular dependencies
- Use `interface.zig` for VTable/protocol definitions

## Common Pitfalls

### Circular Import Prevention
- Never use `@import("abi")` from within `src/` — causes circular import error
- Within `src/`: use relative imports only (`@import("../../foundation/mod.zig")`)
- From `test/`: use `@import("abi")` and `@import("build_options")`
- Cross-feature: use comptime gate, never import another feature's `mod.zig` directly

### Memory Ownership
- Always pair allocation with deallocation
- Use `defer` for cleanup: `defer x.deinit()` is preferred
- Arena allocators for temporary parsing work
- `defer { ... multiple statements ... }` when cleanup spans multiple lines

### Thread Safety
- Database `Engine`: every public method must acquire `db_lock` before reading shared state
- Use `foundation.sync.Mutex` not `std.Thread.Mutex` (may be unavailable in Zig 0.16)
- Platform-gated externs: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`
