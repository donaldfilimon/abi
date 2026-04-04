# AGENTS.md

Guidance for AI coding agents working in this repository.

## Quick Reference

| Command | Description |
|---------|-------------|
| `./build.sh` | Build (macOS 26.4+) |
| `zig build test --summary all` | All tests |
| `zig build test -- --test-filter "pattern"` | Single test |
| `zig build lint` | Check formatting |
| `zig build fix` | Auto-format in place |
| `zig build check` | Full gate (lint + test + parity) |
| `zig build check-parity` | Verify mod/stub parity |

**Do NOT run `zig fmt .` at repo root** — use `zig build fix` which scopes to `src/`, `build.zig`, `build/`, and `test/`.

### Running Single Tests

```bash
# Run a specific test by name pattern
zig build test --summary all -- --test-filter "test_name_pattern"

# On macOS 26.4+:
./build.sh test --summary all -- --test-filter "test_name_pattern"
```

### Test Lanes

```bash
zig build test --summary all                        # All tests
zig build feature-tests messaging-tests agents-tests orchestration-tests
zig build gateway-tests inference-tests secrets-tests pitr-tests
zig build mcp-tests cli-tests tui-tests multi-agent-tests
```

27 focused test lanes exist: `acp-tests`, `agents-tests`, `auth-tests`, `cache-tests`, `cloud-tests`, `compute-tests`, `connectors-tests`, `database-tests`, `desktop-tests`, `documents-tests`, `gateway-tests`, `gpu-tests`, `ha-tests`, `inference-tests`, `lsp-tests`, `messaging-tests`, `multi-agent-tests`, `network-tests`, `observability-tests`, `orchestration-tests`, `pipeline-tests`, `pitr-tests`, `search-tests`, `secrets-tests`, `storage-tests`, `tasks-tests`, `web-tests`.

**Known pre-existing test failures**: inference engine connector backend tests (2 failures), auth integration tests (1 failure, 3 leaks).

**Resolved**: MCP integration tests (fixed `.len` → `.items.len` for `ArrayListUnmanaged`), pipeline tests (fixed `builder.build()` → `try builder.build()` for error union).

---

## Critical Rules

1. **Never use `@import("abi")` from `src/`** — causes circular import
2. **Cross-feature imports**: use comptime gates `if (build_options.feat_X) mod else stub`
3. **Mod/stub parity**: update both together, run `zig build check-parity`
4. Use `.empty` not `.{}` for `ArrayListUnmanaged`/`HashMapUnmanaged` init
5. Use `foundation.time.unixMs()` not `std.time.milliTimestamp`
6. Use `foundation.sync.Mutex` not `std.Thread.Mutex`
7. On macOS 26.4+, use `./build.sh` not `zig build`
8. All path imports need explicit `.zig` extensions
9. `.zigversion` is the toolchain source of truth; `./build.sh` resolves it through `tools/zigly` and prefers `~/.zvm/bin/zig` when the active ZVM version matches

---

## Architecture

- **Entrypoint**: `src/root.zig` re-exports domains as `abi.<domain>`
- **Features**: 21 dirs under `src/features/`, mod/stub/types pattern
- **Comptime gating**: `root.zig` uses `if (build_options.feat_X) mod else stub`
- **Core/Foundation/Runtime**: config, logging, task scheduling
- **Connectors**: OpenAI, Anthropic, Discord, etc.
- **Protocols**: MCP, LSP, ACP, HA — all comptime-gated

---

## Code Style

### Naming
- camelCase: functions/methods
- PascalCase: types/structs/enums
- SCREAMING_SNAKE_CASE: constants
- snake_case: enum variants
- One main type per file named after filename (e.g., `bar.zig` → `Bar`)

### Formatting
- 4-space indentation, no tabs, ~80 char line width

### Comments
- Doc comments (`//!`, `///`) for public API only
- No inline comments unless implementation is non-obvious

### Error Handling
- `@compileError` — compile-time contract violations only
- `@panic` — unrecoverable invariant violations; never in library code (`src/`), only in CLI entry points and tests
- `unreachable` — provably impossible branches verified at comptime
- Error unions (`!`) — all runtime failure paths in library code; prefer `error.FeatureDisabled` in stubs

### Memory
- Always pair allocation/deallocation with `defer`
- Use arena allocators for temporary parsing
- String literals in structs with `deinit()` → always `allocator.dupe()`

### Testing
- `test {}` blocks: include `std.testing.refAllDecls(@This())`
- Integration tests in `test/mod.zig` import `@import("abi")`
- Use public API accessors (e.g., `manager.getStatus()`) not direct struct field access

### refAllDecls Convention
Most files end with `test { std.testing.refAllDecls(@This()); }`. If a sub-module has pre-existing compilation errors, use a deferred comment instead:
```zig
// refAllDecls deferred — sub_module.zig has pre-existing Zig 0.16 API errors
```
Files with known pre-existing errors: `features/ai/abbey/mod.zig`, `features/cloud/mod.zig`, `features/gpu/mod.zig`, `features/network/mod.zig`, `features/web/mod.zig`, `foundation/utils.zig`.

---

## CLI Commands

```bash
abi               # Smart status
abi version       # Version info
abi doctor        # Diagnostics
abi features      # List all 60 features
abi platform      # Platform detection
abi connectors    # List LLM providers
abi chat <msg>    # Multi-profile pipeline
abi db <cmd>      # Vector database ops
abi serve         # Start ACP HTTP server (127.0.0.1:8080)
abi dashboard     # Developer diagnostics shell (requires -Dfeat-tui=true)
```

---

## Imports

- **Never use `@import("abi")` from `src/`** — causes circular import
- Use relative imports within src: `@import("../../foundation/mod.zig")`
- Use `@import("abi")` from `test/` only
- All path imports need explicit `.zig` extensions

---

## Common Patterns

### Feature Gate
```zig
pub const my_feature = if (build_options.feat_X)
    @import("features/my_feature/mod.zig")
else
    @import("features/my_feature/stub.zig");
```

### String Ownership
```zig
// WRONG: crashes on deinit
const response = ProfileResponse{ .content = "Hello", ... };

// CORRECT
const response = ProfileResponse{
    .content = try allocator.dupe(u8, "Hello"),
    ...
};
```

### GPU VTable
```zig
pub const AiOps = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        sgemm: *const fn (ctx: *anyopaque, ...) AiOpsError!void,
        deinit: *const fn (ctx: *anyopaque) void,
    },
    pub fn sgemm(self: AiOps, ...) AiOpsError!void {
        return self.vtable.sgemm(self.ptr, ...);
    }
};
```

---

## Important Safety Notes

- **Thread safety**: every `Engine` public method must acquire `db_lock` before reading vectors/index/client
- **JSON escaping**: use `foundation/utils/json.zig`, never reimplement
- **Emotion files**: use `emotions.zig`, not `emotion.zig`
- **Platform-gated externs**: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`
- **Wall-clock vs monotonic**: `timestampSec()` is monotonic; use `clock_gettime(.REALTIME)` for persisted data
- **Runtime env vars**: use `std.c.getenv(name.ptr)` which returns `?[*:0]const u8`
- **Signal handlers**: use `std.posix.Sigaction` with `callconv(.c)` handler functions

---

## Zig 0.16 Gotchas

- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- Entry points use `pub fn main(init: std.process.Init) !void`; access args via `init.minimal.args`, allocator via `init.gpa` or `init.arena`
- IO operations: use `std.Io.Threaded` + `std.Io.Dir.cwd()` pattern (not removed `std.fs.cwd()`)
- `std.mem.trimRight` renamed to `std.mem.trimEnd`
- `std.process.getEnvVarOwned` removed: use `b.graph.environ_map.get("KEY")` in build.zig

---

## Module Decomposition

- Split files >300 lines into sub-modules
- Keep parent as thin re-export layer (see `src/features/gpu/ai_ops.zig`)
- Define shared types in `types.zig` to avoid circular deps
