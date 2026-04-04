# AGENTS.md

Guidance for AI coding agents working in this repository.

## Quick Reference

| Command | Description |
|---------|-------------|
| `./build.sh` | Build (macOS 26.4+) |
| `zig build test --summary all` | All tests |
| `zig build test -- --test-filter "pattern"` | Single test |
| `zig build lint` | Check formatting |
| `zig build fix` | Auto-format |
| `zig build check` | Full gate (lint + test + parity) |
| `zig build check-parity` | Verify mod/stub parity |

### Test Lanes

```bash
zig build test --summary all                        # All tests
zig build test -- --test-filter "pattern"          # Single test
zig build feature-tests messaging-tests agents-tests orchestration-tests
zig build gateway-tests inference-tests secrets-tests pitr-tests
zig build mcp-tests cli-tests tui-tests multi-agent-tests
```

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
- **Features**: 20 dirs under `src/features/`, mod/stub/types pattern
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
- One main type per file named after filename (e.g., `bar.zig` → `Bar`)

### Formatting
- 4-space indentation, no tabs, ~80 char line width

### Comments
- Doc comments (`//!`, `///`) for public API only
- No inline comments unless implementation is non-obvious

### Error Handling
- Prefer error unions (`!`) for recoverable failures
- Use `error.FeatureDisabled` in stubs
- `@panic` only in CLI entry points and tests

### Memory
- Always pair allocation/deallocation with `defer`
- Use arena allocators for temporary parsing
- String literals in structs with `deinit()` → always `allocator.dupe()`

### Testing
- `test {}` blocks: include `std.testing.refAllDecls(@This())`
- Integration tests in `test/mod.zig` import `@import("abi")`

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
abi dashboard     # Developer diagnostics shell (overview, features, runtime)
```

`abi dashboard` requires `-Dfeat-tui=true`; non-interactive launches point users to `abi doctor`.

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

---

## Module Decomposition

- Split files >300 lines into sub-modules
- Keep parent as thin re-export layer (see `src/features/gpu/ai_ops.zig`)
- Define shared types in `types.zig` to avoid circular deps
