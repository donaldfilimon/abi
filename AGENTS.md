# AGENTS.md

Guidelines for AI coding agents operating in this Zig 0.16 repository.
See also: `CLAUDE.md` (detailed architecture), `CONTRIBUTING.md` (PR workflow).

## Toolchain

- **Zig version**: `0.16.0-dev` pinned in `.zigversion`. Do NOT use 0.13/0.14/0.15 APIs.
- **Compiler**: `~/.zvm/master/zig` (managed via `zvm use master`).
- **Std lib source**: `~/.zvm/master/lib/std/` -- read these files when unsure about an API.

## Build Commands

```bash
zig build                                # Build the project
zig build run -- --help                  # Run CLI (30 commands + 8 aliases)
zig build run -- tui                     # Open TUI interface
zig fmt .                                # Format all source files
zig build lint                           # Check formatting (CI mode, no writes)
zig build fix                            # Auto-format in place
```

## Test Commands

```bash
zig build test --summary all             # Main test suite (1261 pass, 5 skip)
zig build feature-tests --summary all    # Feature inline tests (2082 pass)
zig build cli-tests                      # CLI smoke tests
zig build validate-flags                 # Compile-check 34 feature flag combos
zig build vnext-compat                   # vNext compatibility tests
```

### Running a Single Test

```bash
# Test a single file directly:
zig test src/path/to/file.zig

# Filter tests by name pattern within the main test suite:
zig test src/services/tests/mod.zig --test-filter "pattern"

# Test with a specific feature flag:
zig build test -Denable-gpu=true
zig build test -Denable-ai=false
```

### Quality Gates

```bash
zig build full-check                     # REQUIRED before PRs: format + tests + flags + CLI smoke + imports + consistency
zig build verify-all                     # Extended: full-check + examples + check-wasm + docs
zig build validate-baseline              # Verify test counts match tools/scripts/baseline.zig
zig build check-consistency              # Zig version/baseline/0.16 pattern checks
zig build check-imports                  # No circular @import("abi") in feature modules
zig build toolchain-doctor               # Diagnose local Zig PATH/version drift
```

## Project Structure

```
build.zig                    Top-level build (delegates to build/)
build/                       Modular build system (options, modules, flags, targets, gpu, wasm)
src/abi.zig                  Public API entry point, comptime feature selection
src/core/                    Framework lifecycle, config, registry, errors
src/features/<name>/         mod.zig (real) + stub.zig (disabled stub) per feature
src/services/                Always-available: runtime, platform, shared utils, connectors
src/services/tests/          Integration/system tests (mod.zig is main test root)
src/feature_test_root.zig    Second test root for feature inline tests
tools/cli/                   CLI framework: main.zig, commands/, spec.zig, tui/
tools/scripts/               CI quality gate scripts, baseline.zig
examples/                    Runnable example programs
```

## Code Style

### Formatting
- 4 spaces indentation, no tabs. Target lines under ~100 characters.
- Always run `zig fmt .` before committing. CI enforces via `zig build lint`.

### Naming Conventions
- **Types/enums/structs**: `PascalCase` (e.g., `CacheConfig`, `EvictionPolicy`)
- **Functions/variables**: `camelCase` (e.g., `readFileAlloc`, `isEnabled`)
- **Config structs**: suffix with `*Config` (e.g., `GpuConfig`, `CacheConfig`)
- **Files**: `snake_case.zig` (e.g., `feature_catalog.zig`, `stub_context.zig`)

### Imports
- Use explicit imports only. Never use `usingnamespace`.
- Public API consumers import via `@import("abi")`, not deep file paths.
- Access types via namespaced submodule paths (e.g., `abi.ai.agent.Agent`, `abi.gpu.unified.MatrixDims`). There are no flat convenience aliases at the `abi.ai` or `abi.gpu` level.
- AI sub-features: `abi.ai.core`, `abi.ai.llm`, `abi.ai.training`, `abi.ai.reasoning` (not `abi.ai_core`, `abi.inference`, etc.).
- Feature modules must NOT `@import("abi")` (circular). Use relative paths to `services/shared/`.
- Standard library: `const std = @import("std");` at top of every file.

### Error Handling
- Prefer specific error sets over `anyerror`. Define domain errors per module.
- Use `errdefer` for cleanup on error paths. Watch for `defer free(x)` then `return x` (use-after-free).
- Stubs return `error.FeatureDisabled` for all operations.
- Use `error.SkipZigTest` to skip hardware-gated tests.

### Zig 0.16 Required Patterns (DO NOT use old APIs)
- `std.Io.Dir.cwd()` NOT `std.fs.cwd()`
- `std.ArrayList(T) = .empty` NOT `.init(allocator)` -- pass allocator per-call
- `std.json.Stringify.valueAlloc(...)` NOT `std.json.stringifyAlloc`
- `pub fn main(init: std.process.Init) !void` NOT `pub fn main() !void`
- `std.c.arc4random_buf(...)` NOT `std.crypto.random`
- `std.c.getenv(...)` NOT `std.posix.getenv`
- `{t}` format specifier for enums/errors, NOT `@tagName()` with `{s}`
- `std.Io.Writer.fixed(&buf)` NOT `std.io.fixedBufferStream()`

### Logging
- `std.log.*` in library code. `std.debug.print` only in CLI tools and display functions.

## Feature Module Architecture

Each of the 21 features uses comptime gating in `src/abi.zig`:

```zig
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");
```

**API break (v0.4.0):** Facade aliases (`abi.ai_core`, `abi.inference`, `abi.training`,
`abi.reasoning`) removed. Flat type re-exports removed from `ai` (~156) and `gpu` (~173).
Use submodule paths: `abi.ai.core`, `abi.ai.llm`, `abi.ai.training`, `abi.ai.reasoning`,
`abi.gpu.unified.MatrixDims`, `abi.gpu.profiling.Profiler`, etc.

### Editing a Feature Module
1. Update `mod.zig` with your changes.
2. Update sibling `stub.zig` to keep the same public API signatures.
3. Build with feature off: `zig build -Denable-<feature>=false`
4. Build with feature on: `zig build -Denable-<feature>=true`
5. Run `zig build validate-flags` then `zig build full-check`.

### Stub Conventions
- Use anonymous parameter discard: `fn foo(_: *@This(), _: []const u8) !void`
- Return `error.FeatureDisabled` for all operations.
- Use `StubContext(ConfigT)` from `src/core/stub_context.zig` for Context structs.

## Testing

### Test Baselines (must be maintained)
- **Main tests**: 1261 pass, 5 skip (1266 total) -- source of truth: `tools/scripts/baseline.zig`
- **Feature tests**: 2082 pass, 4 skip (2086 total)

### Test Discovery
- Use `test { _ = @import(...); }` to include submodule tests. `comptime {}` does NOT work.
- End every source file with: `test { std.testing.refAllDecls(@This()); }`
- Two test roots exist because module path restrictions prevent cross-imports.

### Test Utilities
```zig
const allocator = std.testing.allocator;
const io = std.testing.io;
var tmp = std.testing.tmpDir(.{}); defer tmp.cleanup();
```

### After Changing Test Counts
Update `tools/scripts/baseline.zig`, then run `zig build validate-baseline`.

## CLI Commands

New CLI command flow:
1. Implement in `tools/cli/commands/<name>.zig`
2. Export in `tools/cli/commands/mod.zig`
3. Register help/completion metadata in `tools/cli/spec.zig`
4. I/O-heavy commands should use the `std.Io` path from `tools/cli/mod.zig`.

## Commits

Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
Keep commits focused. Do not mix refactors with behavior changes.

## Security

Do not commit secrets, `.env` files, or credentials. Report vulnerabilities via `SECURITY.md`.
