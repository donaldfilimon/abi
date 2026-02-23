# AGENTS.md

Guidelines for AI coding agents operating in this Zig 0.16 repository.
See also: `CLAUDE.md` (detailed architecture), `CONTRIBUTING.md` (PR workflow).

## Toolchain

- **Zig version**: `0.16.0-dev` pinned in `.zigversion`. Do NOT use 0.13/0.14/0.15 APIs.
- **Compiler**: `~/.zvm/master/zig` (managed via `zvm use master`).
- **Std lib source**: `~/.zvm/master/lib/std/` — read these files when unsure about an API.

## Build Commands

```bash
zig build                                # Build the project
zig build run -- --help                  # Run CLI (29 commands + 9 aliases)
zig fmt .                                # Format all source files
zig build lint                           # Check formatting (CI mode, no writes)
zig build fix                            # Auto-format in place
```

## Test Commands

```bash
zig build test --summary all             # Main test suite (1290 pass, 6 skip)
zig build feature-tests --summary all    # Feature inline tests (2360 pass, 5 skip)
zig build cli-tests                      # CLI smoke tests
zig build validate-flags                 # Compile-check 34 feature flag combos
```

### Running a Single Test

```bash
zig test src/path/to/file.zig                              # Test one file directly
zig test src/services/tests/mod.zig --test-filter "pattern" # Filter by name
zig build test -Denable-gpu=true                            # Test with feature flag
zig build test -Denable-ai=false                            # Test with feature off
```

### Quality Gates

```bash
zig build full-check                     # REQUIRED before PRs: fmt + tests + flags + CLI + imports
zig build verify-all                     # Extended: full-check + examples + wasm + docs
zig build validate-baseline              # Verify test counts match tools/scripts/baseline.zig
zig build check-consistency              # Zig version/baseline/0.16 pattern checks
zig build check-imports                  # No circular @import("abi") in feature modules
zig build toolchain-doctor               # Diagnose local Zig PATH/version drift
```

### After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Anything (full gate) | `zig build full-check` |
| Test counts | Update `tools/scripts/baseline.zig`, run `zig build validate-baseline` |

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
- 4 spaces, no tabs. Lines under ~100 characters. Always `zig fmt .` before committing.

### Naming Conventions
- **Types/enums/structs**: `PascalCase` (`CacheConfig`, `EvictionPolicy`)
- **Functions/variables**: `camelCase` (`readFileAlloc`, `isEnabled`)
- **Config structs**: suffix `*Config` (`GpuConfig`, `CacheConfig`)
- **Files**: `snake_case.zig` (`feature_catalog.zig`, `stub_context.zig`)

### Imports
- Explicit imports only. Never use `usingnamespace`.
- Public API: `@import("abi")`, not deep file paths.
- Access types via namespaced paths: `abi.ai.core`, `abi.gpu.unified.MatrixDims`.
- Feature modules must NOT `@import("abi")` (circular). Use relative paths.
- Standard library: `const std = @import("std");` at top of every file.

### Error Handling
- Prefer specific error sets over `anyerror`. Define domain errors per module.
- Use `errdefer` for cleanup on error paths.
- Watch for `defer free(x)` then `return x` (use-after-free — use `errdefer` instead).
- Stubs return `error.FeatureDisabled` for all operations.
- Use `error.SkipZigTest` to skip hardware-gated tests.

### Logging
- `std.log.*` in library code. `std.debug.print` only in CLI tools and TUI display functions.

## Zig 0.16 Required Patterns

DO NOT use old APIs. The left column is correct, the right is wrong:

| Correct (0.16) | Wrong (old) |
|----------------|-------------|
| `std.Io.Dir.cwd()` | `std.fs.cwd()` |
| `std.ArrayList(T) = .empty` | `.init(allocator)` |
| `std.json.Stringify.valueAlloc(...)` | `std.json.stringifyAlloc` |
| `pub fn main(init: std.process.Init) !void` | `pub fn main() !void` |
| `std.c.arc4random_buf(...)` | `std.crypto.random` |
| `std.c.getenv(...)` | `std.posix.getenv` |
| `{t}` format specifier for enums/errors | `@tagName()` with `{s}` |
| `std.Io.Writer.fixed(&buf)` | `std.io.fixedBufferStream()` |

### I/O Backend (required for file/network ops)

```zig
// In CLI (has real environ):
var io_backend: std.Io.Threaded = .init(allocator, .{ .environ = init.environ });
// In library code:
var io_backend: std.Io.Threaded = .init(allocator, .{ .environ = std.process.Environ.empty });
const io = io_backend.io();
```

## Feature Module Architecture

Each feature uses comptime gating in `src/abi.zig`:
```zig
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");
```

### Editing a Feature Module
1. Update `mod.zig` with your changes.
2. Update sibling `stub.zig` — keep matching public API signatures.
3. Build with feature off: `zig build -Denable-<feature>=false`
4. Build with feature on: `zig build -Denable-<feature>=true`
5. Run `zig build validate-flags` then `zig build full-check`.

### Stub Conventions
- Discard unused params: `fn foo(_: *@This(), _: []const u8) !void`
- Return `error.FeatureDisabled` for all operations.
- Use `StubContext(ConfigT)` from `src/core/stub_context.zig` for Context structs.

## Testing

### Baselines (source of truth: `tools/scripts/baseline.zig`)
- **Main tests**: 1290 pass, 6 skip (1296 total)
- **Feature tests**: 2360 pass, 5 skip (2365 total)

### Conventions
- Test discovery: `test { _ = @import(...); }` — `comptime {}` does NOT discover tests.
- End every source file with: `test { std.testing.refAllDecls(@This()); }`
- Two test roots exist because module path restrictions prevent cross-imports.

### Test Utilities
```zig
const allocator = std.testing.allocator;
const io = std.testing.io;
var tmp = std.testing.tmpDir(.{}); defer tmp.cleanup();
```

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
