# Repository Guidelines

A Zig 0.16 framework for AI services, semantic storage, GPU acceleration, and distributed runtime.

## Project Structure

```
src/              # Framework source (single `abi` module)
├── root.zig      # Public entrypoint (@import("abi"))
├── features/     # 21 feature directories (mod/stub/types pattern)
├── foundation/   # Shared utilities (logging, security, time, SIMD)
├── runtime/      # Task scheduling, event loops
├── platform/     # OS detection, capabilities
├── connectors/   # External service adapters
├── protocols/    # MCP, LSP, ACP, HA implementations
├── tasks/        # Task management
└── inference/    # ML inference engine
build.zig         # Self-contained build root
test/             # Integration tests (50 modules)
```

## Build & Test Commands

| Command | Description |
|---------|-------------|
| `./build.sh` | Build (macOS 26.4+ auto-relinks) |
| `zig build` | Build (Linux / older macOS) |
| `zig build test --summary all` | All unit + integration tests |
| `zig build test -- --test-filter "pattern"` | Single test by name |
| `./build.sh test --summary all` | All tests (macOS 26.4+) |
| `zig build lint` | Check formatting |
| `zig build fix` | Auto-format |
| `zig build check` | Full gate (lint + test + parity) |
| `zig build check-parity` | Verify mod/stub declaration parity |

## Build Options

| Flag | Description |
|------|-------------|
| `-Dfeat-<name>=false` | Disable feature (e.g., `-Dfeat-gpu=false`) |
| `-Dgpu-backend=metal` | Single GPU backend |
| `-Dgpu-backend=cuda,vulkan` | Multiple GPU backends |

## Tools

| Command | Description |
|---------|-------------|
| `tools/zigly --status` | Show Zig path |
| `tools/zigly --link` | Symlink Zig/ZLS to `~/.local/bin` |
| `tools/crossbuild.sh` | Cross-compile (linux, wasi, x86_64) |

## CLI

| Command | Description |
|---------|-------------|
| `zig build cli` | Build CLI binary (`zig-out/bin/abi`) |
| `./build.sh cli` | Build CLI (macOS 26.4+) |
| `abi doctor` | Build configuration report |
| `abi features` | List all 60 features |

## Code Style

- **Indentation**: 4 spaces, no tabs; ~80 char line width
- **Naming**: camelCase (functions), PascalCase (types), SCREAMING_SNAKE_CASE (constants), snake_case (enum variants)
- **Doc comments**: `///` on public API only
- **Error handling**: `!` unions for runtime failures; `@panic` only in CLI/tests
- **Imports**: Relative paths within `src/`; never `@import("abi")` from `src/`

## Testing

- Unit tests: `std.testing.refAllDecls(@This())` in `test {}` blocks
- Integration tests: `test/mod.zig` covers all features
- Known pre-existing failures: inference engine connectors (2), auth integration (1)

## Critical Rules

1. **Mod/stub contract**: Update both together; run `zig build check-parity`
2. **Feature gates**: `if (build_options.feat_X) mod else stub`
3. **macOS 26.4+**: Use `./build.sh`, never `zig build` directly
4. **Memory**: Use `allocator.dupe()` for string literals in structs
5. **Paths**: All imports need explicit `.zig` extensions

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
// WRONG
const response = ProfileResponse{ .content = "Hello", ... };

// CORRECT
const response = ProfileResponse{
    .content = try allocator.dupe(u8, "Hello"),
    ...
};
```
