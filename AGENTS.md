# AGENTS.md

A Zig 0.16 framework for AI services, semantic storage, GPU acceleration, and distributed runtime.

## Build Commands

| Command | When |
|---------|------|
| `./build.sh` | macOS 26.4+ (auto-relinks with Apple ld) |
| `zig build` | Linux / older macOS |
| `./build.sh lib` | Build library only |
| `./build.sh cli` | Build CLI binary |
| `./build.sh mcp` | Build MCP server |

## Test Commands

| Command | When |
|---------|------|
| `./build.sh test --summary all` | Full test suite (macOS) |
| `zig build test --summary all` | Full test suite (Linux) |
| `zig build test -- --test-filter "pattern"` | Single test by name |
| `zig build check-parity` | Verify mod/stub parity |

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import
2. **macOS 26.4+**: Use `./build.sh`, never `zig build` directly
3. **Mod/stub contract**: Update both together; run parity check
4. **Feature gates**: `if (build_options.feat_X) mod else stub`
5. **String ownership**: Use `allocator.dupe()` for string literals in structs
6. **Imports**: Explicit `.zig` extension required

## Common Patterns

```zig
// Feature gate
pub const my_feature = if (build_options.feat_X)
    @import("features/my_feature/mod.zig")
else
    @import("features/my_feature/stub.zig");

// String in struct
const response = ProfileResponse{
    .content = try allocator.dupe(u8, "Hello"),
    ...
};
```

## MCP Server

- Config: `.mcp.json` (root) and `zig-abi-plugin/.mcp.json`
- Binary: `./zig-out/bin/abi-mcp`
- Build: `./build.sh mcp`

## Known Pre-existing Issues

- 2 inference engine connector tests (expected failures)
- 1 auth integration test (expected failure)