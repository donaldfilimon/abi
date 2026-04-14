# AGENTS.md

Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Build Commands

| Command | When |
|---------|------|
| `./build.sh` | macOS 26.4+ (auto-relinks with Apple ld) |
| `zig build` | Linux / older macOS |
| `./build.sh cli` | CLI binary (`zig-out/bin/abi`) |
| `./build.sh mcp` | MCP server (`zig-out/bin/abi-mcp`) |

## Test Commands

| Command | Description |
|---------|-------------|
| `./build.sh test --summary all` | Full suite (macOS) |
| `zig build test --summary all` | Full suite (Linux) |
| `zig build test -- --test-filter "pattern"` | Single test |
| `zig build check-parity` | Verify mod/stub parity |

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import
2. **macOS 26.4+**: Use `./build.sh`, never `zig build` directly
3. **Mod/stub contract**: Update both together; run `zig build check-parity`
4. **Feature gates**: `if (build_options.feat_X) mod else stub`
5. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`
6. **Imports**: Explicit `.zig` extension required on all path imports

## Feature Flags

All enabled by default except `feat-mobile` and `feat-tui`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
```

## MCP Server

- Config: `.mcp.json` (root) and `zig-abi-plugin/.mcp.json`
- Binary: `./zig-out/bin/abi-mcp`
- Entry: `src/mcp_main.zig`

## Known Pre-existing Issues

- 2 inference engine connector tests (expected failures)
- 1 auth integration test (expected failure)