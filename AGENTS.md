# AGENTS.md
## Entry Points
- abi (CLI)
  - Build: ./build.sh cli (macOS) or zig build cli (Linux)
  - Binary: zig-out/bin/abi
  - Entry: src/main.zig
- abi-mcp (MCP server)
  - Build: ./build.sh mcp
  - Binary: zig-out/bin/abi-mcp
  - Entry: src/mcp_main.zig
## Workflow
Would an agent likely miss this without help? Ramp-up: Build → Test → Check parity → Run smoke tests (macOS: ./build.sh test --summary all; Linux: zig build test --summary all).
Would an agent likely miss this without help? Verifier: After any public API change, run zig build check-parity and a targeted test subset (zig build test -- --test-filter "parity").

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

## General Next Steps
Would an agent likely miss this without help? Validate that the Build Commands and Test Commands sections align with current repository tooling (build.sh, zig, cross-platform parity checks).
Would an agent likely miss this without help? Cross-link CLAUDE.md and QWEN.md for current project context and references.
Would an agent likely miss this without help? Validation pass: cross-check CLAUDE.md, QWEN.md, README.md for alignment with AGENTS.md and ensure terminology is consistent (parity checks, feature flags).
Would an agent likely miss this without help? Optional consolidation: consider wrapping the General Next Steps in a collapsible details block if your renderer supports it.
Would an agent likely miss this without help? Keep MCP Server notes up-to-date: Config: .mcp.json (root) and zig-abi-plugin/.mcp.json, Entry: src/mcp_main.zig.
Would an agent likely miss this without help? Ensure Known Pre-existing Issues section is current and clearly flagged before work begins.
Would an agent likely miss this without help? Verify feature flag guidance examples (-Dfeat-<name>=false) are consistent with the 60-feature catalog.
