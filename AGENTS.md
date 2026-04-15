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

## Entry Points

| Target | Build Command | Binary | Source |
|--------|--------------|--------|--------|
| CLI | `./build.sh cli` (macOS 26.4+) or `zig build cli` (Linux) | `zig-out/bin/abi` | `src/main.zig` |
| MCP server | `./build.sh mcp` | `zig-out/bin/abi-mcp` | `src/mcp_main.zig` |

## Build Commands

**macOS 26.4+ (Darwin 25.x)**: Always use `./build.sh`, never `zig build` directly — stock Zig's LLD cannot link. The wrapper auto-relinks with Apple's native linker.

**Linux / older macOS**: Use `zig build` directly.

| Command | Description |
|---------|-------------|
| `./build.sh` / `zig build` | Build static library |
| `./build.sh test --summary all` / `zig build test --summary all` | Run all tests |
| `zig build test -- --test-filter "pattern"` | Run single test |
| `./build.sh check` / `zig build check` | Lint + test + stub parity |
| `zig build check-parity` | Verify mod/stub declaration parity |
| `zig build fix` | Auto-format |
| `zig build cli` | Build CLI binary |
| `zig build mcp` | Build MCP server |
| `zig build feature-tests` | Run feature integration + parity tests |
| `zig build full-check` | Full validation gate |

### Focused Test Lanes (27 total)

Run unit + integration tests for specific features:

```bash
zig build {messaging,secrets,pitr,agents,multi-agent,orchestration,gateway,inference,gpu,network,web,observability,search,auth,storage,cloud,cache,database,connectors,lsp,acp,ha,tasks,documents,compute,desktop,pipeline}-tests
```

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import. Use relative imports only.
2. **macOS 26.4+**: Use `./build.sh`, never `zig build` directly.
3. **Mod/stub contract**: Every feature has `mod.zig` (real), `stub.zig` (no-ops), `types.zig` (shared). Update both `mod.zig` and `stub.zig` together for any public API change.
4. **After any public API change**: Run `zig build check-parity` before committing.
5. **Feature gates**: Use pattern `if (build_options.feat_X) @import("features/X/mod.zig") else @import("features/X/stub.zig")`.
6. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`.
7. **Imports**: Explicit `.zig` extension required on all path imports.

## Feature Flags

All enabled by default except `feat-mobile` and `feat-tui`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

## MCP Server

- Config: `.mcp.json` (root) and `zig-abi-plugin/.mcp.json`
- Binary: `./zig-out/bin/abi-mcp`
- Entry: `src/mcp_main.zig`

## Known Pre-existing Issues

- 2 inference engine connector tests (expected failures)
- 1 auth integration test (expected failure)

## Toolchain

- **Zig version**: Pinned in `.zigversion` (currently `0.16.0`)
- **Zig manager**: `tools/zigly` — prefers `~/.zvm/bin/zig` when version matches

## See Also

- `CLAUDE.md` — Detailed architecture, conventions, and agent/skill references
- `QWEN.md` — Quick reference and Zig 0.16 gotchas
