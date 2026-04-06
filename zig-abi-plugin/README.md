# zig-abi-plugin v0.11.0

Claude Code plugin for ABI Framework development. Provides smart build routing, Zig 0.16 patterns, feature module scaffolding, pipeline DSL guidance, cross-platform builds, and real-time verification.

## Installation

```bash
claude --plugin-dir zig-abi-plugin
```

## Components

### Skills (5)

| Skill | Trigger |
|-------|---------|
| `build-troubleshooting` | Build failures, linker errors, Darwin workarounds, flag validation |
| `zig-016-patterns` | Writing Zig code, compilation errors, removed API usage |
| `feature-scaffolding` | Creating new feature modules, mod/stub/types structure |
| `cross-check` | Cross-compilation targets, platform feature matrix |
| `pipeline-dsl` | Pipeline DSL usage, step wiring, WDBX persistence, memory safety |

### Agents (4)

| Agent | Purpose |
|-------|---------|
| `build-troubleshooter` | Diagnoses build failures — linker errors, version mismatch, feature flags |
| `feature-scaffolder` | Scaffolds complete new features (mod.zig, stub.zig, types.zig, build wiring) |
| `parity-checker` | Compares mod/stub public API, reports missing declarations |
| `pipeline-auditor` | Audits pipeline code for memory safety, ownership bugs, Zig 0.16 violations |

### Hooks (4)

| Event | Hook | Action |
|-------|------|--------|
| PreToolUse (Edit/Write) | `check_abi_import.sh` | Blocks `@import("abi")` inside `src/` (prevents circular imports) |
| PreToolUse (Edit/Write) | `check_zigversion.sh` | Warns on `.zigversion` edits to update build.zig.zon and CI atomically |
| PostToolUse (Edit/Write) | prompt | Flags mod.zig public API changes that need stub.zig sync |
| PostToolUse (Write) | `check_test_mod.sh` | Reminds to wire new integration tests into test/mod.zig |

## Build Commands

```bash
# macOS 26.4+ (auto-relinks with Apple ld)
./build.sh test --summary all     # Run tests
./build.sh lib                    # Build static library
./build.sh cli                    # Build CLI binary
./build.sh mcp                    # Build MCP server
./build.sh --link lib             # Build + symlink zig/zls to ~/.local/bin

# Linux / older macOS
zig build test --summary all      # Run tests
zig build check                   # Lint + test + parity (full gate)
zig build check-parity            # Verify mod/stub declaration parity
zig build cross-check             # Cross-compilation (linux, wasi, x86_64)

# Focused test lanes (27 total)
zig build pipeline-tests          # Pipeline DSL tests
zig build inference-tests         # Inference engine tests
zig build database-tests          # Database tests
zig build gpu-tests               # GPU backend tests
# ... see CLAUDE.md for full list

# Version management
zigly --bootstrap        # One-command project setup
zigly --status           # Auto-install zig if missing
zigly --link             # Symlink zig + zls to ~/.local/bin
```

## Feature Flags

60 features total (including AI sub-features). All default to `true` except `feat-mobile` and `feat-tui`. Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
```

## Platform Notes

- **macOS 26.4+** (Darwin 25+): Stock Zig linker fails. Use `./build.sh` which auto-relinks with Apple's native linker.
- **Zig version**: Pinned in `.zigversion` (`0.16.0-dev.3091+557caecaa`). `zigly` auto-downloads the correct version.
- **WASM**: Timer APIs unavailable; foundation time utilities return 0.
