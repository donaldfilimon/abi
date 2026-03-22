# zig-abi-plugin v0.9.0

Claude Code plugin for ABI Framework development. Provides smart build routing, Zig 0.16 patterns, feature module scaffolding, build troubleshooting, cross-platform build guidance, and real-time verification.

## Installation

```bash
claude --plugin-dir zig-abi-plugin
```

## Components

### Skills

| Skill | Trigger |
|-------|---------|
| `build-troubleshooting` | Build failures, linker errors, Darwin workarounds, flag validation |
| `zig-016-patterns` | Writing Zig code, compilation errors, API questions |
| `feature-scaffolding` | Creating new feature modules, mod/stub/types structure |

### Build Commands

```bash
# macOS 26.4+ (auto-relinks with Apple ld)
./build.sh lib                    # Build static library
./build.sh test --summary all     # Run tests
./build.sh --link lib             # Build + symlink zig/zls to ~/.local/bin

# Linux / older macOS
zig build                         # Build static library
zig build test --summary all      # Run tests
zig build check                   # Lint + test + parity (full gate)

# Cross-platform
tools/crossbuild.sh linux         # Build for Linux
tools/crossbuild.sh --all         # Build all platforms
tools/crossbuild.sh --list        # List available targets

# Version management
tools/zigup.sh --status           # Auto-install zig if missing
tools/zigup.sh --link             # Symlink zig + zls to ~/.local/bin
tools/zigup.sh --update           # Check for newer zig
```

### Build Steps (build.zig)

| Step | Purpose |
|------|---------|
| `zig build` / `zig build lib` | Build static library |
| `zig build test --summary all` | Run tests (src/ + test/) |
| `zig build check` | Lint + test + parity |
| `zig build check-parity` | Verify mod/stub declaration parity |
| `zig build cross-check` | Cross-compilation (linux, wasi, x86_64) |
| `zig build lint` | Check formatting |
| `zig build fix` | Auto-format in place |
| `zig build doctor` | Report feature flags, GPU config, platform |

### Hooks

| Event | Action |
|-------|--------|
| `PreToolUse` (Edit/Write) | Blocks `@import("abi")` inside `src/features/` |
| `PostToolUse` (Edit/Write) | Auto-formats .zig files, warns about stub.zig sync |
| `PreToolUse` (.zigversion) | Warns to update build.zig.zon and CI atomically |

## Feature Flags

The ABI Framework uses comptime feature gating with 19 feature flags. All flags default to `true` except `feat-mobile` (defaults to `false`). Disable with `-Dfeat-<name>=false`.

## Platform Notes

On macOS 26.4+ (Darwin 25+), the stock Zig linker fails. Use `./build.sh` which auto-relinks with Apple's native linker. `tools/zigup.sh` auto-downloads the correct zig version matching `.zigversion`.
