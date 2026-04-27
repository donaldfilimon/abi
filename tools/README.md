# ABI Tools

Helper scripts for managing the Zig toolchain and build environment.

## zigly -- Zig Version Manager

Reads `.zigversion` from the repo root, resolves the exact pinned Zig compiler, caches toolchains under `~/.zigly/versions/<version>/`, and optionally symlinks Zig and ZLS onto your PATH.

### Quick Start

```bash
# One-command project setup (recommended for new contributors)
tools/zigly --bootstrap
```

This checks prerequisites, resolves the pinned zig toolchain, installs or reuses a matching ZLS when available, symlinks the tools to `~/.local/bin`, verifies the installation, and prints platform-specific build instructions.

### Commands

| Flag | Description |
|------|-------------|
| `--status` | Print the exact pinned zig binary path (auto-installs if missing) |
| `--install` | Resolve and install the pinned zig toolchain, plus ZLS when available |
| `--update` | Check for newer zig and update `.zigversion` if available |
| `--check` | Report if update is available (no download) |
| `--link` | Symlink zig + zls into `~/.local/bin` (or `/usr/local/bin`) |
| `--unlink` | Remove zig + zls symlinks |
| `--clean` | Remove all cached zigly versions |
| `--bootstrap` | One-command project setup: prereqs + install + link + verify |
| `--doctor` | Report toolchain health diagnostics |

### Doctor

`--doctor` prints a diagnostic report covering:

- `.zigversion` value and whether the installed binary matches
- zig and zls binary paths and versions
- Whether zig is on your PATH
- Platform detection (OS/arch)
- macOS: Xcode CLI tools status
- macOS 26.4+: LLD workaround note (use `./build.sh`)
- Quick `zig build doctor` smoke test

Example output:

```
=== ABI Toolchain Doctor ===

.zigversion:  0.17.0-dev.135+9df02121d
zig binary:   /Users/you/.zigly/versions/0.17.0-dev.135+9df02121d/bin/zig  [OK]
zls binary:   /Users/you/.zigly/versions/0.17.0-dev.135+9df02121d/bin/zls  [OK]
zig on PATH:  YES (/Users/you/.local/bin/zig)
platform:     Darwin/arm64
xcode-cli:    INSTALLED
macos 26.4+:  YES -- use ./build.sh (stock LLD fails)

Build check:
zig build doctor: OK
```

### ZLS Download Strategy

The script resolves ZLS using this priority:

1. **Exact prebuilt match** -- try the exact `.zigversion` on `builds.zigtools.org` and the matching GitHub release path
2. **ZVM fallback** -- if `~/.zvm/bin/zls` exists, copy it into the pinned zigly cache
3. **Zig-only continuation** -- if no exact ZLS is available, keep the Zig toolchain usable and warn instead of failing the install

## Other Tools

| Script | Description |
|--------|-------------|
| `crossbuild.sh` | Cross-compile for linux, wasi, x86_64 targets |
| `verify_changes.sh` | Verify shell scripts syntax and DOS line endings |
| `compile_zig_codeberg.sh` | Compile Zig from source (Codeberg mirror, requires `brew install llvm`) |
| `auto_update.sh` | Check and apply updates for zig + zls |
| `feature_tests.sh` | Run feature-specific test suites |
| `lessons_review.sh` | Review lessons learned from development sessions |

## Modernization notes
- The crossbuild.sh script has been refreshed to a safe, minimal cross-build helper with a dry-run capability to avoid accidental long builds in environments without proper toolchains.
- The hf_discord_models.sh tool was rewritten to a POSIX-compliant shell, improving portability and reliability across shells.
- The crossbuild.ps1 script now includes Set-StrictMode to catch potential runtime errors early.
- Plan.md and CodeMap.md were added to improve planning, onboarding, and codebase understanding.
