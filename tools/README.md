# ABI Tools

Helper scripts for managing the Zig toolchain and build environment.

## zigup.sh -- Zig Version Manager

Reads `.zigversion` from the repo root, downloads the matching Zig compiler and ZLS language server to `~/.cache/abi-zig/<version>/`, and optionally symlinks them onto your PATH.

### Quick Start

```bash
# One-command project setup (recommended for new contributors)
tools/zigup.sh --bootstrap
```

This checks prerequisites, downloads zig + zls, symlinks them to `~/.local/bin`, verifies the installation, and prints platform-specific build instructions.

### Commands

| Flag | Description |
|------|-------------|
| `--status` | Print path to correct zig binary (auto-installs if missing) |
| `--install` | Force (re-)download and install zig + zls |
| `--update` | Check for newer zig and update `.zigversion` if available |
| `--check` | Report if update is available (no download) |
| `--link` | Symlink zig + zls into `~/.local/bin` (or `/usr/local/bin`) |
| `--unlink` | Remove zig + zls symlinks |
| `--clean` | Remove all cached abi-zig versions |
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

.zigversion:  0.16.0-dev.2979+e93834410
zig binary:   /Users/you/.cache/abi-zig/0.16.0-dev.2979+e93834410/bin/zig  [OK]
zls binary:   /Users/you/.cache/abi-zig/0.16.0-dev.2979+e93834410/bin/zls  [OK (0.16.0-dev.1)]
zig on PATH:  YES (/Users/you/.local/bin/zig)
platform:     Darwin/arm64
xcode-cli:    INSTALLED
macos 26.4+:  YES -- use ./build.sh (stock LLD fails)

Build check:
zig build doctor: OK
```

### ZLS Download Strategy

The script resolves the best ZLS version using this priority:

1. **zvm copy** -- if `~/.zvm/bin/zls` exists, copy it (best for dev zig versions)
2. **GitHub API** -- query `zigtools/zls` releases for the latest tag matching your zig major.minor
3. **Hardcoded fallback** -- try known versions (`0.16.0-dev.1`, `0.15.0`, `0.14.0`)
4. **Manual install** -- print instructions for `zvm install master` or building from source

The GitHub API query requires `python3` for JSON parsing. If python3 is not available, the script skips directly to the hardcoded fallback list.

## Other Tools

| Script | Description |
|--------|-------------|
| `crossbuild.sh` | Cross-compile for linux, wasi, x86_64 targets |
| `compile_zig_codeberg.sh` | Compile Zig from source (Codeberg mirror, requires `brew install llvm`) |
| `auto_update.sh` | Check and apply updates for zig + zls |
| `feature_tests.sh` | Run feature-specific test suites |
| `lessons_review.sh` | Review lessons learned from development sessions |
