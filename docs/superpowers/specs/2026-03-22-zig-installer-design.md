# Zig Installer for ABI — Design Spec

**Date:** 2026-03-22
**Status:** Draft
**Problem:** ABI pins a Zig 0.16-dev nightly, but stock prebuilt Zig fails to link on Darwin 25+ (macOS 26.4) due to a `.tbd` stub parsing bug in Zig's embedded LLD. The project has scattered workarounds (`use_zvm_master.sh`, `emergency_bootstrap`, manual cache population) but no unified solution.

## Goals

1. Single command to install the correct pinned Zig for any supported platform
2. Automatically detect and work around the Darwin 25+ LLD bug
3. Support both a local patch-and-rebuild path and an upstream-fix tracking path
4. Replace fragmented toolchain scripts with one modular Zig tool
5. Preserve the existing `$HOME/.cache/abi-host-zig/<version>/` cache convention

## Non-Goals

- Hosting pre-built patched binaries (deferred to a future iteration)
- Replacing CI's `mlugg/setup-zig@v1` (Linux CI is unaffected)
- Supporting platforms not already in `build/link.zig`

## Architecture

### Two-Stage Bootstrap

**Stage 1 — Shell shim** (`install.sh` / `install.ps1`):
A minimal shell script (~40 lines) that requires no Zig. It:
1. Reads `.zigversion` from the repo root
2. Checks if the target version is already cached (early exit)
3. Detects OS + arch
4. Downloads stock Zig from `ziglang.org/builds` to a temp directory
5. Uses that stock Zig to compile and run the real installer
6. Prints `PATH` export instructions

**Stage 2 — Zig installer** (`tools/zig-install/`):
A modular Zig package compiled by the stock Zig from stage 1. Handles all platform-specific logic.

### Directory Structure

```
tools/zig-install/
├── build.zig            # Build the installer package
├── main.zig             # CLI entry: arg parsing, command dispatch
├── platform.zig         # OS/arch detection, download URL construction
├── cache.zig            # Canonical cache management + .meta provenance
├── patcher.zig          # LLD patch application + zig-bootstrap orchestration
├── upstream.zig         # Query ziglang.org/builds for newer nightlies
├── patches/
│   └── lld-darwin25-tbd.patch   # LLD .tbd stub parsing fix
└── shim/
    ├── install.sh       # Unix bootstrap shim (bash)
    └── install.ps1      # Windows bootstrap shim (PowerShell)
```

## CLI Interface

```bash
# Primary usage — install pinned Zig for this repo:
./tools/zig-install/shim/install.sh

# Once bootstrapped, explicit commands via the Zig installer:
zig build run -- install                    # Install pinned version (default)
zig build run -- install --force            # Re-install even if cached
zig build run -- install --use-upstream-fix # Try newer nightly with Darwin fix
zig build run -- status                     # Show toolchain state
zig build run -- clean                      # Remove old cached versions
zig build run -- clean --all                # Remove all cached versions
```

## Install Flow

```
Read .zigversion
    │
    ▼
Check cache ($HOME/.cache/abi-host-zig/<version>/bin/zig)
    │
    ├── Cached + valid + not --force ──► Done (print PATH)
    │
    ▼
Detect platform (os + arch + kernel version)
    │
    ├── NOT affected by LLD bug ──► Download stock Zig ──► Place in cache ──► Done
    │
    ▼ (Darwin 25+)
    │
    ├── --use-upstream-fix? ──► Query ziglang.org/builds ──► Download newer nightly
    │                              │                              │
    │                              │                         Run smoke test
    │                              │                              │
    │                              │                    ┌── Pass ──► Install ──► Done
    │                              │                    └── Fail ──► Warn, fall through
    │
    ▼
Validate prerequisites (cmake, llvm@21, zstd)
    │
    ▼
Clone zig-bootstrap (shallow, matching LLVM tag)
    │
    ▼
Apply patches/lld-darwin25-tbd.patch
    │
    ▼
Run bootstrap build (aarch64-macos-none)
    │
    ▼
Run smoke test (compile + link trivial program)
    │
    ├── Pass ──► Place in cache ──► Done
    └── Fail ──► Preserve build log, error with instructions
```

## Module Details

### `platform.zig` — Platform Detection

Detects:
- **OS**: linux, macos, windows, freebsd (from `builtin.os.tag`)
- **Arch**: x86_64, aarch64 (from `builtin.cpu.arch`)
- **Darwin kernel version**: parsed from `uname -r` output; major >= 25 means LLD-affected

Constructs download URLs:
```
https://ziglang.org/builds/zig-{os}-{arch}-{version}.tar.xz   (Unix)
https://ziglang.org/builds/zig-{os}-{arch}-{version}.zip      (Windows)
```

Validates downloads via SHA256 hash from `ziglang.org/builds/index.json`.

Supported platforms (matching `build/link.zig`):

| OS       | Arch              | Stock Download | LLD Bug Possible |
|----------|-------------------|----------------|-------------------|
| linux    | x86_64, aarch64   | Yes            | No                |
| macos    | aarch64, x86_64   | Yes            | Yes (Darwin 25+)  |
| windows  | x86_64            | Yes            | No                |
| freebsd  | x86_64            | Yes            | No                |

### `cache.zig` — Cache Management

Cache layout (preserves existing convention):
```
$HOME/.cache/abi-host-zig/
└── 0.16.0-dev.2962+08416b44f/
    ├── bin/zig              # The working binary
    ├── lib/                 # Zig standard library
    └── .meta                # Provenance metadata (JSON)
```

`.meta` schema:
```json
{
  "source": "stock|patched|upstream-fix",
  "version": "0.16.0-dev.2962+08416b44f",
  "installed_at": "2026-03-22T10:30:00Z",
  "sha256": "abc123...",
  "upstream_version": null,
  "patch_applied": null
}
```

The `source` field is critical for the hybrid model — when binary hosting is added later, the installer can check `.meta` to decide whether to replace a locally-patched build with an official pre-built one.

`clean` behavior:
- Default: removes all versions except the currently pinned one
- `--all`: removes the entire `abi-host-zig/` directory

### `patcher.zig` — LLD Patch & Bootstrap

Responsibilities:
1. Validate build prerequisites are installed
2. Clone `zig-bootstrap` (shallow, depth=1, matching LLVM tag for the pinned version)
3. Apply `patches/lld-darwin25-tbd.patch` via `git apply`
4. Run the bootstrap build: `./build aarch64-macos-none native`
5. Run a smoke test on the resulting binary
6. Install to cache on success

**Prerequisite validation** (Darwin):
- `cmake` — required for zig-bootstrap
- `llvm@21` — Zig 0.16-dev.2962 requires LLVM 21.x (Homebrew default is LLVM 22)
- `zstd` — required for LLVM/LLD linking

On missing prerequisites, prints:
```
Missing build prerequisites for Darwin patched build:
  brew install cmake llvm@21 zstd
```

**Build environment:**
```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm@21;/opt/homebrew/opt/zstd" \
  ...
```

**Smoke test:** Compiles and links a trivial Zig program targeting the native host:
```zig
pub fn main() void {}
```
If this links successfully, the LLD fix is working. If it fails with the characteristic undefined symbols (`_malloc_size`, `_nanosleep`), the build is no good.

**Build log preservation:** On failure, the full build log is saved to `$HOME/.cache/abi-host-zig/build.log` and its path is printed.

### `upstream.zig` — Upstream Fix Tracker

Activated by `--use-upstream-fix`. Responsibilities:
1. Fetch `ziglang.org/builds/index.json`
2. Find the latest nightly newer than the pinned version
3. Download it for the current platform
4. Run the smoke test (same as patcher)
5. If the smoke test passes, install with `source: "upstream-fix"` in `.meta`
6. Print a warning: version differs from `.zigversion`, tests may behave differently

If the smoke test fails, warn that the upstream hasn't fixed the issue yet and suggest using the patch path instead.

### `shim/install.sh` — Unix Bootstrap

~40 lines of bash. Requirements: `curl` or `wget`, `tar`, `sha256sum` or `shasum`.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(cat "$REPO_ROOT/.zigversion" | tr -d '\n')"
CACHE_DIR="$HOME/.cache/abi-host-zig/$VERSION"

# Early exit if already installed
if [ -x "$CACHE_DIR/bin/zig" ]; then
    CACHED_VER="$("$CACHE_DIR/bin/zig" version 2>/dev/null || echo "")"
    if [ "$CACHED_VER" = "$VERSION" ]; then
        echo "Zig $VERSION already installed at $CACHE_DIR/bin/zig"
        echo "export PATH=\"$CACHE_DIR/bin:\$PATH\""
        exit 0
    fi
fi

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in arm64) ARCH="aarch64" ;; esac
case "$OS" in darwin) OS="macos" ;; esac

# Download stock Zig to temp
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://ziglang.org/builds/zig-${OS}-${ARCH}-${VERSION}.tar.xz"
echo "Downloading stock Zig from $URL..."
curl -fSL "$URL" | tar -xJ -C "$TMP" --strip-components=1

# Run the installer
"$TMP/zig" build run --build-file "$REPO_ROOT/tools/zig-install/build.zig" -- install
```

### `shim/install.ps1` — Windows Bootstrap

~40 lines of PowerShell. Same flow, uses `.zip` instead of `.tar.xz`, uses `Invoke-WebRequest` for download.

## Integration with Existing Tooling

### Replaces

| Current Tool | Replacement |
|---|---|
| `tools/scripts/use_zvm_master.sh` | `install.sh` — no ZVM dependency needed |
| `tools/scripts/emergency_bootstrap` (binary) | `patcher.zig` — source-based, maintainable |
| Manual `$HOME/.cache/abi-host-zig/` population | `cache.zig` — automated with provenance tracking |

### Preserves

| Current Tool | Status |
|---|---|
| `tools/scripts/zig` shim | Kept — points to installer-managed binary via PATH |
| `tools/scripts/toolchain_doctor.zig` | `status` command absorbs its diagnostics |
| `tools/scripts/toolchain_support.zig` | Reused by `status` command initially |
| `.zigversion` | Unchanged — single source of truth |
| `$HOME/.cache/abi-host-zig/<version>/` | Unchanged — installer writes here |
| CI `mlugg/setup-zig@v1` on Ubuntu | Unchanged — Linux is unaffected |

### Future: macOS CI

A macOS CI runner could use `./tools/zig-install/shim/install.sh` to validate the Darwin path. Not in scope for v1.

## Error Handling

| Scenario | Behavior |
|---|---|
| Missing prerequisites (Darwin) | List missing packages with `brew install` command |
| Network failure | Retry up to 3 times with exponential backoff |
| Patch doesn't apply | Error with instructions to file issue or try `--use-upstream-fix` |
| Bootstrap build fails | Save log to `$HOME/.cache/abi-host-zig/build.log`, print path |
| Smoke test fails (upstream) | Warn nightly doesn't fix issue, suggest patch path |
| Unsupported platform | Error with list of supported platforms |

## Progress Output

Long-running operations (especially the ~30-60 min bootstrap build) show progress:

```
[1/5] Reading .zigversion: 0.16.0-dev.2962+08416b44f
[2/5] Darwin 25.4.0 detected — patched build required
[3/5] Cloning zig-bootstrap (shallow)...
[4/5] Applying LLD patch and building (this takes 30-60 minutes)...
[5/5] Smoke test passed — installing to cache

Zig installed to: /Users/you/.cache/abi-host-zig/0.16.0-dev.2962+08416b44f/bin/zig
Run: export PATH="$HOME/.cache/abi-host-zig/0.16.0-dev.2962+08416b44f/bin:$PATH"
```

## The LLD Patch

The patch targets Zig's LLVM fork's LLD, specifically the Mach-O `.tbd` stub file parser. On Darwin 25+, Apple's `.tbd` stubs moved to a format or location that LLD doesn't handle correctly, causing symbols like `_malloc_size`, `_nanosleep`, `_arc4random_buf` to be unresolvable.

The patch file (`patches/lld-darwin25-tbd.patch`) must:
1. Fix `.tbd` stub resolution for Darwin 25+ SDK paths
2. Apply cleanly against the LLVM tag used by the pinned Zig version
3. Not break linking on other platforms

**Research needed:** The exact patch content requires analysis of Zig's LLVM fork and the Darwin 25 `.tbd` format changes. This is a key implementation risk — if the fix is non-trivial, the `--use-upstream-fix` path becomes the primary Darwin solution until the patch is developed.

## Future Extensions

1. **Binary hosting:** Add a `donaldfilimon/abi-zig-builds` repo with pre-built patched binaries. The installer checks for hosted binaries before falling back to local build. The `.meta` `source` field enables smooth transitions.
2. **macOS CI runner:** Validate the Darwin path in CI using `install.sh`.
3. **Auto-PATH integration:** Optionally modify shell profile to add the cache to PATH permanently.
4. **Version manager mode:** Support multiple pinned versions for branch switching.
