# Zig Installer for ABI — Design Spec

**Date:** 2026-03-22
**Status:** Draft (rev 2 — post-review fixes)
**Problem:** ABI pins a Zig 0.16-dev nightly, but stock prebuilt Zig fails to link on Darwin 25+ (macOS 26.4) due to a `.tbd` stub parsing bug in Zig's embedded LLD. The project has scattered workarounds (`use_zvm_master.sh`, `emergency_bootstrap`, manual cache population) but no unified solution.

## Goals

1. Single command to install the correct pinned Zig for any supported platform
2. Automatically detect and work around the Darwin 25+ LLD bug
3. Support an upstream-fix tracking path as the primary Darwin strategy, with a speculative local patch-and-rebuild path as a secondary option
4. Replace fragmented toolchain scripts with one cohesive tool
5. Preserve the existing `$HOME/.cache/abi-host-zig/<version>/` cache convention

## Non-Goals

- Hosting pre-built patched binaries (deferred to a future iteration)
- Replacing CI's `mlugg/setup-zig@v1` (Linux CI is unaffected)
- Platforms beyond the installer support table (illumos, Haiku, Android, NetBSD, OpenBSD, DragonFly are supported by `build/link.zig` for cross-compilation but not as installer host platforms)

## Architecture

### Platform-Adaptive Bootstrap

The installer uses different strategies depending on whether stock Zig can link binaries on the host platform.

**Unaffected platforms (Linux, Windows, FreeBSD, macOS < Darwin 25):**
Two-stage bootstrap — a shell shim downloads stock Zig, then compiles and runs the Zig installer package for cache management and status commands.

**Darwin 25+ (LLD-affected):**
The shell shim handles the entire Darwin path directly in bash, because stock Zig cannot link any binary on the affected platform — including the installer itself. This avoids the chicken-and-egg problem. The Zig installer modules (`platform.zig`, `cache.zig`, etc.) are only compiled and used on platforms where stock Zig works.

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
│   └── lld-darwin25-tbd.patch   # LLD .tbd stub parsing fix (speculative)
└── shim/
    ├── install.sh       # Unix bootstrap shim (bash) — handles Darwin path natively
    ├── install.ps1      # Windows bootstrap shim (PowerShell)
    └── darwin_fix.sh    # Darwin-specific logic sourced by install.sh
```

## CLI Interface

```bash
# Primary usage — install pinned Zig for this repo:
./tools/zig-install/shim/install.sh

# With flags:
./tools/zig-install/shim/install.sh --force              # Re-install even if cached
./tools/zig-install/shim/install.sh --use-upstream-fix    # Try newer nightly with Darwin fix

# On unaffected platforms, once Zig is installed, the Zig-native CLI is also available:
cd tools/zig-install && zig build run -- status           # Show toolchain state
cd tools/zig-install && zig build run -- clean             # Remove old cached versions
cd tools/zig-install && zig build run -- clean --all       # Remove all cached versions
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
    ├── NOT affected by LLD bug ──► Download stock Zig ──► SHA256 verify
    │                                    │
    │                                    ▼
    │                              Place in cache ──► Done
    │
    ▼ (Darwin 25+, handled entirely in bash)
    │
    ├── --use-upstream-fix? ──► Query ziglang.org/builds/index.json
    │                              │
    │                              ▼
    │                         Download newer nightly (same minor version)
    │                              │
    │                         Run smoke test (compile + link trivial .zig)
    │                              │
    │                    ┌── Pass ──► Install to cache ──► Done
    │                    └── Fail ──► Warn, fall through to patch path
    │
    ▼ (speculative patch path — requires validated patch)
    │
    ├── Patch file exists and prerequisites met?
    │   │
    │   ├── No ──► Error: "No validated LLD patch available.
    │   │          Use --use-upstream-fix or wait for upstream Zig fix."
    │   │
    │   ▼ Yes
    │
    ▼
Validate prerequisites (cmake, llvm@21, zstd)
    │
    ▼
Clone zig-bootstrap (shallow, depth=1)
    │
    ▼
Apply patches/lld-darwin25-tbd.patch via git apply
    │
    ▼
Run zig-bootstrap's ./build script (${ARCH}-macos-none native)
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

Supported installer host platforms:

| OS       | Arch              | Stock Download | LLD Bug Possible |
|----------|-------------------|----------------|-------------------|
| linux    | x86_64, aarch64   | Yes            | No                |
| macos    | aarch64, x86_64   | Yes            | Yes (Darwin 25+)  |
| windows  | x86_64            | Yes            | No                |
| freebsd  | x86_64            | Yes            | No                |

Note: `build/link.zig` supports additional platforms (illumos, Haiku, Android, BSD variants) for cross-compilation targets, but these are not installer host platforms.

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
  "sha256": "abc123def456...",
  "upstream_version": "0.16.0-dev.3010+...",
  "patch_applied": "lld-darwin25-tbd.patch"
}
```

Field types:
- `source`: string enum — `"stock"`, `"patched"`, or `"upstream-fix"`
- `version`: string — the `.zigversion` pin this was installed for
- `installed_at`: string — ISO 8601 timestamp
- `sha256`: string — hex SHA256 of the downloaded/built `zig` binary
- `upstream_version`: string or null — if `source` is `"upstream-fix"`, the actual nightly version installed
- `patch_applied`: string or null — if `source` is `"patched"`, the patch filename used

The `source` field is critical for the hybrid model — when binary hosting is added later, the installer can check `.meta` to decide whether to replace a locally-patched build with an official pre-built one.

`clean` behavior:
- Default: removes all versions except the currently pinned one
- `--all`: removes the entire `abi-host-zig/` directory

### `patcher.zig` — LLD Patch & Bootstrap (Speculative)

**Status:** This module is speculative. The LLD patch does not yet exist. Per the research in `docs/ZIG_MACOS_LINKER_RESEARCH.md`, building Zig from source alone does not fix the Darwin 25+ issue because stage 3 uses the bootstrap's embedded LLD. The patch must target LLD's Mach-O `.tbd` stub parser *before* the bootstrap builds the final Zig binary. Until a validated patch exists, `--use-upstream-fix` is the primary Darwin strategy.

**If/when a patch is developed**, this module's responsibilities are:
1. Validate build prerequisites are installed
2. Clone `zig-bootstrap` (shallow, depth=1)
3. Apply `patches/lld-darwin25-tbd.patch` via `git apply`
4. Run `./build ${ARCH}-macos-none native` (zig-bootstrap's own build script — not cmake directly; the `./build` script invokes cmake internally with the correct flags)
5. Run a smoke test on the resulting binary
6. Install to cache on success

**Prerequisite validation** (Darwin):
- `cmake` — required by zig-bootstrap internally
- `llvm@21` — Zig 0.16-dev.2962 requires LLVM 21.x (Homebrew default is LLVM 22)
- `zstd` — required for LLVM/LLD linking
- ~10 GB free disk space

On missing prerequisites, prints:
```
Missing build prerequisites for Darwin patched build:
  brew install cmake llvm@21 zstd
```

**Smoke test:** Compiles and links a trivial Zig program targeting the native host:
```zig
pub fn main() void {}
```
If this links successfully, the LLD fix is working. If it fails with the characteristic undefined symbols (`_malloc_size`, `_nanosleep`), the build is no good.

**Build log preservation:** On failure, the full build log is saved to `$HOME/.cache/abi-host-zig/build.log` and its path is printed.

**Interrupt handling:** The build runs in a temp directory. Only on success is the result copied to the cache. Ctrl-C during the build leaves no partial state in the cache — the trap handler removes the temp directory.

### `upstream.zig` — Upstream Fix Tracker

Activated by `--use-upstream-fix`. This is the **primary Darwin 25+ strategy**.

Responsibilities:
1. Fetch `ziglang.org/builds/index.json`
2. Find the latest 0.16.x nightly newer than the pinned version (same minor version to limit API breakage risk)
3. Download it for the current platform
4. Run the smoke test (same as patcher — compile + link a trivial program)
5. If the smoke test passes, install with `source: "upstream-fix"` in `.meta`
6. Print a warning: version differs from `.zigversion`, tests may behave differently

Version constraint: only considers nightlies matching `0.16.0-dev.*` to avoid major API breakage from a hypothetical 0.17.x release.

If the smoke test fails, warn that the upstream hasn't fixed the issue yet and suggest waiting or checking for a validated LLD patch.

### `shim/install.sh` — Unix Bootstrap

Requirements: `bash`, `curl` or `wget`, `tar`, `sha256sum` or `shasum`.

The shim has two code paths:

**Path A — Unaffected platforms (~50 lines):**
```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERSION="$(cat "$REPO_ROOT/.zigversion" | tr -d '\n')"
CACHE_DIR="$HOME/.cache/abi-host-zig/$VERSION"

# Early exit if already installed
if [ -x "$CACHE_DIR/bin/zig" ]; then
    CACHED_VER="$("$CACHE_DIR/bin/zig" version 2>/dev/null || echo "")"
    if [ "$CACHED_VER" = "$VERSION" ] && [ "${1:-}" != "--force" ]; then
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

# Check if Darwin 25+ (LLD-affected)
if [ "$OS" = "macos" ]; then
    KERN_MAJOR="$(uname -r | cut -d. -f1)"
    if [ "$KERN_MAJOR" -ge 25 ] 2>/dev/null; then
        source "$(dirname "${BASH_SOURCE[0]}")/darwin_fix.sh"
        darwin_install "$VERSION" "$CACHE_DIR" "$ARCH" "$@"
        exit $?
    fi
fi

# Download stock Zig to temp
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://ziglang.org/builds/zig-${OS}-${ARCH}-${VERSION}.tar.xz"
echo "[1/3] Downloading stock Zig from $URL..."
curl -fSL "$URL" -o "$TMP/zig.tar.xz"

# Verify SHA256
echo "[2/3] Verifying SHA256..."
EXPECTED_HASH="$(curl -fsSL "https://ziglang.org/builds/index.json" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['${VERSION}']['${OS}-${ARCH}']['shasum'])" 2>/dev/null || echo "")"
if [ -n "$EXPECTED_HASH" ]; then
    ACTUAL_HASH="$(shasum -a 256 "$TMP/zig.tar.xz" | cut -d' ' -f1)"
    if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
        echo "error: SHA256 mismatch" >&2; exit 1
    fi
fi

tar -xJf "$TMP/zig.tar.xz" -C "$TMP" --strip-components=1

# Install to cache
echo "[3/3] Installing to $CACHE_DIR..."
mkdir -p "$CACHE_DIR"
cp -r "$TMP/zig" "$TMP/lib" "$CACHE_DIR/" 2>/dev/null || cp -r "$TMP"/* "$CACHE_DIR/"

echo "Zig $VERSION installed."
echo "export PATH=\"$CACHE_DIR/bin:\$PATH\""
```

**Path B — Darwin 25+ (`darwin_fix.sh`, sourced by `install.sh`):**
Handles upstream-fix download + smoke test, and speculative patch-and-rebuild, entirely in bash. ~100 lines covering:
- `--use-upstream-fix`: download newer 0.16.x nightly, smoke test, install
- Speculative patch path: validate prerequisites, clone zig-bootstrap, apply patch, build, smoke test
- Temp directory cleanup on interrupt (trap)

### `shim/install.ps1` — Windows Bootstrap

~50 lines of PowerShell. Same flow as Path A (stock Zig download), using `.zip` instead of `.tar.xz` and `Invoke-WebRequest` for download. No Darwin path needed.

## Integration with Existing Tooling

### Replaces

| Current Tool | Replacement |
|---|---|
| `tools/scripts/use_zvm_master.sh` | `install.sh` — no ZVM dependency needed |
| `tools/scripts/emergency_bootstrap` (binary) | `darwin_fix.sh` patch path — source-based, maintainable |
| Manual `$HOME/.cache/abi-host-zig/` population | Automated by `install.sh` with provenance tracking |

### Preserves

| Current Tool | Status |
|---|---|
| `tools/scripts/zig` shim | Kept — points to installer-managed binary via PATH |
| `tools/scripts/toolchain_doctor.zig` | `status` command wraps `toolchain_support.zig`'s `inspect()` function |
| `tools/scripts/toolchain_support.zig` | Reused — `status` calls `inspect()`, does not reimplement |
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
| SHA256 mismatch | Hard error — do not install |
| Patch doesn't apply | Error: "No validated LLD patch available. Use --use-upstream-fix or wait for upstream Zig fix." |
| Bootstrap build fails | Save log to `$HOME/.cache/abi-host-zig/build.log`, print path |
| Smoke test fails (upstream) | Warn nightly doesn't fix issue yet |
| Disk space < 10 GB (Darwin build) | Warn before starting the 30-60 minute build |
| Unsupported platform | Error with list of supported platforms |
| Ctrl-C during build | Trap removes temp directory; cache is never left in partial state |

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

**Status: Speculative — research needed before implementation.**

The patch would target Zig's LLVM fork's LLD, specifically the Mach-O `.tbd` stub file parser. On Darwin 25+, Apple's `.tbd` stubs moved to a format or location that LLD doesn't handle correctly, causing symbols like `_malloc_size`, `_nanosleep`, `_arc4random_buf` to be unresolvable.

Per `docs/ZIG_MACOS_LINKER_RESEARCH.md`, building Zig from source without patching does NOT fix the issue — the C++ bootstrap (`zig2`) links fine with system clang/ld, but stage 3 uses `zig2`'s embedded LLD which has the same bug. The patch must fix LLD *before* the bootstrap builds stage 3.

The patch file (`patches/lld-darwin25-tbd.patch`) would need to:
1. Fix `.tbd` stub resolution for Darwin 25+ SDK paths
2. Apply cleanly against the LLVM tag used by the pinned Zig version's zig-bootstrap
3. Not break linking on other platforms

**Implementation risk:** If the fix is non-trivial or requires deep LLVM/LLD expertise, `--use-upstream-fix` remains the primary Darwin solution until either (a) the patch is developed and validated, or (b) upstream Zig fixes the issue in a newer nightly.

## Future Extensions

1. **Binary hosting:** Add a `donaldfilimon/abi-zig-builds` repo with pre-built patched binaries. The installer checks for hosted binaries before falling back to local build. The `.meta` `source` field enables smooth transitions.
2. **macOS CI runner:** Validate the Darwin path in CI using `install.sh`.
3. **Auto-PATH integration:** Optionally modify shell profile to add the cache to PATH permanently.
4. **Version manager mode:** Support multiple pinned versions for branch switching.
