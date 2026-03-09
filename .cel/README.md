# .cel — Patchable Zig Toolchain Fork

`.cel` (Custom Environment Linker) provides a way to build a patched Zig compiler
from source, targeting the macOS 26 (Tahoe) linker incompatibilities that prevent
the stock Zig 0.16-dev binary from linking on this platform.

## Problem

macOS 26+ introduced changes that break Zig's self-hosted Mach-O linker:
- `__availability_version_check` / `_arc4random_buf` undefined symbol errors
- `__CONST_ZIG` segment ordering issues (upstream #25521)
- SDK version detection mismatches in `BUILD_VERSION` load commands

## Quick Start

### One-command migration (recommended)

```bash
./tools/scripts/cel_migrate.sh
```

This will check prerequisites, build Zig and ZLS, activate them, and validate.
On macOS 26+, `.cel/build.sh` now prefers a repo-local bootstrap-host Zig
from `zig-bootstrap-emergency/out/host/bin/zig` when available so the stage3
build runner does not depend on the broken prebuilt native linker path.

### Manual build + activate

```bash
# Build the patched toolchain
./.cel/build.sh

# Activate in current shell
eval "$(./tools/scripts/use_cel.sh)"

# Verify
zig version
zls --version
zig build full-check
```

### Build system integration

The ABI build system has native CEL support:

```bash
zig build cel-check     # Quick platform & toolchain status
zig build cel-doctor    # Full diagnostics with remediation
zig build cel-status    # Detailed source/patch/binary info
zig build cel-verify    # Verify CEL Zig/ZLS status
zig build cel-build     # Trigger CEL build from zig build
```

### Build options

```bash
./.cel/build.sh --clean       # Wipe source and rebuild from scratch
./.cel/build.sh --patch-only  # Clone + apply patches, skip build
./.cel/build.sh --verify      # Print current .cel/bin/zig + .cel/bin/zls status
./.cel/build.sh --status      # Show source, patches, binary, and version info
./.cel/build.sh --zig-only    # Build only CEL Zig
./.cel/build.sh --zls-only    # Build only ZLS using .cel/bin/zig
```

Set `CMAKE_JOBS=N` to control parallel build jobs (defaults to nproc/2).

## Adding Patches

1. Create a `.patch` file in `.cel/patches/` with a numeric prefix for ordering:
   ```
   .cel/patches/003-my-fix.patch
   ```

2. Patches are applied in lexicographic order (`001-...`, `002-...`, etc.).

3. Generate patches against the cloned source:
   ```bash
   cd .cel/.src
   # make your changes
   git diff > ../patches/003-my-fix.patch
   ```

### Current patches

| Patch | Purpose |
|-------|---------|
| `001-darwin26-force-lld.patch` | Force LLVM backend on Darwin 26+ hosts |
| `002-sdk-version-clamp.patch` | Force LLD on Darwin 26+ even if use_llvm=false |
| `003-macho-segment-ordering.patch` | Restore synthetic `__*_ZIG` segment load-command ordering to match vmaddr order |

## Updating the Upstream Pin

1. Edit `.cel/config.sh` and update:
   - `ZIG_UPSTREAM_COMMIT` — the new commit hash
   - `ZIG_VERSION` — the version string (should match `.zigversion`)

2. Update the root `.zigversion` file to match.

3. Rebuild:
   ```bash
   ./.cel/build.sh --clean
   ```

4. Re-test patches — they may need rebasing against the new upstream.

## Version Consistency Contract

These files must all agree on the Zig version:
- `.zigversion`
- `.cel/config.sh` (`ZIG_VERSION`)
- `build.zig.zon` (`minimum_zig_version`)
- `tools/scripts/baseline.zig` (`zig_version`)

Run `zig build check-zig-version` to verify, or `zig build cel-doctor` for
full CEL-specific diagnostics.

## LLVM Reuse

If `zig-bootstrap-emergency/out/build-llvm-host/` exists, the build script
automatically reuses those LLVM artifacts for a static build. Otherwise, it
falls back to a compatible system LLVM (preferably Homebrew `llvm@21` on
macOS).

If `zig-bootstrap-emergency/out/host/bin/zig` exists, the build script uses
that host-built Zig as the stage3 driver on macOS 26+ instead of relying on
the upstream `zig2` build runner.

## Directory Layout

```
.cel/
  config.sh          # Upstream pin and version config
  build.sh           # Build script (executable)
  README.md          # This file
  patches/           # Patch files applied in lexicographic order
    001-*.patch
    002-*.patch
    003-*.patch
  bin/               # Build output (git-ignored)
    zig              # Patched Zig binary
    zls              # ZLS built with CEL Zig
  .src/              # Cloned upstream source (git-ignored)
  .zls-src/          # Cloned ZLS source (git-ignored)
```

## Build System Module

The CEL integration lives in `build/cel.zig` and provides:
- `detectCelStatus()` — Build-time CEL detection
- `addCelCheckStep()` — Platform status reporting
- `addCelBuildStep()` — Trigger CEL build via `zig build`
- `emitCelSuggestion()` — Contextual guidance on blocked hosts

The `cel_doctor.zig` script at `tools/scripts/cel_doctor.zig` provides
comprehensive diagnostics including prerequisite checks, version consistency
validation, and actionable remediation steps.
