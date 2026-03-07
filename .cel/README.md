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

### Build the patched toolchain

```bash
./.cel/build.sh
```

This will:
1. Clone upstream Zig at the pinned commit (matching `.zigversion`)
2. Apply all patches from `.cel/patches/` in order
3. Build Zig from source using cmake
4. Output the binary to `.cel/bin/zig`

### Use the patched toolchain

```bash
export PATH="$(pwd)/.cel/bin:$PATH"
```

Or, if a convenience script is available:

```bash
eval "$(./tools/scripts/use_cel.sh)"
```

### Build options

```bash
./.cel/build.sh --clean       # Wipe source and rebuild from scratch
./.cel/build.sh --patch-only  # Clone + apply patches, skip build
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

4. Placeholder patches (files containing only comment lines starting with `#`)
   are skipped automatically during build.

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

## LLVM Reuse

If `zig-bootstrap-emergency/out/build-llvm-host/` exists, the build script
automatically reuses those LLVM artifacts for a static build. Otherwise, it
falls back to system LLVM (e.g., Homebrew `llvm` on macOS).

## Directory Layout

```
.cel/
  config.sh          # Upstream pin and version config
  build.sh           # Build script (executable)
  README.md          # This file
  patches/           # Patch files applied in lexicographic order
    001-*.patch
    002-*.patch
  bin/               # Build output (git-ignored)
    zig              # Patched Zig binary
  .src/              # Cloned upstream source (git-ignored)
```
