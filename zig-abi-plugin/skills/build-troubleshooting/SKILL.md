---
name: build-troubleshooting
description: This skill diagnoses and resolves ABI build failures. Trigger when the user encounters Darwin linker errors (undefined symbols like _malloc_size, __availability_version_check), Zig version mismatch warnings, feature flag validation failures, Metal backend linking issues, "no module named 'abi'" circular import errors, stub compilation failures, format check failures, or any build step that exits non-zero. Also trigger on questions about "build fails", "linker error", "zig build broken", "wrong zig version", "flag combo", "validate-flags", "full-check fails", or "Darwin 25".
---

# Build Troubleshooting for ABI

Pinned at Zig `0.16.0-dev.2962+08416b44f` (`.zigversion`), package version `0.1.0` (`build.zig.zon`). The build.zig is **self-contained** — no external build/ modules.

On macOS 26.4+ (Darwin 25.x), use `./build.sh` which auto-relinks with Apple's native linker. On Linux / older macOS, `zig build` works directly.

Run `zig build doctor` for a quick configuration report showing all feature flags, GPU backends, and platform info.

## Darwin 25+ Linker Failure

### Symptoms

Stock prebuilt Zig's internal LLD linker fails on Darwin 25+ (macOS 26+) with undefined symbols:
```
error: undefined symbol: _malloc_size
error: undefined symbol: __availability_version_check
error: undefined symbol: _arc4random_buf
```
Compilation succeeds. Only the linking step fails. No `build.zig` workaround exists because the linker itself is broken before any build logic can intervene.

### Diagnosis

1. Check macOS version: `sw_vers -productVersion`. Darwin 25+ corresponds to macOS 26+.
2. Check which Zig is active: `which zig && zig version`. If it reports a stock prebuilt binary, it will use the internal LLD linker, which cannot produce Mach-O binaries on Darwin 25+.
3. Confirm the failure is link-only: run `zig build test --summary all` and observe that compilation completes but linking fails.

### Fix

Use `./build.sh` which auto-relinks with Apple ld + compiler_rt on macOS 26.4+:
```bash
./build.sh lib                    # Build static library
./build.sh test --summary all     # Run tests
./build.sh -Dfeat-gpu=false       # Build with flags
```

Ensure zig is installed via the version manager:
```bash
tools/zigup.sh --status    # Auto-install if missing
tools/zigup.sh --link      # Symlink to ~/.local/bin
```

Never set `use_lld = true` on macOS. LLD has zero Mach-O support.

### Fallback Verification

When the build runner cannot link, use these partial validation paths:

| Gate | Command | What it checks |
|------|---------|----------------|
| Format | `zig build lint` | Style conformance (no linking) |
| Compile-only | `zig test src/path/to/file.zig -fno-emit-bin` | Syntax + type correctness |
| Library | `./build.sh lib` | Static library (Apple ld wrapper) |

Format checks always succeed regardless of platform or linker state. Use them as the minimum verification gate on Darwin 25+ when the linker is blocked.

## Zig Version Mismatch

### Symptoms

Build output shows a version mismatch warning, or only format-check steps (`lint`, `fix`) are registered. On very old builds (dev < 2000), `build.zig` intentionally restricts available steps.

### Diagnosis

1. Check the pinned version: read `.zigversion` (should contain `0.16.0-dev.2962+08416b44f`).
2. Check the active Zig version: `zig version`.
3. Compare the two. Any mismatch triggers the detection logic in `build.zig`.

### Fix

Install the exact pinned Zig version and ensure it is first on PATH. Do not attempt to use a different Zig version with this codebase. The build system, API patterns, and all source code are tuned to the exact pinned dev build.

If upgrading the pin, update all of these atomically: `.zigversion`, `build.zig.zon`, and CI configuration (`.github/workflows/ci.yml`). Run `tools/zigup.sh --install` to re-download the new version.

## Feature Flag Validation Errors

### Symptoms

Build fails with warnings about invalid flag combinations or missing dependencies.

### Diagnosis

The build.zig defines all feature flags inline (no external modules). Each flag defaults to enabled except `feat-mobile` (defaults false). The build validates GPU backend combinations and feature dependencies at build time with warnings.

1. Check which flags are being passed: review the `zig build` invocation for `-Dfeat-*=false` or `-Dfeat-*=true` arguments.
2. Run `zig build doctor` to see the current flag configuration.
3. Cross-reference with `build.zig` to identify the invalid combination.

### Fix

Correct the flag combination. Common issues:
- Disabling a flag that another enabled flag depends on.
- Enabling `feat-mobile` without its required dependencies.
- Setting `-Dgpu-backend=metal` on non-macOS (see Metal Validation below).

Run `zig build doctor` after each change to confirm validity.

## Metal Backend Linking Failures

### Symptoms

Build fails with Metal/CoreML/MPS framework not found, or `validateMetalBackendRequest()` panics with a message about Metal unavailability.

### Diagnosis

1. Check platform: Metal is macOS/iOS only. Requesting `-Dgpu-backend=metal` on Linux or Windows will always fail.
2. Run `xcrun --show-sdk-path` to verify Xcode command-line tools are installed.
3. Platform linking is in `build.zig` — check the macOS linking block for framework declarations.

### Fix

- On macOS: install Xcode or Command Line Tools (`xcode-select --install`). Verify frameworks exist at the SDK path.
- On non-macOS: do not request `-Dgpu-backend=metal`. Use `vulkan`, `cuda`, or the default backend for the platform.

Platform linking in `build.zig` covers:
- **macOS**: System, c, objc, IOKit, Accelerate (when feat_gpu), Metal/MPS/CoreGraphics (when gpu_metal)
- **Linux**: Default Zig libc linking
- **Windows**: Default Zig linking

## Circular Import: "no module named 'abi'"

### Symptoms

Compilation fails with:
```
error: no module named 'abi' available within module 'abi'
```

### Diagnosis

A file inside `src/` is using `@import("abi")`. All files under `src/` belong to the single `abi` named module. Importing `abi` from within itself creates a circular dependency.

### Fix

Replace `@import("abi")` with a relative path import. For example:
```zig
// WRONG (inside src/)
const shared = @import("abi").foundation;
// CORRECT (inside src/)
const shared = @import("../../services/shared/mod.zig");
```

Only external code (CLI in `tools/cli/`, test roots) may use `@import("abi")`.

For cross-feature imports, never import another feature's `mod.zig` directly (bypasses the comptime gate). Use a conditional import with `build_options`:
```zig
const obs = if (build_options.feat_observability)
    @import("../../observability/mod.zig")
else
    @import("../../observability/stub.zig");
```

## Stub Compilation Failures

### Symptoms

Build fails when a feature is disabled (`-Dfeat-<name>=false`) because `stub.zig` does not match `mod.zig` signatures.

### Diagnosis

1. Compare the public API surface of `src/features/<name>/mod.zig` with `src/features/<name>/stub.zig`.
2. Look for missing functions, mismatched return types, or missing re-exports of sub-modules.
3. Run the parity check: `zig build check-parity`.

### Fix

Update `stub.zig` to match every public declaration in `mod.zig`. Shared types belong in `types.zig` (imported by both). Use `StubFeature` or `StubFeatureNoConfig` from `src/core/stub_helpers.zig` for common boilerplate. Sub-modules in mod.zig must be re-exported from stub.zig as matching structs.

Verify the fix: `zig build test -Dfeat-<name>=false --summary all`.

## Format Check Failures

### Symptoms

`zig fmt --check` fails, or pre-commit hooks reject formatting.

### Diagnosis

Never run `zig fmt .` from the repository root. The repo contains vendored fixtures with intentionally invalid code that will cause `zig fmt .` to fail or corrupt files.

### Fix

Always specify explicit paths:
```bash
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
```

After any bulk find-replace operation, run the format check immediately. Bulk replacements must exclude string literal interiors to avoid corrupting quoted text.

## Import "File Not Found" Errors

### Symptoms

Compilation fails with an import path not found error after adding a `.zig` suffix or referencing a new file.

### Diagnosis

1. Verify the target file actually exists at the expected path.
2. Check for gated imports: the file might only be available when a feature flag is enabled.
3. Confirm the import path depth matches the source file's location in the directory tree.
4. Ensure explicit `.zig` extensions are present on all path imports (required in Zig 0.16 / dev.2962+).

### Fix

Correct the path to match the actual file location. Count directory levels carefully for relative imports. If the file is behind a feature gate, use a conditional import with `build_options`. If a file was recently moved or renamed, update all import sites.

## Unmanaged Collection Initialization

### Symptoms

Compilation fails with:
```
error: missing struct field: items
```

### Diagnosis

Zig 0.16 changed initialization for `ArrayListUnmanaged` and `AutoHashMapUnmanaged`. The old `.{}` initialization no longer works.

### Fix

Use `.empty` instead of `.{}`:
```zig
// WRONG
var list: std.ArrayListUnmanaged(u8) = .{};
// CORRECT
var list: std.ArrayListUnmanaged(u8) = .empty;
```

## Build Step Reference

Quick reference for available build steps and when to use each:

| Step | Purpose | Requires linking |
|------|---------|-----------------|
| `./build.sh` / `zig build` | Build static library (default) | No (build.sh uses Apple ld) |
| `./build.sh lib` / `zig build lib` | Build static library artifact | No |
| `./build.sh test --summary all` | Run tests (src/ + test/) | Yes |
| `zig build check` | Lint + test + parity (full gate) | Yes |
| `zig build check-parity` | Verify mod/stub declaration parity | Yes (compile only) |
| `zig build cross-check` | Verify cross-compilation (linux, wasi, x86_64) | No |
| `zig build lint` | Check formatting (read-only) | No |
| `zig build fix` | Auto-format in place | No |
| `zig build doctor` | Report feature flags, GPU config, platform | No |
| `./build.sh --link lib` | Build + symlink zig/zls to ~/.local/bin | No |

When the linker is blocked (Darwin 25+), use `./build.sh` which auto-relinks with Apple ld.

## build.sh AUTO_LINK Behavior

The `--link` flag in `build.sh` is opt-in. If `build.sh` always symlinks zig/zls into `~/.local/bin` even without `--link`, check that the `AUTO_LINK` variable defaults to `false` at the top of the script. The intended behavior is:
- `./build.sh` — build only, no symlinks
- `./build.sh --link lib` — build and symlink zig/zls to ~/.local/bin

## Escalation Checklist

When a build failure does not match any symptom above:

1. Run `zig build doctor` to inspect the build configuration.
2. Check `.zigversion` for version pin consistency.
3. Run `tools/zigup.sh --status` to verify zig installation.
4. Check `build.zig` for the full flag list (all flags defined inline).
5. Check `src/core/feature_catalog.zig` for feature metadata.
