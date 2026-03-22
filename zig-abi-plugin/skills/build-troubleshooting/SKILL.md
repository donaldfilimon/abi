---
name: build-troubleshooting
description: This skill diagnoses and resolves ABI build failures. Trigger when the user encounters Darwin linker errors (undefined symbols like _malloc_size, __availability_version_check), Zig version mismatch warnings, feature flag validation failures, Metal backend linking issues, "no module named 'abi'" circular import errors, stub compilation failures, format check failures, or any build step that exits non-zero. Also trigger on questions about "build fails", "linker error", "zig build broken", "wrong zig version", "flag combo", "validate-flags", "full-check fails", or "Darwin 25".
---

# Build Troubleshooting for ABI

Pinned at Zig `0.16.0-dev.2962+08416b44f` (`.zigversion`), package version `0.4.0` (`build.zig.zon`). All legacy build wrappers (`build.sh`, `build/compat.zig`, `build/darwin.zig`, `tools/scripts/run_build.sh`, `tools/scripts/bootstrap_host_zig.sh`, `tools/scripts/zig_toolchain.sh`, `tools/scripts/zig_darwin26_wrapper.sh`) have been removed. The only supported build path is direct `zig build` with the pinned Zig on PATH.

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

Use a host-built or known-good Zig matching the `.zigversion` pin (`0.16.0-dev.2962+08416b44f`) that uses the system linker instead of LLD. Ensure this Zig is first on PATH.

Never set `use_lld = true` on macOS. LLD has zero Mach-O support.

### Fallback Verification

When no working linker is available, use these partial validation paths:

| Gate | Command | What it checks |
|------|---------|----------------|
| Format | `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` | Style conformance (always works, no linking) |
| Compile-only | `zig test src/path/to/file.zig -fno-emit-bin` | Syntax + type correctness (no binary, no link) |
| Linux CI | `zig build full-check` on Linux | Full binary gate (working linker available) |

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

If upgrading the pin, update all of these atomically: `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, and CI configuration.

## Feature Flag Validation Errors

### Symptoms

`zig build validate-flags` fails, or `full-check` fails at the flag validation step. Error messages reference invalid flag combinations.

### Diagnosis

The build system defines 27 feature flags in `build/options.zig` and validates 58 flag combinations in `build/flags.zig` via `CanonicalFlags`. Each flag defaults to enabled except `feat-mobile` (defaults false).

1. Check which flags are being passed: review the `zig build` invocation for `-Dfeat-*=false` or `-Dfeat-*=true` arguments.
2. Run `zig build validate-flags` in isolation to see the specific failure.
3. Cross-reference with `build/flags.zig` to identify the invalid combination.

### Fix

Correct the flag combination. Common issues:
- Disabling a flag that another enabled flag depends on.
- Enabling `feat-mobile` without its required dependencies.
- Setting `-Dgpu-backend=metal` on non-macOS (see Metal Validation below).

Run `zig build validate-flags` after each change to confirm validity.

## Metal Backend Linking Failures

### Symptoms

Build fails with Metal/CoreML/MPS framework not found, or `validateMetalBackendRequest()` panics with a message about Metal unavailability.

### Diagnosis

1. Check platform: Metal is macOS/iOS only. Requesting `-Dgpu-backend=metal` on Linux or Windows will always fail.
2. Run `xcrun --show-sdk-path` to verify Xcode command-line tools are installed.
3. Check framework paths: `build/link.zig` uses `canLinkMetalFrameworks()` which verifies both xcrun availability and framework directory existence.

### Fix

- On macOS: install Xcode or Command Line Tools (`xcode-select --install`). Verify frameworks exist at the SDK path.
- On non-macOS: do not request `-Dgpu-backend=metal`. Use `vulkan`, `cuda`, or the default backend for the platform.

Platform linking in `build/link.zig` via `applyAllPlatformLinks()` covers:
- **macOS/iOS**: Accelerate, Foundation, Metal, CoreML, MPS, AppKit/UIKit
- **Linux**: libc, libm, CUDA (libcuda/cublas/cudart/cudnn), Vulkan, OpenGL
- **Windows**: CUDA, Vulkan
- **BSD**: Vulkan, OpenGL
- **Android**: log, android, EGL, GLESv2
- **illumos**: socket, nsl, OpenGL (Mesa)
- **Haiku**: OpenGL

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
const obs = if (build_options.feat_profiling)
    @import("../../observability/mod.zig")
else
    @import("../../observability/stub.zig");
```

## Stub Compilation Failures

### Symptoms

Build fails when a feature is disabled (`-Dfeat-<name>=false`) because `stub.zig` does not match `mod.zig` signatures.

### Diagnosis

1. Compare the public API surface of `src/features/<name>/mod.zig` with `src/features/<name>/stub.zig`.
2. Look for missing functions, mismatched return types, or missing re-exports of CLI-accessed sub-modules.
3. Run the parity check: `zig build check-stub-parity`.

### Fix

Update `stub.zig` to match every public declaration in `mod.zig`. Shared types belong in `types.zig` (imported by both). Use `StubFeature` or `StubFeatureNoConfig` from `src/core/stub_context.zig` for common boilerplate. CLI-accessed sub-modules must be re-exported from both `mod.zig` and `stub.zig`.

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
| `zig build` | Build all targets | Yes |
| `zig build test --summary all` | Run primary tests | Yes |
| `zig build feature-tests --summary all` | Exercise all 20 feature gates | Yes |
| `zig build full-check` | Pre-commit gate (typecheck + test + docs + CLI + flags + catalog) | Yes |
| `zig build validate-flags` | Check 58 flag combinations | No |
| `zig build check-stub-parity` | Verify mod/stub declaration parity | No |
| `zig build check-zig-version` | Verify version pin consistency | No |
| `zig build toolchain-doctor` | Inspect active Zig resolution | No |
| `zig build check-docs` | Verify docs consistency | Yes |
| `zig build refresh-cli-registry` | Sync CLI registry after command changes | Yes |
| `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` | Format check (always works) | No |

When the linker is blocked, start with steps that do not require linking. Escalate to full checks once a working toolchain is available.

## Escalation Checklist

When a build failure does not match any symptom above:

1. Run `zig build toolchain-doctor` to inspect the active Zig installation.
2. Run `zig build check-zig-version` to confirm version pin consistency.
3. Check `build/options.zig` for the full flag list (27 flags).
4. Check `build/flags.zig` for the validation matrix (58 combos).
5. Check `build/link.zig` for platform-specific linking logic.
6. Examine `build/targets.zig` for target resolution details.
7. Review `build/test_discovery.zig` for test manifest wiring.
8. Consult `tasks/lessons.md` for previously encountered and resolved issues.
