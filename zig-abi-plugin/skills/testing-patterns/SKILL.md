---
name: testing-patterns
description: This skill should be used when writing, running, or debugging tests in the ABI project. Covers test types, test roots, test helpers, feature test discovery, narrowing test scope with flags, Darwin workarounds, baseline sync, and common test pitfalls. Trigger when user asks about running tests, writing tests, test failures, test helpers, TestAllocator, skipIfNoGpu, feature-tests, cli-tests, full-check, test discovery, baseline sync, or "how to test" in the ABI codebase.
---

# ABI Testing Patterns

ABI is a Zig 0.16 framework (pinned `0.16.0-dev.2934+47d2e5de9`) with a layered test architecture spanning unit tests, feature tests, CLI tests, integration tests, and a full-check gate. This skill documents how tests are structured, discovered, run, and debugged.

## Test Types

### Unit Tests

Inline `test` blocks within source files. Run them with:

```bash
zig build test --summary all
```

Unit tests live alongside production code. The Zig test runner discovers all `test {}` blocks reachable from the test root module. Use `std.testing.expect*` functions for assertions and `std.testing.allocator` for allocation (it detects leaks automatically).

### Feature Tests

Exercise all 20 comptime-gated feature imports through a unified test root:

```bash
zig build feature-tests --summary all
```

The feature test step uses the `abi` module (`src/root.zig`) as its test root. All `test {}` blocks reachable from the root are compiled and run. The `src/services/tests/mod.zig` file force-references submodules inside `test {}` blocks to ensure the test runner discovers them. See the "Feature Test Discovery Architecture" section below for details.

### CLI Tests

Two tiers of CLI testing exist:

- **Smoke tests** (~53 vectors): `zig build cli-tests` -- descriptor-driven coverage using `build/cli_smoke_runner.zig`.
- **Exhaustive tests**: `zig build cli-tests-full` -- runs integration vectors from `tests/integration/matrix_manifest.zig`, excluding TUI vectors that require a PTY.

Both tiers build the CLI binary and exercise it end-to-end.

### Integration Tests

Located in `tests/integration/` with a matrix manifest (`matrix_manifest.zig`, 91+ lines). These tests cover cross-feature interactions and environment-specific behavior. Run via the full-check gate or directly through `cli-tests-full`.

### Full Check Gate

The strongest pre-commit verification gate:

```bash
zig build full-check
```

This runs typecheck + test + check-docs + check-cli + validate-flags + check-feature-catalog in sequence. Use it before completing any significant change. On Darwin 25+, where stock Zig cannot link, use `zig fmt --check ...` and `zig test <file> -fno-emit-bin` as fallback evidence.

## Test Roots and Import Rules

### The Single-Module Ownership Constraint

Zig 0.16 enforces that every `.zig` file belongs to exactly one named module. All files under `src/` belong to the `abi` module. This constraint has direct consequences for test architecture.

### Primary Test Root

`src/services/tests/mod.zig` is a separate test root with named imports (`abi` and `build_options`) injected by `build.zig`. Its child files must use `@import("abi")`, never relative imports into `src/`.

Switching to relative imports (e.g., `@import("../../root.zig")`) creates duplicate module ownership: the test runner would see both an `abi` module and a `root` module claiming the same files, causing compilation failure.

### Import Rule Summary for Tests

| Location | Import Style | Reason |
|----------|-------------|--------|
| `src/services/tests/mod.zig` and children | `@import("abi")` | Separate test root, external to `abi` module |
| Inline `test {}` blocks within `src/` | Relative imports (`@import("../foo.zig")`) | Part of the `abi` module |
| `tools/cli/tests/` | `@import("abi")` | CLI is a separate module |
| `tests/integration/` | `@import("abi")` | External test code |

## Test Helpers

`src/services/tests/helpers.zig` provides shared utilities for the test suite. Import it via `@import("abi")` from test roots or via relative import from within `src/`.

### TestAllocator

Wraps `std.heap.DebugAllocator` with automatic leak detection. Panics on `deinit()` if any allocations were not freed:

```zig
var ta = helpers.TestAllocator.init();
defer ta.deinit(); // panics if leaks detected
const alloc = ta.allocator();
```

Use `TestAllocator` for tests that manage their own allocations. For simpler cases, `std.testing.allocator` (which also detects leaks) is sufficient.

### Platform-Aware Skip

`skipIfNoGpu()` returns `error.SkipZigTest` on WASM and freestanding targets, preventing GPU-dependent tests from failing on unsupported platforms:

```zig
test "GPU compute" {
    try helpers.skipIfNoGpu();
    // ... GPU test logic
}
```

`skipIfNoTimer()` skips tests that require high-resolution timing, which may not be available on all platforms.

### Vector Utilities

- `generateRandomVector(rng, buffer)` -- fill a buffer with random `f32` values in `[-1, 1]`.
- `generateRandomVectorAlloc(allocator, rng, dims)` -- allocate and fill a vector.
- `normalizeVector(vec)` -- normalize to unit length in-place.

### Temporary Directory Management

- `createTempDir(allocator)` -- create a uniquely-named temp directory under the platform temp path.
- `removeTempDir(allocator, path)` -- delete a temp directory and free its path.
- `TempDir` -- scoped wrapper with automatic cleanup on `deinit()`:

```zig
var temp = try helpers.TempDir.init(allocator);
defer temp.deinit(); // removes directory automatically
const path = temp.getPath();
```

### Time Utilities

`sleepMs` and `sleepNs` are re-exported from `abi.foundation.time` for test convenience. On WASM, `sleepMs` is a no-op.

## Running Tests

### Full Suite

```bash
zig build test --summary all          # unit tests
zig build feature-tests --summary all # feature module tests
zig build cli-tests                   # CLI smoke (~53 vectors)
zig build cli-tests-full              # exhaustive CLI tests
zig build full-check                  # all gates combined
```

### Single File (Standalone)

For files that do not depend on the `abi` module's import graph:

```bash
zig test src/path/to/file.zig -fno-emit-bin
```

The `-fno-emit-bin` flag performs compile-only checking without producing a binary, which avoids linker issues on Darwin 25+.

### Narrowing Scope with Feature Flags

Disable features to skip their tests and reduce compilation time:

```bash
zig build test -Dfeat-ai=false -Dfeat-gpu=false --summary all
```

All 27 feature flags default to `true` (except `feat-mobile`). Disabling a flag causes the corresponding feature to compile its `stub.zig` instead of `mod.zig`, effectively removing that feature's test blocks from the runner. This is the recommended way to isolate test failures to a specific feature domain.

The 58-combo validation matrix in `build/flags.zig` defines which flag combinations are tested. Run the full matrix with:

```bash
zig build validate-flags
```

## Feature Test Discovery Architecture

### Current Design: Unified Module Root

`build/test_discovery.zig` creates the `feature-tests` step using the `abi` module as the test root. The `addFeatureTests` function passes the pre-configured `abi_module` (with `build_options` already wired) to `b.addTest()`:

```zig
const feature_tests = b.addTest(.{ .root_module = abi_module });
```

All `test {}` blocks reachable from `src/root.zig` are compiled and run. The `src/services/tests/mod.zig` file uses force-references inside `test {}` blocks to pull in submodule tests:

```zig
test {
    if (build_options.feat_ai) {
        _ = abi.ai.llm.io;
        _ = abi.ai.eval;
        // ... more submodules
    }
}
```

Force-references must appear inside `test {}` blocks, not `comptime {}` blocks. In Zig 0.16, `comptime {}` forces compilation but does not include test blocks in the test runner.

### Historical Context

The `feature_test_manifest` in `build/module_catalog.zig` remains as documentation. The previous per-entry module approach created N separate modules (one per manifest entry), causing ownership conflicts whenever entries shared files through their import graphs. The unified approach eliminates this by having a single `abi` module own all source files.

## Darwin 25+ Workarounds

Stock prebuilt Zig's internal LLD linker fails on Darwin 25+ with undefined symbols (`_malloc_size`, `__availability_version_check`, etc.). Compilation succeeds -- only linking is blocked. This affects all test steps that produce binaries.

### What Works Without Linking

- **Format checks**: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` always works.
- **Compile-only checks**: `zig test <file> -fno-emit-bin` verifies syntax and types without linking.

### Full Test Runs on Darwin

Use a host-built Zig matching `.zigversion` and prepend its bin directory to `PATH`:

```bash
export PATH="$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin:$PATH"
hash -r
zig build full-check
```

Never set `use_lld = true` on macOS (zero Mach-O support).

## Baseline Sync

`tools/scripts/baseline.zig` tracks expected pass/skip/fail counts for both main tests and feature tests. These counts are updated during baseline synchronization waves. The file also holds the canonical Zig version pin (must match `.zigversion` and `build.zig.zon`).

When test counts change (new tests added, tests removed, skip conditions updated), synchronize the baseline using the `baseline-sync` skill to keep the expected counts current. Mismatched baselines cause `full-check` to report unexpected results.

## Common Test Pitfalls

| Pitfall | Cause | Fix |
|---------|-------|-----|
| "no module named 'abi' available within module 'abi'" | Using `@import("abi")` inside `src/` inline tests | Use relative imports within `src/`. Only external test roots use `@import("abi")`. |
| Duplicate module ownership during `zig build test` | Test root uses relative imports into `src/` instead of `@import("abi")` | Ensure `src/services/tests/mod.zig` and children use `@import("abi")` exclusively. |
| Tests not discovered by the runner | Force-references placed in `comptime {}` instead of `test {}` | Move force-references (`_ = abi.foo.bar;`) into `test {}` blocks. |
| Feature test fails when feature disabled | `stub.zig` does not match `mod.zig` public signatures | Run `zig build test -Dfeat-<name>=false --summary all` to verify stub parity. |
| Linker failure on Darwin 25+ | Stock Zig LLD cannot link on modern macOS | Use host-built Zig, or `-fno-emit-bin` for compile-only validation. |
| Leak detection panic in test | `TestAllocator.deinit()` found unfreed memory | Track down the missing `allocator.free()` or `deinit()` call. Use `errdefer` for cleanup on error paths. |
| `error.SkipZigTest` not recognized | Test function does not declare `error{SkipZigTest}` in return type | Use `!void` return type (which includes the full error set) for test functions. |
| Test creates files but does not clean up | Manual temp file management | Use `helpers.TempDir` with `defer temp.deinit()` for automatic cleanup. |
| Cross-feature test breaks when dependency disabled | Direct import of another feature's `mod.zig` | Use conditional import with `build_options`: `if (build_options.feat_x) @import("...mod.zig") else @import("...stub.zig")`. |
