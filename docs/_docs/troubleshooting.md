---
title: Troubleshooting
description: Common errors, Zig 0.16 gotchas, and FAQ
section: Reference
order: 4
---

# Troubleshooting

This page covers the most common issues when building, testing, and developing
with ABI on Zig 0.16.

---

## Zig 0.16 Gotchas

These are the top sources of build failures. Zig 0.16 changed several APIs
from earlier versions.

### 1. `std.fs.cwd()` is removed

**Error:** `error: no member named 'cwd' in 'std.fs'`

**Cause:** Zig 0.16 moved filesystem access behind an I/O backend that must
be explicitly initialized.

**Fix:** Use `std.Io.Dir.cwd()` with an initialized I/O backend:

```zig
// Wrong (Zig 0.15 and earlier)
const dir = std.fs.cwd();

// Correct (Zig 0.16)
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io, path, allocator, .limited(10 * 1024 * 1024),
);
```

### 2. Editing `mod.zig` without updating `stub.zig`

**Error:** Compilation fails when building with a feature disabled (e.g.,
`-Denable-gpu=false`).

**Cause:** Every feature module has `mod.zig` (real implementation) and
`stub.zig` (disabled stub). Their public signatures must match exactly.

**Fix:** After changing any public function in `mod.zig`, update the
corresponding `stub.zig` with the same signature:

```zig
// mod.zig
pub fn query(self: *@This(), input: []const u8) !Result { ... }

// stub.zig (must match signature, returns error)
pub fn query(_: *@This(), _: []const u8) !Result {
    return error.FeatureDisabled;
}
```

Verify with:
```bash
zig build -Denable-<feature>=false
```

### 3. `defer allocator.free(x)` then `return x`

**Error:** Use-after-free or test allocator detects double-free.

**Cause:** `defer` runs unconditionally at scope exit, including on successful
return. If you `defer free(x)` and then `return x`, the caller gets freed memory.

**Fix:** Use `errdefer` instead of `defer` when returning allocated memory:

```zig
// Wrong
const buf = try allocator.alloc(u8, 256);
defer allocator.free(buf);
// ... populate buf ...
return buf; // Bug: buf is freed before caller can use it

// Correct
const buf = try allocator.alloc(u8, 256);
errdefer allocator.free(buf); // Only frees if function returns an error
// ... populate buf ...
return buf; // Caller now owns the memory
```

### 4. `@tagName()` and `@errorName()` in format strings

**Error:** `error: expected type '[]const u8', found '@TagType(...)'` or
runtime format failures.

**Cause:** Zig 0.16 provides the `{t}` format specifier for enum and error
values, replacing manual `@tagName()` calls.

**Fix:** Use `{t}` in format strings:

```zig
// Wrong (Zig 0.15 pattern)
std.debug.print("Status: {s}\n", .{@tagName(status)});
std.debug.print("Error: {s}\n", .{@errorName(err)});

// Correct (Zig 0.16)
std.debug.print("Status: {t}\n", .{status});
std.debug.print("Error: {t}\n", .{err});
```

Other modern format specifiers:
- `{B}` / `{Bi}` for byte sizes (e.g., `1.5 MB`)
- `{D}` for durations
- `{b64}` for base64 encoding

### 5. `std.io.fixedBufferStream()` is removed

**Error:** `error: no member named 'fixedBufferStream' in 'std.io'`

**Cause:** The function was removed in Zig 0.16.

**Fix:** Use `std.Io.Writer.fixed()`:

```zig
// Wrong (Zig 0.15)
var stream = std.io.fixedBufferStream(&buf);
var writer = stream.writer();

// Correct (Zig 0.16)
var buf: [256]u8 = undefined;
var writer = std.Io.Writer.fixed(&buf);
try writer.writeAll("hello");
const written = buf[0..writer.end];
```

### 6. `@field()` requires comptime context

**Error:** `error: unable to evaluate comptime expression` when using
`@field(build_options, name)` in a runtime loop.

**Fix:** Use `inline for` instead of a regular `for` loop:

```zig
// Wrong
for (field_names) |name| {
    const val = @field(build_options, name); // Error: not comptime
}

// Correct
inline for (field_names) |name| {
    const val = @field(build_options, name); // OK: unrolled at comptime
}
```

---

## Common Build Errors

### Zig version mismatch

**Symptom:** Unexpected compilation errors, missing standard library symbols.

**Fix:** Ensure your Zig matches `.zigversion`:

```bash
cat .zigversion
zig version

# If they differ, update:
zvm install master
zvm use master
export PATH="$HOME/.zvm/bin:$PATH"

# Or run the diagnostic script:
zig run tools/scripts/toolchain_doctor.zig
```

### "module not found" for `@import("abi")`

**Symptom:** `error: module 'abi' not found`

**Cause:** You are running `zig test` directly on a file that expects the
named `abi` module, but the module path is not configured.

**Fix:** Use the build system instead of raw `zig test`:

```bash
# Instead of: zig test src/my_file.zig
# Use:
zig build test --summary all

# Or for a specific filter:
zig build test -- --test-filter "my test name"
```

### Feature flag validation failures

**Symptom:** `zig build validate-flags` reports errors.

**Cause:** A new feature was added without updating all 8 integration points.

**Fix:** Check the integration checklist:

1. `build/options.zig` -- `enable_<name>` field + CLI option
2. `build/flags.zig` -- `FlagCombo`, `validation_matrix`, `comboToBuildOptions()`
3. `src/features/<name>/mod.zig` + `stub.zig`
4. `src/abi.zig` -- comptime conditional import
5. `src/core/config/mod.zig` -- Feature enum, Config field, Builder, validation
6. `src/core/registry/types.zig` -- `isFeatureCompiledIn` switch case
7. `src/core/framework.zig` -- import, context, init/deinit, getter, builder
8. `src/services/tests/stub_parity.zig` -- parity test

---

## Test Failures

### Test baseline regression

**Expected:** 1270 pass, 5 skip (1275 total).

If tests drop below baseline, check recent changes. The test baseline is
enforced by CI hooks.

```bash
zig build test --summary all
```

### Feature test regression

**Expected:** 1534 pass (1534 total).

```bash
zig build feature-tests --summary all
```

### Hardware-gated test skips

Tests that require specific hardware (GPU, network interfaces) use
`error.SkipZigTest` to skip gracefully:

```zig
test "gpu kernel dispatch" {
    if (!abi.gpu.backends.detect.moduleEnabled()) return error.SkipZigTest;
    // ... test body ...
}
```

This is normal and does not indicate a failure.

### GPU/database test compilation issues

GPU and database backend source files have known Zig 0.16 migration gaps
(37+ errors in backend files related to `*const DynLib`, stale struct fields,
extern enum tag width). These modules compile through the named `abi` module
in `zig build test` but cannot be directly registered in `feature_test_root.zig`.

This is tracked for a dedicated migration pass.

---

## Feature Flag Issues

### Feature returns `error.FeatureDisabled`

This is expected behavior when a feature module is disabled at compile time.
Check which flags are active:

```bash
zig build run -- system-info
```

Enable the feature:

```bash
zig build -Denable-gpu=true -Dgpu-backend=metal
```

### Mobile is disabled by default

Unlike other features, mobile defaults to `false`:

```bash
zig build -Denable-mobile=true
```

### GPU backend conflicts

Prefer one primary GPU backend to avoid conflicts. On macOS, `metal` is the
natural choice:

```bash
zig build -Denable-gpu=true -Dgpu-backend=metal
```

Available backends: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`,
`webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`.

The `simulated` backend is always enabled as a software fallback for testing.

### WASM targets

WASM targets auto-disable `database`, `network`, and `gpu`:

```bash
zig build check-wasm
```

---

## FAQ

**Q: What Zig version do I need?**
A: `0.16.0-dev.2611+f996d2866` or newer. Check `.zigversion` for the exact
pinned version.

**Q: Can I use ABI from C/C++?**
A: Yes. Build the shared library with `zig build lib` and link against it.
See [C API Bindings](c-bindings.html) for details on the 36 exported functions.

**Q: How do I add a new feature module?**
A: Follow the 8-point integration checklist in the
[Architecture](architecture.html) docs. Verify with `zig build validate-flags`.

**Q: Why are there two test roots?**
A: Module path restrictions prevent `src/services/tests/mod.zig` from importing
files outside its directory tree. The feature test root at `src/feature_test_root.zig`
can reach both `features/` and `services/` subdirectories.

**Q: How do I run a full validation?**
A: Use the full-check target, which runs formatting, tests, feature tests,
flag validation, and CLI smoke tests:

```bash
zig build full-check
```

For the most comprehensive gate:

```bash
zig build verify-all
```

**Q: My feature module cannot `@import("abi")`. Why?**
A: Feature modules cannot import the root `abi` module because it would create
a circular dependency (abi imports the feature, feature imports abi). Use
relative imports to `services/shared/` instead.

**Q: How do I debug test failures?**
A: Use test filtering to isolate the failing test:

```bash
zig test src/services/tests/mod.zig --test-filter "my failing test"
```

For the testing allocator to detect leaks, ensure all allocations are properly
freed with `defer` or `errdefer`.

---

## Related Pages

- [Installation](installation.html) -- Setting up the Zig toolchain
- [Configuration](configuration.html) -- Feature flags and build options
- [Contributing](contributing.html) -- Development workflow and PR checklist

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
