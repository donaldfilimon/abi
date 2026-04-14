---
name: cross-check
description: Diagnoses cross-compilation failures from `zig build cross-check`. Trigger on cross-compilation errors, undefined symbols for non-native targets, feature availability issues across platforms, or questions about "cross-check fails", "cross-compilation error", "wasm build fails", "linux build fails", "target not supported", "platform not supported".
---

# Cross-Check Troubleshooting

ABI's `zig build cross-check` validates that the framework compiles as a static library for 4 target platforms. This skill helps diagnose failures.

Pinned at Zig `0.16.0-dev.3153+d6f43caad` (`.zigversion`). The build.zig is self-contained.

## Targets

| Target | Arch | OS | Name in build |
|--------|------|----|---------------|
| ARM Linux | aarch64 | linux | `cross-aarch64-linux` |
| Intel Linux | x86_64 | linux | `cross-x86_64-linux` |
| WebAssembly | wasm32 | wasi | `cross-wasm32-wasi` |
| Intel macOS | x86_64 | macos | `cross-x86_64-macos` |

## Feature Availability Matrix

Features set per target during cross-check (from `build.zig` lines 304-333):

| Feature | aarch64-linux | x86_64-linux | wasm32-wasi | x86_64-macos |
|---------|:---:|:---:|:---:|:---:|
| feat_gpu | yes | yes | **no** | yes |
| feat_ai | yes | yes | yes | yes |
| feat_database | yes | yes | **no** | yes |
| feat_network | yes | yes | **no** | yes |
| feat_observability | yes | yes | **no** | yes |
| feat_web | yes | yes | **no** | yes |
| feat_pages | yes | yes | **no** | yes |
| feat_analytics | yes | yes | yes | yes |
| feat_cloud | yes | yes | **no** | yes |
| feat_auth | yes | yes | yes | yes |
| feat_messaging | yes | yes | yes | yes |
| feat_cache | yes | yes | yes | yes |
| feat_storage | yes | yes | **no** | yes |
| feat_search | yes | yes | yes | yes |
| feat_mobile | no | no | no | no |
| feat_gateway | yes | yes | yes | yes |
| feat_benchmarks | yes | yes | yes | yes |
| feat_compute | yes | yes | **no** | yes |
| feat_documents | yes | yes | yes | yes |
| feat_desktop | **no** | **no** | **no** | yes |
| feat_lsp | yes | yes | **no** | yes |
| feat_mcp | yes | yes | **no** | yes |

**Key patterns:**
- WASM disables all features requiring OS syscalls (filesystem, networking, threading)
- `feat_desktop` is macOS-only (uses NSStatusItem / ObjC runtime)
- `feat_mobile` is always false (opt-in only)
- AI sub-features (`feat_llm`, `feat_training`, `feat_vision`, `feat_explore`, `feat_reasoning`) are always true

## GPU Backend Matrix

All GPU backends are disabled in cross-check except `gpu_stdgpu` on non-WASM targets:

| Backend | aarch64-linux | x86_64-linux | wasm32-wasi | x86_64-macos |
|---------|:---:|:---:|:---:|:---:|
| gpu_stdgpu | yes | yes | **no** | yes |
| gpu_metal | no | no | no | no |
| gpu_cuda | no | no | no | no |
| gpu_vulkan | no | no | no | no |
| gpu_webgpu | no | no | no | no |
| All others | no | no | no | no |

Hardware-specific backends (Metal, CUDA, Vulkan) are disabled because cross-check validates compilation only — no hardware drivers are available during cross-compilation.

## Common Failure Patterns

### 1. WASM: Undefined Symbols for Syscalls

**Symptom:** `error: undefined symbol: _read`, `_write`, `_socket`, `_pthread_create`, or similar libc/POSIX symbols when building `cross-wasm32-wasi`.

**Cause:** A feature that requires OS syscalls is enabled for WASM. Check the matrix above — WASM should disable: gpu, database, network, profiling, web, pages, cloud, storage, compute, lsp, mcp.

**Fix:** Verify the feature's stub.zig compiles cleanly with no syscall references. Check `build.zig` — the WASM target uses `const is_wasm = ct.os == .wasi` to gate features.

### 2. Linux: Desktop Feature Failures

**Symptom:** `error: undefined symbol: _objc_msgSend`, `_sel_registerName`, or ObjC-related symbols on `cross-aarch64-linux` or `cross-x86_64-linux`.

**Cause:** `feat_desktop` is enabled on Linux. Desktop uses macOS-specific NSStatusItem / ObjC runtime.

**Fix:** `feat_desktop` must be false on Linux. In `build.zig`, this is: `cross_opts.addOption(bool, "feat_desktop", !is_wasm and !is_linux)`.

### 3. Stub Parity Violations

**Symptom:** `error: <feature>/stub.zig is missing declarations from mod.zig: <name>` during cross-check.

**Cause:** When a feature is disabled for a target, `src/root.zig` imports stub.zig instead of mod.zig. If stub.zig is missing declarations that other enabled features reference, compilation fails.

**Fix:** Update stub.zig to match mod.zig's public declarations. Run `zig build check-parity` for a quick check. See also the `stub-parity-reviewer` agent.

### 4. Conditional Import Errors

**Symptom:** `error: no module named 'abi'` in a file under `src/features/`.

**Cause:** Code inside `src/` must use relative imports, never `@import("abi")`. The `abi` module is only available in `test/` via build.zig's named module import.

**Fix:** Use relative paths: `@import("../../foundation/mod.zig")`. For cross-feature imports, always use the comptime gate:
```zig
const obs = if (build_options.feat_observability)
    @import("../../features/observability/mod.zig")
else
    @import("../../features/observability/stub.zig");
```

### 5. Missing Platform Framework Links

**Symptom:** `error: undefined symbol` for system framework functions on macOS targets (e.g., `_IOServiceOpen`, `_vDSP_vadd`).

**Cause:** macOS targets need explicit framework linking in `build.zig`. Cross-check macOS target may be missing a framework link that the native build has.

**Fix:** Check `build.zig` platform linking section (lines 136-156). Ensure cross-check macOS target links the same frameworks: System, c, objc, IOKit, Accelerate (when gpu), Metal/MPS/CoreGraphics (when gpu_metal).

## Diagnostic Procedure

1. **Run cross-check and capture output:**
   ```bash
   zig build cross-check 2>&1 | tee /tmp/cross-check.log
   ```

2. **Identify which target failed:** Look for `cross-<target>` in the error output. The artifact name encodes the target.

3. **Check if it's a feature availability issue:** Compare the failing feature against the matrix above. Is it supposed to be disabled for that target?

4. **Check if it's a stub issue:** If the feature is disabled for the target, verify stub.zig has all needed declarations: `zig build check-parity`

5. **Check if it's an import issue:** Search the failing file for `@import("abi")` or direct mod.zig imports that bypass the comptime gate.

6. **Reproduce with a single target:** Manually build for just one target by adding the appropriate flags:
   ```bash
   zig build -Dtarget=wasm32-wasi -Dfeat-gpu=false -Dfeat-database=false ...
   ```

## Quick Reference

| Error Pattern | Likely Cause | Fix |
|---------------|-------------|-----|
| `undefined symbol: _pthread*` on WASM | Threading feature enabled for WASM | Disable feature for WASM in build.zig |
| `undefined symbol: _objc*` on Linux | Desktop feature enabled for Linux | Ensure `feat_desktop=false` for Linux |
| `missing declarations from mod.zig` | Stub parity drift | Update stub.zig, run `zig build check-parity` |
| `no module named 'abi'` | `@import("abi")` inside src/ | Use relative imports |
| `undefined symbol: _IOService*` | Missing IOKit framework link | Add `linkFramework("IOKit")` for macOS target |
| `error: AnalyzeFail` | Zig analysis error in stubbed code | Check stub returns correct error type |
