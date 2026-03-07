# Zig Broken on macOS (Linker / Undefined Symbols) — Research Summary

This document summarizes why Zig can fail to **link** on macOS 26+ (Tahoe) and related Darwin environments, and what workarounds exist.

---

## Symptoms

When building or testing with Zig on Darwin (e.g. macOS 26+, Apple Silicon or x86_64), you may see:

- **Undefined symbol: `__availability_version_check`**  
  Referenced by `libcompiler_rt.a` (e.g. from `___isPlatformVersionAtLeast`).
- **Undefined symbol: `_arc4random_buf`**  
  Referenced by the build output or compiler_rt.
- **Undefined symbol: `_abort`**  
  And other libc/libSystem symbols.

Binary-emitting steps (`zig build`, `zig build test`, `zig test` without `-fno-emit-bin`) fail at **link** time, not compile time.

---

## Root Causes

### 1. compiler_rt and `__availability_version_check`

- Zig’s **compiler_rt** (see `zig/lib/compiler_rt/os_version_check.zig`) implements `__isPlatformVersionAtLeast` for Darwin when the target OS version is **≥ 10.15**.
- That implementation **calls** the system API `_availability_version_check` (exposed as `__availability_version_check` in SDK stubs). It is used for `@available(macOS 10.15, *)`-style checks.
- **Issue #18818** (ziglang/zig): “compiler_rt: avoid referencing symbol on versions where it doesn't exist” — for targets **below** 10.15, Zig **does not** export `__isPlatformVersionAtLeast`, so compiler_rt does not pull in `__availability_version_check` for those targets. For 10.15+, the symbol is expected to come from the **system** (libSystem) at link time.

So the failure is not that the symbol was removed from Zig, but that the **link step** does not resolve it. That usually means:

- The linker is not being given the correct **SDK / sysroot** or **libSystem** for the host (e.g. macOS 26), or  
- The **prebuilt Zig** (e.g. from ziglang.org or ZVM) was built with an SDK or toolchain that doesn’t match the current OS (e.g. Tahoe), so the link command or library search path is wrong.

### 2. System symbols (`_abort`, `_arc4random_buf`, etc.)

- These are normal libc/libSystem symbols. If they are undefined at link time, the linker is not being pointed at the system libraries (e.g. **libSystem**) correctly.
- This matches the class of bugs seen in **Issue #16118** (“zig ld: undefined symbols for macOS 14”), where “resolve library paths in the frontend” and SDK detection fixes were needed so that Zig’s driver passes the right libraries to the linker on macOS.

So on macOS 26+ you are effectively hitting the same class of issue: **Zig’s linker invocation or SDK/library detection is not correct for this OS version**, so system and compiler_rt-related symbols are left undefined.

### 3. macOS 26 (Tahoe) specific

- **Issue #25521** (ziglang/zig): “zig build fails with a dyld error on x86_64 macOS Tahoe” — **segment `__CONST_ZIG` vm address out of order** when **running** the built binary. This is a **different** bug (self-hosted linker / layout on Tahoe), not the same as undefined-symbol at link time, but it shows that **macOS 26 is a sensitive target** for the current Zig toolchain.
- **Issue #25463**: “Compilation failed on macos 26.0 with llvm 21” — turned out to be a wrong LLVM version; closed.
- **#25152**, **#25813**: Fixes for duplicate LC_RPATH and SDK 26.0/26.1 headers — indicate ongoing Tahoe compatibility work.

---

## Upstream references

| Issue | Summary |
|-------|--------|
| **#16118** | zig ld: undefined symbols for macOS 14 (`_arc4random_buf`, `__availability_version_check`, `_abort`, etc.). Fixed by SDK/library path resolution in the frontend. |
| **#18818** | compiler_rt: avoid referencing `__availability_version_check` on OS versions where it doesn’t exist (target &lt; 10.15). Documents that 10.15+ relies on the system symbol. |
| **#25521** | `zig build` fails on x86_64 macOS Tahoe with dyld “segment `__CONST_ZIG` vm address out of order” (self-hosted linker / backend). **Open**, milestone 0.16.0. |
| **#25152** | Fix duplicate LC_RPATH entries on macOS Tahoe. |
| **#25813** | libc: Update macOS headers to SDK 26.1. |

---

## Why “this Zig version” is broken on macOS

- **Prebuilt** Zig (official tarballs, ZVM, etc.) is built on a specific host and against a specific SDK. If that build was made for an older macOS/SDK, the **linker driver logic** (paths, sysroot, which libs to pass) may not be correct for **macOS 26**, so you get undefined `__availability_version_check`, `_arc4random_buf`, `_abort`, etc.
- **Zig’s compiler_rt** intentionally depends on the system’s `__availability_version_check` for Darwin targets ≥ 10.15; if the link environment doesn’t provide it (wrong SDK or missing libSystem), the link fails.

So the “broken” behavior is not necessarily a single bug in “this Zig version,” but the combination of:

1. compiler_rt requiring a system symbol that must be resolved at link time,  
2. linker/SDK handling that hasn’t been fully adapted for macOS 26 (Tahoe), and  
3. possible self-hosted linker bugs on Tahoe (e.g. #25521).

---

## Workarounds (what we use in ABI)

### Recommended: `.cel` toolchain fork

The `.cel/` directory contains a patchable Zig fork that builds from source with macOS 26 fixes applied. This is the recommended approach:

```bash
./.cel/build.sh                          # Build patched Zig (reuses bootstrap LLVM)
eval "$(./tools/scripts/use_cel.sh)"     # Set PATH to use .cel/bin/zig
zig build full-check                     # Validate everything
```

The `.cel` fork pins the same upstream commit as `.zigversion` and applies patches from `.cel/patches/`. See `.cel/README.md` for details.

### Alternative: zig-bootstrap (legacy)

1. **When the build runner fails to link**
   The first binary Zig builds is the **build runner** (the program that runs your `build.zig`). If you see undefined symbols when running `zig build` or `zig build test` and the references include `build_zcu.o` or similar, the failure is in that first link. **You cannot fix that from build.zig** (your code has not run yet). The only fix is to use a Zig that was **built on this machine** so its linker/SDK match the OS:
   - Use **zig-bootstrap**: see **`zig-bootstrap-emergency/ABI-USAGE.md`**. From that directory run `./build aarch64-macos-none baseline` (Apple Silicon) or `./build x86_64-macos-none baseline` (Intel). Ensure the `build` script is executable (`chmod +x build`). Then point `PATH` at the resulting `out/zig-<target>-baseline/bin` and run `zig build test` from the ABI repo root.

2. **Compile-only tests**  
   When you have a working Zig (e.g. from bootstrap) but want to validate code without linking:  
   `zig test path/to/file.zig -fno-emit-bin`.  
   Use this for unit tests when you want to skip the link step.

3. **Older Zig (e.g. 0.14)**  
   Some users report that `brew install zig@0.14` and using that Zig avoids the failure on newer macOS, at the cost of using an older language/stdlib.

4. **Force LLVM backend for ABI artifacts**  
   When the build runner *does* link (e.g. you are using a bootstrap-built Zig), our `build.zig` sets `use_llvm = true` and `use_lld = true` on macOS 26+ for all host executables and tests we define. That may avoid self-hosted Mach-O linker issues (see #25521) for those artifacts. It does **not** affect the build runner itself.

---

## References

- Zig compiler_rt OS version check: `zig/lib/compiler_rt/os_version_check.zig` (in zig-bootstrap or upstream).
- Darwin libSystem stub (symbol list): `zig/lib/libc/darwin/libSystem.tbd` (contains `__availability_version_check`, `_arc4random_buf`, etc.).
- Zig GitHub: [ziglang/zig](https://github.com/ziglang/zig) (mirror; development moved to Codeberg).
- Zig bootstrap: [ziglang/zig-bootstrap](https://codeberg.org/ziglang/zig-bootstrap) (build Zig from source with minimal deps).

---

*Summary written 2026-03; issues and milestones as of that date.*
