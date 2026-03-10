---
name: build
description: Smart build command that auto-detects Darwin linker issues and routes to the correct build method
argument-hint: "[step]  e.g. test, lint, full-check, validate-flags"
allowed-tools:
  - Bash
  - Read
---

# ABI Smart Build

Run a build step with automatic Darwin workaround detection.

## Instructions

1. Read the build step argument. If none provided, default to `full-check`.

2. Detect the platform:
   ```bash
   uname -s
   ```

3. **If Darwin**: Check for CEL toolchain first, then fall back:
   ```bash
   # Check if CEL toolchain is available
   if [ -x ".zig-bootstrap/bin/zig" ]; then
       # Use CEL — full fidelity build
       PATH="$(pwd)/.zig-bootstrap/bin:$PATH" zig build <step> --summary all
   else
       # Fall back to run_build.sh wrapper
       ./tools/scripts/run_build.sh <step>
   fi
   ```
   If both fail with the known linker error (`__availability_version_check`), fall back to:
   - For `lint`/`fmt`: `./tools/scripts/fmt_repo.sh --check`
   - For `fix`: `./tools/scripts/fmt_repo.sh`
   - For `test`: `zig test src/services/tests/mod.zig -fno-emit-bin` (compile-only — no actual test execution)
   - For `validate-flags`: Report that this requires a linking-capable toolchain
   - Otherwise: Report that this step requires the CEL toolchain or Linux CI

4. **If Linux**: Run directly:
   ```bash
   zig build <step> --summary all
   ```

5. Report the result clearly. If using a fallback, warn about reduced validation scope:
   - `zig fmt --check` validates formatting only (no type checking)
   - `-fno-emit-bin` validates types and compilation but runs zero tests
   - Only `zig build` with a linking-capable toolchain runs actual tests

## Build Steps Quick Reference

| Step | What it does |
|------|-------------|
| `test` | Main test suite (~1290 tests) |
| `feature-tests` | Feature test suite (~2836 tests) |
| `full-check` | format + tests + feature tests + flag validation + CLI smoke |
| `verify-all` | full-check + examples + wasm + cross + docs (release gate) |
| `lint` | Check formatting (alias for format check) |
| `fix` | Auto-format |
| `validate-flags` | Check 42 feature flag combos |
| `refresh-cli-registry` | Regenerate CLI registry snapshot |
| `check-cli-registry` | Verify registry is current |
| `check-docs` | Docs consistency check |
| `benchmarks` | Run benchmarks |

## Tips

- After CLI changes, always run `refresh-cli-registry`
- `cel-status` / `cel-check` / `cel-doctor` diagnose the CEL toolchain state
