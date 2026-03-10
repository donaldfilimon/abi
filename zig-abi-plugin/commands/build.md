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

3. **If Darwin**: Use the workaround wrapper:
   ```bash
   ./tools/scripts/run_build.sh <step>
   ```
   If `run_build.sh` fails with the known linker error, fall back to:
   - For `lint`: `zig fmt --check build.zig build/ src/ tools/`
   - For `fix`: `zig fmt build.zig build/ src/ tools/`
   - For `test`: `zig test src/services/tests/mod.zig -fno-emit-bin` (compile-only)
   - Otherwise: Report that this step requires the CEL toolchain

4. **If Linux**: Run directly:
   ```bash
   zig build <step> --summary all
   ```

5. Report the result clearly, including any warnings about partial validation.

## Tips

- `full-check` = format + tests + feature tests + flag validation + CLI smoke
- `verify-all` = full-check + examples + wasm + cross + docs (release gate)
- After CLI changes, always run `refresh-cli-registry`
