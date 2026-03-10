---
name: check
description: Run verification checks on the ABI codebase with platform-aware routing
argument-hint: "[scope]  e.g. format, imports, registry, stub-sync, modules, all"
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# ABI Verification Checks

Run targeted or full verification of the ABI codebase.

## Instructions

Based on the scope argument:

### `format` (or `fmt`)
```bash
zig fmt --check build.zig build/ src/ tools/
```
Report any files with format violations.

### `imports`
Search for import rule violations:
- Grep `src/features/` for `@import("abi")` — features must use relative imports or named modules
- Report violations with file paths and line numbers

### `modules`
Check for cross-module import violations:
- Grep `src/features/` for relative paths to `wdbx/wdbx.zig` — these should use `@import("wdbx")`
- A file can only belong to ONE Zig module; relative paths to named module roots cause conflicts
- Check `src/` broadly for any `@import("../../wdbx/wdbx.zig")` patterns

### `registry`
Check if CLI registry is current. Read `tools/cli/generated/` and compare against command files in `tools/cli/commands/`.

### `stub-sync`
For each directory in `src/features/*/`:
1. Extract `pub fn` signatures from `mod.zig`
2. Extract `pub fn` signatures from `stub.zig`
3. Report any mismatches (missing functions, wrong signatures)

### `deprecated`
Scan for Zig 0.16 deprecated patterns:
- `std.time.timestamp` — use `time.unixSeconds()` from services/shared/time.zig
- `std.posix.getenv` — use `std.c.getenv`
- `std.meta.intToEnum` — use `@enumFromInt`
- `usingnamespace` — removed in 0.16
- Invalid format specifiers like `{t}` in `std.log` / `std.fmt`

### `all` (default)
Run all checks above in sequence. Report a summary table:

| Check | Status | Issues |
|-------|--------|--------|
| format | PASS/FAIL | N violations |
| imports | PASS/FAIL | N violations |
| modules | PASS/FAIL | N conflicts |
| registry | CURRENT/STALE | N missing |
| stub-sync | PASS/FAIL | N mismatches |
| deprecated | PASS/FAIL | N patterns |

## Tips
- On Darwin 25+, `zig build full-check` won't work directly — use this command instead
- For full build-system verification, use `/zig-abi:build full-check` which wraps `run_build.sh`
