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
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
```
Report any files with format violations.

### `imports`
Search for import rule violations:
- Grep `src/features/` for `@import("abi")` — features must use relative imports, never `@import("abi")` (circular dependency)
- Report violations with file paths and line numbers

### `modules`
Check for cross-module import violations. Named modules registered in `build.zig` must not be imported via relative paths from other modules — a Zig file can only belong to ONE module.

Search for these patterns:
- Grep `src/` broadly for relative imports to any named module root:
  - Any `../` chain ending in a build.zig-registered module root
- Grep for `@import("shared_services")` or `@import("core")` — these named modules no longer exist
- Named modules in build.zig: `abi` (root: `src/root.zig`), `build_options`, `cli` (root: `tools/cli/mod.zig`)
- Note: `foundation` is NOT a named module — it is a namespace within `abi` at `src/services/shared/mod.zig`

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

### `flags`
Run the feature flag sync validation script:
```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/check-flag-sync.sh "${CLAUDE_PROJECT_DIR:-.}"
```
Reports mismatches between `build/options.zig`, `build/flags.zig`, and `src/core/feature_catalog.zig` flag counts.

### `darwin`
Run the Darwin relink audit:
```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/audit-darwin-targets.sh "${CLAUDE_PROJECT_DIR:-.}"
```
Checks that every `addExecutable()` in `build.zig` has `darwinRelink()` wiring or an `is_blocked_darwin` guard.

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
| flags | PASS/FAIL | N mismatches |
| darwin | PASS/FAIL | N missing |

## Tips
- On Darwin 25+, `zig build full-check` won't work directly due to linker incompatibility — use this command instead
- For full build-system verification, use `/zig-abi:build full-check` which wraps `run_build.sh`
