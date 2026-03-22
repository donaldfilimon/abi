---
name: abi-code-review
description: This skill should be used when reviewing ABI code changes, PRs, or diffs — applies Zig 0.16 (dev.2962+) review patterns, path-to-gate mapping, and ABI-specific heuristics for feature modules, build system, CLI, and database changes.
---

# ABI Code Review

Use this skill to review ABI changes with the repo's actual workflow, Zig pin, and validation coupling in mind. Keep the review focused on discrete regressions, not style, and prefer ABI-specific evidence over generic Zig advice.

## Quick Start

1. Confirm the checkout looks like ABI by checking for `AGENTS.md`, `CLAUDE.md`, `.zigversion`, `build.zig`, and `src/root.zig`.
2. Read `tasks/lessons.md` and `tasks/todo.md` before reviewing the diff.
3. Run `python3 <skill-dir>/scripts/review_prep.py --repo <abi-repo> --base <ref>` to collect merge-base diff metadata, subsystem tags, and recommended checks.
4. Review the diff against the merge base and report only actionable bugs or regressions.

## Review Workflow

- Run `review_prep.py` before writing findings. Use its output to choose which checks matter.
- If the script reports no changes to review, return no findings instead of inventing work.
- Treat ABI's Zig pin as 0.16-dev/master-era guidance from `.zigversion`, not as stable-release Zig 0.16 guidance.
- Treat the known Darwin stock-Zig linker failure as an environment constraint unless the patch worsens `.cel`, bootstrap, or toolchain-detection behavior.
- For `src/features/` changes, verify `mod.zig` and `stub.zig` stay aligned when public surfaces move, and keep feature code on relative imports.
- When CLI, docs, or registry files move together, check the coupling rather than reviewing each file in isolation.

## Zig 0.16 Review Patterns

### Pin and Framing

- ABI pins Zig at `0.16.0-dev.2962+08416b44f` via `.zigversion`; read that file instead of assuming a stable release.
- Describe the repo as Zig `0.16.0-dev` or Zig 0.16-dev/master-era when reviewing API usage and toolchain behavior.
- Use `CLAUDE.md` as the repo-local summary of known 0.16 migration constraints and Darwin caveats.
- Build scripts (`build.sh`, `tools/scripts/run_build.sh`, `build/compat.zig`, `build/darwin.zig`) have been removed — direct `zig build` is the primary build path.

### Build API Patterns

- Prefer `b.createModule(.{ .root_source_file = b.path(...), ... })`.
- Prefer `addTest` and `addExecutable` with `.root_module = mod`.
- Do not use deprecated `LazyPath.path`.
- When build options change, check whether the option wiring, docs, and validation steps remain aligned.

### Known 0.16 Traps

- Prefer validated `switch` parsing or `@enumFromInt` over removed `std.meta.intToEnum`.
- Use `std.mem.Alignment` in allocator vtables, not raw `u8` alignment values.
- Review `std.Io` changes with the current backend patterns in mind; avoid reintroducing older file-I/O assumptions.
- Treat time helpers carefully. ABI often passes `now` from the caller or uses shared time helpers rather than old `std.time` shortcuts.
- `std.time.unixSeconds()` not `timestamp()`.
- `file.writeStreamingAll(io, data)` not `writeAll`.
- `std.Io.Dir.createDirPath(.cwd(), io, path)` not `makeDirAbsolute`.
- `.cwd_relative` / `.src_path` not `.path` on `LazyPath`.
- `root_module` field not `root_source_file`.
- `valueIterator()` not `.values()` on hash maps.
- `@enumFromInt(x)` not `intToEnum`.

### Feature-Gated Modules

- When a public function changes in `src/features/<name>/mod.zig`, check whether `src/features/<name>/stub.zig` still exposes a matching signature and behavior contract.
- Keep feature-module imports relative inside `src/features/`; direct `@import("abi")` inside feature code is an ABI rule violation.
- When feature flags, catalog entries, or build options move, expect `zig build validate-flags` to matter.
- 27 flags in `CanonicalFlags` (including `feat_lsp`, `feat_mcp`); catalog has 29 features with sub-features.

### Test Discovery

- `build/test_discovery.zig` uses the unified `abi` module as test root (not per-entry modules).
- This avoids single-module file ownership violations when test entries share files.
- The `feature_test_manifest` in `module_catalog.zig` remains as documentation.

### Toolchain Caveat

- The repo documents a Darwin/macOS 25+ stock-Zig linker failure outside normal patch scope.
- Only flag toolchain-related issues when a change breaks CEL, version consistency, or diagnostics.
- The legacy `build.sh`, `run_build.sh`, `compat.zig`, and `darwin.zig` wrapper scripts have been removed. Platform-specific linking now lives in `build/link.zig`.

## Path-to-Gate Mapping

Use this map to choose validation expectations from the changed paths.

| Changed path pattern | Review focus | Expected checks |
| --- | --- | --- |
| `tools/cli/` or CLI command metadata | Command wiring, registry metadata, help/output drift | `zig build test`, `zig build refresh-cli-registry`, `zig build check-cli-registry` |
| `tools/cli/terminal/` or TUI panels | Dashboard behavior, async loop, non-blocking input, renderer coupling | `zig build test` |
| `tools/gendocs/`, `docs/`, `README.md`, `CLAUDE.md` | Docs generation coupling, stale registry/module data | `zig build check-docs` |
| `src/features/*/mod.zig`, `src/features/*/stub.zig`, `build/options.zig`, `build/flags.zig`, `src/core/feature_catalog.zig` | Feature-gate parity, public surface drift, disabled-build compatibility | `zig build validate-flags` |
| `src/core/database/` or `src/features/database/` | WDBX engine behavior, database correctness, replication, graph logic | `zig build test` |
| `build.zig`, `build/`, `.zigversion`, `build.zig.zon` | Zig 0.16 build API usage, pin consistency, platform linking integrity | `zig build full-check`, `zig build verify-all` on a host where the toolchain links |
| Any non-trivial code change | End-to-end ABI correctness | `zig build full-check`, `zig build verify-all` on a host where the toolchain links |

### Gate Notes

- `full-check` is ABI's default pre-close gate.
- `verify-all` is the release-style umbrella gate and may be blocked on the known Darwin linker issue.
- On the blocked Darwin host, treat binary-emitting failures as environment noise unless the patch changes the workaround path itself.

## Review Discipline

- Review against the merge base, not just against the named base ref.
- Prioritize correctness, safety, performance, and ABI-specific maintainability regressions.
- Ignore style unless it hides a real bug or violates a documented ABI rule.
- When verification is relevant, cite the exact ABI gate that should have covered the behavior.

## Output Rules

- Report only discrete bugs or behavioral regressions the author would likely fix.
- Keep comments short, concrete, and scoped to the exact failure mode.
- Mention blocked verification when environment issues prevent a gate from running.
- Do not flag the known Darwin linker issue unless the patch expands or breaks the existing `.cel` or bootstrap workaround path.
- If no findings remain after ABI-specific checks, say so plainly.

## Subsystem Categories

The `review_prep.py` script tags changed files into these ordered categories for triage:

1. `build-system` -- `build.zig`, `build.zig.zon`, `.zigversion`, `build/`
2. `toolchain` -- `build.zig.zon`, `.zigversion`, `.cel/`, `tools/scripts/`
3. `cli` -- `tools/cli/`
4. `tui` -- `tools/cli/terminal/`, `*tui_tests_root.zig`
5. `docs` -- `tools/gendocs/`, `docs/`, `README.md`, `CLAUDE.md`, `AGENTS.md`
6. `features` -- `src/features/`
7. `feature-flag-surface` -- `src/features/*/mod.zig`, `src/features/*/stub.zig`, `build/options.zig`, `build/flags.zig`, `src/core/feature_catalog.zig`
8. `wdbx` -- `src/core/database/`
9. `database` -- `src/features/database/`
10. `network-dist` -- `src/core/database/dist/`, `src/features/network/`
11. `training` -- `src/features/ai/training/`, any path containing "training"
12. `tasks-planning` -- `tasks/`
