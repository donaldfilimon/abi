# ABI Project Memory

## Project Overview

- **Repo**: `/Users/donaldfilimon/abi` — Zig 0.16 framework with 24 comptime-gated feature modules
- **Version**: 0.4.0 | **Zig**: 0.16.0-dev (pinned in `.zigversion`)
- **Entry**: `src/abi.zig` (public API, comptime feature selection)
- **CLI**: `tools/cli/` — 30 commands + 8 aliases
- **AI entry**: `CLAUDE.md` (repo root) — build, test, gotchas, consistency markers; `.claude/rules/zig.md` for Zig 0.16

## Build Commands (Quick Ref)

| Command | Purpose |
|---------|---------|
| `zig build test --summary all` | Main tests (1290 pass, 6 skip) |
| `zig build feature-tests --summary all` | Feature tests (2836 pass, 9 skip); can take several minutes |
| `zig build full-check` | Format + tests + feature tests + flag validation + CLI smoke |
| `zig build validate-flags` | Check 34 feature flag combos |
| `zig build verify-all` | Release gate (full-check + consistency + examples + wasm) |
| `zig fmt .` | Format all source |

## Architecture Essentials

- Each feature: `src/features/<name>/mod.zig` (real) + `stub.zig` (disabled)
- Comptime gating: `build_options.enable_<name>` selects mod vs stub
- Two test roots: `src/services/tests/mod.zig` (main) + `src/feature_test_root.zig` (inline)
- Import rule: public API via `@import("abi")`, features use relative imports (no circular)
- Baselines source of truth: `tools/scripts/baseline.zig`

## Zig 0.16 Key Learnings

- `std.Io.Threaded.init(gpa, .{})` — returns Threaded, call `.io()` for Io interface
- `std.Io.Dir.cwd()` — returns Dir (no io param needed for cwd itself)
- File ops: `dir.readFileAlloc(io, path, alloc, .limited(N))`, `dir.writeFile(io, .{...})`
- ArrayList/HashMap: `.empty` init, allocator per-call
- JSON: `std.json.parseFromSlice(T, alloc, data, .{})` / `std.json.Stringify.valueAlloc(...)`
- Testing: `std.testing.allocator`, `std.testing.io`, `std.testing.tmpDir(.{})`
- Random: `std.c.arc4random_buf(&buf, buf.len)` (no std.crypto.random)
- Env: `std.c.getenv("KEY")` returns `?[*:0]const u8`
- `@tagName()` returns `[*:0]const u8` — use `std.mem.sliceTo(@tagName(x), 0)` before `dupe()` or any length-dependent op
- **Mutex**: No `std.Thread.Mutex` in 0.16 — use project `sync.Mutex` from `services/shared/sync.zig`

## Multi-Agent Module Notes

- **blackboard.zig**: Uses `sync.Mutex` (from `../../../services/shared/sync.zig`), not `std.Thread.Mutex`. Init with `sync.Mutex{}`.
- **supervisor.zig**: When publishing to `messaging.EventBus`, use only `Event` fields: `event_type`, `task_id`, `success`, `detail`. No `agent_id`, `data`, or `timestamp` on `Event`.
- **workflow.zig**: For pointer-to-enum (e.g. `*StepStatus`), call methods as `ptr.*.isTerminal()` not `ptr.isTerminal()`.

## Common Debugging Patterns

- **Stub parity error**: `mod.zig` has fn that `stub.zig` doesn't — always update both
- **"member not found" on std**: API moved in 0.16 — read `~/.zvm/master/lib/std/`
- **Test not discovered**: Using `comptime { }` instead of `test { }` for imports
- **Feature flag build fail**: Check `build/options.zig` and `build/flags.zig`
- **realpathAlloc**: Use `realPathFileAlloc(io, sub_path, allocator)` — file must exist first
- **Test paths before file creation**: Use `std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/file.ext", .{tmp.sub_path})`
- **Variable shadowing**: Zig 0.16 errors on `const x` inside else branch when outer scope has `var x` — rename inner
