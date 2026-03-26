# Workflow Lessons

## Planning Discipline

### Lesson 1: Respect Plan-First Workflow Before Implementation
- Pattern: when the user requires plan-first workflow for non-trivial work, do not proceed into implementation until planning expectations and task tracking are established.
- Root cause: I executed corrective code changes before making the repo-local workflow explicit and before persisting the operating rules in `tasks/`.
- Prevention rules:
  - For any non-trivial task, read `tasks/lessons.md` first and update `tasks/todo.md` before implementation.
  - If the user corrects process, capture the root cause here immediately instead of relying on conversational memory.
  - Recheck `git status --short` immediately before edits and immediately after verification so shifting repo state does not invalidate the plan.
  - Treat verification as part of the plan, not as a postscript.

## Review Startup Rules
- Use in-thread AGENTS instructions plus `CLAUDE.md` as the workflow source of truth while repo-root `AGENTS.md` is absent.
- Treat ABI `review_prep.py` as blocked in this checkout unless the required repo markers are added or the helper is updated.
- Keep repo-root `tasks/` for workflow notes only; do not confuse it with `src/tasks`.

## Zig 0.16 API Changes

### Lesson 2: Removed APIs in Zig 0.16
- `std.process.getEnvVarOwned` — removed. Use `b.graph.environ_map.get("KEY")` in build.zig context.
- `std.mem.trimRight` — renamed to `std.mem.trimEnd`.
- `std.fs.cwd()` — removed. Use `std.Io.Threaded` + `std.Io.Dir.cwd()`.
- `std.time.milliTimestamp` — removed. Use `foundation.time.unixMs()`.
- `std.BoundedArray` — removed. Use manual `buffer: [N]T = undefined` + `len: usize = 0`.
- Entry point signature: `pub fn main(init: std.process.Init) !void` (not `pub fn main() !void`).
- Prevention: grep for the old API name before using it. Check CLAUDE.md "Zig 0.16 Gotchas" section.

## Mod/Stub Parity

### Lesson 3: Always Update Both mod.zig AND stub.zig
- When adding a public declaration to any feature's mod.zig, the corresponding stub.zig must be updated.
- Run `zig build check-parity` (or `./build.sh check-parity`) after ANY public API change.
- Prevention: before committing, check that the declaration count matches between mod and stub.

### Lesson 4: New Comptime-Gated Modules Need Parity Coverage
- When adding a new comptime-gated module in root.zig (like connectors, tasks, inference), add an `assertParity` call in `src/feature_parity_tests.zig`.
- Root cause: connectors, tasks, and inference were gated in root.zig but had no parity check, allowing drift to go undetected.

## macOS 26.4+ Build

### Lesson 5: LLD Cannot Link on macOS 26.4+
- Always use `./build.sh` instead of `zig build` on macOS 26.4+ (Darwin 25.x).
- `build.sh` auto-relinks with Apple ld and auto-retries with `-Dfeat-gpu=false` on Accelerate symbol failures.
- `use_lld = false` in build.zig helps but doesn't fully resolve — some system symbols still fail.
- Prevention: never recommend raw `zig build` on macOS 26.4+.

## Database Engine Thread Safety

### Lesson 6: Lock All Shared State Access
- All public methods on `Engine` must acquire `db_lock` before reading shared state (`vectors_array`, `hnsw_index`, `ai_client`, `cache`).
- Root cause: `deinit`, `indexWithPolicy`, and `search` read `ai_client` and `cache` outside the lock, creating data races with concurrent `connectAI` calls.
- Prevention: when adding a new public method to engine.zig, always acquire the lock first.

## Build System

### Lesson 7: Use linkIfDarwin() for macOS Framework Linking
- Always use `linkIfDarwin()` from `build/linking.zig` instead of inline `if (os.tag == .macos)` checks.
- The inner macOS check in `linkDarwinArtifact` is defensive — don't remove it even though `linkIfDarwin` gates.

## Stub Patterns

### Lesson 8: AI Sub-Feature Stub Conventions
- AI sub-feature stubs: check `src/core/stub_helpers.zig` before writing custom init/deinit/isEnabled boilerplate.
- Complex domain stubs (with rich types like EmbeddingModel) should stay custom — don't force StubFeature on them.
- Aviva stub should import shared types from `../types.zig` instead of redefining them.

## Protocol Patterns

### Lesson 9: JSON and Error Handling in Protocols
- JSON escaping: use `foundation/utils/json.zig`, don't reimplement in protocol-specific utils.
- ACP json_utils.zig had a silent `continue` bug — always propagate or handle format errors explicitly.

## AI Pipeline

### Lesson 10: AI Pipeline Safety
- String literals in ProfileResponse.content will crash on deinit — always `allocator.dupe()` heap copies.
- Check for double imports when adding new type references.
- Abbey has emotion.zig AND emotions.zig — use emotions.zig (the canonical one).
