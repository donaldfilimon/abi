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

## Zig 0.17 API Changes

### Lesson 2: Removed APIs in Zig 0.17
- `std.process.getEnvVarOwned` — removed. Use `b.graph.environ_map.get("KEY")` in build.zig context.
- `std.mem.trimRight` — renamed to `std.mem.trimEnd`.
- `std.fs.cwd()` — removed. Use `std.Io.Threaded` + `std.Io.Dir.cwd()`.
- `std.time.milliTimestamp` — removed. Use `foundation.time.unixMs()`.
- `std.BoundedArray` — removed. Use manual `buffer: [N]T = undefined` + `len: usize = 0`.
- Entry point signature: `pub fn main(init: std.process.Init) !void` (not `pub fn main() !void`).
- Prevention: grep for the old API name before using it. Check CLAUDE.md "Zig 0.17 Gotchas" section.

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

## Import Patterns

### Lesson 11: Avoid Circular Imports
- Never use `@import("abi")` from within `src/` — causes circular import error.
- Within `src/`: use relative imports only (`@import("../../foundation/mod.zig")`)
- From `test/`: use `@import("abi")` and `@import("build_options")`
- Cross-feature: use comptime gate, never import another feature's `mod.zig` directly
- All path imports require explicit `.zig` extensions

### Lesson 12: Comptime Feature Gating
- Cross-feature imports must use comptime gates: `if (build_options.feat_X) mod else stub`
- When adding new comptime-gated modules, ensure both mod.zig and stub.zig exist
- Run `zig build check-parity` after any public API change

## Code Style

### Lesson 13: Naming Conventions
- camelCase for functions/methods
- PascalCase for types/structs/enums
- SCREAMING_SNAKE_CASE for constants
- Avoid abbreviations unless universally understood (e.g., `num` is fine, `cnt` is not)

### Lesson 14: Struct and File Naming
- One main type per file named after the filename (e.g., `src/foo/bar.zig` defines `Bar`)
- Helper types can be nested or defined nearby
- Use `test {}` blocks with `std.testing.refAllDecls(@This())` at the end of every public type file

### Lesson 15: Memory Patterns
- Always paired allocation/deallocation
- Use `defer` for cleanup: `defer x.deinit()` is preferred
- Prefer arena allocators for temporary parsing work
- Use `defer { ... multiple statements ... }` when cleanup spans multiple lines

### Lesson 16: Error Handling
- Prefer error unions (`!`) for recoverable failures
- Use `error.FeatureDisabled` in stubs
- `@panic` only in CLI entry points and tests, never in library code

## Module Decomposition

### Lesson 17: VTable Pattern for Backend-Agnostic Interfaces
- Use VTable pattern (like `AiOps`) for backend-agnostic interfaces
- Define types in separate `types.zig` to avoid circular dependencies
- Implement concrete backends that satisfy the VTable interface
- Use adapters (`adapters.zig`) to wrap concrete implementations

### Lesson 18: Large File Decomposition
- When a file exceeds ~300 lines, consider splitting into sub-modules
- Keep the parent file as a thin re-export layer
- Move tests to a dedicated `tests.zig` sub-module
- Preserve public API surface in the parent file

## Test Patterns

### Lesson 19: Test Organization
- Unit tests: `src/root.zig` uses `refAllDecls` to walk all `test` blocks in `src/`
- Integration tests: `test/mod.zig` imports `@import("abi")` as external consumer
- Add new integration tests by importing them from `test/mod.zig`
- Both suites link macOS frameworks: System, IOKit, Accelerate, Metal, objc

### Lesson 20: Focused Test Lanes
- Use focused test lanes for faster iteration:
  - `zig build messaging-tests` — Messaging unit + integration
  - `zig build agents-tests` — Agents unit + integration
  - `zig build inference-tests` — Inference unit + integration
  - `zig build gpu-tests` — GPU unit + integration
- Full gate: `zig build check` (lint + test + parity)

## Module Decomposition Patterns

### Lesson 21: Extraction of Private Methods as Free Functions
- When extracting private methods from a struct into a sub-module, prefer free functions that take the relevant data as parameters rather than creating a context struct.
- Example: Fusion detection methods were extracted as `pub fn detectElementWiseChains(allocator, nodes, buffer_refs, patterns)` instead of creating a `DetectionContext` struct.
- The parent struct calls these functions by passing `self.nodes.items`, `&self.buffer_refs`, `&self.patterns`.
- This avoids circular dependencies and keeps the sub-module stateless.

### Lesson 22: Preserving Tests in Parent File
- When decomposing a file, keep the tests in the parent re-export file rather than splitting them across sub-modules.
- The parent file re-exports all public types, so tests can use the same import paths as before.
- Sub-modules can have their own `test { std.testing.refAllDecls(@This()); }` blocks for their own types.
- This preserves test discoverability and avoids test duplication.

### Lesson 23: Import Path Adjustment in Sub-Moving
- When extracting code into sub-directories, adjust relative imports:
  - `@import("../occupancy.zig")` becomes `@import("../../occupancy.zig")` when moving into a sub-directory
  - `@import("../time.zig")` becomes `@import("../../time.zig")` when moving into a sub-directory
  - `@import("../csprng.zig")` becomes `@import("../csprng.zig")` when staying at the same level
- Always verify with `./build.sh typecheck` after extraction.

## Documentation Sync

### Lesson 24: Keep Feature Count Docs Derived and Synchronized
- Treat `src/core/feature_catalog.zig` as the canonical source for feature counts used in user-facing docs and planning notes.
- When feature or directory counts change, update every relevant `.md` file in one pass so README, CLI docs, and planning/spec artifacts do not drift.
- Prefer explicit count derivation over stale hardcoded values; if a doc must preserve historical context, label it clearly as historical.
- After doc count edits, run a grep sweep for the old count strings and finish with `git diff --check`.

### Lesson 25: Markdown Allowlist Controls Tracking
- New Markdown files under `docs/` are ignored unless they are explicitly allowlisted in `.gitignore`.
- Before treating a docs edit as complete, confirm `git check-ignore -v <path>` does not match the new file.
- If a doc is intended to be part of the repository, add the allowlist entry in the same workflow slice as the content update.
- Keep archival docs separate from live guides so ignored local notes do not masquerade as tracked repository documentation.

### Lesson 26: Preserve Helper Surfaces During Facade Splits
- When a large module is decomposed into submodules, keep any helper functions that integration tests or external call sites import on the parent facade until those consumers are updated in the same change.
- Re-exporting the helper surface is acceptable if the split is purely structural; dropping it without a call-site migration turns a refactor into a behavior regression.
- Verify the parent facade still compiles under the public test lane before considering the split complete.
