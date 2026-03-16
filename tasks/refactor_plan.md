# Refactoring Plan for Zig 0.16-dev Syntax Perfection

## 1. Specific Fixes Needed

### 1.1 Missing .zig Extensions in Imports
- All `@import("path")` statements must have explicit `.zig` extensions.
- Example: Change `@import("std")` to `@import("std.zig")` (though std is special, see note below).
- Note: `@import("std")` is allowed without .zig because it's a special module. The rule applies to user modules.
- For local imports: `@import("../core/database/mod")` → `@import("../core/database/mod.zig")`
- For same-directory imports: `@import("helper")` → `@import("helper.zig")`

### 1.2 Return Types for Error Handling Functions
- Functions that can return errors must have explicit error sets.
- Example: Change `fn foo() void` to `fn foo() !void` if it can return an error.
- Propagate errors with `try` or `catch`, never silently swallow.

### 1.3 Other Zig 0.16 Deviations from Lessons Learned
- Replace `GeneralPurposeAllocator` with `DebugAllocator`.
- Replace `std.time.timestamp()` with `std.time.unixSeconds()`.
- Replace `File.writeAll` with `writeStreamingAll(io, data)`.
- Replace `makeDirAbsolute*` with `createDirPath(.cwd(), io, path)`.
- Remove `usingnamespace`; pass parent context as parameters to submodule init functions.
- For `LazyPath`: use `.cwd_relative`/`.src_path`, not `.path`.
- In `addTest`/`addExecutable`: use `root_module`, not `root_source_file`.
- For ZON parsing: use arena-backed `fromSliceAlloc`, deinit arena at scope end.
- Replace `valueIterator()`/`keyIterator()` with `.values()`? (Note: lessons say not `.values()`, but actually `valueIterator()`/`keyIterator()` are not `.values()`; we need to check the correct replacement)
- Replace `@enumFromInt(x)` with `intToEnum`? (Lessons say `@enumFromInt(x)` not `intToEnum`, so we should use `@enumFromInt(x)`)
- Replace `std.time.sleep` in event loops with `std.posix.poll` on STDIN.

### 1.4 Feature Module Contract Violations
- Ensure every `src/features/<name>/` has `mod.zig`, `stub.zig`, and optionally `types.zig`.
- When changing `mod.zig` public signatures, update `stub.zig` immediately.
- Shared types must be in `types.zig` imported by both `mod.zig` and `stub.zig`.
- Feature-gated sub-modules must not directly import other feature modules via relative paths; use `build_options` conditional imports.

### 1.5 Single-Module File Ownership
- Every `.zig` file must belong to exactly one named module.
- Consolidate all `src/` into single `abi` module (already done per lessons, but verify).
- Ensure no cross-module relative-path imports that bypass the single module rule.

## 2. Steps to Implement Each Fix Safely

### 2.1 For Each File:
1. Read the file to understand its content.
2. Identify Zig 0.16 deviations.
3. Plan the changes.
4. Apply changes incrementally.
5. After each change, run `zig fmt --check` on the file to ensure formatting.
6. After a set of changes, run the appropriate verification gate.

### 2.2 Ensuring mod/stub Parity:
- When modifying `mod.zig`, immediately check `stub.zig` for matching public signatures.
- If `types.zig` exists, ensure both import it.
- After changes, verify that the code compiles with the feature enabled and disabled (if applicable).

### 2.3 Verification Gates:
- Format check: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`
- Full check: `zig build full-check` (when available)
- Darwin fallback: `./tools/scripts/run_build.sh typecheck --summary all`
- Feature flag parity: Ensure that for each feature flag, the corresponding stub matches the mod.

## 3. Order of Operations

### 3.1 Phase 1: Import Fixes (.zig extensions)
- Start with leaf modules (those with no internal imports) to avoid breaking dependencies.
- Work upwards in the dependency tree.
- Batch changes by directory to minimize disruption.

### 3.2 Phase 2: Error Handling and API Updates
- Fix return types and error propagation.
- Update deprecated API calls (timestamp, writeAll, etc.).
- Again, batch by directory or feature module.

### 3.3 Phase 3: Feature Module Contract and Single-Module Ownership
- Verify each feature module has proper mod/stub/types structure.
- Ensure no cross-feature imports that bypass gates.
- Check for single-module ownership violations.

### 3.4 Phase 4: Liong-Term Debt and Patterns
- Address any remaining patterns from lessons (usingnamespace, LazyPath, etc.).
- Update build.zig and test files to use root_module.

## 4. Validation Steps

### 4.1 After Each Batch of Changes:
1. Run `zig fmt --check` on the entire codebase to ensure no formatting issues.
2. If on Darwin and linker-blocked, run `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence.
3. If not linker-blocked, run `zig build full-check`.
4. Run tests for the modified modules to ensure functionality is preserved.

### 4.2 After Major Phases:
- Verify mod/stub parity for all touched feature modules.
- Check that `tasks/todo.md` is updated with completion evidence.
- Ensure no duplicative content introduced across governance docs.

### 4.3 Pre-Commit:
- Run the strongest available verification gate.
- Update `tasks/lessons.md` if any new lessons are learned.
- Create conventional commits with atomic scope.

## 5. Documentation of Changes

### 5.1 Update tasks/todo.md:
- Add completion evidence for each batch of fixes.
- Example: `fix: added .zig extensions to imports in src/database/`

### 5.2 Conventional Commits:
- Use `fix:` for bug fixes (including syntax corrections).
- Use `refactor:` for API updates and structural changes.
- Keep commits atomic and focused.

### 5.3 Update tasks/lessons.md:
- After fixing any mistake that could recur, add a lesson to prevent future drift.

## Dependencies Between Fixes
- Import fixes (.zig extensions) must come first because they affect module visibility.
- Error handling fixes depend on correct imports.
- API updates (timestamp, etc.) depend on correct imports and error handling.
- Feature module contract fixes may require import fixes first.

## Files to Fix First
1. Start with `src/core/` and `src/services/shared/` as they are foundational.
2. Then move to feature modules in `src/features/`.
3. Finally, address `build.zig`, `tools/`, and `tests/`.

This plan ensures systematic refactoring while maintaining verification gate compliance.