---
name: merge-conflict-resolver
description: Resolve git merge conflicts by grouping files by architecture layer, auto-resolving simple additions, and presenting manual conflicts with clear choices
---

# Merge Conflict Resolver

Use this skill to resolve git merge conflicts in the ABI repository. It groups conflicting files by architecture layer, auto-resolves simple additive conflicts, and presents manual conflicts with clear choices for the user.

## Quick Start

1. Run `git status` to identify all files with merge conflicts.
2. Group the conflicting files by architecture layer.
3. For each file, read the conflict markers and determine resolution strategy.
4. Auto-resolve where possible, present manual choices otherwise.
5. Run `zig fmt --check` on resolved `.zig` files.
6. Stage resolved files with `git add`.

## Step 1: Identify Conflicting Files

Run `git status` and collect every file listed as "both modified", "both added", "deleted by us", "deleted by them", or "both deleted". These are the unmerged paths.

If `git status` shows no merge conflicts (no "Unmerged paths" section), report that there are no conflicts to resolve and stop.

## Step 2: Group by Architecture Layer

Sort conflicting files into these ordered groups for triage:

| Layer | Path pattern | Priority |
| --- | --- | --- |
| **build** | `build.zig`, `build.zig.zon`, `.zigversion`, `build/` | 1 (resolve first) |
| **core** | `src/core/`, `src/root.zig` | 2 |
| **features** | `src/features/` | 3 |
| **services** | `src/services/` | 4 |
| **inference** | `src/inference/` | 5 |
| **tools** | `tools/` | 6 |
| **bindings** | `bindings/`, `lang/` | 7 |
| **tests** | `tests/` | 8 |
| **docs** | `docs/`, `*.md` | 9 |
| **plugin** | `zig-abi-plugin/` | 10 |
| **other** | everything else | 11 |

Resolve in priority order because earlier layers (build system, core) affect later layers (features, tools). Report the grouping to the user before proceeding.

## Step 3: Read and Classify Each Conflict

For each conflicting file, read the full file content and locate every conflict block delimited by `<<<<<<<`, `=======`, and `>>>>>>>` markers.

Classify each conflict block into one of two categories:

### Auto-resolvable

A conflict is auto-resolvable when **both** of the following hold:

- **Additive-only**: both sides add new lines (imports, list items, array entries, struct fields, enum variants, test declarations, feature catalog entries) without modifying or deleting existing lines.
- **Non-overlapping**: the additions from each side are to logically independent items (e.g., two different features added to a list, two different imports added to a block, two different test entries).

Common auto-resolvable patterns in this codebase:

- Both sides add different entries to an array/list in `src/core/feature_catalog.zig`, `build/module_catalog.zig`, `build/flags.zig`, or `plugin.json`.
- Both sides add different `@import` lines or `pub const` re-exports in `src/root.zig` or a `mod.zig`.
- Both sides add different `test` blocks or force-reference lines in test files.
- Both sides add different CLI command registrations in `tools/cli/registry/`.
- Both sides add different entries to `.gitignore` or similar declarative files.
- Both sides add different doc entries or table rows in markdown files.

### Manual review required

A conflict requires manual review when any of the following hold:

- One or both sides modify or delete existing lines (not purely additive).
- The additions are to the same logical item (e.g., both sides change the same function signature, the same config field, the same build option).
- The conflict involves control flow, type definitions, or function bodies where merging both sides could produce incorrect behavior.
- The conflict is in `build.zig` core logic, `src/root.zig` namespace wiring, or any file where ordering or uniqueness constraints apply beyond simple list membership.

## Step 4: Resolve Auto-resolvable Conflicts

For each auto-resolvable conflict:

1. Keep both sides' additions. Place them in a consistent order:
   - For alphabetically sorted lists (imports, catalog entries): maintain alphabetical order.
   - For positionally ordered lists (test blocks, CLI registrations): place the "ours" additions first, then "theirs" additions after.
   - For `.zig` files: maintain `zig fmt`-compatible formatting.
2. Remove all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
3. Write the resolved file.
4. Report what was auto-resolved: file path, conflict description, and resolution taken.

## Step 5: Present Manual Conflicts

For each conflict requiring manual review, present:

1. **File path** and architecture layer.
2. **Conflict context**: 5 lines before and after the conflict block for surrounding context.
3. **"Ours" side**: the content between `<<<<<<<` and `=======`, with a label identifying which branch it comes from.
4. **"Theirs" side**: the content between `=======` and `>>>>>>>`, with a label identifying which branch it comes from.
5. **Analysis**: a brief explanation of what each side changed and why they conflict.
6. **Recommendation**: if one side is clearly more complete or correct (e.g., it includes a stub update that the other side missed), say so.

Then ask the user which resolution to apply:

- **Keep ours**: use only the "ours" side.
- **Keep theirs**: use only the "theirs" side.
- **Keep both**: include both sides' changes (with suggested ordering).
- **Custom**: the user provides their own resolution.

Apply the chosen resolution, remove conflict markers, and write the file.

## Step 6: Format Check

After resolving all conflicts, run `zig fmt --check` on every resolved `.zig` file:

```bash
zig fmt --check <resolved-file-1.zig> <resolved-file-2.zig> ...
```

If formatting errors are found:

1. Run `zig fmt <file>` to fix them.
2. Report which files were reformatted.

Do **not** run `zig fmt .` from the repository root (it walks vendored fixtures with intentionally invalid code). Always pass explicit file paths.

## Step 7: Stage Resolved Files

For each resolved file, run:

```bash
git add <file>
```

After staging all resolved files, run `git status` again and report:

- How many conflicts were auto-resolved.
- How many required manual resolution.
- Whether any unresolved conflicts remain.
- Whether the merge/rebase can now continue (e.g., `git merge --continue` or `git rebase --continue`).

## ABI-Specific Considerations

### mod/stub Parity

When a conflict involves a `mod.zig` file under `src/features/`, check whether the corresponding `stub.zig` also has conflicts or needs updates to maintain signature parity. If `mod.zig` gains new public functions from the merge, `stub.zig` must gain matching stubs. Flag this even if `stub.zig` itself has no conflict markers.

### Feature Catalog Consistency

When conflicts touch `src/core/feature_catalog.zig`, `build/options.zig`, or `build/module_catalog.zig`, all three may need coordinated updates. After resolution, verify the feature count is consistent across these files.

### Root Namespace Wiring

When conflicts touch `src/root.zig`, verify that every feature's conditional import (`if (build_options.feat_*) @import(...mod.zig) else @import(...stub.zig)`) is intact and that no duplicate or missing namespace bindings result from the merge.

### Import Rules

When resolving conflicts that add imports:

- Inside `src/`: only relative imports are valid. Never introduce `@import("abi")`.
- Outside `src/` (CLI, tests, tools): use `@import("abi")`.
- Cross-feature imports must use `build_options` conditionals, never direct `mod.zig` imports.

## Output Summary

After all conflicts are resolved, provide a final summary:

```
Merge Conflict Resolution Summary
==================================
Total conflicts: <N>
Auto-resolved:   <N> (<list of files>)
Manual:          <N> (<list of files>)
Remaining:       <N>

Format check:    PASS / <N files reformatted>
Staged files:    <list>

Next step:       git merge --continue | git rebase --continue | git cherry-pick --continue
```
