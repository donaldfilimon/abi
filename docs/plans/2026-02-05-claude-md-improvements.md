# CLAUDE.md & AGENTS.md Improvements Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 known issues in CLAUDE.md and AGENTS.md that cause confusion or omit important information for AI agents.

**Architecture:** Direct edits to two documentation files. No code changes. Each task is an independent edit that can be verified by reading the file.

**Tech Stack:** Markdown only. No build/test steps needed (docs-only changes).

---

### Task 1: Add commit message convention to CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:179-190` (insert new section before "Testing Patterns")

**Step 1: Add the section**

Insert before the `## Testing Patterns` line (line 179):

```markdown
## Commit Convention

Format: `<type>: <short summary>`

| Type | Use for |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change (no feature/fix) |
| `test` | Adding or updating tests |
| `chore` | Maintenance, deps, CI |

```

**Step 2: Verify**

Read CLAUDE.md and confirm the new section appears between "Key File Locations" / "Environment Variables" area and "Testing Patterns".

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add commit convention section to CLAUDE.md"
```

---

### Task 2: Clarify the conflicting time API gotcha rows

**Files:**
- Modify: `CLAUDE.md:48,54` (two rows in the Critical Gotchas table)

**Step 1: Reword the two rows**

The current table has:
- Row: `std.time.Instant.now()` → `std.time.Timer.start()`
- Row: `std.time.nanoTimestamp()` → `std.time.Instant.now()` + `.since(anchor)`

`Instant.now()` appearing in both "Mistake" and "Fix" columns is confusing.

Replace these two rows with:

```markdown
| `std.time.Instant.now()` for elapsed time | `std.time.Timer.start()` — use Timer for benchmarks/elapsed |
| `std.time.nanoTimestamp()` | Doesn't exist in 0.16 — use `Instant.now()` + `.since(anchor)` for absolute time |
```

**Step 2: Verify**

Read the gotchas table and confirm both rows now have unambiguous context about when each applies.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: clarify time API gotcha rows to avoid Instant.now() ambiguity"
```

---

### Task 3: Add `validate-flags` to the "Add a new feature module" row

**Files:**
- Modify: `CLAUDE.md:158` (Key File Locations table, "Add a new feature module" row)

**Step 1: Append validation reminder**

Change:
```
| Add a new feature module | 6 files: `mod.zig` + `stub.zig`, `build.zig` (5 places), `src/abi.zig`, `src/core/flags.zig`, `src/core/config/mod.zig`, `src/core/registry/types.zig` |
```

To:
```
| Add a new feature module | 6 files: `mod.zig` + `stub.zig`, `build.zig` (5 places), `src/abi.zig`, `src/core/flags.zig`, `src/core/config/mod.zig`, `src/core/registry/types.zig`. **Verify:** `zig build validate-flags` |
```

**Step 2: Verify**

Read the row and confirm the validate-flags reminder is present.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add validate-flags reminder to feature module checklist"
```

---

### Task 4: Make test baseline approximate instead of exact

**Files:**
- Modify: `CLAUDE.md:181` (Testing Patterns section)

**Step 1: Replace exact count with approximate**

Change:
```
**Current baseline**: 921 tests total, 916 pass, 5 skip (observability stress tests).
```

To:
```
**Current baseline**: ~920 tests, 5 skipped (observability stress tests). Run `zig build test --summary all` to get exact counts.
```

**Step 2: Verify**

Read the line and confirm it no longer has a brittle exact count.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: use approximate test count to avoid staleness"
```

---

### Task 5: Add missing feature flags to AGENTS.md table

**Files:**
- Modify: `AGENTS.md:87-98` (Feature Flags table)

**Step 1: Add missing rows**

Add these rows to the feature flags table (after the `-Denable-profiling` row, before `-Denable-mobile`):

```markdown
| `-Denable-analytics` | true | Event tracking, funnels, experiments |
| `-Denable-explore` | true | Exploration/discovery features |
```

**Step 2: Verify**

Read the table and confirm all flags mentioned in CLAUDE.md prose now have rows.

**Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add enable-analytics and enable-explore to AGENTS.md flag table"
```

---

### Task 6: Add cloud module to architecture sections

**Files:**
- Modify: `AGENTS.md:52` (Project Structure feature list)
- Modify: `CLAUDE.md:97` (Module Hierarchy)

**Step 1: Update AGENTS.md structure comment**

Change:
```
├── features/            # Feature modules (ai, gpu, database, network, web, observability)
```

To:
```
├── features/            # Feature modules (ai, gpu, database, network, web, observability, analytics, cloud)
```

**Step 2: Update CLAUDE.md module hierarchy**

Change:
```
src/features/<name>/     → mod.zig + stub.zig per feature
```

To:
```
src/features/<name>/     → mod.zig + stub.zig per feature (8 modules: ai, gpu, database, network, web, observability, analytics, cloud)
```

**Step 3: Verify**

Read both files and confirm the cloud and analytics modules are listed.

**Step 4: Commit**

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: add cloud and analytics to architecture module lists"
```

---

### Task 7: Final combined commit (alternative to per-task commits)

If you prefer a single commit for all changes:

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: improve CLAUDE.md and AGENTS.md with 6 fixes

- Add commit convention section
- Clarify conflicting time API gotcha rows
- Add validate-flags to feature module checklist
- Use approximate test count to avoid staleness
- Add enable-analytics and enable-explore flags to AGENTS.md
- Add cloud and analytics to architecture module lists"
```
