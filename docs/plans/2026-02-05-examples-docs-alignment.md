# Examples & Docs Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all example compilation errors, verify API alignment, and ensure CLI documentation matches current behavior.

**Architecture:** Fix Zig 0.16.0-dev.2535+b5bd49460 Mutex/RwLock API migration issues across 8+ modules, verify all examples build successfully, audit CLI help text against actual command implementations.

**Tech Stack:** Zig 0.16.0-dev.2535+b5bd49460, std.Thread.Mutex (new location), build system validation

---

## Task 1: Fix AI Personas Module Mutex API

**Files:**
- Modify: `src/features/ai/personas/loadbalancer.zig:124`
- Modify: `src/features/ai/personas/registry.zig:16`
- Reference: Search codebase for correct Mutex usage examples

**Step 1: Identify correct Zig 0.16.0-dev.2535+b5bd49460 Mutex API**

Run: `grep -r "Mutex" src/ --include="*.zig" | grep -v "std.Thread.Mutex" | head -20`
Expected: Find files using the correct API (likely imports or different path)

**Step 2: Check CLAUDE.md for Zig 0.16.0-dev.2535+b5bd49460 patterns**

Read: `CLAUDE.md` section "Critical Gotchas"
Look for: Mutex API migration guidance

**Step 3: Update loadbalancer.zig**

In `src/features/ai/personas/loadbalancer.zig:124`:
```zig
// OLD
mutex: std.Thread.Mutex,

// NEW (verify correct API from step 1)
mutex: std.sync.Mutex,
```

**Step 4: Update registry.zig**

In `src/features/ai/personas/registry.zig:16`:
```zig
// OLD
mutex: std.Thread.Mutex,

// NEW
mutex: std.sync.Mutex,
```

**Step 5: Verify compilation**

Run: `zig build -Denable-ai=true 2>&1 | grep personas`
Expected: No errors in personas modules

**Step 6: Commit**

```bash
git add src/features/ai/personas/loadbalancer.zig src/features/ai/personas/registry.zig
git commit -m "fix: update AI personas to Zig 0.16.0-dev.2535+b5bd49460 Mutex API"
```

---

## Task 2: Fix Analytics and Database Sync Primitives

**Files:**
- Modify: `src/features/analytics/mod.zig:72`
- Modify: `src/features/database/database.zig:439`

**Step 1: Fix analytics Mutex**

In `src/features/analytics/mod.zig:72`:
```zig
// OLD
mutex: std.Thread.Mutex = .{},

// NEW
mutex: std.sync.Mutex = .{},
```

**Step 2: Fix database RwLock**

In `src/features/database/database.zig:439`:
```zig
// OLD
rw_lock: std.Thread.RwLock,

// NEW
rw_lock: std.sync.RwLock,
```

**Step 3: Verify compilation**

Run: `zig build -Denable-analytics=true -Denable-database=true 2>&1 | grep -E "(analytics|database)"`
Expected: No errors in these modules

**Step 4: Commit**

```bash
git add src/features/analytics/mod.zig src/features/database/database.zig
git commit -m "fix: update analytics and database to Zig 0.16.0-dev.2535+b5bd49460 sync primitives API"
```

---

## Task 3: Fix GPU Modules Mutex API

**Files:**
- Modify: `src/features/gpu/dispatcher.zig:219`
- Modify: `src/features/gpu/metrics.zig:163`
- Modify: `src/features/gpu/multi_device.zig:125`
- Modify: `src/features/gpu/stream.zig:296`
- Check: Any other GPU files with same issue

**Step 1: Search for all GPU Mutex usages**

Run: `grep -n "std.Thread.Mutex" src/features/gpu/*.zig`
Expected: List all files needing updates

**Step 2: Update dispatcher.zig**

In `src/features/gpu/dispatcher.zig:219`:
```zig
// OLD
queue_mutex: std.Thread.Mutex,

// NEW
queue_mutex: std.sync.Mutex,
```

**Step 3: Update metrics.zig**

In `src/features/gpu/metrics.zig:163`:
```zig
// OLD
mutex: std.Thread.Mutex,

// NEW
mutex: std.sync.Mutex,
```

**Step 4: Update multi_device.zig**

In `src/features/gpu/multi_device.zig:125`:
```zig
// OLD
mutex: std.Thread.Mutex,

// NEW
mutex: std.sync.Mutex,
```

**Step 5: Update stream.zig**

In `src/features/gpu/stream.zig:296`:
```zig
// OLD
mutex: std.Thread.Mutex,

// NEW
mutex: std.sync.Mutex,
```

**Step 6: Check for any missed files**

Run: `grep -r "std.Thread.Mutex\|std.Thread.RwLock" src/features/gpu/`
Expected: No matches (all fixed)

**Step 7: Verify GPU compilation**

Run: `zig build -Denable-gpu=true 2>&1 | head -30`
Expected: No Mutex-related errors

**Step 8: Commit**

```bash
git add src/features/gpu/dispatcher.zig src/features/gpu/metrics.zig \
        src/features/gpu/multi_device.zig src/features/gpu/stream.zig
git commit -m "fix: update GPU modules to Zig 0.16.0-dev.2535+b5bd49460 sync primitives API"
```

---

## Task 4: Verify All Examples Build

**Files:**
- Test: All `examples/*.zig` files
- Reference: `build.zig` examples step

**Step 1: Build all examples**

Run: `zig build examples 2>&1`
Expected: All examples compile successfully (0 errors)

**Step 2: If errors remain, list them**

Run: `zig build examples 2>&1 | grep "error:" | head -20`
Expected: Empty output or list of remaining issues

**Step 3: Document any remaining issues**

Create: `docs/examples-audit-2026-02-05.md`
List: Any examples that still fail and why

**Step 4: Update plan.md with results**

In `docs/plan.md`, update "Next Sprint" section:
```markdown
### Completed (2026-02-05)
- [x] **Zig 0.16.0-dev.2535+b5bd49460 API migration** - Fixed Mutex/RwLock across 8+ modules
- [x] **Examples build verification** - All 18 examples compile successfully
```

**Step 5: Commit**

```bash
git add docs/plan.md docs/examples-audit-2026-02-05.md
git commit -m "docs: update sprint progress - examples now build"
```

---

## Task 5: CLI Documentation Audit

**Files:**
- Read: `tools/cli/commands/*.zig` (all command implementations)
- Verify: CLI help text matches actual behavior
- Document: Any discrepancies found

**Step 1: Generate CLI help documentation**

Run: `zig build run -- --help > docs/cli-help-output.txt 2>&1`
Expected: Full CLI help text captured

**Step 2: Test each command's help**

Run: `for cmd in model db gpu train agent; do zig build run -- $cmd --help >> docs/cli-help-output.txt; done`
Expected: All subcommand help text captured

**Step 3: Compare help text to implementations**

Manual review:
- Check `tools/cli/commands/model.zig` matches `abi model --help`
- Check `tools/cli/commands/db.zig` matches `abi db --help`
- Check `tools/cli/commands/gpu.zig` matches `abi gpu --help`
- Check `tools/cli/commands/train.zig` matches `abi train --help`
- Check `tools/cli/commands/agent.zig` matches `abi agent --help`

**Step 4: Document any discrepancies**

Create: `docs/cli-audit-2026-02-05.md`
Format:
```markdown
# CLI Documentation Audit - 2026-02-05

## Discrepancies Found

### Command: `abi model download`
- **Help says**: "Downloads model from HuggingFace"
- **Code does**: Downloads with native HTTP (as of 2026-02-05)
- **Fix needed**: Update help text to mention native HTTP

## All Commands Verified

- [x] abi model
- [x] abi db
- [x] abi gpu
...
```

**Step 5: Commit audit results**

```bash
git add docs/cli-help-output.txt docs/cli-audit-2026-02-05.md
git commit -m "docs: CLI help text audit - 2026-02-05"
```

---

## Task 6: Update CLAUDE.md with Zig 0.16.0-dev.2535+b5bd49460 Gotcha

**Files:**
- Modify: `CLAUDE.md` - Critical Gotchas table

**Step 1: Add Mutex/RwLock to gotchas table**

In `CLAUDE.md` "Critical Gotchas" section, add row:

```markdown
| `std.Thread.Mutex` / `std.Thread.RwLock` | `std.sync.Mutex` / `std.sync.RwLock` — Zig 0.16.0-dev.2535+b5bd49460 moved sync primitives to std.sync |
```

**Step 2: Verify table formatting**

Run: `cat CLAUDE.md | grep -A 15 "Critical Gotchas"`
Expected: New row appears in table with correct formatting

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Mutex/RwLock migration to CLAUDE.md gotchas"
```

---

## Task 7: Run Full Test Suite

**Files:**
- Test: All tests via `zig build test`

**Step 1: Run full test suite**

Run: `zig build test --summary all 2>&1 | tail -30`
Expected: 944 pass, 5 skip (949 total) - baseline maintained

**Step 2: If test count changes, investigate**

If not 944/949:
- Run: `zig build test --summary all 2>&1 | grep -E "(FAIL|pass|skip)" | tail -50`
- Document: Which tests changed and why

**Step 3: Update plan.md with test results**

In `docs/plan.md`:
```markdown
### Completed This Sprint (2026-02-05)
- [x] **Examples & Docs Alignment** - Fixed Mutex API migration, all examples build
- [x] **Test baseline verified** - 944/949 tests passing (baseline maintained)
```

**Step 4: Commit**

```bash
git add docs/plan.md
git commit -m "docs: sprint complete - examples and docs aligned"
```

---

## Success Criteria

✅ All 18 examples compile successfully
✅ Zero Mutex/RwLock API errors
✅ Test baseline maintained (944/949)
✅ CLI help text audit completed
✅ CLAUDE.md updated with new gotcha
✅ Sprint documented in plan.md

---

## Notes

- **Parallel execution**: Tasks 1-3 can run in parallel (independent modules)
- **Test early**: Run `zig build examples` after each module fix to catch issues
- **Git workflow**: Frequent small commits (one per task)
- **Documentation**: Keep plan.md and audit docs up to date

## Related Skills

- @superpowers:executing-plans - For task-by-task execution
- @superpowers:subagent-driven-development - For parallel subagent execution
- @superpowers:verification-before-completion - Before marking sprint complete
