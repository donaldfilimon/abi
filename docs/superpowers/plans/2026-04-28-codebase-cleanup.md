# Codebase Cleanup and 0.17 Modernization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove verified Zig 0.17-dev cleanup debt in small, independently validated waves.

**Architecture:** Prefer targeted cleanup of proven stale code, dormant tests, empty orphan directories, and documented `refAllDecls` blockers. Do not replace ABI foundation compatibility wrappers unless the current pinned Zig and repository lessons prove the native primitive is ready.

**Tech Stack:** Zig 0.17, POSIX/Darwin shell tools.

---

### Task 1: Preserve ABI Sync Wrappers Until Proven Obsolete

**Files:**
- Inspect: `src/foundation/sync.zig`
- Inspect: `tasks/lessons.md`

- [ ] **Step 1: Confirm whether the current Zig 0.17-dev pin exposes stable native primitives for every ABI sync use case**
- [ ] **Step 2: Keep `foundation.sync` wrappers when they remain required by current lessons and cross-platform gates**

### Task 2: Reconcile Orphaned Feature Artifacts

**Files:**
- Inspect: `src/features/`
- Inspect: `src/features/core/feature_catalog.zig`
- Inspect: `build/validation.zig`
- Action: Remove empty legacy orphan directories only after confirming no tracked files remain.

- [ ] **Step 1: Audit feature catalog vs. `src/features`**
- [ ] **Step 2: Remove empty orphan directories or register live features with matching mod/stub parity**
- [ ] **Step 3: Validate with `./build.sh typecheck --summary all` and parity when public surfaces move**

### Task 3: Refresh Stale Cleanup Documentation

**Files:**
- Modify: cleanup plan docs only unless source comments are proven stale by current code

- [ ] **Step 1: Supersede stale instructions that conflict with current ABI lessons**
- [ ] **Step 2: Keep historical compatibility comments when they explain active Zig 0.17-dev constraints**

---

Plan complete and saved to `docs/superpowers/plans/2026-04-28-codebase-cleanup.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
