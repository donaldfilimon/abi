# Codebase Cleanup and 0.17 Modernization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy Zig 0.16 technical debt and synchronize the feature catalog with the build system.

**Architecture:** Systematic removal of 0.16 compatibility wrappers and reconciling orphaned features.

**Tech Stack:** Zig 0.17, POSIX/Darwin shell tools.

---

### Task 1: Remove Zig 0.16 Sync Workarounds

**Files:**
- Modify: `src/foundation/sync.zig`

- [ ] **Step 1: Replace legacy sync workarounds with native Zig 0.17 primitives**
- [ ] **Step 2: Commit cleanup**

### Task 2: Reconcile Orphaned Features

**Files:**
- Modify: `build/validation.zig`
- Action: Inspect `src/features/` and move orphaned folders to `src/features/legacy_orphans/` or register them.

- [ ] **Step 1: Audit feature catalog vs. `src/features`**
- [ ] **Step 2: Update `build/validation.zig`**
- [ ] **Step 3: Commit structural changes**

### Task 3: Refresh Legacy Documentation

**Files:**
- Modify: `src/features/ai/abbey/engine.zig`

- [ ] **Step 1: Purge 0.16 comments and stale docstrings**
- [ ] **Step 2: Commit documentation cleanup**

---

Plan complete and saved to `docs/superpowers/plans/2026-04-28-codebase-cleanup.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
