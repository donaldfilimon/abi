# Phase 3: Zig 0.17 Perfection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaustively update the ABI codebase to align with Zig 0.17-dev idioms and ensure all files meet "perfection" standards for production.

**Architecture:** Global codebase sweep using targeted regex for known hotspots. Focus on ArrayListUnmanaged, time functions, manual buffer safety, and import path consistency.

**Tech Stack:** Zig 0.17-dev, standard library, ABI foundation.

---

### Task 1: ArrayListUnmanaged Perfection

**Files:**
- Modify: `src/**/*.zig` (multiple files)

- [ ] **Step 1: Identify all ArrayListUnmanaged initializations missing `.empty`**

Run: `grep -r "ArrayListUnmanaged" abi/src | grep -v "\.empty"`

- [ ] **Step 2: Update initializations to use `.empty`**

```zig
// Before
var list = std.ArrayListUnmanaged(u8){};
// After
var list = std.ArrayListUnmanaged(u8).empty;
```

- [ ] **Step 3: Verify build**

Run: `./build.sh check`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "perfection: migrate ArrayListUnmanaged to .empty idiom"
```

### Task 2: Time Function Migration

**Files:**
- Modify: `src/**/*.zig` (multiple files)

- [ ] **Step 1: Identify direct std.time usage for timestamps**

Run: `grep -r "std\.time\." abi/src` (looking for milliTimestamp, nanoTimestamp, etc.)

- [ ] **Step 2: Replace with foundation.time.unixMs() or equivalent**

```zig
// Before
const ts = std.time.milliTimestamp();
// After
const ts = foundation.time.unixMs();
```

- [ ] **Step 3: Update time constants if necessary**

Ensure `std.time.ns_per_ms` etc. are used consistently or via `foundation.time`.

- [ ] **Step 4: Verify build**

Run: `./build.sh check`

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "perfection: migrate to foundation.time wrappers"
```

### Task 3: Manual Buffer & "Bounded" Patterns

**Files:**
- Modify: `src/**/*.zig` (multiple files)

- [ ] **Step 1: Scan for manual buffer/len patterns that replaced BoundedArray**

Run: `grep -r "\[.*\]u8 = undefined" abi/src`

- [ ] **Step 2: Ensure these patterns follow the recommended 0.17 idiom**

Verify that manual buffers are accompanied by an explicit `len: usize` and that no remnants of `std.BoundedArray` remain.

- [ ] **Step 3: Verify build**

Run: `./build.sh check`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "perfection: normalize manual buffer management patterns"
```

### Task 4: Import Path Perfection

**Files:**
- Modify: `src/**/*.zig` (multiple files)

- [ ] **Step 1: Scan for imports missing .zig extension**

Run: `grep -r "@import(\"[^\"]*[^\.][^z][^i][^g]\"" abi/src` (or similar regex)

- [ ] **Step 2: Add .zig extension where missing**

```zig
// Before
const types = @import("types");
// After
const types = @import("types.zig");
```

- [ ] **Step 3: Verify build**

Run: `./build.sh check`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "perfection: ensure all import paths use .zig extension"
```

### Task 5: Global Perfection Review

**Files:**
- Modify: All files in `src/`

- [ ] **Step 1: Run `zig build fix` to normalize formatting**

Run: `cd abi && zig build fix`

- [ ] **Step 2: Run `zig build check-parity`**

Ensure no mod/stub drifts were introduced.

- [ ] **Step 3: Run full test suite**

Run: `./build.sh test --summary all`

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "perfection: final global formatting and parity normalization"
```
