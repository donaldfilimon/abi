# ABI Post-Merge Verification & Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to perform these verification and cleanup tasks.

**Goal:** Verify the integrity of the consolidated ABI repository and remove leftover backup artifacts.

**Architecture:** Systematic verification of the build state, followed by removal of the temporary backup directory.

**Tech Stack:** Zig 0.17-dev (master).

---

### Task 1: Repository Integrity Check

- [ ] **Step 1: Run full clean build**
Run: `cd abi && ./build.sh clean && ./build.sh check`
Expected: SUCCESS

- [ ] **Step 2: Verify parity check**
Run: `cd abi && zig build check-parity`
Expected: SUCCESS (API parity between mod/stub verified)

### Task 2: Artifact Cleanup

- [ ] **Step 1: Remove backup directory**
Run: `rm -rf /Users/donaldfilimon/abi_backups`
(Note: Confirmed that all relevant worktrees are merged into /abi)

- [ ] **Step 2: Final status report**
Run: `git status` (inside /abi)
Expected: Clean working state (or expected tracked changes only)

---
