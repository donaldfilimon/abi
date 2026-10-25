# Governed Process Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task.

**Goal:** Deliver a startup-pinned process executor and real OS evidence bridge
through the existing exact approval host.
**Architecture:** Closed WorkingDirectory operation over a private Unix bounded
supervisor, with immutable plan/source binding and an opaque single-use receipt.
**Tech Stack:** Rust 1.99.1, std, existing capability/host contracts, Unix libc.
**Spec:** `docs/superpowers/specs/2026-10-23-governed-process-design.md`.

## Global Constraints

Canonical dirty main; preserve others' edits. No commits, CI, deployment, live
accounts or user store access. Frozen CLI14/MCP12 and existing encryption unchanged.
Use tools/cargo.sh, locked resolution, tests stdin EOF; at most 1,000 lines/Rust file.
All exact values, signatures and evidence rules are in the spec and bind tasks.
Unsupported platforms refuse; no sandbox/general containment claim.

## Review Focus

- Cwd replacement after observation must not redirect the retained directory FD.
- Child leader exit cannot cause stale-PGID group kill or inherited-pipe hangs.
- Aggregate stdout/stderr or JSON expansion cannot evade output ceilings.
- Forged/upstream process facts or reused receipt cannot prove postconditions.
- Fresh authority/local approval cannot be bypassed by source/plan mismatch.

### Task 1: Process adapter and supervisor

**Files:** Create `crates/abi-agent-host/src/platform_process.rs` and focused
`platform_process/{plan,unix,facts,tests}.rs` modules as needed. Modify only
host lib export and target-Unix libc dependency; lock only if resolver needs it.
**Consumes:** Existing CompiledSource, GovernedLiveExecutor, GovernedFactProvider,
Clock, Request/PreparedExecution and ToolExecutionContext contracts.
**Produces:** ProcessLimits, ProcessPlan, GovernedProcessExecutor,
ProcessObservationHandle, GovernedProcessFacts with exact spec methods.

- [ ] Write core/unit/process helper tests for the spec acceptance first; run
  them, preserving actual RED failures separately from compile/harness failures.
- [ ] Implement sealed plan binding, OS observations, bounded runner and receipts.
- [ ] Run `./tools/cargo.sh test --locked -p abi-agent-host --lib < /dev/null`;
  expected all existing/new cases pass. Run package Clippy and fmt, size gate.
- [ ] Independent spec/quality review and resolve findings. No commit per ruling.

### Task 2: Governed host/local approval acceptance

**Files:** Create `crates/abi-agent-host/tests/governed_process.rs` and modules
under `tests/governed_process/` only. No existing Task8 or sibling files.
**Consumes:** Task1 exact public APIs; existing host registration/A1 lifecycle.
**Produces:** Actual pwd approval/refusal/replay and postcondition acceptance.

- [ ] Write exact A1 pending/approved/denied/expired/changed-binding tests using
  scratch approval/lifecycle state and real process execution/counters.
- [ ] Verify output plus actual trusted process predicates, no second launch on
  replay and no launches on all refusal paths in the spec.
- [ ] Run `./tools/cargo.sh test --locked -p abi-agent-host < /dev/null` and
  warning-denied package Clippy; expected zero failed/ignored new cases.
- [ ] Independent review; one final complete slice review after both tasks.

### Task 3: Qualification and honest ledger

- [ ] Full `./tools/check.sh`, actual dependency scan, docs-policy tests,
  `.agents/skills/docs-validate/validate.sh`, `git diff --check`.
- [ ] Record exact source/modes/log digests and failed receipts. Update only
  process-specific spec/plan/evidence and tasks entry; preserve other chat's
  Task8 documents. Parent Task7 remains open for all other adapters/platforms.
