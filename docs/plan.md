---
title: "ABI Multi-Agent Execution Plan"
status: "active"
updated: "2026-02-08"
tags: [planning, execution, zig-0.16, multi-agent]
---
# ABI Multi-Agent Execution Plan

## Current Objective
Deliver a stable ABI baseline on Zig 0.16 with verified feature-gated parity, reproducible
build/test outcomes, and a clear release-readiness decision in February 2026.

## Execution Update (2026-02-08)
- Completed ownership-scoped refactor passes across:
  - `src/features/web/**` (response/request helpers and middleware tests)
  - `src/features/cloud/**` (header normalization for case-insensitive lookup and tests)
  - `src/services/runtime/**` (channel/thread-pool/pipeline cleanup and focused tests)
  - `src/services/shared/**` (utility cleanup and benchmark tests)
- Closed a feature-toggle parity regression discovered during explicit spot-checking:
  - `zig build -Denable-web=false` initially failed due cloud stub error-set mismatch.
  - Fixed by extending `Framework.Error` cloud variants in `src/core/framework.zig`.
  - Revalidated with `zig build -Denable-web=false` and `zig build -Denable-web=true`.
- Post-fix gate evidence:
  - `zig build validate-flags` -> success
  - `zig build cli-tests` -> success
  - `zig build test --summary all` -> success (`944 pass`, `5 skip`)
  - `zig build full-check` -> success

## Assumptions
- Zig toolchain is `0.16.0-dev.2471+e9eadee00` or a compatible newer Zig 0.16 build.
- Public API usage stays on `@import("abi")`; deep internal imports are not relied on.
- Parallel execution is done by explicit file/module ownership per agent.

## Constraints
- Feature-gated parity is required: each changed `src/features/*/mod.zig` and `stub.zig` pair
  must expose matching public signatures and compatible error behavior.
- Every touched feature must compile in both enabled and disabled flag states.
- During parallel execution, formatting must stay ownership-scoped: use `zig fmt <owned-paths>`
  per agent; reserve `zig build full-check` for integration coordinator gates.
- No completion claim without formatting, full tests, flag validation, and CLI smoke checks.

## Multi-Agent Roles and Responsibilities
- **A0 Coordinator**: Ownership: Cross-cutting. Responsibilities: Own phase sequencing,
  conflict resolution, and go/no-go decisions. Outputs: Daily status and final readiness call.
- **A1 Feature Parity**: Ownership: `src/features/**`. Responsibilities: Keep `mod.zig` and
  `stub.zig` API parity and fix flag-conditional compile failures. Outputs: Parity fixes with
  passing toggle builds.
- **A2 Core Runtime**: Ownership: `src/core/**`, `src/services/**`. Responsibilities: Protect
  runtime/config contracts and integration boundaries. Outputs: Stable runtime behavior and
  focused tests.
- **A3 API and CLI**: Ownership: `src/api/**` and CLI surfaces. Responsibilities: Keep command
  behavior/help coherent with implementation. Outputs: Passing CLI smoke and verified help output.
- **A4 Validation**: Ownership: Test and gate execution. Responsibilities: Run verification
  matrix, publish failures with repro commands. Outputs: Final verification checklist and evidence.

## Phased Execution Plan

### Phase 0: Baseline Capture (2026-02-08)
Run once before new changes are merged.

```sh
zig version
zig fmt <owned-paths>
zig build
zig build run -- --help
zig build test --summary all
```

Exit criteria:
- Baseline pass/fail state recorded.
- Existing failures labeled as baseline, not regression.

### Phase 1: Feature-Gated Parity Closure (2026-02-09 to 2026-02-11)
Run for all touched feature areas.
Use the matrix below as the current baseline from `build.zig`; if additional feature flags
exist in a branch, add both `true` and `false` checks for those flags.

```sh
zig build validate-flags
zig build -Denable-ai=true
zig build -Denable-ai=false
zig build -Denable-gpu=true
zig build -Denable-gpu=false
zig build -Denable-database=true
zig build -Denable-database=false
zig build -Denable-network=true
zig build -Denable-network=false
zig build -Denable-web=true
zig build -Denable-web=false
zig build -Denable-profiling=true
zig build -Denable-profiling=false
zig build -Denable-analytics=true
zig build -Denable-analytics=false
```

Exit criteria:
- All touched features compile in both flag states.
- No unresolved `mod.zig` vs `stub.zig` public API drift.

### Phase 2: Integration and Regression Gates (2026-02-12 to 2026-02-14)

```sh
zig build cli-tests
zig build test --summary all
zig build full-check
```

Exit criteria:
- No regression versus baseline behavior.
- Formatting, tests, flag validation, and CLI smoke gates are green together.

### Phase 3: Release Readiness Decision (2026-02-15 to 2026-02-16)

```sh
zig build full-check
```

Exit criteria:
- Final rerun of release gates (including formatting) is green.
- Coordinator issues go/no-go decision with evidence.

## Risk Controls and Rollback Policy
- Keep changes small and isolated to owned modules.
- Re-run the narrowest relevant command set after each merge.
- If parity breaks, stop feature expansion and restore parity first.
- Rollback policy:
  - Revert only the smallest offending commit set.
  - Continue unaffected agent tracks when isolation is clear.
  - If root cause is unclear, roll back to last known green state and reapply incrementally.

## Definition of Done
- Zig 0.16 path is stable for normal and feature-gated builds.
- Feature-gated parity is confirmed on touched modules.
- Full validation matrix passes with no unresolved regressions.
- Plan references remain accurate and current.

## Verification Checklist
- [x] `zig fmt <owned-paths>`
- [ ] Example owned-path formatting: `zig fmt docs/plan.md prompts/*.md`
- [x] `zig build`
- [x] `zig build run -- --help`
- [x] `zig build validate-flags`
- [x] `zig build cli-tests`
- [x] `zig build test --summary all`
- [x] `zig build full-check`
- [x] Spot-check changed features with `-Denable-<feature>=true/false`
- [x] Example feature spot-check: `zig build -Denable-web=false`

## Remaining Risks (As of 2026-02-08)
- The test harness output still prints `failed command ... --listen=-` during
  `zig build test --summary all` / `zig build full-check` even when the build step exits `0`
  and reports `944/949` passing (`5` skipped). Treat as a known harness artifact unless exit
  status changes.

## Near-Term Milestones (February 2026)
- 2026-02-08: Baseline captured and ownership map confirmed.
- 2026-02-10: First full parity pass complete for active feature workstreams.
- 2026-02-12: Validation matrix run and failures triaged with assigned owners.
- 2026-02-14: Shared integration branch reaches green gate state.
- 2026-02-16: Release-readiness review and go/no-go outcome.

## Quick Links
- [Roadmap](roadmap.md)
- [CLAUDE.md](../CLAUDE.md)
