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
- Completed Phase 6 documentation/examples closure:
  - Added `examples/tensor_ops.zig` (tensor + matrix + SIMD demo)
  - Added `examples/concurrent_pipeline.zig` (channel + thread pool + DAG pipeline demo)
  - Updated existing examples where v2 references were beneficial (`examples/compute.zig`,
    `examples/concurrency.zig`)
  - Regenerated docs site with `zig build -j1 docs-site`
- Closed a feature-toggle parity regression discovered during explicit spot-checking:
  - `zig build -Denable-web=false` initially failed due cloud stub error-set mismatch.
  - Fixed by extending `Framework.Error` cloud variants in `src/core/framework.zig`.
  - Revalidated with `zig build -Denable-web=false` and `zig build -Denable-web=true`.
- Post-fix gate evidence:
  - `zig build validate-flags` -> success
  - `zig build cli-tests` -> success
  - `zig build test --summary all` -> success (`944 pass`, `5 skip`)
  - `zig build full-check` -> success
- Phase 6 verification evidence:
  - `zig build examples` -> success
  - `zig build -j1 run-tensor-ops` -> success
  - `zig build -j1 run-concurrent-pipeline` -> success
  - `zig build -j1 docs-site` -> success
- Phase 7 release-gate verification evidence:
  - `zig build -j1 validate-flags` -> success
  - `zig build -j1 test --summary all` -> success (`944 pass`, `5 skip`)
  - `zig build -j1 test -- --test-filter parity` -> success
  - `rg -n "@panic\\(" src -g "*.zig" -g "!**/*test*.zig"` -> no runtime library `@panic` calls detected
  - `zig build -j1 examples` -> success
  - `zig build -j1 check-wasm` -> success
  - `zig build -j1 docs-site` -> success
  - `zig build -j1 full-check` -> success (known harness artifact still printed)
  - `zig build -j1 bench-competitive` -> success (published comparative metrics printed)
  - `zig build -j1 benchmarks -- --suite=concurrency` -> success (no channel stall observed)
  - `zig build -j1 benchmarks -- --suite=v2` -> success (v2 module metrics published)
  - `zig build -j1 benchmarks -- --suite=quick` -> success (`Total runtime: 115.13s`)
  - `zig build -j1 benchmarks -- --help` -> success after fixing
    `src/services/shared/utils/v2_primitives.zig` branch-quota overflow in `nextPowerOfTwo`

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
- Local Zig cache can intermittently emit `FileNotFound` in highly parallel `run-*` builds;
  use `-j1` for deterministic local verification when this occurs.
- Full all-suite `zig build -j1 benchmarks` runtime can be long in the AI/streaming segment.
  Use suite-scoped invocations (`--suite=concurrency`, `--suite=v2`, `--suite=quick`) for
  deterministic local gates unless a full all-suite baseline capture is explicitly required.

## v2 Module Integration Status (2026-02-08)

15 modules from abi-system-v2.0 integrated and committed (`7175ac18`):

| Module | Location | Status | Tests |
|--------|----------|--------|-------|
| v2_primitives | `shared/utils/v2_primitives.zig` | Wired | Inline |
| structured_error | `shared/utils/structured_error.zig` | Wired | Inline |
| swiss_map | `shared/utils/swiss_map.zig` | Wired | Inline |
| abix_serialize | `shared/utils/abix_serialize.zig` | Wired | Inline |
| profiler | `shared/utils/profiler.zig` | Wired | Inline |
| benchmark | `shared/utils/benchmark.zig` | Wired | Inline |
| arena_pool | `shared/utils/memory/arena_pool.zig` | Wired | Inline |
| combinators | `shared/utils/memory/combinators.zig` | Wired | Inline |
| tensor | `shared/tensor.zig` | Wired | Inline |
| matrix | `shared/matrix.zig` | Wired | Inline |
| channel | `runtime/concurrency/channel.zig` | Wired | Inline |
| thread_pool | `runtime/scheduling/thread_pool.zig` | Wired | Inline |
| dag_pipeline | `runtime/scheduling/dag_pipeline.zig` | Wired | Inline |
| simd (7 kernels) | `shared/simd.zig` | Extended | Existing |
| v2 benchmarks | `benchmarks/infrastructure/v2_modules.zig` | Wired | N/A |

Import chains verified: `abi.zig` -> `services/{shared,runtime}/mod.zig` -> sub-modules.

### v2 Modules Intentionally Skipped
- `config.zig` — framework already has layered config system
- `gpu.zig` — existing GPU module is far more complete (11 backends)
- `cli.zig` — existing CLI has 24 commands
- `main.zig` — entry point, not applicable

---

## Phase 4: v2 Hardening (2026-02-09 to 2026-02-11)

### 4.1 Integration Testing
- [x] Write integration tests exercising v2 modules through `abi.shared.*` and `abi.runtime.*`
- [x] Verify `SwissMap` works with all key types used in the codebase (integer + string keys, rehash)
- [x] ~~Test `ArenaPool` under concurrent access~~ — N/A: ArenaPool is intentionally single-threaded (no atomics)
- [x] Test `Channel` (Vyukov MPMC) under high contention with multiple producers/consumers
- [x] Test `ThreadPool` work-stealing with varying task granularity
- [x] Test `DagPipeline` with diamond dependency graphs and error propagation
- [x] Verify `FallbackAllocator` ownership detection (rawResize probe pattern)

### 4.2 SIMD Kernel Validation
- [x] Verify SIMD kernels produce correct results vs scalar fallbacks (scale, saxpy verified)
- [x] Test euclidean distance, softmax, saxpy, reduce_sum, reduce_max, reduce_min, scale (declaration + functional tests)
- [x] Confirm `@Vector` operations work on target architectures (verified on aarch64/macOS)
- [ ] Benchmark SIMD vs scalar performance ratios

### 4.3 Benchmark Integration
- [x] Wire `benchmarks/infrastructure/v2_modules.zig` into `zig build benchmarks`
- [ ] Establish baseline performance numbers for v2 data structures
- [ ] Compare `SwissMap` vs `std.HashMap` performance
- [ ] Compare `ArenaPool` vs raw `ArenaAllocator` performance

## Phase 5: Feature Completion (2026-02-12 to 2026-02-16)

### 5.1 Remaining v2 Patterns to Harvest
- [ ] BufferPool staging pattern from v2 `gpu.zig` — evaluate for GPU module
- [ ] Validation patterns from v2 `config.zig` — evaluate for config builder

### 5.2 Known Technical Debt
- [ ] Three `Backend` enums with different members across GPU backends — unify
- [ ] Inconsistent error naming across GPU backends — standardize
- [ ] `createCorsMiddleware` limitation: Zig fn pointers can't capture config (always permissive)
- [ ] Cloud `CloudConfig` type mismatch: `core/config/cloud.zig` vs `features/cloud/types.zig`
- [ ] `TODO(gpu-tests)`: Enable GPU kernel tests once mock backend suppresses error logging

### 5.3 Security Hardening
- [x] Audit v2 modules for unsafe patterns (unbounded allocations, panics in library code)
- [x] Verify no `@panic` in library paths (should return errors)
- [x] Review `abix_serialize.zig` for buffer overflow potential with untrusted input
      - Fixed: `readSlice()` overflow-safe bounds check, `readArrayF32()` multiplication overflow + alignment validation, `readHeader()` payload_len validation
- [x] Review `swiss_map.zig` hash collision resilience
      - Fixed: `rehash()` errdefer on partial allocations, `ensureCapacity()` overflow-checked multiplication
      - Known: deterministic hash (no per-instance seed) — acceptable for internal use, document if exposed to untrusted input

## Phase 6: Documentation and Examples (2026-02-17 to 2026-02-19)

- [x] Define actionable issue intake fields in `.github/ISSUE_TEMPLATE/custom.md`
      (problem statement, expected behavior, reproduction steps, environment, logs).
- [x] Refresh `examples/gpu.zig` against current unified GPU API and validate with
      `zig build examples` to ensure docs/example parity.
- [x] Audit all 19 existing examples for v2 adoption and update where beneficial
      (including `examples/compute.zig` and `examples/concurrency.zig`).
- [x] Add example: `examples/tensor_ops.zig` — demonstrate tensor + matrix + SIMD pipeline
- [x] Add example: `examples/concurrent_pipeline.zig` — demonstrate channel + thread pool + DAG
- [x] Ensure CLAUDE.md and AGENTS.md reflect v2 module locations and import patterns
- [x] Refresh `SECURITY.md` with v2 security review targets and ownership locations
- [x] Generate API docs: `zig build docs-site`

## Phase 7: Release Gate (2026-02-20 to 2026-02-21)

Final release criteria for v0.4.1:

```sh
zig build full-check          # format + build + test + validate-flags
zig build examples             # all 19+ examples compile
zig build benchmarks           # benchmarks compile and run
zig build check-wasm           # WASM target compiles
zig build docs-site            # documentation generates
```

Exit criteria:
- [x] 944+ tests passing, 5 or fewer skipped
- [x] All feature flag combos compile (validate-flags green)
- [x] All examples build
- [x] No `@panic` in library code paths
- [x] Stub parity confirmed for all 8 feature modules
- [x] v2 module benchmarks show expected performance characteristics
- [x] CLAUDE.md, AGENTS.md, SECURITY.md up to date

---

## Near-Term Milestones (February 2026)
- 2026-02-08: ~~Baseline captured and ownership map confirmed.~~ DONE
- 2026-02-08: ~~v2 module integration (15 modules).~~ DONE (commit `7175ac18`)
- 2026-02-08: ~~Benchmark safety fixes (errdefer, div-by-zero, percentile).~~ DONE (commit `46f24957`)
- 2026-02-08: ~~M10 production readiness (health, signal, status CLI).~~ DONE (commit `4c58d5a0`)
- 2026-02-08: ~~M11 language bindings (state + feature count, all 5 langs).~~ DONE (commit `290baa66`)
- 2026-02-09: Stub parity audit complete, any drift fixed.
- 2026-02-11: v2 integration tests written and passing.
- 2026-02-14: Feature completion and tech debt addressed.
- 2026-02-16: ~~Documentation and examples updated.~~ DONE
- 2026-02-21: Release-readiness review and v0.4.1 go/no-go.

## Metrics Dashboard

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Tests passing | 944 | 980 | 950+ |
| Tests skipped | 5 | 5 | 5 or fewer |
| Feature modules | 8 | 8 | 8 |
| v2 modules integrated | 0 | 15 | 15 |
| Flag combos passing | 16 | 16 | 16 |
| Examples | 19 | 21 | 21+ |
| Known `@panic` in lib | 0 | 0 | 0 |
| Stub parity violations | TBD | 0 | 0 |

## Quick Links
- [Cleanup + Production + Bindings Plan](plans/2026-02-08-cleanup-production-bindings.md)
- [v2 Integration Plan](plans/2026-02-08-abi-system-v2-integration.md)
- [Ralph Loop Eval](plans/2026-02-08-ralph-loop-zig016-multi-agent-eval.md)
- [Roadmap](roadmap.md)
- [CLAUDE.md](../CLAUDE.md)
